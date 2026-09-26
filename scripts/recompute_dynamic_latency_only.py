from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "baselines"))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "pytorch-a2c-ppo-acktr-gail"))

from a2c_ppo_acktr.model import Policy

from scripts.reevaluate_dynamic_run_shared_protocol import (
    _load_source_args,
)
from src.dynamic_morl.metrics import compute_trace_shift_metrics
from src.dynamic_morl.mopg import _make_eval_env, evaluation

_DYNAMIC_WORKER_STATE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute OW-CMORL trace latency without shared-regime/front reevaluation.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--episode-seeds-json", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    return parser.parse_args()


def _policy_index_from_name(name: str) -> int:
    match = re.search(r"_(\d+)\.pt$", name)
    if match is None:
        raise ValueError(f"unexpected policy name: {name}")
    return int(match.group(1))


def _episode_seeds(source_run_dir: Path, episode_seeds_json: Path | None) -> list[int]:
    if episode_seeds_json is not None:
        payload = json.loads(episode_seeds_json.read_text())
        if isinstance(payload, dict):
            payload = payload.get("shared_eval_episode_seeds", [])
        return [int(seed) for seed in payload]
    summary_path = source_run_dir / "final" / "shared_eval_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing shared eval summary: {summary_path}")
    payload = json.loads(summary_path.read_text())
    seeds = payload.get("shared_eval_episode_seeds", [])
    if not seeds:
        raise ValueError(f"missing shared_eval_episode_seeds in {summary_path}")
    return [int(seed) for seed in seeds]


def _policy_indices(source_run_dir: Path) -> list[int]:
    final_dir = source_run_dir / "final"
    indices = sorted(_policy_index_from_name(path.name) for path in final_dir.glob("EP_policy_*.pt"))
    if not indices:
        raise FileNotFoundError(f"no EP policies found in {final_dir}")
    return indices


def _load_single_sample(source_run_dir: Path, policy_idx: int, run_args: argparse.Namespace) -> SimpleNamespace:
    final_dir = source_run_dir / "final"
    policy_path = final_dir / f"EP_policy_{policy_idx}.pt"
    env_param_path = final_dir / f"EP_env_params_{policy_idx}.pkl"
    if not policy_path.exists():
        raise FileNotFoundError(f"missing policy: {policy_path}")
    if not env_param_path.exists():
        raise FileNotFoundError(f"missing env params: {env_param_path}")

    actor_critic = Policy(
        _DYNAMIC_WORKER_STATE["observation_shape"],
        _DYNAMIC_WORKER_STATE["action_space"],
        base_kwargs={"layernorm": getattr(run_args, "layernorm", False)},
        obj_num=run_args.obj_num,
    ).double()
    actor_critic.load_state_dict(torch.load(policy_path, map_location="cpu"))
    with env_param_path.open("rb") as fp:
        env_params = pickle.load(fp)
    return SimpleNamespace(env_params=env_params, actor_critic=actor_critic)


def _init_worker(source_run_dir: str, episode_seeds: list[int]) -> None:
    source_path = Path(source_run_dir)
    _source_args, run_args = _load_source_args(source_path)
    torch.set_default_dtype(torch.float64)
    temp_env = _make_eval_env(run_args)
    try:
        observation_shape = temp_env.observation_space.shape
        action_space = temp_env.action_space
    finally:
        temp_env.close()
    _DYNAMIC_WORKER_STATE.clear()
    _DYNAMIC_WORKER_STATE.update(
        {
            "source_run_dir": source_path,
            "run_args": run_args,
            "episode_seeds": [int(seed) for seed in episode_seeds],
            "observation_shape": observation_shape,
            "action_space": action_space,
        }
    )


def _evaluate_one(policy_idx: int) -> dict[str, Any]:
    source_path = Path(_DYNAMIC_WORKER_STATE["source_run_dir"])
    run_args = _DYNAMIC_WORKER_STATE["run_args"]
    sample = _load_single_sample(source_path, policy_idx, run_args)
    objs, trace = evaluation(
        run_args,
        sample,
        return_trace=True,
        env_name=run_args.env_name,
        env_kwargs={},
        episode_seeds=_DYNAMIC_WORKER_STATE["episode_seeds"],
    )
    trace_metrics = compute_trace_shift_metrics(
        trace,
        run_args.eval_delta_weight,
        recovery_window=run_args.trace_recovery_window,
    )
    return {
        "policy_idx": int(policy_idx),
        "objs": np.asarray(objs, dtype=np.float64).tolist(),
        "trace_metrics": {key: float(value) for key, value in trace_metrics.items()},
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    rows = list(payload.get("per_policy_trace_metrics", []))
    lines = [
        "# Dynamic Latency Recompute",
        "",
        f"- source: `{payload['source_run_dir']}`",
        f"- env: `{payload['env_name']}`",
        f"- policies: `{payload['policy_count']}`",
        f"- mean Lat: `{payload['trace_recovery_latency_mean']:.6f}`",
        f"- std Lat: `{payload['trace_recovery_latency_std']:.6f}`",
        "",
        "| policy_idx | trace_recovery_latency | trace_shift_regret | trace_shift_count |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["trace_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row["policy_idx"])),
                    f"{float(metrics['trace_recovery_latency']):.6f}",
                    f"{float(metrics['trace_shift_regret']):.6f}",
                    f"{float(metrics['trace_shift_count']):.0f}",
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    episode_seeds = _episode_seeds(source_run_dir, args.episode_seeds_json.resolve() if args.episode_seeds_json else None)
    policy_indices = _policy_indices(source_run_dir)
    _source_args, run_args = _load_source_args(source_run_dir)

    rows: list[dict[str, Any]] = []
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.workers),
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(str(source_run_dir), episode_seeds),
    ) as executor:
        future_to_idx = {
            executor.submit(_evaluate_one, int(policy_idx)): int(policy_idx)
            for policy_idx in policy_indices
        }
        total = len(future_to_idx)
        completed = 0
        for future in as_completed(future_to_idx):
            row = future.result()
            rows.append(row)
            completed += 1
            if completed % 4 == 0 or completed == total:
                print(f"[lat-only] {run_args.env_name} policy {completed}/{total}", flush=True)

    rows.sort(key=lambda row: int(row["policy_idx"]))
    latencies = np.asarray(
        [float(row["trace_metrics"]["trace_recovery_latency"]) for row in rows],
        dtype=np.float64,
    )
    payload = {
        "source_run_dir": str(source_run_dir),
        "env_name": str(run_args.env_name),
        "policy_count": int(len(rows)),
        "shared_eval_episode_seeds": [int(seed) for seed in episode_seeds],
        "trace_recovery_latency_mean": float(latencies.mean()),
        "trace_recovery_latency_std": float(latencies.std(ddof=0)),
        "per_policy_trace_metrics": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    if args.out_md is not None:
        _write_md(args.out_md.resolve(), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
