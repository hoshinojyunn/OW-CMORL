from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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
sys.path.insert(0, str(PROJECT_ROOT / "CAPQL"))
sys.path.insert(0, str(PROJECT_ROOT / "Q-Pensieve"))

from scripts.run_capql_dynamic_baseline import (
    build_config as build_capql_config,
    infer_hidden_size_from_checkpoint,
    install_capql_stubs,
)
from scripts.run_pgmorl_dynamic_baseline import (
    PGMORL_ROOT,
    build_pg_args,
)
from scripts.run_qpensieve_dynamic_baseline import install_qpensieve_stubs
from src.baseline_envs import get_env_spec, make_scalar_env, make_vector_env, register_baseline_envs
from src.baseline_eval import (
    build_old_gym_scalar_env_from_name,
    build_old_gym_vector_env_from_name,
    evaluate_old_gym_policy,
)
from src.dynamic_morl.metrics import compute_trace_shift_metrics
from src.morl_ca_baseline import MORLCABaseline, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute baseline trace latency without regime/front reevaluation.")
    parser.add_argument("--method", choices=["capql", "pgmorl", "q_pensieve", "morlca"], required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) // 2)))
    return parser.parse_args()


def _load_source_payload(source_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = source_run_dir / "summary.json"
    config_path = source_run_dir / "config.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing config: {config_path}")
    return json.loads(summary_path.read_text()), json.loads(config_path.read_text())


def _episode_seeds(summary: dict[str, Any]) -> list[int]:
    seeds = list(summary.get("shared_eval_episode_seeds", []))
    if not seeds:
        raise ValueError("missing shared_eval_episode_seeds")
    return [int(seed) for seed in seeds]


def _capql_policy_fn(source_run_dir: Path, config: dict[str, Any]):
    register_baseline_envs()
    install_capql_stubs()
    spec = get_env_spec(str(config["env_key"]))
    env_kwargs = dict(config.get("env_kwargs", {}))
    args = argparse.Namespace(**config)
    hidden_size = infer_hidden_size_from_checkpoint(source_run_dir / "model", int(config.get("hidden_size", 256)))
    setattr(args, "hidden_size", int(hidden_size))
    config_obj = build_capql_config(args, spec.vector_env_id)
    train_env = make_vector_env(str(config["env_key"]), seed=int(config["seed"]), env_kwargs=env_kwargs)
    from training_pipeline import TrainingProcess

    trainer = TrainingProcess(config_obj, train_env)
    trainer.writer.close()
    trainer.agent.policy.load_state_dict(torch.load(source_run_dir / "model" / "policy.pt", map_location="cpu"))
    trainer.agent.critic.load_state_dict(torch.load(source_run_dir / "model" / "critic.pt", map_location="cpu"))
    train_env.close()
    return spec, env_kwargs, (lambda obs, pref: trainer.agent.select_action(obs, pref, evaluate=True))


def _qpensieve_policy_fn(source_run_dir: Path, config: dict[str, Any]):
    register_baseline_envs()
    install_qpensieve_stubs()
    spec = get_env_spec(str(config["env_key"]))
    env_kwargs = dict(config.get("env_kwargs", {}))
    from agent import SacAgent

    env = make_vector_env(str(config["env_key"]), seed=int(config["seed"]), env_kwargs=env_kwargs)
    torch.set_num_threads(1)
    agent = SacAgent(
        env=env,
        log_dir=str(source_run_dir),
        num_steps=int(config.get("total_timesteps", 200000)),
        batch_size=int(config.get("batch_size", 256)),
        lr=3e-4,
        hidden_units=list(config.get("hidden_size", [256, 256])),
        memory_size=max(50000, int(config.get("total_timesteps", 200000)) * 2),
        prefer_num=int(config.get("prefer_num", 4)),
        buf_num=0,
        gamma=0.99,
        tau=0.005,
        entropy_tuning=True,
        ent_coef=0.2,
        multi_step=1,
        per=False,
        grad_clip=None,
        updates_per_step=int(config.get("updates_per_step", 1)),
        start_steps=min(int(config.get("start_steps", 5000)), max(128, int(config.get("total_timesteps", 200000)) // 5)),
        log_interval=10,
        target_update_interval=1,
        eval_interval=max(1000000, int(config.get("total_timesteps", 200000)) * 100),
        cuda=False,
        seed=int(config["seed"]),
        cuda_device=0,
        q_frequency=int(config.get("q_frequency", 1000)),
        model_saved_step=max(1, int(config.get("total_timesteps", 200000))),
    )
    model_dir = source_run_dir / "model"
    agent.policy.load(str(model_dir / "policy_final.pth"))
    agent.critic.load(str(model_dir / "critic_final.pth"))
    agent.critic_target.load(str(model_dir / "critic_target.pth"))
    env.close()
    return spec, env_kwargs, (lambda obs, pref: agent.exploit(obs, pref))


def _morlca_policy_fn(source_run_dir: Path, config: dict[str, Any]):
    register_baseline_envs()
    set_global_seed(int(config["seed"]))
    spec = get_env_spec(str(config["env_key"]))
    env_kwargs = dict(config.get("env_kwargs", {}))
    env = make_vector_env(str(config["env_key"]), seed=int(config["seed"]), env_kwargs=env_kwargs)
    model = MORLCABaseline(
        obs_dim=int(env.observation_space.shape[0]),
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        obj_num=int(spec.obj_num),
        hidden_dim=int(config.get("hidden_size", 256)),
        gamma=float(config.get("gamma", 0.99)),
        tau=float(config.get("tau", 0.005)),
        actor_lr=float(config.get("actor_lr", 3e-4)),
        critic_lr=float(config.get("critic_lr", 3e-4)),
        alpha_lr=float(config.get("alpha_lr", 3e-4)),
        alpha_init=float(config.get("alpha_init", 0.2)),
        batch_size=int(config.get("batch_size", 256)),
        replay_size=max(50000, int(config.get("total_timesteps", 200000)) * 2),
        start_steps=min(int(config.get("start_steps", 5000)), max(128, int(config.get("total_timesteps", 200000)) // 5)),
        updates_per_step=int(config.get("updates_per_step", 1)),
        aow_aux_coef=float(config.get("aow_aux_coef", 0.2)),
        eval_pref_bias_scale=float(config.get("eval_pref_bias_scale", 0.75)),
        device="cpu",
    )
    model.load(source_run_dir / "model" / "morlca.pt")
    env.close()
    return spec, env_kwargs, (lambda obs, pref: model.act(obs, pref=np.asarray(pref, dtype=np.float32), deterministic=True))


def _pgmorl_policy_fn(source_run_dir: Path, config: dict[str, Any], policy_idx: int):
    register_baseline_envs()
    sys.path.insert(1, str(PGMORL_ROOT))
    sys.path.insert(1, str(PGMORL_ROOT / "morl"))
    sys.path.insert(1, str(PGMORL_ROOT / "externals" / "baselines"))
    sys.path.insert(1, str(PGMORL_ROOT / "externals" / "pytorch-a2c-ppo-acktr-gail"))
    from a2c_ppo_acktr.model import Policy

    spec = get_env_spec(str(config["env_key"]))
    env_kwargs = dict(config.get("env_kwargs", {}))
    args = argparse.Namespace(**config)
    pg_args = build_pg_args(args, spec.scalar_env_id, spec.obj_num)
    env = make_scalar_env(spec.env_key, seed=int(config["seed"]))
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    env.close()

    final_dir = source_run_dir / "final"
    env_params_path = final_dir / f"EP_env_params_{policy_idx}.pkl"
    with env_params_path.open("rb") as fp:
        env_params = pickle.load(fp)
    actor_critic = Policy(
        obs_shape,
        action_space,
        base_kwargs={"layernorm": bool(pg_args.layernorm)},
        obj_num=spec.obj_num,
    )
    actor_critic.to(torch.device("cpu")).double()
    actor_critic.load_state_dict(torch.load(final_dir / f"EP_policy_{policy_idx}.pt", map_location="cpu"))
    actor_critic.eval()
    ob_rms = env_params.get("ob_rms")

    def _policy_fn(obs):
        ob = np.asarray(obs, dtype=np.float64)
        if bool(pg_args.ob_rms) and ob_rms is not None:
            ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
        with torch.no_grad():
            _, action, _, _ = actor_critic.act(
                torch.as_tensor(ob, dtype=torch.float64).unsqueeze(0),
                None,
                None,
                deterministic=True,
            )
        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)

    return spec, env_kwargs, _policy_fn


def _evaluate_conditioned_one(
    method: str,
    source_run_dir: str,
    preference: list[float],
) -> dict[str, Any]:
    source_path = Path(source_run_dir)
    summary, config = _load_source_payload(source_path)
    episode_seeds = _episode_seeds(summary)
    pref = [float(x) for x in preference]

    if method == "capql":
        spec, env_kwargs, policy_fn = _capql_policy_fn(source_path, config)
    elif method == "q_pensieve":
        spec, env_kwargs, policy_fn = _qpensieve_policy_fn(source_path, config)
    elif method == "morlca":
        spec, env_kwargs, policy_fn = _morlca_policy_fn(source_path, config)
    else:
        raise ValueError(f"unsupported conditioned method: {method}")

    _objs, dyn_trace = evaluate_old_gym_policy(
        lambda: build_old_gym_vector_env_from_name(
            spec.dynamic_env_name,
            seed=int(config["seed"]),
            env_kwargs=env_kwargs,
        ),
        lambda obs: policy_fn(obs, np.asarray(pref, dtype=np.float32)),
        episodes=len(episode_seeds),
        episode_seeds=episode_seeds,
        record_trace=True,
    )
    trace_metrics = compute_trace_shift_metrics(
        dyn_trace,
        float(config.get("eval_delta_weight", 0.5)),
        recovery_window=int(config.get("trace_recovery_window", 12)),
    )
    return {
        "preference": pref,
        "trace_metrics": {key: float(value) for key, value in trace_metrics.items()},
    }


def _evaluate_pgmorl_one(source_run_dir: str, policy_idx: int) -> dict[str, Any]:
    source_path = Path(source_run_dir)
    summary, config = _load_source_payload(source_path)
    episode_seeds = _episode_seeds(summary)
    spec, env_kwargs, policy_fn = _pgmorl_policy_fn(source_path, config, int(policy_idx))
    _objs, dyn_trace = evaluate_old_gym_policy(
        lambda: build_old_gym_scalar_env_from_name(
            spec.dynamic_env_name,
            seed=int(config["seed"]),
            env_kwargs=env_kwargs,
        ),
        policy_fn,
        episodes=len(episode_seeds),
        episode_seeds=episode_seeds,
        record_trace=True,
    )
    trace_metrics = compute_trace_shift_metrics(
        dyn_trace,
        float(config.get("eval_delta_weight", 0.5)),
        recovery_window=int(config.get("trace_recovery_window", 12)),
    )
    return {
        "policy_idx": int(policy_idx),
        "trace_metrics": {key: float(value) for key, value in trace_metrics.items()},
    }


def _write_md(path: Path, payload: dict[str, Any]) -> None:
    rows = list(payload.get("rows", []))
    lines = [
        "# Baseline Latency Recompute",
        "",
        f"- method: `{payload['method']}`",
        f"- env: `{payload['env_key']}`",
        f"- source: `{payload['source_run_dir']}`",
        f"- count: `{payload['row_count']}`",
        f"- mean Lat: `{payload['trace_recovery_latency_mean']:.6f}`",
        f"- std Lat: `{payload['trace_recovery_latency_std']:.6f}`",
        "",
    ]
    if payload["method"] == "pgmorl":
        lines.append("| policy_idx | trace_recovery_latency | trace_shift_regret | trace_shift_count |")
    else:
        lines.append("| preference | trace_recovery_latency | trace_shift_regret | trace_shift_count |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows:
        metrics = row["trace_metrics"]
        label = row.get("policy_idx", row.get("preference"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(label),
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
    summary, _config = _load_source_payload(source_run_dir)

    mp_context = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    if args.method == "pgmorl":
        tasks = [int(row.get("policy_id", idx)) for idx, row in enumerate(summary.get("per_policy", []))]
        if not tasks:
            final_dir = source_run_dir / "final"
            tasks = sorted(int(path.stem.split("_")[-1]) for path in final_dir.glob("EP_policy_*.pt"))
        task_fn = _evaluate_pgmorl_one
        task_args = [(str(source_run_dir), int(policy_idx)) for policy_idx in tasks]
    else:
        tasks = [list(pref) for pref in summary.get("preferences", [])]
        if not tasks:
            raise ValueError("missing preferences in source summary")
        task_fn = _evaluate_conditioned_one
        task_args = [(args.method, str(source_run_dir), pref) for pref in tasks]

    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=mp_context) as executor:
        future_map = {executor.submit(task_fn, *task_arg): task_arg for task_arg in task_args}
        total = len(future_map)
        completed = 0
        for future in as_completed(future_map):
            rows.append(future.result())
            completed += 1
            if completed % 4 == 0 or completed == total:
                print(f"[lat-only] {args.method} {summary['env_key']} {completed}/{total}", flush=True)

    if args.method == "pgmorl":
        rows.sort(key=lambda row: int(row["policy_idx"]))
    latencies = np.asarray(
        [float(row["trace_metrics"]["trace_recovery_latency"]) for row in rows],
        dtype=np.float64,
    )
    payload = {
        "method": str(args.method),
        "env_key": str(summary["env_key"]),
        "env_name": str(summary["env_name"]),
        "source_run_dir": str(source_run_dir),
        "row_count": int(len(rows)),
        "trace_recovery_latency_mean": float(latencies.mean()),
        "trace_recovery_latency_std": float(latencies.std(ddof=0)),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    if args.out_md is not None:
        _write_md(args.out_md.resolve(), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
