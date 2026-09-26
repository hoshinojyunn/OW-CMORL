from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "baselines"))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "pytorch-a2c-ppo-acktr-gail"))

from a2c_ppo_acktr.model import Policy

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl.metrics import (
    DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    append_metrics_row,
    compute_front_metrics,
    compute_trace_shift_metrics,
    pareto_front,
    shared_regime_seed_plan as build_shared_regime_seed_plan,
    summarize_regime_metrics,
    summarize_trace_metrics,
)
from src.dynamic_morl.fine_regimes import (
    assign_episode_returns_to_plan,
    coerce_regime_seed_plan,
    selected_episode_seeds_from_plan,
)
from src.dynamic_morl.chlor_shared_protocol import (
    canonical_chlor_regime_seed_plan,
    canonical_chlor_regime_rows,
    chlor_regime_eval_env_kwargs,
)
from src.dynamic_morl.mopg import _make_eval_env
from src.dynamic_morl.mopg import evaluation
from src.ood_protocol import load_env_config_json
from src.dynamic_morl.morl import (
    _save_shared_regime_artifacts,
    _write_obj_rows,
    eval as overall_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate an existing OW-CMORL run under shared protocol.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--shared-regime-eval-episodes", type=int, required=True)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Only materialize shared regime fronts. Intended for fast chlor candidate screening.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing shared_partial/shared_final artifacts in target-run-dir when possible.",
    )
    parser.add_argument(
        "--max-eval-policies",
        type=int,
        default=0,
        help="Evaluate at most this many nondominated/diverse final policies. 0 means evaluate all.",
    )
    parser.add_argument(
        "--shared-plan-json",
        type=Path,
        default=None,
        help="Reuse an existing shared regime plan JSON/payload so this reevaluation matches other methods exactly.",
    )
    parser.add_argument(
        "--env-config-json",
        type=Path,
        default=None,
        help="Optional evaluation-only env kwargs JSON. Intended for OOD shared-plan reevaluation.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _policy_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _load_source_args(source_run_dir: Path) -> tuple[list[str], argparse.Namespace]:
    args_path = source_run_dir / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.txt: {args_path}")
    raw_args = ast.literal_eval(args_path.read_text())
    if not isinstance(raw_args, list):
        raise ValueError(f"unexpected args.txt payload in {args_path}")
    parser = get_parser()
    parsed, _unknown = parser.parse_known_args(raw_args)
    return [str(token) for token in raw_args], parsed


def _write_incremental_regime_outputs(
    save_dir: str | Path,
    env_name: str,
    seed: int,
    selection_method: str,
    regime_rows: list[dict[str, object]],
) -> None:
    # Rewrite the shared-regime artifact set after each completed regime so partial
    # progress survives interruptions.
    _save_shared_regime_artifacts(
        str(save_dir),
        env_name,
        int(seed),
        selection_method,
        regime_rows,
    )

    regime_front_dir = Path(save_dir) / "regime_fronts"
    regime_front_dir.mkdir(parents=True, exist_ok=True)
    with (regime_front_dir / "shared_partial.json").open("w") as fp:
        json.dump(
            {
                "stage": "shared_partial",
                "seed": int(seed),
                "env_name": env_name,
                "selection_method": selection_method,
                "completed_regimes": len(regime_rows),
                "regimes": regime_rows,
            },
            fp,
            indent=2,
        )


def _prepare_target_dir(source_run_dir: Path, target_run_dir: Path, force: bool, resume: bool) -> None:
    if resume and target_run_dir.exists():
        return
    if target_run_dir.exists():
        if not force:
            raise FileExistsError(f"target exists: {target_run_dir}")
        shutil.rmtree(target_run_dir)
    shutil.copytree(source_run_dir, target_run_dir)
    shutil.rmtree(target_run_dir / "shared_regime_returns", ignore_errors=True)
    shutil.rmtree(target_run_dir / "regime_fronts", ignore_errors=True)
    os.makedirs(target_run_dir / "regime_fronts", exist_ok=True)
    metrics_path = target_run_dir / "metrics_history.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    regime_metrics_path = target_run_dir / "regime_metrics.csv"
    if regime_metrics_path.exists():
        regime_metrics_path.unlink()
    shared_summary = target_run_dir / "final" / "shared_eval_summary.json"
    if shared_summary.exists():
        shared_summary.unlink()


def _rewrite_args_record(
    target_run_dir: Path,
    source_args: list[str],
    shared_eval_episodes: int,
    seed_offset: int,
    shared_plan_json: Path | None = None,
) -> None:
    updated_args = list(source_args)
    updated_args.extend(
        [
            "--shared-regime-eval-episodes",
            str(shared_eval_episodes),
            "--shared-regime-seed-offset",
            str(seed_offset),
        ]
    )
    if shared_plan_json is not None:
        updated_args.extend(["--shared-plan-json", str(shared_plan_json)])
    (target_run_dir / "args.txt").write_text(str(updated_args))
    (target_run_dir / "shared_protocol_reeval.json").write_text(
        json.dumps(
            {
                "source_args": source_args,
                "shared_regime_eval_episodes": int(shared_eval_episodes),
                "shared_regime_seed_offset": int(seed_offset),
                "shared_plan_json": None if shared_plan_json is None else str(shared_plan_json),
            },
            indent=2,
        )
    )


def _load_existing_regime_rows(target_run_dir: Path) -> list[dict[str, object]]:
    final_path = target_run_dir / "regime_fronts" / "shared_final.json"
    partial_path = target_run_dir / "regime_fronts" / "shared_partial.json"
    payload_path = final_path if final_path.exists() else partial_path
    if not payload_path.exists():
        return []
    try:
        payload = json.loads(payload_path.read_text())
    except Exception:
        return []
    rows = list(payload.get("regimes", []))
    if not isinstance(rows, list):
        return []
    rows.sort(key=lambda row: int(row.get("regime_id", -1)))
    return rows


def _select_diverse_subset(points: np.ndarray, max_points: int) -> list[int]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return list(range(len(points)))
    mins = points.min(axis=0, keepdims=True)
    spans = np.maximum(points.max(axis=0, keepdims=True) - mins, 1e-8)
    norm = (points - mins) / spans

    selected: list[int] = []
    for dim in range(norm.shape[1]):
        selected.append(int(np.argmax(norm[:, dim])))
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = [0]

    while len(selected) < min(max_points, len(norm)):
        remaining = [idx for idx in range(len(norm)) if idx not in selected]
        if not remaining:
            break
        distances = []
        for idx in remaining:
            dists = np.linalg.norm(norm[idx] - norm[selected], axis=1)
            distances.append((float(np.min(dists)), idx))
        distances.sort(reverse=True)
        selected.append(int(distances[0][1]))
    return selected[:max_points]


def _select_policy_indices(final_dir: Path, max_eval_policies: int) -> list[int]:
    if max_eval_policies <= 0:
        return []
    obj_path = final_dir / "objs.txt"
    if not obj_path.exists():
        return []
    objs = np.loadtxt(obj_path, delimiter=",")
    if objs.ndim == 1:
        objs = objs[None, :]
    nd_idx = NonDominatedSorting().do(-np.asarray(objs, dtype=np.float64), only_non_dominated_front=True)
    nd_idx = [int(idx) for idx in np.atleast_1d(nd_idx).tolist()]
    if len(nd_idx) <= max_eval_policies:
        return nd_idx
    nd_points = np.asarray(objs[nd_idx], dtype=np.float64)
    subset = _select_diverse_subset(nd_points, max_eval_policies)
    return [nd_idx[idx] for idx in subset]


def _load_final_samples(
    args: argparse.Namespace,
    artifact_run_dir: Path | None = None,
    *,
    max_eval_policies: int = 0,
) -> list[SimpleNamespace]:
    final_dir = (Path(artifact_run_dir) if artifact_run_dir is not None else Path(args.save_dir)) / "final"
    policy_paths = sorted(final_dir.glob("EP_policy_*.pt"), key=_policy_index)
    if not policy_paths:
        raise FileNotFoundError(f"no final policies found in {final_dir}")
    selected_indices = set(_select_policy_indices(final_dir, int(max_eval_policies)))
    if selected_indices:
        policy_paths = [path for path in policy_paths if _policy_index(path) in selected_indices]
    if not policy_paths:
        raise FileNotFoundError(f"no selected final policies found in {final_dir}")

    torch.set_default_dtype(torch.float64)
    temp_env = _make_eval_env(args)
    try:
        observation_shape = temp_env.observation_space.shape
        action_space = temp_env.action_space
    finally:
        temp_env.close()

    final_samples: list[SimpleNamespace] = []
    for policy_path in policy_paths:
        idx = _policy_index(policy_path)
        env_param_path = final_dir / f"EP_env_params_{idx}.pkl"
        if not env_param_path.exists():
            raise FileNotFoundError(f"missing env params for policy {policy_path.name}")
        actor_critic = Policy(
            observation_shape,
            action_space,
            base_kwargs={"layernorm": args.layernorm},
            obj_num=args.obj_num,
        ).double()
        state_dict = torch.load(policy_path, map_location="cpu")
        actor_critic.load_state_dict(state_dict)
        with open(env_param_path, "rb") as fp:
            env_params = pickle.load(fp)
        final_samples.append(SimpleNamespace(env_params=env_params, actor_critic=actor_critic))
    print(
        f"[shared-reeval] selected {len(final_samples)} final policies"
        f"{' (subset)' if selected_indices else ''}",
        flush=True,
    )
    return final_samples


def _shared_protocol_plan_and_seeds(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[int]]:
    override_path = getattr(args, "shared_plan_json", None)
    if override_path:
        plan = coerce_regime_seed_plan(override_path)
        return plan, selected_episode_seeds_from_plan(plan)
    if args.env_name == "chlor_alkali_dynamic":
        plan, shared_episode_seeds = canonical_chlor_regime_seed_plan(
            env_name=args.env_name,
            base_seed=int(args.seed),
            shared_regime_eval_episodes=int(getattr(args, "shared_regime_eval_episodes", 0)),
            seed_offset=int(getattr(args, "shared_regime_seed_offset", 0)),
            num_regime_clusters=int(getattr(args, "chlor_regime_clusters", 20)),
        )
        return plan, shared_episode_seeds
    return build_shared_regime_seed_plan(args)


def _evaluate_non_chlor_shared(
    args: argparse.Namespace,
    final_samples: list[SimpleNamespace],
    shared_episode_seeds: list[int],
    shared_regime_seed_plan: list[dict[str, object]],
    *,
    eval_env_kwargs: dict[str, object] | None = None,
):
    reevaluated_objs: list[np.ndarray] = []
    trace_rows: list[dict[str, float]] = []
    grouped: dict[str, dict[str, object]] = {}

    for idx, sample in enumerate(final_samples, start=1):
        objs, trace = evaluation(
            args,
            sample,
            return_trace=True,
            env_name=args.env_name,
            env_kwargs=dict(eval_env_kwargs or {}),
            episode_seeds=shared_episode_seeds,
        )
        reevaluated_objs.append(np.asarray(objs, dtype=np.float64))
        trace_rows.append(
            compute_trace_shift_metrics(
                trace,
                args.eval_delta_weight,
                recovery_window=args.trace_recovery_window,
            )
        )
        assignments = assign_episode_returns_to_plan(
            trace,
            plan=shared_regime_seed_plan,
        )
        for row in assignments:
            bucket = grouped.setdefault(
                row["regime"],
                {
                    "regime": row["regime"],
                    "regime_id": int(row["regime_id"]),
                    "regime_meta": dict(row["regime_meta"]),
                    "solutions": [],
                },
            )
            bucket["solutions"].append(np.asarray(row["return"], dtype=np.float64))
        if idx % 4 == 0 or idx == len(final_samples):
            print(f"[shared-reeval] {args.env_name} sample {idx}/{len(final_samples)}", flush=True)

    regime_rows = []
    ref_point = np.asarray(args.ref_point, dtype=np.float64)
    for regime_name in sorted(grouped):
        bucket = grouped[regime_name]
        solutions = np.asarray(bucket["solutions"], dtype=np.float64)
        if solutions.ndim != 2 or len(solutions) == 0:
            continue
        front = pareto_front(solutions)
        if len(front) == 0:
            continue
        hv, eu, sp = compute_front_metrics(front, ref_point, args.eval_delta_weight)
        regime_rows.append(
            {
                "regime": regime_name,
                "regime_id": int(bucket["regime_id"]),
                "regime_meta": dict(bucket["regime_meta"]),
                "hv": hv,
                "eu": eu,
                "sp": sp,
                "points": int(len(front)),
                "num_solutions": int(len(solutions)),
                "front_points": front.tolist(),
                "solution_points": solutions.tolist(),
            }
        )
        _write_incremental_regime_outputs(
            args.save_dir,
            args.env_name,
            int(args.seed),
            args.selection_method,
            regime_rows,
        )
        print(f"[shared-reeval] {args.env_name} regime {len(regime_rows)}/{len(grouped)}", flush=True)
    return np.stack(reevaluated_objs, axis=0), trace_rows, regime_rows


def _evaluate_chlor_shared(
    args: argparse.Namespace,
    final_samples: list[SimpleNamespace],
    shared_episode_seeds: list[int],
    *,
    skip_trace: bool = False,
    existing_regime_rows: list[dict[str, object]] | None = None,
    eval_env_kwargs: dict[str, object] | None = None,
):
    reevaluated_objs: list[np.ndarray] = []
    trace_rows: list[dict[str, float]] = []
    ref_point = np.asarray(args.ref_point, dtype=np.float64)

    regime_rows = list(existing_regime_rows or [])
    completed_ids = {
        int(row.get("regime_id", -1))
        for row in regime_rows
        if int(row.get("regime_id", -1)) >= 0
    }
    canonical_rows = canonical_chlor_regime_rows(
        num_regime_clusters=int(getattr(args, "chlor_regime_clusters", 20)),
    )
    plan, _flat = canonical_chlor_regime_seed_plan(
        env_name=args.env_name,
        base_seed=int(args.seed),
        shared_regime_eval_episodes=int(getattr(args, "shared_regime_eval_episodes", 0)),
        seed_offset=int(getattr(args, "shared_regime_seed_offset", 0)),
        num_regime_clusters=int(getattr(args, "chlor_regime_clusters", 20)),
    )
    regime_by_id = {int(row["regime_id"]): row for row in canonical_rows}
    for plan_row in plan:
        if int(plan_row["regime_id"]) in completed_ids:
            continue
        base_row = regime_by_id.get(int(plan_row["regime_id"]))
        if base_row is None:
            continue
        solutions = []
        for sample in final_samples:
            regime_env_kwargs = {
                **dict(eval_env_kwargs or {}),
                **chlor_regime_eval_env_kwargs(
                    regime_name=str(base_row["regime"]),
                    regime_id=int(base_row["regime_id"]),
                    episode_length=int((eval_env_kwargs or {}).get("episode_length", args.chlor_episode_length)),
                    schedule="random",
                    dynamic_price=bool((eval_env_kwargs or {}).get("dynamic_price", True)),
                    aging_dynamics=bool((eval_env_kwargs or {}).get("aging_dynamics", True)),
                    num_regime_clusters=int((eval_env_kwargs or {}).get("num_regime_clusters", getattr(args, "chlor_regime_clusters", 20))),
                ),
            }
            eval_out = evaluation(
                args,
                sample,
                return_trace=False,
                env_name="chlor_alkali_dynamic",
                env_kwargs=regime_env_kwargs,
                episode_seeds=[int(seed) for seed in plan_row.get("episode_seeds", [])],
            )
            objs = eval_out[0] if isinstance(eval_out, tuple) else eval_out
            solutions.append(np.asarray(objs, dtype=np.float64))
        solutions_arr = np.asarray(solutions, dtype=np.float64)
        if solutions_arr.ndim != 2 or len(solutions_arr) == 0:
            continue
        front = pareto_front(solutions_arr)
        if len(front) == 0:
            continue
        hv, eu, sp = compute_front_metrics(front, ref_point, args.eval_delta_weight)
        regime_rows.append(
            {
                "regime": str(base_row["regime"]),
                "regime_id": int(base_row["regime_id"]),
                "regime_meta": dict(base_row["regime_meta"]),
                "hv": hv,
                "eu": eu,
                "sp": sp,
                "points": int(len(front)),
                "num_solutions": int(len(solutions_arr)),
                "front_points": front.tolist(),
                "solution_points": solutions_arr.tolist(),
            }
        )
        _write_incremental_regime_outputs(
            args.save_dir,
            args.env_name,
            int(args.seed),
            args.selection_method,
            regime_rows,
        )
        print(f"[shared-reeval] {args.env_name} regime {len(regime_rows)}/{len(plan)}", flush=True)

    if skip_trace:
        return np.zeros((0, args.obj_num), dtype=np.float64), trace_rows, regime_rows

    for idx, sample in enumerate(final_samples, start=1):
        objs, trace = evaluation(
            args,
            sample,
            return_trace=True,
            env_name=args.env_name,
            env_kwargs=dict(eval_env_kwargs or {}),
            episode_seeds=shared_episode_seeds,
        )
        reevaluated_objs.append(np.asarray(objs, dtype=np.float64))
        trace_rows.append(
            compute_trace_shift_metrics(
                trace,
                args.eval_delta_weight,
                recovery_window=args.trace_recovery_window,
            )
        )
        if idx % 4 == 0 or idx == len(final_samples):
            print(f"[shared-reeval] {args.env_name} sample {idx}/{len(final_samples)}", flush=True)
    return np.stack(reevaluated_objs, axis=0), trace_rows, regime_rows


def _write_shared_outputs(
    args: argparse.Namespace,
    obj_array: np.ndarray,
    trace_rows: list[dict[str, float]],
    regime_rows: list[dict[str, object]],
    shared_episode_seeds: list[int],
    shared_regime_seed_plan: list[dict[str, object]],
) -> None:
    if len(obj_array) > 0:
        hv, eu, sp = overall_eval(obj_array, np.asarray(args.ref_point, dtype=np.float64), args.obj_num, args.eval_delta_weight)
    else:
        hv = eu = sp = float("nan")
    dynamic_summary = summarize_regime_metrics(regime_rows)
    trace_summary = summarize_trace_metrics(trace_rows)

    row = {
        "stage": "shared_final",
        "iteration": int(10**9),
        "seed": int(args.seed),
        "env_name": args.env_name,
        "selection_method": args.selection_method,
        "hv": hv,
        "eu": eu,
        "sp": sp,
        "drift_score": 0.0,
        "regime_loss": 0.0,
        "ep_size": int(len(obj_array)),
        "selected_size": int(len(obj_array)),
        "prediction_error": 0.0,
        "context_gap_norm": 0.0,
        "online_trace_steps": 0.0,
        "online_shift_count": 0.0,
        "context_volatility": 0.0,
        "context_support_strength": 0.0,
        "matched_recovery": 0.0,
        "shared_eval_episodes": int(len(shared_episode_seeds)),
    }
    row.update(dynamic_summary)
    row.update(trace_summary)
    append_metrics_row(os.path.join(args.save_dir, "metrics_history.csv"), row)

    _save_shared_regime_artifacts(
        args.save_dir,
        args.env_name,
        args.seed,
        args.selection_method,
        regime_rows,
    )

    regime_front_dir = os.path.join(args.save_dir, "regime_fronts")
    os.makedirs(regime_front_dir, exist_ok=True)
    with open(os.path.join(regime_front_dir, "shared_final.json"), "w") as fp:
        json.dump(
            {
                "stage": "shared_final",
                "iteration": int(10**9),
                "seed": int(args.seed),
                "env_name": args.env_name,
                "selection_method": args.selection_method,
                "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                "shared_regime_seed_plan": shared_regime_seed_plan,
                "regimes": regime_rows,
            },
            fp,
            indent=2,
        )

    final_dir = os.path.join(args.save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if len(obj_array) > 0:
        _write_obj_rows(os.path.join(final_dir, "objs.txt"), obj_array, args.obj_num)
    with open(os.path.join(final_dir, "shared_eval_summary.json"), "w") as fp:
        json.dump(
            {
                "stage": "shared_final",
                "seed": int(args.seed),
                "env_name": args.env_name,
                "selection_method": args.selection_method,
                "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                "shared_regime_seed_plan": shared_regime_seed_plan,
                "front_points": obj_array.tolist(),
                "trace_metrics": trace_summary,
                "per_sample_trace_metrics": trace_rows,
                "regime_metrics": dynamic_summary,
            },
            fp,
            indent=2,
        )


def main() -> None:
    cli_args = parse_args()
    if cli_args.env_config_json is not None and cli_args.shared_plan_json is None:
        raise ValueError("--env-config-json requires --shared-plan-json so evaluation seeds match the overridden environment.")
    source_args, run_args = _load_source_args(cli_args.source_run_dir)
    _prepare_target_dir(
        cli_args.source_run_dir,
        cli_args.target_run_dir,
        cli_args.force,
        cli_args.resume,
    )

    run_args.save_dir = str(cli_args.target_run_dir)
    run_args.shared_regime_eval_episodes = int(cli_args.shared_regime_eval_episodes)
    run_args.shared_regime_seed_offset = int(cli_args.shared_regime_seed_offset)
    run_args.shared_plan_json = cli_args.shared_plan_json
    eval_env_kwargs = load_env_config_json(cli_args.env_config_json) if cli_args.env_config_json else {}

    _rewrite_args_record(
        cli_args.target_run_dir,
        source_args,
        cli_args.shared_regime_eval_episodes,
        cli_args.shared_regime_seed_offset,
        cli_args.shared_plan_json,
    )

    # Always load final archived policies from the source run. The target directory
    # is only the destination for repaired shared-protocol artifacts.
    final_samples = _load_final_samples(
        run_args,
        cli_args.source_run_dir,
        max_eval_policies=int(cli_args.max_eval_policies),
    )
    shared_regime_seed_plan, shared_episode_seeds = _shared_protocol_plan_and_seeds(run_args)
    existing_regime_rows = _load_existing_regime_rows(cli_args.target_run_dir) if cli_args.resume else []
    if cli_args.skip_trace and run_args.env_name != "chlor_alkali_dynamic":
        raise ValueError("--skip-trace is currently only supported for chlor_alkali_dynamic")
    if run_args.env_name == "chlor_alkali_dynamic" and cli_args.shared_plan_json is None:
        obj_array, trace_rows, regime_rows = _evaluate_chlor_shared(
            run_args,
            final_samples,
            shared_episode_seeds,
            skip_trace=cli_args.skip_trace,
            existing_regime_rows=existing_regime_rows,
            eval_env_kwargs=eval_env_kwargs,
        )
    else:
        obj_array, trace_rows, regime_rows = _evaluate_non_chlor_shared(
            run_args,
            final_samples,
            shared_episode_seeds,
            shared_regime_seed_plan,
            eval_env_kwargs=eval_env_kwargs,
        )
    _write_shared_outputs(
        run_args,
        obj_array,
        trace_rows,
        regime_rows,
        shared_episode_seeds,
        shared_regime_seed_plan,
    )


if __name__ == "__main__":
    main()
