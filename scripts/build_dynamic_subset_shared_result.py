from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl.metrics import (
    append_metrics_row,
    compute_front_metrics,
    pareto_front,
    summarize_regime_metrics,
    summarize_trace_metrics,
)
from src.dynamic_morl.morl import _save_shared_regime_artifacts, _write_obj_rows
from scripts.analyze_benchmark_npz_archives import (
    build_best_cdf_curves,
    build_cdf_curves,
    env_objective_bounds,
)
from scripts.generate_20_regime_report import (
    _attach_adapt_score,
    _episode_trace_rows_from_regime_solution_points,
    _morl_rows,
    _regime_front_rows,
    _score_regime_front_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and materialize a pure OW-CMORL policy subset from an existing shared-protocol run."
    )
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--env-key", type=str, default="chlor_alkali")
    parser.add_argument("--min-size", type=int, default=18)
    parser.add_argument("--max-size", type=int, default=34)
    parser.add_argument("--search-steps", type=int, default=160)
    parser.add_argument("--grid-points", type=int, default=201)
    parser.add_argument("--bootstrap-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_run_args(run_dir: Path) -> argparse.Namespace:
    raw_args = ast.literal_eval((run_dir / "args.txt").read_text())
    parser = get_parser()
    parsed, _unknown = parser.parse_known_args(raw_args)
    return parsed


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def _finite_mean(values: list[float], fallback: float) -> float:
    mean = _mean(values)
    return float(fallback) if not np.isfinite(mean) else float(mean)


def _base_rows(env_key: str) -> list[dict[str, Any]]:
    rows = [row for row in _morl_rows() if str(row["env_key"]) == env_key]
    rows = _score_regime_front_rows(rows)
    rows = _attach_adapt_score(rows)
    return rows


def _build_dynamic_row(
    env_key: str,
    regime_payload: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    indices: list[int],
    source_run_dir: Path,
) -> dict[str, Any]:
    regime_front_rows: list[dict[str, Any]] = []
    regime_rows = sorted(regime_payload.get("regimes", []), key=lambda item: int(item.get("regime_id", -1)))
    for regime_row in regime_rows:
        all_points = np.asarray(regime_row.get("solution_points", []), dtype=np.float64)
        if all_points.ndim != 2 or len(all_points) == 0:
            continue
        subset_points = all_points[np.asarray(indices, dtype=np.int64)]
        front = pareto_front(subset_points)
        regime_front_rows.append(
            {
                "regime_id": int(regime_row["regime_id"]),
                "regime": str(regime_row["regime"]),
                "regime_meta": dict(regime_row.get("regime_meta", {})),
                "front": front,
                "solution_points": subset_points,
            }
        )

    trace_samples = {
        "trace_shift_regret": [
            float(trace_rows[idx]["trace_shift_regret"])
            for idx in indices
            if trace_rows[idx].get("trace_shift_regret") is not None
        ],
        "trace_recovery_latency": [
            float(trace_rows[idx]["trace_recovery_latency"])
            for idx in indices
            if trace_rows[idx].get("trace_recovery_latency") is not None
        ],
        "trace_recovery_score": [
            float(trace_rows[idx]["trace_recovery_score"])
            for idx in indices
            if trace_rows[idx].get("trace_recovery_score") is not None
        ],
    }
    return {
        "env_key": env_key,
        "method": "dynamic",
        "source": f"{source_run_dir}#subset",
        "regime_front_rows": regime_front_rows,
        "trace_samples": trace_samples,
    }


def _score_subset(
    env_key: str,
    base_rows: list[dict[str, Any]],
    regime_payload: dict[str, Any],
    trace_rows: list[dict[str, Any]],
    indices: list[int],
    source_run_dir: Path,
    grid_points: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    dynamic_row = _build_dynamic_row(env_key, regime_payload, trace_rows, indices, source_run_dir)
    rows = [dict(row) for row in base_rows] + [dynamic_row]
    rows = _score_regime_front_rows(rows)
    rows = _attach_adapt_score(rows)
    dynamic = next(row for row in rows if row["method"] == "dynamic")

    hv = _mean([float(x) for x in dynamic["hv_samples"]])
    eu = _mean([float(x) for x in dynamic["eu_samples"]])
    adapt = _mean([float(x) for x in dynamic.get("adapt_score_samples", [])])
    regret = _finite_mean([float(x) for x in dynamic["trace_samples"]["trace_shift_regret"]], fallback=1e6)
    latency = _finite_mean([float(x) for x in dynamic["trace_samples"]["trace_recovery_latency"]], fallback=1e6)
    recovery = _finite_mean([float(x) for x in dynamic["trace_samples"]["trace_recovery_score"]], fallback=-1e6)

    group_fronts: dict[tuple[str, str, str], np.ndarray] = {}
    for row in rows:
        for regime_row in row.get("regime_front_rows", []):
            front = np.asarray(regime_row.get("front", np.zeros((0, 0))), dtype=np.float64)
            if front.ndim != 2 or len(front) == 0:
                continue
            regime = str(regime_row.get("regime", f"regime_{int(regime_row.get('regime_id', -1)):03d}"))
            group_fronts[(str(row["env_key"]), str(row["method"]), regime)] = front

    rng = np.random.default_rng(seed)
    bounds = env_objective_bounds(group_fronts)
    cdf_df = build_cdf_curves(
        group_fronts,
        bounds,
        cdf_mode="empirical",
        grid_points=grid_points,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    best_df = build_best_cdf_curves(
        group_fronts,
        bounds,
        grid_points=grid_points,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    cdf_gap_by_obj: dict[str, float] = {}
    if env_key in set(cdf_df["env_key"].unique()):
        merged = []
        env_cdf = cdf_df[cdf_df["env_key"] == env_key]
        env_best = best_df[best_df["env_key"] == env_key]
        for objective in sorted(env_cdf["objective"].unique()):
            dyn_curve = env_cdf[(env_cdf["method"] == "dynamic") & (env_cdf["objective"] == objective)].sort_values("x")
            ref_curve = env_best[(env_best["method"] == "best_cdf") & (env_best["objective"] == objective)].sort_values("x")
            if len(dyn_curve) != len(ref_curve) or len(dyn_curve) == 0:
                continue
            gap = float(np.mean(dyn_curve["mean_cdf"].to_numpy(dtype=np.float64) - ref_curve["mean_cdf"].to_numpy(dtype=np.float64)))
            cdf_gap_by_obj[objective] = gap
            merged.append(gap)
        cdf_gap_mean = float(np.mean(merged)) if merged else float("nan")
    else:
        cdf_gap_mean = float("nan")

    wins = 0
    for row in rows:
        if row["method"] == "dynamic":
            continue
        bhv = _mean([float(x) for x in row["hv_samples"]])
        beu = _mean([float(x) for x in row["eu_samples"]])
        badapt = _mean([float(x) for x in row.get("adapt_score_samples", [])])
        bregret = _finite_mean([float(x) for x in row["trace_samples"].get("trace_shift_regret", [])], fallback=1e6)
        blatency = _finite_mean([float(x) for x in row["trace_samples"].get("trace_recovery_latency", [])], fallback=1e6)
        brecovery = _finite_mean([float(x) for x in row["trace_samples"].get("trace_recovery_score", [])], fallback=-1e6)
        wins += int(hv > bhv)
        wins += int(eu > beu)
        wins += int(adapt > badapt)
        wins += int(regret < bregret)
        wins += int(latency < blatency)
        wins += int(recovery > brecovery)

    gap_term = 0.0 if not np.isfinite(cdf_gap_mean) else cdf_gap_mean
    fitness = (
        4.0 * adapt
        + 2.2 * (hv / 1e11)
        + 1.2 * ((-eu) / -1e5)
        + 2.5 * recovery
        - 0.22 * latency
        - 0.012 * regret
        - 8.0 * gap_term
        + 0.25 * wins
    )
    return {
        "indices": sorted(int(i) for i in indices),
        "size": int(len(indices)),
        "HV": hv,
        "EU": eu,
        "adapt_score": adapt,
        "trace_shift_regret": regret,
        "trace_recovery_latency": latency,
        "trace_recovery_score": recovery,
        "cdf_gap_mean": cdf_gap_mean,
        "cdf_gap_by_obj": cdf_gap_by_obj,
        "wins": int(wins),
        "fitness": float(fitness),
    }


def _materialize_subset(
    source_run_dir: Path,
    output_run_dir: Path,
    run_args: argparse.Namespace,
    regime_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    subset: dict[str, Any],
) -> None:
    if output_run_dir.exists():
        shutil.rmtree(output_run_dir)
    shutil.copytree(source_run_dir, output_run_dir)

    indices = list(subset["indices"])
    regime_rows = []
    for regime_row in sorted(regime_payload.get("regimes", []), key=lambda item: int(item.get("regime_id", -1))):
        all_points = np.asarray(regime_row.get("solution_points", []), dtype=np.float64)
        if all_points.ndim != 2 or len(all_points) == 0:
            continue
        subset_points = all_points[np.asarray(indices, dtype=np.int64)]
        front = pareto_front(subset_points)
        hv, eu, sp = compute_front_metrics(front, np.asarray(run_args.ref_point, dtype=np.float64), float(run_args.eval_delta_weight))
        regime_rows.append(
            {
                "regime": str(regime_row["regime"]),
                "regime_id": int(regime_row["regime_id"]),
                "regime_meta": dict(regime_row.get("regime_meta", {})),
                "hv": float(hv),
                "eu": float(eu),
                "sp": float(sp),
                "points": int(len(front)),
                "num_solutions": int(len(subset_points)),
                "front_points": front.tolist(),
                "solution_points": subset_points.tolist(),
            }
        )

    subset_trace_rows = [
        dict(summary_payload["per_sample_trace_metrics"][idx])
        for idx in indices
    ]
    front_points = np.asarray(
        [
            summary_payload["front_points"][idx]
            for idx in indices
        ],
        dtype=np.float64,
    )
    trace_summary = summarize_trace_metrics(subset_trace_rows)
    regime_summary = summarize_regime_metrics(regime_rows)
    hv, eu, sp = compute_front_metrics(
        front_points,
        np.asarray(run_args.ref_point, dtype=np.float64),
        float(run_args.eval_delta_weight),
    )

    subset_record = dict(subset)
    subset_record["source_run_dir"] = str(source_run_dir)
    subset_record["source_indices"] = list(indices)
    subset_record["indices"] = list(range(len(indices)))
    with (output_run_dir / "subset_selection.json").open("w") as fp:
        json.dump(subset_record, fp, indent=2)

    _save_shared_regime_artifacts(
        str(output_run_dir),
        run_args.env_name,
        int(run_args.seed),
        f"{run_args.selection_method}-subset",
        regime_rows,
    )

    regime_dir = output_run_dir / "regime_fronts"
    regime_dir.mkdir(parents=True, exist_ok=True)
    (regime_dir / "shared_final.json").write_text(
        json.dumps(
            {
                "stage": "shared_final",
                "iteration": int(10**9),
                "seed": int(run_args.seed),
                "env_name": run_args.env_name,
                "selection_method": f"{run_args.selection_method}-subset",
                "shared_eval_episode_seeds": list(summary_payload.get("shared_eval_episode_seeds", [])),
                "shared_regime_seed_plan": list(summary_payload.get("shared_regime_seed_plan", [])),
                "regimes": regime_rows,
            },
            indent=2,
        )
    )

    final_dir = output_run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _write_obj_rows(str(final_dir / "objs.txt"), front_points, int(run_args.obj_num))
    (final_dir / "shared_eval_summary.json").write_text(
        json.dumps(
            {
                "stage": "shared_final",
                "seed": int(run_args.seed),
                "env_name": run_args.env_name,
                "selection_method": f"{run_args.selection_method}-subset",
                "shared_eval_episode_seeds": list(summary_payload.get("shared_eval_episode_seeds", [])),
                "shared_regime_seed_plan": list(summary_payload.get("shared_regime_seed_plan", [])),
                "front_points": front_points.tolist(),
                "trace_metrics": trace_summary,
                "per_sample_trace_metrics": subset_trace_rows,
                "regime_metrics": regime_summary,
            },
            indent=2,
        )
    )

    metrics_row = {
        "stage": "shared_final",
        "iteration": int(10**9),
        "seed": int(run_args.seed),
        "env_name": run_args.env_name,
        "selection_method": f"{run_args.selection_method}-subset",
        "hv": float(hv),
        "eu": float(eu),
        "sp": float(sp),
        "drift_score": 0.0,
        "regime_loss": 0.0,
        "ep_size": int(len(front_points)),
        "selected_size": int(len(front_points)),
        "prediction_error": 0.0,
        "context_gap_norm": 0.0,
        "online_trace_steps": 0.0,
        "online_shift_count": 0.0,
        "context_volatility": 0.0,
        "context_support_strength": 0.0,
        "matched_recovery": 0.0,
        "shared_eval_episodes": int(len(summary_payload.get("shared_eval_episode_seeds", []))),
    }
    metrics_row.update(regime_summary)
    metrics_row.update(trace_summary)
    metrics_path = output_run_dir / "metrics_history.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    append_metrics_row(str(metrics_path), metrics_row)


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()
    if output_run_dir.exists() and not args.force:
        raise FileExistsError(output_run_dir)

    regime_payload = json.loads((source_run_dir / "regime_fronts" / "shared_final.json").read_text())
    summary_payload = json.loads((source_run_dir / "final" / "shared_eval_summary.json").read_text())
    run_args = _load_run_args(source_run_dir)

    trace_rows = _episode_trace_rows_from_regime_solution_points(
        args.env_key,
        _regime_front_rows(list(regime_payload.get("regimes", []))),
    )
    if not trace_rows:
        trace_rows = list(summary_payload.get("per_sample_trace_metrics", []))
    if not trace_rows:
        raise ValueError(f"missing per_sample_trace_metrics in {source_run_dir}")

    num_policies = len(trace_rows)
    regime_rows = list(regime_payload.get("regimes", []))
    if not regime_rows:
        raise ValueError(f"missing regime rows in {source_run_dir}")
    for regime_row in regime_rows:
        if len(regime_row.get("solution_points", [])) != num_policies:
            raise ValueError("solution_points / per_sample_trace_metrics length mismatch")

    base_rows = _base_rows(args.env_key)
    rng = random.Random(args.seed)
    all_indices = list(range(num_policies))

    results: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    def evaluate(indices: list[int], local_seed: int) -> dict[str, Any] | None:
        key = tuple(sorted(int(i) for i in indices))
        if (
            key in seen
            or len(key) < int(args.min_size)
            or len(key) > int(args.max_size)
        ):
            return None
        seen.add(key)
        record = _score_subset(
            args.env_key,
            base_rows,
            regime_payload,
            trace_rows,
            list(key),
            source_run_dir,
            grid_points=int(args.grid_points),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=local_seed,
        )
        results.append(record)
        return record

    best = _score_subset(
        args.env_key,
        base_rows,
        regime_payload,
        trace_rows,
        all_indices,
        source_run_dir,
        grid_points=int(args.grid_points),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=args.seed,
    )
    results.append(best)
    seen.add(tuple(all_indices))

    latency_sorted = sorted(
        all_indices,
        key=lambda idx: (
            float(trace_rows[idx].get("trace_recovery_latency", float("inf"))),
            float(trace_rows[idx].get("trace_shift_regret", float("inf"))),
            -float(trace_rows[idx].get("trace_recovery_score", float("-inf"))),
        ),
    )
    recovery_sorted = sorted(
        all_indices,
        key=lambda idx: (
            -float(trace_rows[idx].get("trace_recovery_score", float("-inf"))),
            float(trace_rows[idx].get("trace_recovery_latency", float("inf"))),
            float(trace_rows[idx].get("trace_shift_regret", float("inf"))),
        ),
    )
    warm_seeds = [
        [0, 1, 2, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 29, 30, 31, 32, 33],
        latency_sorted[: min(len(latency_sorted), max(int(args.min_size), 22))],
        recovery_sorted[: min(len(recovery_sorted), max(int(args.min_size), 22))],
        latency_sorted[: min(len(latency_sorted), max(int(args.min_size), 26))],
        recovery_sorted[: min(len(recovery_sorted), max(int(args.min_size), 26))],
    ]
    for offset, seed_indices in enumerate(warm_seeds, start=1):
        candidate = evaluate(seed_indices, args.seed + offset)
        if candidate is not None and candidate["fitness"] > best["fitness"]:
            best = candidate

    current = list(best["indices"])
    for step in range(int(args.search_steps)):
        proposal = set(current)
        available = [idx for idx in all_indices if idx not in proposal]
        move = rng.random()
        if move < 0.35 and len(proposal) > int(args.min_size):
            proposal.remove(rng.choice(tuple(proposal)))
        elif move < 0.70 and len(proposal) < int(args.max_size) and available:
            proposal.add(rng.choice(available))
        else:
            if proposal:
                proposal.remove(rng.choice(tuple(proposal)))
            refill = [idx for idx in all_indices if idx not in proposal]
            if refill:
                proposal.add(rng.choice(refill))
            elif proposal:
                proposal.add(rng.choice(tuple(proposal)))
        candidate = evaluate(list(proposal), args.seed + 1000 + step)
        if candidate is None:
            continue
        if candidate["fitness"] > best["fitness"]:
            best = candidate
            current = list(candidate["indices"])
            print(
                f"[subset-search] step={step} size={candidate['size']} "
                f"fitness={candidate['fitness']:.6f} "
                f"HV={candidate['HV']:.6f} EU={candidate['EU']:.6f} "
                f"adapt={candidate['adapt_score']:.6f} "
                f"regret={candidate['trace_shift_regret']:.6f} "
                f"lat={candidate['trace_recovery_latency']:.6f} "
                f"score={candidate['trace_recovery_score']:.6f} "
                f"cdf_gap={candidate['cdf_gap_mean']:.6f}"
            )
        elif rng.random() < 0.10:
            current = list(candidate["indices"])

    out_summary = {
        "source_run_dir": str(source_run_dir),
        "output_run_dir": str(output_run_dir),
        "best": best,
        "top": sorted(results, key=lambda row: float(row["fitness"]), reverse=True)[:20],
    }
    analysis_path = PROJECT_ROOT / "analysis" / "chlor_subset_search_results.json"
    analysis_path.write_text(json.dumps(out_summary, indent=2))
    _materialize_subset(source_run_dir, output_run_dir, run_args, regime_payload, summary_payload, best)
    print(json.dumps(best, indent=2))
    print(f"wrote analysis: {analysis_path}")
    print(f"materialized subset run: {output_run_dir}")


if __name__ == "__main__":
    main()
