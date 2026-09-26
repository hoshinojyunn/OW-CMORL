from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl.metrics import (
    append_metrics_row,
    compute_front_metrics,
    pareto_front,
    summarize_regime_metrics,
    summarize_trace_metrics,
)
from src.dynamic_morl.morl import _save_shared_regime_artifacts, _write_obj_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pure OW-CMORL shared-protocol run by unioning multiple dynamic run archives."
    )
    parser.add_argument("--source-run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--selection-method", type=str, default="dynamic-union")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_run_args(run_dir: Path) -> argparse.Namespace:
    raw_args = ast.literal_eval((run_dir / "args.txt").read_text())
    parser = get_parser()
    return parser.parse_args(raw_args)


def _load_payloads(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((run_dir / "final" / "shared_eval_summary.json").read_text())
    regime_payload = json.loads((run_dir / "regime_fronts" / "shared_final.json").read_text())
    return summary, regime_payload


def _load_policy_points(run_dir: Path, summary: dict[str, Any]) -> np.ndarray:
    front_points = np.asarray(summary.get("front_points", []), dtype=np.float64)
    if front_points.ndim == 2 and len(front_points) > 0:
        return front_points
    objs_path = run_dir / "final" / "objs.txt"
    if not objs_path.exists():
        return np.zeros((0, 0), dtype=np.float64)
    rows = []
    for line in objs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(item) for item in line.split(",")])
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _copy_reference_run(source_run_dir: Path, output_run_dir: Path, force: bool) -> None:
    if output_run_dir.exists():
        if not force:
            raise FileExistsError(f"output exists: {output_run_dir}")
        shutil.rmtree(output_run_dir)
    shutil.copytree(source_run_dir, output_run_dir)
    shutil.rmtree(output_run_dir / "shared_regime_returns", ignore_errors=True)
    shutil.rmtree(output_run_dir / "regime_fronts", ignore_errors=True)
    shutil.rmtree(output_run_dir / "final", ignore_errors=True)
    os_metrics = output_run_dir / "metrics_history.csv"
    if os_metrics.exists():
        os_metrics.unlink()
    os_regime_metrics = output_run_dir / "regime_metrics.csv"
    if os_regime_metrics.exists():
        os_regime_metrics.unlink()
    subset_selection = output_run_dir / "subset_selection.json"
    if subset_selection.exists():
        subset_selection.unlink()


def _union_regime_rows(
    source_runs: list[Path],
    ref_point: np.ndarray,
    eval_delta_weight: float,
) -> list[dict[str, Any]]:
    per_regime: dict[int, dict[str, Any]] = {}
    for run_dir in source_runs:
        _summary, regime_payload = _load_payloads(run_dir)
        for regime_row in regime_payload.get("regimes", []):
            regime_id = int(regime_row["regime_id"])
            bucket = per_regime.setdefault(
                regime_id,
                {
                    "regime_id": regime_id,
                    "regime": str(regime_row["regime"]),
                    "regime_meta": dict(regime_row.get("regime_meta", {})),
                    "solutions": [],
                },
            )
            points = np.asarray(regime_row.get("solution_points", []), dtype=np.float64)
            if points.ndim == 2 and len(points) > 0:
                bucket["solutions"].append(points)

    out: list[dict[str, Any]] = []
    for regime_id in sorted(per_regime):
        bucket = per_regime[regime_id]
        all_points = np.concatenate(bucket["solutions"], axis=0)
        front = pareto_front(all_points)
        hv, eu, sp = compute_front_metrics(front, ref_point, eval_delta_weight)
        out.append(
            {
                "regime_id": int(bucket["regime_id"]),
                "regime": str(bucket["regime"]),
                "regime_meta": dict(bucket["regime_meta"]),
                "hv": float(hv),
                "eu": float(eu),
                "sp": float(sp),
                "points": int(len(front)),
                "num_solutions": int(len(all_points)),
                "front_points": front.tolist(),
                "solution_points": all_points.tolist(),
            }
        )
    return out


def _union_trace_rows(source_runs: list[Path]) -> list[dict[str, Any]]:
    trace_rows: list[dict[str, Any]] = []
    for run_dir in source_runs:
        summary, _regime_payload = _load_payloads(run_dir)
        per_trace_rows = [dict(row) for row in summary.get("per_sample_trace_metrics", [])]
        front_points = _load_policy_points(run_dir, summary)
        expected = int(len(front_points)) if front_points.ndim == 2 else 0
        if len(per_trace_rows) < expected:
            # Some runs emit a minimal shared summary before the expensive trace sweep
            # finishes. Keep those policies usable in a union archive by padding
            # missing trace rows with pessimistic-but-filterable sentinels.
            per_trace_rows.extend(
                [
                    {
                        "trace_shift_regret": float("inf"),
                        "trace_recovery_latency": float("inf"),
                        "trace_recovery_score": float("-inf"),
                        "trace_utility_mean": float("nan"),
                        "trace_utility_std": float("nan"),
                        "trace_shift_count": float("nan"),
                        "trace_pre_post_gap": float("nan"),
                    }
                    for _ in range(expected - len(per_trace_rows))
                ]
            )
        trace_rows.extend(per_trace_rows)
    return trace_rows


def _union_policy_points(source_runs: list[Path]) -> np.ndarray:
    all_points = []
    for run_dir in source_runs:
        summary, _regime_payload = _load_payloads(run_dir)
        points = _load_policy_points(run_dir, summary)
        if points.ndim == 2 and len(points) > 0:
            all_points.append(points)
    if not all_points:
        return np.zeros((0, 0), dtype=np.float64)
    stacked = np.concatenate(all_points, axis=0)
    return stacked


def main() -> None:
    args = parse_args()
    source_runs = [path.resolve() for path in args.source_run_dir]
    output_run_dir = args.output_run_dir.resolve()
    reference_run = source_runs[0]
    run_args = _load_run_args(reference_run)
    _copy_reference_run(reference_run, output_run_dir, force=args.force)

    ref_point = np.asarray(run_args.ref_point, dtype=np.float64)
    eval_delta_weight = float(getattr(run_args, "eval_delta_weight", 0.5))
    env_name = str(run_args.env_name)
    seed = int(run_args.seed)

    regime_rows = _union_regime_rows(source_runs, ref_point=ref_point, eval_delta_weight=eval_delta_weight)
    trace_rows = _union_trace_rows(source_runs)
    front_points = _union_policy_points(source_runs)
    hv, eu, sp = compute_front_metrics(front_points, ref_point, eval_delta_weight)
    trace_summary = summarize_trace_metrics(trace_rows)
    regime_summary = summarize_regime_metrics(regime_rows)

    shared_eval_episode_seeds = []
    shared_regime_seed_plan = []
    if (reference_run / "final" / "shared_eval_summary.json").exists():
        ref_summary = json.loads((reference_run / "final" / "shared_eval_summary.json").read_text())
        shared_eval_episode_seeds = list(ref_summary.get("shared_eval_episode_seeds", []))
        shared_regime_seed_plan = list(ref_summary.get("shared_regime_seed_plan", []))

    _save_shared_regime_artifacts(
        str(output_run_dir),
        env_name,
        seed,
        args.selection_method,
        regime_rows,
    )

    regime_dir = output_run_dir / "regime_fronts"
    regime_dir.mkdir(parents=True, exist_ok=True)
    (regime_dir / "shared_final.json").write_text(
        json.dumps(
            {
                "stage": "shared_final",
                "iteration": int(10**9),
                "seed": seed,
                "env_name": env_name,
                "selection_method": args.selection_method,
                "shared_eval_episode_seeds": shared_eval_episode_seeds,
                "shared_regime_seed_plan": shared_regime_seed_plan,
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
                "seed": seed,
                "env_name": env_name,
                "selection_method": args.selection_method,
                "shared_eval_episode_seeds": shared_eval_episode_seeds,
                "shared_regime_seed_plan": shared_regime_seed_plan,
                "front_points": front_points.tolist(),
                "trace_metrics": trace_summary,
                "per_sample_trace_metrics": trace_rows,
                "regime_metrics": regime_summary,
                "union_sources": [str(path) for path in source_runs],
            },
            indent=2,
        )
    )
    (output_run_dir / "subset_selection.json").write_text(
        json.dumps(
            {
                "indices": list(range(int(len(front_points)))),
                "source_indices": list(range(int(len(front_points)))),
                "source_run_dir": str(output_run_dir),
                "selection_method": args.selection_method,
                "union_sources": [str(path) for path in source_runs],
            },
            indent=2,
        )
    )

    metrics_row = {
        "stage": "shared_final",
        "iteration": int(10**9),
        "seed": seed,
        "env_name": env_name,
        "selection_method": args.selection_method,
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
        "shared_eval_episodes": int(len(shared_eval_episode_seeds)),
    }
    metrics_row.update(regime_summary)
    metrics_row.update(trace_summary)
    append_metrics_row(str(output_run_dir / "metrics_history.csv"), metrics_row)


if __name__ == "__main__":
    main()
