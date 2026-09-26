from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl.morl import eval as front_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a hybrid shared-protocol run directory from one dynamic archive and one or more baseline summaries.")
    parser.add_argument("--dynamic-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--selection-method", type=str, default="hybrid-union")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[idx], dtype=np.float64)


def load_dynamic_args(run_dir: Path) -> argparse.Namespace:
    raw_args = ast.literal_eval((run_dir / "args.txt").read_text())
    return get_parser().parse_args(raw_args)


def load_dynamic_payload(run_dir: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    summary = json.loads((run_dir / "final" / "shared_eval_summary.json").read_text())
    regime_payload = json.loads((run_dir / "regime_fronts" / "shared_final.json").read_text())
    with open(run_dir / "metrics_history.csv", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"empty metrics history: {run_dir}")
    return summary, regime_payload, rows[-1]


def load_baseline_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def weighted_trace_metrics(weighted_rows: list[tuple[int, dict[str, object]]]) -> dict[str, float]:
    keys = sorted({key for _count, row in weighted_rows for key in row.keys() if key.startswith("trace_")})
    out: dict[str, float] = {}
    total = float(sum(count for count, _row in weighted_rows))
    if total <= 0:
        return out
    for key in keys:
        numer = 0.0
        for count, row in weighted_rows:
            if key in row:
                numer += float(count) * float(row[key])
        out[key] = numer / total
    return out


def merge_regime_rows(dynamic_rows: list[dict[str, object]], baseline_summaries: list[dict[str, object]], ref_point: np.ndarray, obj_num: int, eval_delta_weight: float) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}

    def extend_rows(rows: list[dict[str, object]]) -> None:
        for row in rows:
            regime = str(row["regime"])
            bucket = grouped.setdefault(
                regime,
                {
                    "regime": regime,
                    "regime_id": int(row.get("regime_id", -1)),
                    "regime_meta": dict(row.get("regime_meta", {})),
                    "points": [],
                },
            )
            raw_points = row.get("solution_points", row.get("points", []))
            bucket["points"].extend(np.asarray(raw_points, dtype=np.float64).tolist())

    extend_rows(dynamic_rows)
    for summary in baseline_summaries:
        extend_rows(list(summary.get("regime_fronts", [])))

    merged: list[dict[str, object]] = []
    for regime in sorted(grouped):
        bucket = grouped[regime]
        points = np.asarray(bucket["points"], dtype=np.float64)
        if points.ndim != 2 or len(points) == 0:
            continue
        front = pareto_front(points)
        hv, eu, sp = front_eval(front, ref_point, obj_num, eval_delta_weight)
        merged.append(
            {
                "regime": regime,
                "regime_id": int(bucket["regime_id"]),
                "regime_meta": dict(bucket["regime_meta"]),
                "hv": float(hv),
                "eu": float(eu),
                "sp": float(sp),
                "points": int(len(front)),
                "num_solutions": int(len(points)),
                "front_points": front.tolist(),
                "solution_points": points.tolist(),
            }
        )
    return merged


def save_obj_rows(path: Path, obj_rows: np.ndarray, obj_num: int) -> None:
    with path.open("w") as fp:
        for obj in np.asarray(obj_rows, dtype=np.float64):
            fp.write(("{:5f}" + (obj_num - 1) * ",{:5f}" + "\n").format(*obj.tolist()))


def regime_stem(row: dict[str, object]) -> str:
    regime_id = int(row.get("regime_id", -1))
    regime = str(row.get("regime", "regime")).replace("/", "_").replace(" ", "_")
    if regime_id >= 0:
        return f"regime_{regime_id:03d}_{regime}"
    return regime


def save_shared_regime_npz(output_dir: Path, rows: list[dict[str, object]], env_name: str, seed: int, selection_method: str) -> None:
    shared_dir = output_dir / "shared_regime_returns"
    shared_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = {
            "env_name": env_name,
            "seed": int(seed),
            "selection_method": selection_method,
            "regime": str(row["regime"]),
            "regime_id": int(row["regime_id"]),
            "regime_meta": dict(row.get("regime_meta", {})),
            "num_solutions": int(row["num_solutions"]),
            "hv": float(row["hv"]),
            "eu": float(row["eu"]),
            "sp": float(row["sp"]),
            "front_points": row["front_points"],
            "solution_points": row["solution_points"],
        }
        stem = regime_stem(row)
        (shared_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        np.savez_compressed(
            shared_dir / f"{stem}.npz",
            solutions=np.asarray(row["solution_points"], dtype=np.float32),
            pareto_front_points=np.asarray(row["front_points"], dtype=np.float32),
            regime_id=np.asarray([int(row["regime_id"])], dtype=np.int32),
        )


def write_metrics_history(path: Path, row: dict[str, object]) -> None:
    fieldnames = list(row.keys())
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.output_run_dir.exists():
        if not args.force:
            raise FileExistsError(f"output exists: {args.output_run_dir}")
        shutil.rmtree(args.output_run_dir)
    args.output_run_dir.mkdir(parents=True, exist_ok=True)

    dyn_args = load_dynamic_args(args.dynamic_run_dir)
    dyn_summary, dyn_regimes, dyn_metrics = load_dynamic_payload(args.dynamic_run_dir)
    baseline_summaries = [load_baseline_summary(path) for path in args.baseline_summary]

    obj_num = int(dyn_args.obj_num)
    ref_point = np.asarray(dyn_args.ref_point, dtype=np.float64)
    eval_delta_weight = float(getattr(dyn_args, "eval_delta_weight", 0.5))
    env_name = str(dyn_args.env_name)
    seed = int(dyn_args.seed)

    dynamic_points = np.asarray(dyn_summary["front_points"], dtype=np.float64)
    baseline_points = [np.asarray(summary["front_points"], dtype=np.float64) for summary in baseline_summaries]
    all_points = np.concatenate([dynamic_points, *baseline_points], axis=0)
    front = pareto_front(all_points)
    hv, eu, sp = front_eval(front, ref_point, obj_num, eval_delta_weight)

    trace_rows = [
        (int(len(dynamic_points)), dict(dyn_summary.get("trace_metrics", {}))),
    ]
    for summary in baseline_summaries:
        trace_rows.append((int(len(summary.get("front_points", []))), dict(summary.get("trace_metrics", {}))))
    trace_metrics = weighted_trace_metrics(trace_rows)

    regime_rows = merge_regime_rows(
        list(dyn_regimes.get("regimes", [])),
        baseline_summaries,
        ref_point=ref_point,
        obj_num=obj_num,
        eval_delta_weight=eval_delta_weight,
    )

    final_dir = args.output_run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_obj_rows(final_dir / "objs.txt", np.asarray(all_points, dtype=np.float64), obj_num)

    shared_eval_summary = {
        "stage": "shared_final",
        "seed": seed,
        "env_name": env_name,
        "selection_method": args.selection_method,
        "front_points": np.asarray(all_points, dtype=np.float64).tolist(),
        "trace_metrics": trace_metrics,
        "regime_metrics": {
            "cr_hv": float(np.mean([row["hv"] for row in regime_rows])) if regime_rows else 0.0,
            "cr_eu": float(np.mean([row["eu"] for row in regime_rows])) if regime_rows else 0.0,
            "cr_sp": float(np.mean([row["sp"] for row in regime_rows])) if regime_rows else 0.0,
        },
        "hybrid_sources": {
            "dynamic_run_dir": str(args.dynamic_run_dir),
            "baseline_summaries": [str(path) for path in args.baseline_summary],
        },
    }
    (final_dir / "shared_eval_summary.json").write_text(json.dumps(shared_eval_summary, indent=2))

    regime_dir = args.output_run_dir / "regime_fronts"
    regime_dir.mkdir(parents=True, exist_ok=True)
    regime_payload = {
        "stage": "shared_final",
        "iteration": int(10**9),
        "seed": seed,
        "env_name": env_name,
        "selection_method": args.selection_method,
        "regimes": regime_rows,
    }
    (regime_dir / "shared_final.json").write_text(json.dumps(regime_payload, indent=2))

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
        "ep_size": int(len(all_points)),
        "selected_size": int(len(all_points)),
        "prediction_error": 0.0,
        "context_gap_norm": 0.0,
        "online_trace_steps": 0.0,
        "online_shift_count": 0.0,
        "context_volatility": 0.0,
        "context_support_strength": 0.0,
        "matched_recovery": 0.0,
        "shared_eval_episodes": int(dyn_metrics.get("shared_eval_episodes", 0) or 0),
        "cr_hv": float(np.mean([row["hv"] for row in regime_rows])) if regime_rows else 0.0,
        "cr_eu": float(np.mean([row["eu"] for row in regime_rows])) if regime_rows else 0.0,
        "cr_sp": float(np.mean([row["sp"] for row in regime_rows])) if regime_rows else 0.0,
        "hv_std": float(np.std([row["hv"] for row in regime_rows])) if regime_rows else 0.0,
        "eu_std": float(np.std([row["eu"] for row in regime_rows])) if regime_rows else 0.0,
        "sp_std": float(np.std([row["sp"] for row in regime_rows])) if regime_rows else 0.0,
        "irs": float(1.0 / (1.0 + np.std([row["eu"] for row in regime_rows]))) if regime_rows else 0.0,
    }
    metrics_row.update(trace_metrics)
    write_metrics_history(args.output_run_dir / "metrics_history.csv", metrics_row)
    save_shared_regime_npz(args.output_run_dir, regime_rows, env_name, seed, args.selection_method)

    provenance = {
        "dynamic_run_dir": str(args.dynamic_run_dir),
        "baseline_summaries": [str(path) for path in args.baseline_summary],
        "selection_method": args.selection_method,
    }
    (args.output_run_dir / "hybrid_provenance.json").write_text(json.dumps(provenance, indent=2))
    shutil.copy2(args.dynamic_run_dir / "args.txt", args.output_run_dir / "args.txt")


if __name__ == "__main__":
    main()
