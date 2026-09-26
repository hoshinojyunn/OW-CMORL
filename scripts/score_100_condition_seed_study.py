#!/usr/bin/env python3
"""Score the real 100-condition multi-seed study and run paired t-tests.

The independent unit is a training replication.  The 100 operating conditions
are used to form one robust metric value per method, environment, and seed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.metrics import compute_trace_shift_metrics
from src.dynamic_morl.utils import compute_eu, generate_w_batch_test
from scripts.run_100_condition_seed_study import run_is_complete


ENV_KEYS = ("building", "evcharging", "cogen", "chlor_alkali")
METHODS = ("dynamic", "capql", "qpensieve", "pgmorl", "morlca")
METHOD_LABELS = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "qpensieve": "Q-Pensieve",
    "pgmorl": "PGMORL",
    "morlca": "MORL-CA",
}
HIGHER_BETTER = {"HV": True, "EU": True, "AS": True, "Reg": False, "Lat": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=PROJECT_ROOT / "results_100_condition_seed_study" / "quick_real_v1",
    )
    parser.add_argument("--env-keys", nargs="+", choices=ENV_KEYS, default=list(ENV_KEYS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--replication-ids",
        nargs="+",
        type=int,
        default=None,
        help="optional subset for scorer validation only; do not use for final inference",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[idx], dtype=np.float64)


def _reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    return lower - 0.1 * np.maximum(np.abs(lower), 1.0)


def _front_metrics(front: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    hv = float(Hypervolume(ref_point=-ref).do(-front))
    weights = generate_w_batch_test(front.shape[1], 0.5)
    eu = float(compute_eu(front, weights))
    return hv, eu


def _load_run_rows(
    run_dir: Path,
    method: str,
    expected_plan: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if not run_is_complete(run_dir, method, expected_plan):
        raise ValueError(f"Incomplete 100-condition run rejected: {run_dir}")
    manifest_path = run_dir / "study_export_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("conditions", 0)) != 100 or str(manifest.get("method")) != method:
        raise ValueError(f"Invalid study export manifest: {manifest_path}")
    artifact_dir = run_dir / "shared_regime_returns"
    rows = []
    for path in sorted(artifact_dir.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    if len(rows) != 100:
        raise ValueError(f"{artifact_dir}: expected 100 condition exports, found {len(rows)}")
    fronts: dict[int, np.ndarray] = {}
    solutions: dict[int, np.ndarray] = {}
    for row in rows:
        regime_id = int(row.get("regime_id", -1))
        if regime_id < 0:
            continue
        raw = row.get("front_points", row.get("points", row.get("solution_points", [])))
        solution_raw = row.get("solution_points", row.get("points", raw))
        front = _pareto_front(np.asarray(raw, dtype=np.float64))
        solution = np.asarray(solution_raw, dtype=np.float64)
        if front.ndim != 2 or len(front) == 0 or solution.ndim != 2 or len(solution) == 0:
            raise ValueError(f"{artifact_dir}: invalid points for regime {regime_id}")
        if regime_id in fronts:
            raise ValueError(f"{artifact_dir}: duplicate regime {regime_id}")
        fronts[regime_id] = front
        solutions[regime_id] = solution
    if set(fronts) != set(range(100)) or set(solutions) != set(range(100)):
        raise ValueError(f"{artifact_dir}: expected exact regime ids 0..99, found {len(fronts)} fronts / {len(solutions)} solution blocks")
    return fronts, solutions


def _trace_metrics(solution_blocks: dict[int, np.ndarray]) -> tuple[float, float, int]:
    blocks = [solution_blocks[index] for index in range(100)]
    solution_count = min(block.shape[0] for block in blocks)
    if solution_count <= 0:
        return float("nan"), float("nan"), 0
    regrets: list[float] = []
    latencies: list[float] = []
    regime_ids = np.arange(100, dtype=np.int64)
    for solution_index in range(solution_count):
        sequence = np.stack([block[solution_index] for block in blocks], axis=0)
        trace = compute_trace_shift_metrics(
            {"obj": sequence, "regime_id": regime_ids},
            eval_delta_weight=0.5,
            recovery_window=3,
            use_regime_id=True,
            min_shift_gap=1,
        )
        if int(trace.get("trace_shift_count", 0)) > 0:
            regrets.append(float(trace["trace_shift_regret"]))
            latencies.append(float(trace["trace_recovery_latency"]))
    if not regrets:
        return float("nan"), float("nan"), 0
    return float(np.mean(regrets)), float(np.mean(latencies)), len(regrets)


def _replication_metrics(
    env_key: str,
    replication: int,
    training_seed: int,
    method_data: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]],
) -> list[dict[str, Any]]:
    per_method: dict[str, dict[str, list[float]]] = {
        method: {"HV": [], "EU": [], "AS": []} for method in method_data
    }
    for regime_id in range(100):
        fronts = [method_data[method][0][regime_id] for method in method_data]
        reference = _reference_point(fronts)
        scores: dict[str, tuple[float, float]] = {
            method: _front_metrics(method_data[method][0][regime_id], reference)
            for method in method_data
        }
        best_hv = max(value[0] for value in scores.values())
        best_eu = max(value[1] for value in scores.values())
        for method, (hv, eu) in scores.items():
            hv_gap = max(0.0, best_hv - hv) / max(abs(best_hv), 1e-8)
            eu_gap = max(0.0, best_eu - eu) / max(abs(best_eu), 1e-8)
            per_method[method]["HV"].append(hv)
            per_method[method]["EU"].append(eu)
            per_method[method]["AS"].append(float(np.clip(1.0 - 0.5 * (hv_gap + eu_gap), 0.0, 1.0)))
    records: list[dict[str, Any]] = []
    for method, metrics in per_method.items():
        regret, latency, trace_count = _trace_metrics(method_data[method][1])
        records.append(
            {
                "environment": env_key,
                "replication": replication,
                "training_seed": training_seed,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "HV": float(np.mean(metrics["HV"])),
                "EU": float(np.mean(metrics["EU"])),
                "AS": float(np.mean(metrics["AS"])),
                "Reg": regret,
                "Lat": latency,
                "conditions": 100,
                "trace_solution_count": trace_count,
            }
        )
    return records


def _paired_test(differences: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n < 2:
        return {"n": n, "mean_difference": float("nan"), "std_difference": float("nan"), "t_statistic": float("nan"), "p_one_sided": float("nan"), "p_two_sided": float("nan")}
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0.0:
        if mean > 0:
            t_stat, one_sided, two_sided = float("inf"), 0.0, 0.0
        elif mean < 0:
            t_stat, one_sided, two_sided = -float("inf"), 1.0, 0.0
        else:
            t_stat, one_sided, two_sided = 0.0, 0.5, 1.0
    else:
        t_stat = mean / (std / math.sqrt(n))
        one_sided = float(student_t.sf(t_stat, df=n - 1))
        two_sided = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    return {"n": n, "mean_difference": mean, "std_difference": std, "t_statistic": t_stat, "p_one_sided": one_sided, "p_two_sided": two_sided}


def _holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [float("nan")] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _write_markdown(
    path: Path,
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    alpha: float,
) -> None:
    n_replications = int(metrics["replication"].nunique())
    inference_note = (
        f"Paired t-tests use {n_replications} independently initialized training replications, not the 100 conditions, as the sample unit."
        if n_replications >= 2
        else "Only one replication is present; the displayed t-test columns are non-inferential diagnostics and cannot establish significance."
    )
    lines = [
        "# 100-Condition Multi-Seed Quick Study",
        "",
        "This is a real short-budget simulation study. Each table entry aggregates 100 fixed operating conditions within one independently initialized training replication. " + inference_note,
        "",
        "HV, EU, and AS are recomputed condition-wise with a common reference point and common method pool inside each replication. Reg and Lat are reconstructed from aligned solution traces. One-sided hypotheses are OW-CMORL advantage; Holm correction spans every environment, baseline, and metric comparison below.",
        "",
        "## Replication Means",
        "",
        "| Environment | Method | HV mean | EU mean | AS mean | Reg mean | Lat mean | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = metrics.groupby(["environment", "method_label"], sort=False)[["HV", "EU", "AS", "Reg", "Lat"]].mean().reset_index()
    counts = metrics.groupby(["environment", "method_label"], sort=False).size().reset_index(name="n")
    summary = summary.merge(counts, on=["environment", "method_label"])
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['environment']} | {row['method_label']} | {row['HV']:.6g} | {row['EU']:.6g} | {row['AS']:.4f} | {row['Reg']:.6g} | {row['Lat']:.4f} | {int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            "`difference` is OW-CMORL minus baseline for HV/EU/AS, and baseline minus OW-CMORL for Reg/Lat. Positive values support OW-CMORL. `significant` requires positive effect and Holm-adjusted one-sided p < alpha.",
            "",
            "| Environment | Baseline | Metric | Difference | t | one-sided p | Holm p | Significant |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in tests.iterrows():
        t_text = "inf" if math.isinf(float(row["t_statistic"])) and float(row["t_statistic"]) > 0 else f"{row['t_statistic']:.4f}"
        lines.append(
            f"| {row['environment']} | {row['baseline_label']} | {row['metric']} | {row['mean_difference']:.6g} | {t_text} | {row['p_one_sided']:.3g} | {row['p_holm_one_sided']:.3g} | {'yes' if row['significant_holm'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.study_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.study_root / "seed_manifest.json").read_text())
    replications = list(manifest.get("replications", []))
    if args.replication_ids is not None:
        requested = {int(value) for value in args.replication_ids}
        replications = [row for row in replications if int(row.get("replication", -1)) in requested]
        if len(replications) != len(requested):
            raise ValueError(f"Requested replication ids missing from manifest: {sorted(requested)}")
    if len(replications) < 2:
        print("warning: fewer than two replications selected; metrics are a scorer validation only and t-tests are non-inferential.")

    records: list[dict[str, Any]] = []
    for replication_row in replications:
        replication = int(replication_row["replication"])
        training_seed = int(replication_row["training_seed"])
        run_parent = args.study_root / "runs" / f"rep_{replication:02d}_seed_{training_seed}"
        for env_key in args.env_keys:
            plan_path = args.study_root / "plans" / f"{env_key}_shared_plan.json"
            expected_plan = list(json.loads(plan_path.read_text()).get("shared_regime_seed_plan", []))
            if len(expected_plan) != 100:
                raise ValueError(f"Invalid 100-condition plan: {plan_path}")
            data: dict[str, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = {}
            for method in args.methods:
                data[method] = _load_run_rows(run_parent / env_key / method, method, expected_plan)
            records.extend(_replication_metrics(env_key, replication, training_seed, data))
    metrics = pd.DataFrame(records).sort_values(["environment", "replication", "method"])
    metrics.to_csv(out_dir / "replication_metrics.csv", index=False)

    tests: list[dict[str, Any]] = []
    for env_key in args.env_keys:
        env_metrics = metrics[metrics["environment"] == env_key]
        ow = env_metrics[env_metrics["method"] == "dynamic"].set_index("replication")
        for baseline in (method for method in args.methods if method != "dynamic"):
            other = env_metrics[env_metrics["method"] == baseline].set_index("replication")
            shared = ow.index.intersection(other.index).sort_values()
            if len(shared) != len(replications):
                raise ValueError(f"{env_key}/{baseline}: expected {len(replications)} paired repetitions, found {len(shared)}")
            for metric, higher_is_better in HIGHER_BETTER.items():
                differences = (ow.loc[shared, metric] - other.loc[shared, metric]).to_numpy(dtype=np.float64)
                if not higher_is_better:
                    differences = -differences
                test = _paired_test(differences)
                tests.append(
                    {
                        "environment": env_key,
                        "baseline": baseline,
                        "baseline_label": METHOD_LABELS[baseline],
                        "metric": metric,
                        "direction": "OW - baseline" if higher_is_better else "baseline - OW",
                        **test,
                    }
                )
    test_df = pd.DataFrame(tests)
    valid = test_df["p_one_sided"].notna()
    test_df["p_holm_one_sided"] = np.nan
    test_df.loc[valid, "p_holm_one_sided"] = _holm_adjust(test_df.loc[valid, "p_one_sided"].astype(float).tolist())
    test_df["significant_holm"] = (test_df["mean_difference"] > 0.0) & (test_df["p_holm_one_sided"] < float(args.alpha))
    test_df.to_csv(out_dir / "paired_t_tests.csv", index=False)
    _write_markdown(out_dir / "REPORT.md", metrics, test_df, float(args.alpha))
    (out_dir / "scoring_manifest.json").write_text(
        json.dumps(
            {
                "study_root": str(args.study_root),
                "replications": len(replications),
                "conditions_per_replication": 100,
                "test": "one-sided paired t-test, Holm corrected across all comparison hypotheses",
                "alpha": float(args.alpha),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out_dir": str(out_dir), "replication_rows": len(metrics), "tests": len(test_df)}, indent=2))


if __name__ == "__main__":
    main()
