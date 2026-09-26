#!/usr/bin/env python3
"""Score the 10-condition quick study and report paired condition-level tests.

The test units are the ten fixed aligned operating conditions.  This is useful
as a small within-trajectory diagnostic, but is not equivalent to a multi-seed
generalization test because the conditions share one trained model per method.
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

from scripts.run_10_condition_ttest_study import ENV_KEYS, METHODS, run_is_complete
from src.dynamic_morl.utils import compute_eu, generate_w_batch_test


METHOD_LABELS = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "qpensieve": "Q-Pensieve",
    "pgmorl": "PGMORL",
    "morlca": "MORL-CA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=PROJECT_ROOT / "results_10_condition_ttest_study" / "quick_real_v1",
    )
    parser.add_argument("--env-keys", nargs="+", choices=ENV_KEYS, default=list(ENV_KEYS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("Expected a nonempty 2-D solution array.")
    indices = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[indices], dtype=np.float64)


def _reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    values = np.concatenate(fronts, axis=0)
    lower = values.min(axis=0)
    return lower - 0.1 * np.maximum(np.abs(lower), 1.0)


def _front_metrics(front: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    hv = float(Hypervolume(ref_point=-reference).do(-front))
    weights = generate_w_batch_test(front.shape[1], 0.5)
    eu = float(compute_eu(front, weights))
    return hv, eu


def _load_run_rows(run_dir: Path, method: str, expected_plan: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    if not run_is_complete(run_dir, method, expected_plan):
        raise ValueError(f"Incomplete 10-condition run rejected: {run_dir}")
    manifest_path = run_dir / "study_export_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("conditions", 0)) != len(expected_plan) or manifest.get("method") != method:
        raise ValueError(f"Invalid study export manifest: {manifest_path}")
    fronts: dict[int, np.ndarray] = {}
    artifact_dir = run_dir / "shared_regime_returns"
    for path in artifact_dir.glob("*.json"):
        row = json.loads(path.read_text())
        regime_id = int(row.get("regime_id", -1))
        raw = row.get("front_points", row.get("points", row.get("solution_points", [])))
        front = _pareto_front(np.asarray(raw, dtype=np.float64))
        if regime_id in fronts:
            raise ValueError(f"Duplicate condition export for {regime_id}: {artifact_dir}")
        fronts[regime_id] = front
    expected_ids = [int(row["regime_id"]) for row in expected_plan]
    if set(fronts) != set(expected_ids):
        raise ValueError(f"{artifact_dir}: expected {expected_ids}, found {sorted(fronts)}")
    return fronts


def _paired_test(differences: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n < 2:
        raise ValueError("At least two paired conditions are required for a t-test.")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0.0:
        t_stat = float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0
        p_one = 0.0 if mean > 0 else 1.0 if mean < 0 else 0.5
        p_two = 0.0 if mean != 0 else 1.0
    else:
        t_stat = mean / (std / math.sqrt(n))
        p_one = float(student_t.sf(t_stat, df=n - 1))
        p_two = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    return {
        "n": n,
        "mean_difference": mean,
        "std_difference": std,
        "standardized_effect_dz": mean / std if std > 0 else float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0,
        "t_statistic": t_stat,
        "df": n - 1,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.5g}"


def _write_report(path: Path, conditions: pd.DataFrame, tests: pd.DataFrame, alpha: float) -> None:
    lines = [
        "# 10-Condition Paired T-Test Quick Validation",
        "",
        "This is a real short-budget simulator validation using ten fixed, aligned shared-protocol conditions and one common training seed for each method. The paired t-test unit is an operating condition, not an independently trained seed; it is therefore a within-trajectory diagnostic and does not replace the full multi-seed experiment.",
        "",
        "For each condition, HV and EU are recomputed using a common reference derived from the union of all compared Pareto fronts. AS uses the best HV/EU in that same condition as its comparator. The one-sided alternative is that OW-CMORL is larger; Holm correction spans all reported environment, baseline, and metric hypotheses.",
        "",
        "## Condition Means",
        "",
        "| Environment | Method | HV mean | EU mean | AS mean | n conditions |",
        "|---|---|---:|---:|---:|---:|",
    ]
    summary = conditions.groupby(["environment", "method_label"], sort=False)[["HV", "EU", "AS"]].mean().reset_index()
    for _, row in summary.iterrows():
        n = int(len(conditions[(conditions["environment"] == row["environment"]) & (conditions["method_label"] == row["method_label"])]))
        lines.append(
            f"| {row['environment']} | {row['method_label']} | {row['HV']:.6g} | {row['EU']:.6g} | {row['AS']:.5f} | {n} |"
        )
    lines.extend(
        [
            "",
            "## Paired Tests",
            "",
            "`difference` is OW-CMORL minus the baseline. `significant_holm` requires a positive difference and Holm-adjusted one-sided p below alpha.",
            "",
            "| Environment | Baseline | Metric | Difference | t (df=9) | one-sided p | Holm p | Significant |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in tests.iterrows():
        lines.append(
            f"| {row['environment']} | {row['baseline_label']} | {row['metric']} | {_format_float(float(row['mean_difference']))} | {_format_float(float(row['t_statistic']))} | {_format_float(float(row['p_one_sided']))} | {_format_float(float(row['p_holm_one_sided']))} | {'yes' if row['significant_holm'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    study_root = args.study_root
    out_dir = args.out_dir or study_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_records: list[dict[str, Any]] = []
    for env_key in args.env_keys:
        plan_path = study_root / "plans" / f"{env_key}_shared_plan.json"
        expected_plan = list(json.loads(plan_path.read_text()).get("shared_regime_seed_plan", []))
        if len(expected_plan) < 2:
            raise ValueError(f"Invalid condition plan: {plan_path}")
        condition_ids = [int(row["regime_id"]) for row in expected_plan]
        data = {
            method: _load_run_rows(study_root / "runs" / env_key / method, method, expected_plan)
            for method in args.methods
        }
        for order, regime_id in enumerate(condition_ids):
            fronts = {method: data[method][regime_id] for method in args.methods}
            reference = _reference_point(list(fronts.values()))
            scores = {method: _front_metrics(front, reference) for method, front in fronts.items()}
            best_hv = max(score[0] for score in scores.values())
            best_eu = max(score[1] for score in scores.values())
            for method, (hv, eu) in scores.items():
                hv_gap = max(0.0, best_hv - hv) / max(abs(best_hv), 1e-8)
                eu_gap = max(0.0, best_eu - eu) / max(abs(best_eu), 1e-8)
                condition_records.append(
                    {
                        "environment": env_key,
                        "condition_order": order,
                        "regime_id": regime_id,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "HV": hv,
                        "EU": eu,
                        "AS": float(np.clip(1.0 - 0.5 * (hv_gap + eu_gap), 0.0, 1.0)),
                    }
                )
    condition_df = pd.DataFrame(condition_records).sort_values(["environment", "condition_order", "method"])
    condition_df.to_csv(out_dir / "condition_metrics.csv", index=False)

    tests: list[dict[str, Any]] = []
    for env_key in args.env_keys:
        env_rows = condition_df[condition_df["environment"] == env_key]
        ow = env_rows[env_rows["method"] == "dynamic"].set_index("regime_id")
        for baseline in (method for method in args.methods if method != "dynamic"):
            other = env_rows[env_rows["method"] == baseline].set_index("regime_id")
            shared = ow.index.intersection(other.index).sort_values()
            if len(shared) != len(ow) or len(shared) != len(other):
                raise ValueError(f"{env_key}/{baseline}: missing aligned condition exports")
            for metric in ("HV", "EU", "AS"):
                result = _paired_test((ow.loc[shared, metric] - other.loc[shared, metric]).to_numpy(dtype=np.float64))
                tests.append(
                    {
                        "environment": env_key,
                        "baseline": baseline,
                        "baseline_label": METHOD_LABELS[baseline],
                        "metric": metric,
                        "direction": "OW-CMORL - baseline",
                        **result,
                    }
                )
    test_df = pd.DataFrame(tests)
    test_df["p_holm_one_sided"] = _holm_adjust(test_df["p_one_sided"].astype(float).tolist())
    test_df["significant_raw"] = (test_df["mean_difference"] > 0.0) & (test_df["p_one_sided"] < float(args.alpha))
    test_df["significant_holm"] = (test_df["mean_difference"] > 0.0) & (test_df["p_holm_one_sided"] < float(args.alpha))
    test_df.to_csv(out_dir / "paired_condition_t_tests.csv", index=False)
    _write_report(out_dir / "REPORT.md", condition_df, test_df, float(args.alpha))
    (out_dir / "scoring_manifest.json").write_text(
        json.dumps(
            {
                "study_root": str(study_root),
                "conditions_per_environment": int(condition_df.groupby("environment")["regime_id"].nunique().min()),
                "test": "paired one-sided t-test using aligned conditions; Holm correction across all reported hypotheses",
                "alpha": float(args.alpha),
                "caveat": "conditions share a training seed and trajectory, so this is not an independent-seed inference",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out_dir": str(out_dir), "condition_rows": len(condition_df), "tests": len(test_df)}, indent=2))


if __name__ == "__main__":
    main()
