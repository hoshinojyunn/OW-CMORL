#!/usr/bin/env python3
"""Summarize the reproducible 100-condition lightweight LCPO run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_lcpo_lightweight_100",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "lcpo_lightweight_100_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "LCPO_LIGHTWEIGHT_100_RESULTS.md",
    )
    return parser.parse_args()


def _ci(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0, 1
    half_width = float(student_t.ppf(0.975, df=len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, half_width, int(len(arr))


def _expected_utility(front: np.ndarray) -> float:
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or len(front) == 0:
        return 0.0
    weights = np.random.default_rng(0).dirichlet(np.ones(front.shape[1]), size=64)
    return float(np.mean(np.max(front @ weights.T, axis=0)))


def _format(mean: float, ci: float) -> str:
    if abs(mean) >= 1e4 or (0 < abs(mean) < 1e-3):
        return f"{mean:.4e} +/- {ci:.4e}"
    return f"{mean:.4f} +/- {ci:.4f}"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        path = args.results_root / env_key / "seed0" / "summary.json"
        payload = json.loads(path.read_text())
        if payload.get("method") != "LCPO" or payload.get("env_key") != env_key:
            raise ValueError(f"Unexpected result payload: {path}")
        fronts = [np.asarray(row.get("points", []), dtype=np.float64) for row in payload.get("regime_fronts", [])]
        if len(fronts) != 100:
            raise ValueError(f"{env_key}: expected 100 condition fronts, found {len(fronts)}")
        eu_mean, eu_ci, eu_n = _ci([_expected_utility(front) for front in fronts])
        trace = dict(payload.get("trace_metrics", {}))
        rows.append(
            {
                "environment": env_key,
                "preferences": len(payload.get("preferences", [])),
                "conditions": len(fronts),
                "timesteps_per_preference": int(payload.get("total_timesteps_per_preference", 0)),
                "eu_mean": eu_mean,
                "eu_ci95": eu_ci,
                "eu_n": eu_n,
                "trace_shift_regret": float(trace.get("trace_shift_regret", float("nan"))),
                "trace_recovery_latency": float(trace.get("trace_recovery_latency", float("nan"))),
                "trace_recovery_score": float(trace.get("trace_recovery_score", float("nan"))),
                "source": str(path),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    lines = [
        "# Lightweight LCPO Results on Reconstructed 100-Condition Plans",
        "",
        "Each environment uses the full standard scalarization grid, 128 training steps per preference, and 100 reconstructed ID conditions. The table reports raw expected utility as a condition-wise mean with a two-sided 95% t-confidence interval. Trace metrics are the LCPO evaluator's mean across scalarization-policy trajectories.",
        "",
        "| Environment | Preferences | Conditions | Steps / preference | Raw EU | Trace regret | Trace latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['environment']} | {row['preferences']} | {row['conditions']} | "
            f"{row['timesteps_per_preference']} | {_format(row['eu_mean'], row['eu_ci95'])} | "
            f"{row['trace_shift_regret']:.6g} | {row['trace_recovery_latency']:.6g} |"
        )
    lines.extend(
        [
            "",
            "HV and adaptation score are intentionally not reported here. Table 1 requires a condition-wise common HV reference point and best-front pool formed from the original 100-condition fronts of every compared method. Those external-run artifacts are unavailable in this workspace, so adding this lightweight LCPO run directly to Table 1 would not be numerically valid.",
            "",
            "The per-condition LCPO fronts are retained in `results_lcpo_lightweight_100/<environment>/seed0/shared_regime_returns/`. They can be jointly rescored with the original method fronts once those artifacts are restored.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))
    print(json.dumps({"out_csv": str(args.out_csv), "out_md": str(args.out_md), "rows": len(rows)}))


if __name__ == "__main__":
    main()
