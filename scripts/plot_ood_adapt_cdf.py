#!/usr/bin/env python
"""Plot the protocol's OOD CDF from shared-regime adaptation scores.

The score is the same method-set-relative, per-regime ``adapt_score`` used by
the OOD summary table. It is a comparative proxy, not a calibrated return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_20_regime_report as base_report
import scripts.generate_ood_report as ood_report


ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
METHOD_ORDER = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca", "lcpo"]
METHOD_COLORS = {
    "dynamic": "#0072B2",
    "capql": "#D55E00",
    "pgmorl": "#009E73",
    "q_pensieve": "#CC79A7",
    "morlca": "#7F7F7F",
    "lcpo": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=str, default="ood_v1")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "ood_cdf",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_cdf",
    )
    return parser.parse_args()


def _load_rows(results_root: Path, profile: str) -> list[dict[str, Any]]:
    rows = ood_report._load_dynamic_rows(results_root, profile)
    rows.extend(ood_report._load_baseline_rows(results_root, profile))
    rows = base_report._align_rows_by_shared_plan(rows)
    rows = base_report._score_regime_front_rows(rows)
    return base_report._attach_adapt_score(rows)


def _cdf_rows(rows: list[dict[str, Any]], env_key: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        matched = [row for row in rows if row["env_key"] == env_key and row["method"] == method]
        if not matched:
            continue
        values = np.asarray(matched[0].get("adapt_score_samples", []), dtype=np.float64)
        values = np.sort(values[np.isfinite(values)])
        for rank, value in enumerate(values, start=1):
            records.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "sample_rank": int(rank),
                    "regime_adapt_score": float(value),
                    "cdf": float(rank / len(values)),
                }
            )
    return pd.DataFrame(records)


def _plot(env_key: str, frame: pd.DataFrame, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for method in METHOD_ORDER:
        method_frame = frame[frame["method"] == method]
        if method_frame.empty:
            continue
        axis.step(
            method_frame["regime_adapt_score"],
            method_frame["cdf"],
            where="post",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=ood_report._format_method_name(method),
        )
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel("Regime-level adaptation score")
    axis.set_ylabel("Empirical CDF")
    axis.set_title(f"{env_key}: OOD adaptation-score CDF")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(args.results_root, args.profile)
    manifest: dict[str, Any] = {
        "profile": args.profile,
        "metric": "regime-level adapt_score",
        "interpretation": "method-set-relative comparative adaptation proxy, not calibrated return",
        "environments": {},
    }
    for env_key in ENV_ORDER:
        frame = _cdf_rows(rows, env_key)
        if frame.empty:
            continue
        csv_path = args.analysis_dir / f"{args.profile}_{env_key}_adapt_cdf.csv"
        figure_path = args.figure_dir / f"{args.profile}_{env_key}_adapt_cdf.png"
        frame.to_csv(csv_path, index=False)
        _plot(env_key, frame, figure_path)
        manifest["environments"][env_key] = {
            "methods": sorted(frame["method"].unique().tolist()),
            "regime_samples_per_method": {
                method: int(len(frame[frame["method"] == method]))
                for method in sorted(frame["method"].unique().tolist())
            },
            "csv": str(csv_path),
            "figure": str(figure_path),
        }
        print(json.dumps({"env_key": env_key, "figure": str(figure_path), "rows": int(len(frame))}))
    (args.analysis_dir / f"{args.profile}_adapt_cdf_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
