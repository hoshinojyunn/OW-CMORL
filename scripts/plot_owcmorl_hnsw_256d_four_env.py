#!/usr/bin/env python3
"""Plot four-environment HNSW Top-50 quality retention and batch latency."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("building", "evcharging", "cogen", "chlor_alkali")
LABELS = ("Building", "EV charging", "Cogen", "Chlor-alkali")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_hnsw_256d_four_env" / "summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "owcmorl_hnsw_256d_four_env",
    )
    parser.add_argument("--preview-png", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, dict[str, float | str]]]:
    rows: dict[str, dict[str, dict[str, float | str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["environment"], {})[row["backend"]] = {
                key: value if key in {"environment", "backend"} else float(value)
                for key, value in row.items()
            }
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    hnsw = [rows[env]["hnsw"] for env in ENVIRONMENTS]
    exact = [rows[env]["exact"] for env in ENVIRONMENTS]
    hv_retention = np.asarray(
        [100.0 * float(h["mean_hv"]) / float(e["mean_hv"]) for h, e in zip(hnsw, exact)]
    )
    eu_retention = np.asarray(
        [100.0 * float(h["mean_eu"]) / float(e["mean_eu"]) for h, e in zip(hnsw, exact)]
    )
    hnsw_batch = np.asarray([float(row["route_batch_seconds"]) for row in hnsw])
    exact_batch = np.asarray([float(row["route_batch_seconds"]) for row in exact])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(ENVIRONMENTS))
    width = 0.34

    fig_quality, ax_quality = plt.subplots(figsize=(3.55, 3.1), constrained_layout=True)
    ax_quality.bar(x - width / 2.0, hv_retention, width, label="HV", color="#0072B2")
    ax_quality.bar(x + width / 2.0, eu_retention, width, label="EU", color="#E69F00")
    ax_quality.axhline(100.0, color="#4D4D4D", linewidth=0.8, linestyle="--", label="Exact")
    quality_floor = min(float(hv_retention.min()), float(eu_retention.min())) - 0.015
    ax_quality.set_ylim(quality_floor, 100.006)
    ax_quality.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_quality.set_ylabel("Relative solution quality to Exact (%)")
    ax_quality.set_xticks(x, LABELS, rotation=18, ha="right")
    ax_quality.legend(frameon=False, ncol=3, loc="upper left")
    ax_quality.spines["top"].set_visible(False)
    ax_quality.spines["right"].set_visible(False)
    ax_quality.yaxis.grid(True, linewidth=0.45, color="#D9D9D9")
    ax_quality.set_axisbelow(True)

    fig_time, ax_time = plt.subplots(figsize=(3.55, 3.1), constrained_layout=True)
    ax_time.bar(x - width / 2.0, hnsw_batch, width, label="HNSW", color="#009E73")
    ax_time.bar(x + width / 2.0, exact_batch, width, label="Exact", color="#CC79A7")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("1,000 Top-50 routes (s, log scale)")
    ax_time.set_xticks(x, LABELS, rotation=18, ha="right")
    ax_time.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax_time.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.yaxis.grid(True, linewidth=0.45, color="#D9D9D9")
    ax_time.set_axisbelow(True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quality_output = args.out_dir / "owcmorl_hnsw_256d_quality.pdf"
    time_output = args.out_dir / "owcmorl_hnsw_256d_latency.pdf"
    fig_quality.savefig(quality_output, bbox_inches="tight", pad_inches=0.06)
    fig_time.savefig(time_output, bbox_inches="tight", pad_inches=0.06)
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        preview_quality = args.preview_png.with_name(f"{args.preview_png.stem}_quality.png")
        preview_time = args.preview_png.with_name(f"{args.preview_png.stem}_latency.png")
        fig_quality.savefig(preview_quality, dpi=300, bbox_inches="tight", pad_inches=0.06)
        fig_time.savefig(preview_time, dpi=300, bbox_inches="tight", pad_inches=0.06)
    print(quality_output)
    print(time_output)


if __name__ == "__main__":
    main()
