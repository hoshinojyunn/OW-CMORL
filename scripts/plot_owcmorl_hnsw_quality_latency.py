#!/usr/bin/env python3
"""Plot HNSW quality retention and query latency from the summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT
       
        / "analysis"
        / "owcmorl_h1h2hnsw_million_slot"
        / "summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "owcmorl_h1h2hnsw",
    )
    parser.add_argument(
        "--preview-png",
        type=Path,
        default=None,
        help="Optional raster preview for visual review; the PDF remains the paper artifact.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="") as handle:
        return {
            row["variant"]: {
                key: value if key in {"variant", "backend"} else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    hnsw, exact = rows["full"], rows["exact"]
    labels = ["HNSW", "Exact"]
    relative_hv = np.array([100.0 * hnsw["mean_hv"] / exact["mean_hv"], 100.0])
    relative_eu = np.array([100.0 * hnsw["mean_eu"] / exact["mean_eu"], 100.0])
    p50 = np.array([hnsw["query_p50_ms"], exact["query_p50_ms"]])
    p95 = np.array([hnsw["query_p95_ms"], exact["query_p95_ms"]])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_quality, ax_latency) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.85),
        constrained_layout=True,
    )
    x = np.arange(2)
    width = 0.34
    hv_bars = ax_quality.bar(x - width / 2.0, relative_hv, width, label="HV", color="#0072B2")
    eu_bars = ax_quality.bar(x + width / 2.0, relative_eu, width, label="EU", color="#E69F00")
    for bars in (hv_bars, eu_bars):
        for backend_index, bar in enumerate(bars):
            if backend_index == 1:
                continue
            ax_quality.annotate(
                f"{bar.get_height():.3f}%",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_quality.annotate(
        "100.000%",
        (x[1], 100.0),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    quality_floor = min(float(relative_hv.min()), float(relative_eu.min())) - 0.015
    ax_quality.set_ylim(quality_floor, 100.025)
    ax_quality.ticklabel_format(style="plain", axis="y", useOffset=False)
    ax_quality.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_quality.set_ylabel("Relative solution quality to Exact (%)")
    ax_quality.set_xticks(x, labels)
    ax_quality.legend(frameon=False, ncol=2, loc="upper left")

    p50_bars = ax_latency.bar(x - width / 2.0, p50, width, label="p50", color="#009E73")
    p95_bars = ax_latency.bar(x + width / 2.0, p95, width, label="p95", color="#CC79A7")
    for bars in (p50_bars, p95_bars):
        for bar in bars:
            ax_latency.annotate(
                f"{bar.get_height():.3g}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax_latency.set_yscale("log")
    ax_latency.set_ylabel("Query latency (ms, log scale)")
    ax_latency.set_xticks(x, labels)
    ax_latency.legend(frameon=False, ncol=2, loc="upper left")

    for ax in (ax_quality, ax_latency):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.45, color="#D9D9D9")
        ax.set_axisbelow(True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "owcmorl_hnsw_quality_latency.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(output)


if __name__ == "__main__":
    main()
