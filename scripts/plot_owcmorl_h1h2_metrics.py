#!/usr/bin/env python3
"""Plot the controlled million-slot H1/H2 ablation from its summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def add_labels(ax: plt.Axes, bars: list, *, decimals: int = 3) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary_csv)
    variants = ["full", "h1_current_token_no_encoder", "h2_no_forecast", "h2_no_history"]
    labels = ["Full", "H1", "F=0", "H=0"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    values = {
        metric: np.asarray([rows[variant][metric] for variant in variants])
        for metric in ("mean_hv", "mean_eu", "mean_regret", "mean_recovery_latency_steps")
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_quality, ax_regret, ax_recovery) = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.75),
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
        constrained_layout=True,
    )
    x = np.arange(len(variants))
    width = 0.34
    hv_bars = ax_quality.bar(x - width / 2.0, values["mean_hv"], width, label="HV", color="#0072B2")
    eu_bars = ax_quality.bar(x + width / 2.0, values["mean_eu"], width, label="EU", color="#E69F00")
    ax_quality.set_ylabel("Mean metric value")
    ax_quality.set_xticks(x, labels)
    ax_quality.set_ylim(0.0, 1.05)
    ax_quality.legend(frameon=False, ncol=2, loc="upper center")

    regret_bars = ax_regret.bar(x, values["mean_regret"], color=colors, width=0.62)
    ax_regret.set_ylabel("Mean regret")
    ax_regret.set_xticks(x, labels)
    ax_regret.set_ylim(0.0, 0.95)

    recovery_bars = ax_recovery.bar(x, values["mean_recovery_latency_steps"], color=colors, width=0.62)
    add_labels(ax_recovery, list(recovery_bars), decimals=2)
    ax_recovery.set_ylabel("Recovery (steps)")
    ax_recovery.set_xticks(x, labels)
    ax_recovery.set_ylim(0.0, max(1.0, 1.12 * float(values["mean_recovery_latency_steps"].max())))

    for ax in (ax_quality, ax_regret, ax_recovery):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.45, color="#D9D9D9")
        ax.tick_params(axis="x", pad=3)
        ax.set_axisbelow(True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "owcmorl_h1_h2_metrics.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if args.preview_png is not None:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(output)


if __name__ == "__main__":
    main()
