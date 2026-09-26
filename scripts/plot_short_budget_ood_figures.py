#!/usr/bin/env python
"""Create OW-CMORL short-update OOD figures from generated CSV summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ENV_ORDER = ("building", "evcharging", "cogen", "chlor_alkali")
ENV_LABELS = {
    "building": "Building",
    "evcharging": "EV charging",
    "cogen": "Cogen",
    "chlor_alkali": "Chlor-alkali",
}
METRIC_COLORS = {"HV": "#0072B2", "EU": "#D55E00"}


def _read_csv(path: Path):
    with path.open(newline="") as fp:
        return [dict(row) for row in csv.DictReader(fp)]


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # The four environment PDFs must keep one common canvas.  Tight
            # cropping makes their bounding boxes differ and breaks the
            # two-by-two LaTeX subfigure grid even when all include widths
            # are identical.
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def _pca2(id_values: np.ndarray, ood_values: np.ndarray):
    joined = np.vstack((id_values, ood_values))
    scale = joined.std(axis=0, keepdims=True)
    standard = (joined - joined.mean(axis=0, keepdims=True)) / np.where(scale > 1e-12, scale, 1.0)
    _left, singular, vectors_t = np.linalg.svd(standard, full_matrices=False)
    xy = standard @ vectors_t[:2].T
    explained = singular[:2] ** 2 / max(float(np.sum(singular ** 2)), 1e-12) * 100.0
    return xy[: len(id_values)], xy[len(id_values) :], explained


def _load_trace(path: Path):
    rows = _read_csv(path)
    fields = [key for key in rows[0] if key not in {"regime_index", "context_label"}]
    values = np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=np.float64)
    return fields, values


def plot_context_trajectories(trace_dir: Path, out_dir: Path):
    figure, axes = plt.subplots(1, 4, figsize=(7.05, 2.08))
    figure.subplots_adjust(wspace=0.38, bottom=0.25, right=0.925, top=0.88)
    color_map = plt.get_cmap("viridis")
    order_artist = None
    for panel, (axis, env_key) in enumerate(zip(axes, ENV_ORDER)):
        id_fields, id_values = _load_trace(trace_dir / f"{env_key}_id_trace.csv")
        ood_fields, ood_values = _load_trace(trace_dir / f"{env_key}_ood_trace.csv")
        if id_fields != ood_fields:
            raise ValueError(f"Incompatible trace schema for {env_key}")
        id_xy, ood_xy, explained = _pca2(id_values, ood_values)
        axis.plot(id_xy[:, 0], id_xy[:, 1], color="#9E9E9E", linewidth=0.8, alpha=0.72, zorder=1)
        axis.scatter(id_xy[:, 0], id_xy[:, 1], color="#9E9E9E", s=5, alpha=0.58, linewidths=0, zorder=2)
        for index in range(len(ood_xy) - 1):
            color = color_map(index / max(len(ood_xy) - 1, 1))
            axis.plot(ood_xy[index : index + 2, 0], ood_xy[index : index + 2, 1], color=color, linewidth=1.35, zorder=3)
        order_artist = axis.scatter(
            ood_xy[:, 0], ood_xy[:, 1], c=np.arange(len(ood_xy)), cmap="viridis", s=9, linewidths=0, zorder=4
        )
        axis.scatter(ood_xy[0, 0], ood_xy[0, 1], marker="o", s=27, color="#0072B2", edgecolors="white", linewidths=0.45, zorder=5)
        axis.scatter(ood_xy[-1, 0], ood_xy[-1, 1], marker="s", s=29, color="#009E73", edgecolors="white", linewidths=0.45, zorder=5)
        axis.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
        axis.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
        axis.tick_params(length=2.0, width=0.5)
    if order_artist is not None:
        colorbar = figure.colorbar(order_artist, ax=axes.tolist(), fraction=0.020, pad=0.025)
        colorbar.set_label("OOD condition order")
    for axis, env_key in zip(axes, ENV_ORDER):
        bbox = axis.get_position()
        figure.text((bbox.x0 + bbox.x1) / 2.0, 0.965, ENV_LABELS[env_key], ha="center", va="top", fontsize=7.4)
    figure.legend(
        handles=[
            Line2D([0], [0], color="#9E9E9E", marker="o", markersize=3, linewidth=0.8, label="ID trace"),
            Line2D([0], [0], color="#0072B2", marker="o", markersize=4, linewidth=0, label="OOD start"),
            Line2D([0], [0], color="#009E73", marker="s", markersize=4, linewidth=0, label="OOD end"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.44, -0.12),
    )
    output = out_dir / "ood_short_context_trajectories.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def _lookup(summary, env_key, phase, budget, metric):
    for row in summary:
        if (
            row["env_key"] == env_key
            and row["phase"] == phase
            and int(float(row["update_budget"])) == budget
            and row["metric"] == metric
        ):
            return float(row["mean"]), float(row["std"]), int(float(row["n"]))
    return None


def plot_response_traces(summary, out_dir: Path):
    figure, axes = plt.subplots(4, 2, figsize=(7.05, 6.15))
    for row_index, env_key in enumerate(ENV_ORDER):
        for col_index, metric in enumerate(("HV", "EU")):
            axis = axes[row_index, col_index]
            points = []
            pre = _lookup(summary, env_key, "id_pre", 0, metric)
            if pre is not None:
                points.append((0, "Initial ID", *pre))
            ood_budgets = sorted(
                {
                    int(float(row["update_budget"]))
                    for row in summary
                    if row["env_key"] == env_key and row["phase"] == "ood" and row["metric"] == metric
                }
            )
            for budget in ood_budgets:
                point = _lookup(summary, env_key, "ood", budget, metric)
                if point is not None:
                    points.append((len(points), f"U={budget}", *point))
            if ood_budgets:
                returned = _lookup(summary, env_key, "id_return", ood_budgets[-1], metric)
                if returned is not None:
                    points.append((len(points), "ID after updates", *returned))
            if points:
                x = np.asarray([point[0] for point in points], dtype=float)
                mean = np.asarray([point[2] for point in points], dtype=float)
                std = np.asarray([point[3] for point in points], dtype=float)
                axis.plot(x, mean, marker="o", markersize=3.0, linewidth=1.3, color=METRIC_COLORS[metric])
                if np.any(std > 0):
                    axis.fill_between(x, mean - std, mean + std, color=METRIC_COLORS[metric], alpha=0.18, linewidth=0)
                axis.set_xticks(x, [point[1] for point in points], rotation=28, ha="right")
            axis.set_ylabel(metric)
            axis.tick_params(length=2.0, width=0.5)
    for col_index, metric in enumerate(("HV", "EU")):
        bbox = axes[0, col_index].get_position()
        figure.text((bbox.x0 + bbox.x1) / 2.0, 0.992, metric, ha="center", va="top")
    for row_index, env_key in enumerate(ENV_ORDER):
        bbox = axes[row_index, 0].get_position()
        figure.text(0.012, (bbox.y0 + bbox.y1) / 2.0, ENV_LABELS[env_key], ha="left", va="center", rotation=90)
    figure.subplots_adjust(hspace=0.64, wspace=0.34, bottom=0.08, left=0.13, top=0.96)
    output = out_dir / "ood_short_response_traces.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def plot_diagnostics(diagnostics, out_dir: Path):
    figure, axes = plt.subplots(1, 3, figsize=(7.05, 2.05))
    configs = (
        ("zero_shot_loss", "Zero-shot OOD loss"),
        ("recovery_auc", "Relative recovery AUC"),
        ("id_return_retention", "ID-return retention"),
    )
    x = np.arange(len(ENV_ORDER), dtype=float)
    width = 0.32
    for axis, (key, label) in zip(axes, configs):
        for offset, metric in ((-width / 2, "HV"), (width / 2, "EU")):
            values = []
            for env_key in ENV_ORDER:
                rows = [float(row[f"{key}_{metric}"]) for row in diagnostics if row["env_key"] == env_key]
                values.append(float(np.mean(rows)) if rows else np.nan)
            axis.bar(x + offset, values, width=width, color=METRIC_COLORS[metric], label=metric)
        if key == "id_return_retention":
            axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_xticks(x, [ENV_LABELS[key] for key in ENV_ORDER], rotation=22, ha="right")
        axis.set_ylabel(label)
        axis.tick_params(length=2.0, width=0.5)
    axes[0].legend(frameon=False, loc="upper right")
    figure.subplots_adjust(wspace=0.46, bottom=0.30)
    output = out_dir / "ood_short_diagnostics.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def plot_environment_diagnostics(summary, env_key: str, out_dir: Path) -> Path:
    """Plot the ID anchors and four measured OOD update budgets for one environment."""

    figure, (axis, recovery_axis) = plt.subplots(
        2,
        1,
        figsize=(2.50, 2.95),
        gridspec_kw={"height_ratios": (1.75, 0.85)},
    )
    phase_points = [("id_pre", 0, "Initial ID")]
    phase_points.extend(("ood", budget, f"U={budget}") for budget in (0, 8, 16, 32))
    phase_points.append(("id_return", 32, "ID after\nupdates"))
    positions = np.arange(len(phase_points), dtype=float)
    width = 0.34
    ood_values: dict[str, np.ndarray] = {}
    for offset, metric in ((-width / 2.0, "HV"), (width / 2.0, "EU")):
        anchor = _lookup(summary, env_key, "id_pre", 0, metric)
        if anchor is None or abs(anchor[0]) <= 1e-12:
            raise ValueError(f"{env_key}/{metric} has no nonzero ID anchor for relative diagnostics")
        values = []
        errors = []
        for phase, budget, _label in phase_points:
            point = _lookup(summary, env_key, phase, budget, metric)
            if point is None:
                raise ValueError(f"missing {env_key}/{metric}/{phase}/U={budget} summary point")
            # An affine shift keeps larger raw values aligned with taller
            # bars even when an environment's native EU is negative.
            values.append(1.0 + (point[0] - anchor[0]) / abs(anchor[0]))
            errors.append(point[1] / abs(anchor[0]))
        values = np.asarray(values, dtype=float)
        if np.any(values <= 0.0):
            raise ValueError(f"{env_key}/{metric} has nonpositive relative values; log diagnostic is undefined")
        axis.bar(
            positions + offset,
            values,
            width=width,
            color=METRIC_COLORS[metric],
            label=metric,
            yerr=np.minimum(np.asarray(errors, dtype=float), 0.9 * values),
            error_kw={"elinewidth": 0.55, "capsize": 1.3, "capthick": 0.55},
        )
        ood_values[metric] = values[1:5].copy()
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.75)
    axis.set_yscale("log")
    axis.set_xticks(positions, [label for _phase, _budget, label in phase_points], rotation=30, ha="right")
    axis.set_ylabel("Normalized score\n(Initial ID = 1; log scale)")
    axis.tick_params(length=2.0, width=0.5)
    axis.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.34),
        ncol=2,
        handlelength=1.1,
        columnspacing=0.8,
        borderpad=0.15,
    )

    update_positions = np.arange(4, dtype=float)
    recovery_values: list[float] = []
    for offset, metric in ((-width / 2.0, "HV"), (width / 2.0, "EU")):
        values = ood_values[metric]
        baseline = float(values[0])
        scale = max(abs(baseline), 1e-12)
        # This affine normalization is valid for both positive and negative
        # utilities: an increase in the raw OOD score always raises the bar.
        recovery = 1.0 + (values - baseline) / scale
        recovery_values.extend(recovery.tolist())
        recovery_axis.bar(
            update_positions + offset,
            recovery,
            width=width,
            color=METRIC_COLORS[metric],
        )
    recovery_axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.70)
    low = min(recovery_values)
    high = max(recovery_values)
    span = max(high - low, 0.004)
    margin = max(0.008, 0.30 * span)
    recovery_axis.set_ylim(low - margin, high + margin)
    recovery_axis.set_xticks(update_positions, ["U=0", "U=8", "U=16", "U=32"])
    recovery_axis.set_ylabel("OOD recovery\n(U=0 = 1)")
    recovery_axis.tick_params(length=1.8, width=0.45, labelsize=5.9)

    figure.subplots_adjust(left=0.27, right=0.98, bottom=0.16, top=0.79, hspace=0.78)
    output = out_dir / f"ood_short_diagnostics_{env_key}.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "short_ood_local_refinement_v1",
    )
    parser.add_argument("--trace-dir", type=Path, default=PROJECT_ROOT / "analysis" / "ood_trace100_v3")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "innovation_paper" / "fig")
    return parser.parse_args()


def main():
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(args.analysis_dir / "short_ood_events_summary.csv")
    plot_context_trajectories(args.trace_dir, args.out_dir)
    for env_key in ENV_ORDER:
        plot_environment_diagnostics(summary, env_key, args.out_dir)
    print(f"[short-ood-figures] wrote PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()
