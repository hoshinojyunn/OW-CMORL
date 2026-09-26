#!/usr/bin/env python3
"""Visualize the shared 20-condition comparison underlying Table 3.

The script creates two publication figures from the saved per-condition
evaluation exports and the aggregate summary: (1) HV/EU performance profiles
over the 20 operating conditions and (2) normalized heatmaps of the three
dynamic aggregate metrics. The detailed numerical values remain in Table 3.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ("building", "evcharging", "cogen", "chlor_alkali")
ENVIRONMENT_LABELS = ("Building", "EV charging", "Cogen", "Chlor-alkali")
METHODS = ("capql", "pgmorl", "q_pensieve", "morlca", "dynamic")
METHOD_LABELS = {
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "dynamic": "OW-CMORL",
}
COLORS = {
    "capql": "#4D4D4D",
    "pgmorl": "#E69F00",
    "q_pensieve": "#CC79A7",
    "morlca": "#009E73",
    "dynamic": "#0072B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "experiment_results_20_regime_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "shared20_comparison_profiles",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for 300-dpi PNG previews.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        (row["env_key"], row["method"]): row
        for row in rows
        if row["env_key"] in ENVIRONMENTS and row["method"] in METHODS
    }
    expected = {(environment, method) for environment in ENVIRONMENTS for method in METHODS}
    missing = expected - selected.keys()
    if missing:
        raise ValueError(f"Missing shared-20 summary rows: {sorted(missing)}")
    return selected


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    indices = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[indices], dtype=np.float64)


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    return lower - 0.1 * np.maximum(np.abs(lower), 1.0)


def expected_utility(front: np.ndarray, num_preferences: int = 64) -> float:
    weights = np.random.default_rng(0).dirichlet(np.ones(front.shape[1]), size=num_preferences)
    return float(np.mean(np.max(front @ weights.T, axis=0)))


def front_metrics(front: np.ndarray, reference_point: np.ndarray) -> dict[str, float]:
    return {
        "hv": float(Hypervolume(ref_point=-reference_point).do(-front)),
        "eu": expected_utility(front),
    }


def load_condition_fronts(source: Path) -> dict[int, np.ndarray]:
    export_dir = source / "shared_regime_returns"
    exports = sorted(export_dir.glob("*.json"))
    fronts: dict[int, np.ndarray] = {}
    for export in exports:
        with export.open() as handle:
            row = json.load(handle)
        points = row.get("front_points", row.get("solution_points", []))
        front = pareto_front(np.asarray(points, dtype=np.float64))
        if front.size == 0:
            raise ValueError(f"Missing Pareto-front points in condition export: {export}")
        regime_id = int(row["regime_id"])
        fronts[regime_id] = front
    if len(fronts) != 20:
        raise ValueError(f"Expected 20 condition exports in {export_dir}, found {len(fronts)}")
    return fronts


def build_condition_scores(
    summary: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | str]]]:
    raw: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(dict)
    for environment in ENVIRONMENTS:
        for method in METHODS:
            source = Path(summary[(environment, method)]["source"])
            if source.is_file():
                source = source.parent
            raw[environment][method] = load_condition_fronts(source)

    score_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | str]] = []
    thresholds = np.linspace(0.0, 1.0, 101)
    for environment in ENVIRONMENTS:
        common_conditions = set.intersection(*(set(raw[environment][method]) for method in METHODS))
        if len(common_conditions) != 20:
            raise ValueError(f"Expected 20 common conditions for {environment}, found {len(common_conditions)}")
        normalized: dict[str, dict[str, list[float]]] = {
            metric: {method: [] for method in METHODS} for metric in ("hv", "eu")
        }
        for regime_id in sorted(common_conditions):
            condition_fronts = [raw[environment][method][regime_id] for method in METHODS]
            reference_point = derive_reference_point(condition_fronts)
            per_method_metrics = [front_metrics(front, reference_point) for front in condition_fronts]
            for metric in ("hv", "eu"):
                values = np.asarray([metrics[metric] for metrics in per_method_metrics])
                span = float(values.max() - values.min())
                if span <= np.finfo(float).eps:
                    scores = np.ones_like(values)
                else:
                    scores = (values - values.min()) / span
                for method, value, score in zip(METHODS, values, scores):
                    normalized[metric][method].append(float(score))
                    score_rows.append(
                        {
                            "environment": environment,
                            "condition_id": regime_id,
                            "metric": metric.upper(),
                            "method": METHOD_LABELS[method],
                            "raw_value": float(value),
                            "relative_score": float(score),
                        }
                    )
        for metric in ("hv", "eu"):
            for method in METHODS:
                scores = np.asarray(normalized[metric][method])
                survival = np.asarray([(scores >= threshold).mean() for threshold in thresholds])
                curve_rows.extend(
                    {
                        "environment": environment,
                        "metric": metric.upper(),
                        "method": METHOD_LABELS[method],
                        "threshold": float(threshold),
                        "fraction_conditions": float(value),
                    }
                    for threshold, value in zip(thresholds, survival)
                )
    return score_rows, curve_rows


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_profiles(score_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row["environment"]), str(row["metric"]), str(row["method"]))].append(
            float(row["relative_score"])
        )
    summaries = []
    for (environment, metric, method), values in sorted(grouped.items()):
        scores = np.asarray(values, dtype=np.float64)
        summaries.append(
            {
                "environment": environment,
                "metric": metric,
                "method": method,
                "mean_relative_score": float(scores.mean()),
                "median_relative_score": float(np.median(scores)),
                "condition_win_rate": float(np.mean(scores >= 1.0 - 1e-12)),
                "n_conditions": int(len(scores)),
            }
        )
    return summaries


def plot_profiles(
    curve_rows: list[dict[str, float | str]],
    out_dir: Path,
    preview_dir: Path | None,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(8.25, 4.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.86, bottom=0.19, wspace=0.18, hspace=0.20)
    for column, (environment, environment_label) in enumerate(zip(ENVIRONMENTS, ENVIRONMENT_LABELS)):
        fig.text(0.18 + 0.21 * column, 0.895, environment_label, ha="center", va="bottom", fontsize=9)
        for row_index, metric in enumerate(("HV", "EU")):
            axis = axes[row_index, column]
            for method in METHODS:
                points = [
                    row
                    for row in curve_rows
                    if row["environment"] == environment
                    and row["metric"] == metric
                    and row["method"] == METHOD_LABELS[method]
                ]
                thresholds = np.asarray([float(row["threshold"]) for row in points])
                fractions = np.asarray([float(row["fraction_conditions"]) for row in points])
                axis.step(
                    thresholds,
                    fractions,
                    where="post",
                    color=COLORS[method],
                    linewidth=2.1 if method == "dynamic" else 1.35,
                    label=METHOD_LABELS[method],
                )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.02)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            axis.grid(axis="y", linewidth=0.45, color="#D9D9D9")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("HV: share of conditions")
    axes[1, 0].set_ylabel("EU: share of conditions")
    fig.text(0.55, 0.08, "Condition-wise relative metric score (0 = worst, 1 = best)", ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.55, 0.105), frameon=False)
    output = out_dir / "shared20_hv_eu_performance_profiles.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview_dir is not None:
        fig.savefig(preview_dir / "shared20_hv_eu_performance_profiles.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def relative_scores(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    lower = float(values.min())
    upper = float(values.max())
    if upper - lower <= np.finfo(float).eps:
        return np.ones_like(values)
    if higher_is_better:
        return (values - lower) / (upper - lower)
    return (upper - values) / (upper - lower)


def plot_adaptation_heatmaps(
    summary: dict[tuple[str, str], dict[str, str]],
    out_dir: Path,
    preview_dir: Path | None,
) -> None:
    metrics = (
        ("adapt_score_mean", "adaptation_score", True),
        ("trace_shift_regret_mean", "shift_regret", False),
        ("trace_recovery_latency_mean", "recovery_latency", False),
    )
    for metric_key, output_stem, higher_is_better in metrics:
        fig, axis = plt.subplots(figsize=(3.1, 2.9))
        fig.subplots_adjust(left=0.25, right=0.92, top=0.98, bottom=0.24)
        values = np.asarray(
            [
                [float(summary[(environment, method)][metric_key]) for environment in ENVIRONMENTS]
                for method in METHODS
            ]
        )
        normalized = np.column_stack(
            [relative_scores(values[:, column], higher_is_better) for column in range(values.shape[1])]
        )
        image = axis.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(np.arange(len(ENVIRONMENT_LABELS)), ENVIRONMENT_LABELS, rotation=24, ha="right")
        axis.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
        for row_index in range(normalized.shape[0]):
            for column_index in range(normalized.shape[1]):
                value = normalized[row_index, column_index]
                color = "white" if value >= 0.60 else "black"
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        axis.tick_params(length=0)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
        colorbar.set_label("Relative score")
        output = out_dir / f"shared20_{output_stem}_heatmap.pdf"
        fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
        if preview_dir is not None:
            fig.savefig(preview_dir / f"shared20_{output_stem}_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary_csv)
    score_rows, curve_rows = build_condition_scores(summary)
    save_csv(args.out_dir / "shared20_condition_relative_scores.csv", score_rows)
    save_csv(args.out_dir / "shared20_hv_eu_performance_profile_curves.csv", curve_rows)
    save_csv(args.out_dir / "shared20_hv_eu_profile_summary.csv", summarize_profiles(score_rows))
    plot_profiles(curve_rows, args.out_dir, args.preview_dir)
    plot_adaptation_heatmaps(summary, args.out_dir, args.preview_dir)
    print(args.out_dir / "shared20_hv_eu_performance_profiles.pdf")
    print(args.out_dir / "shared20_adaptation_score_heatmap.pdf")
    print(args.out_dir / "shared20_shift_regret_heatmap.pdf")
    print(args.out_dir / "shared20_recovery_latency_heatmap.pdf")


if __name__ == "__main__":
    main()
