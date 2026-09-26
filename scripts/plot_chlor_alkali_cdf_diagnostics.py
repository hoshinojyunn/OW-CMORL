from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import generate_20_regime_report as report
from analyze_benchmark_npz_archives import env_objective_bounds
from plot_20_regime_comparison_figures import _collect_group_fronts, _selected_comparison_rows


ENV_KEY = "chlor_alkali"
METHOD_ORDER = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca"]
METHOD_NAMES = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
}
METHOD_COLORS = {
    "dynamic": "#E31A1C",
    "capql": "#1F77B4",
    "pgmorl": "#2CA02C",
    "q_pensieve": "#FF7F0E",
    "morlca": "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chlor_alkali CDF diagnostics that expose method gaps.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "comparison_20_regime",
    )
    parser.add_argument(
        "--cdf-csv",
        type=Path,
        default=PROJECT_ROOT / "figures" / "comparison_20_regime" / "cdf_curves.csv",
    )
    parser.add_argument("--grid-points", type=int, default=201)
    return parser.parse_args()


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ordered_methods(methods: list[str]) -> list[str]:
    rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda method: (rank.get(method, len(METHOD_ORDER)), method))


def _empirical_cdf(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros_like(x_grid)
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / float(len(values))


def _normalize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(float(vmax) - float(vmin), 1e-12)
    return np.clip((np.asarray(values, dtype=np.float64) - float(vmin)) / span, 0.0, 1.0)


def _plot_legend(fig: plt.Figure, axes: list[plt.Axes], methods: list[str], y_anchor: float = -0.01) -> None:
    handles = []
    labels = []
    for method in methods:
        line = axes[0].plot([], [], color=METHOD_COLORS[method], linewidth=3.2, label=METHOD_NAMES[method])[0]
        handles.append(line)
        labels.append(METHOD_NAMES[method])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y_anchor), ncol=len(methods), frameon=False)


def plot_edge_cdf_with_gap(cdf_csv: Path, out_dir: Path) -> Path:
    df = pd.read_csv(cdf_csv)
    df = df[df["env_key"] == ENV_KEY].copy()
    methods = _ordered_methods(df["method"].dropna().astype(str).unique().tolist())
    objectives = sorted(df["objective"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(
        2,
        len(objectives),
        figsize=(5.3 * len(objectives), 6.4),
        sharex="col",
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for obj_idx, objective in enumerate(objectives):
        obj_df = df[df["objective"] == objective]
        pivot = obj_df.pivot(index="x", columns="method", values="mean_cdf").sort_index()
        top_ax = axes[0, obj_idx]
        gap_ax = axes[1, obj_idx]
        for method in methods:
            if method not in pivot:
                continue
            lw = 3.8 if method == "dynamic" else 3.0
            top_ax.plot(
                pivot.index.to_numpy(),
                pivot[method].to_numpy(),
                color=METHOD_COLORS[method],
                linewidth=lw,
                alpha=0.98,
                label=METHOD_NAMES[method],
            )
            if method != "dynamic":
                gap_ax.plot(
                    pivot.index.to_numpy(),
                    (pivot[method] - pivot["dynamic"]).to_numpy(),
                    color=METHOD_COLORS[method],
                    linewidth=2.4,
                    alpha=0.98,
                )
        top_ax.set_title(f"{ENV_KEY} | {objective}")
        top_ax.set_xlim(0.0, 1.0)
        top_ax.set_ylim(0.0, 1.0)
        top_ax.grid(alpha=0.22)
        gap_ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
        gap_ax.set_xlabel("Normalized return")
        gap_ax.set_ylabel(r"$\Delta$CDF vs OW")
        gap_ax.grid(alpha=0.22)
        gap = pivot[[method for method in methods if method != "dynamic" and method in pivot]].sub(pivot["dynamic"], axis=0)
        max_abs = float(np.nanmax(np.abs(gap.to_numpy()))) if not gap.empty else 0.1
        max_abs = max(0.05, min(0.45, max_abs * 1.12))
        gap_ax.set_ylim(-max_abs, max_abs)
    axes[0, 0].set_ylabel("Cumulative probability")
    _plot_legend(fig, [axes[0, 0]], methods, y_anchor=-0.015)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out_path = out_dir / "cdf_chlor_alkali_with_gap.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chlor_group_fronts() -> dict[tuple[str, str, str], np.ndarray]:
    rows = _selected_comparison_rows()
    group_fronts = _collect_group_fronts(rows)
    return {
        key: front
        for key, front in group_fronts.items()
        if key[0] == ENV_KEY and key[1] in METHOD_ORDER and front.ndim == 2 and len(front) > 0
    }


def plot_method_best_return_cdf(out_dir: Path, grid_points: int) -> Path:
    group_fronts = _chlor_group_fronts()
    bounds = env_objective_bounds(group_fronts)
    min_vals, max_vals = bounds[ENV_KEY]
    x_grid = np.linspace(0.0, 1.0, grid_points)
    objectives = [f"obj_{idx}" for idx in range(len(min_vals))]
    methods = _ordered_methods(sorted({method for (_env, method, _regime) in group_fronts}))

    fig, axes = plt.subplots(1, len(objectives), figsize=(5.3 * len(objectives), 4.2), sharey=True)
    if len(objectives) == 1:
        axes = [axes]
    for obj_idx, objective in enumerate(objectives):
        ax = axes[obj_idx]
        for method in methods:
            best_values = []
            for (_env, item_method, _regime), front in group_fronts.items():
                if item_method != method:
                    continue
                best_values.append(float(np.max(front[:, obj_idx])))
            norm_best = _normalize(np.asarray(best_values), float(min_vals[obj_idx]), float(max_vals[obj_idx]))
            cdf = _empirical_cdf(norm_best, x_grid)
            lw = 3.8 if method == "dynamic" else 3.0
            ax.plot(x_grid, cdf, color=METHOD_COLORS[method], linewidth=lw, alpha=0.98, label=METHOD_NAMES[method])
        ax.set_title(f"Best per regime | {objective}")
        ax.set_xlabel("Best normalized return")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Cumulative probability")
    _plot_legend(fig, [axes[0]], methods, y_anchor=-0.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / "best_return_cdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _chlor_report_rows() -> list[dict[str, object]]:
    rows = report._dynamic_rows() + report._morl_rows()
    rows = [
        row
        for row in rows
        if str(row["env_key"]) == ENV_KEY and str(row["method"]) in METHOD_ORDER
    ]
    rows = report._score_regime_front_rows(rows)
    rows = report._attach_adapt_score(rows)
    return rows


def _metric_samples(rows: list[dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    samples: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        metric_rows = list(row.get("regime_metric_rows", []))
        samples[method]["HV"] = np.asarray([float(item["hv"]) for item in metric_rows], dtype=np.float64)
        samples[method]["EU"] = np.asarray([float(item["eu"]) for item in metric_rows], dtype=np.float64)
        samples[method]["Adapt score"] = np.asarray(row.get("adapt_score_samples", []), dtype=np.float64)
        trace = row.get("trace_samples", {})
        if isinstance(trace, dict):
            samples[method]["Trace regret"] = np.asarray(trace.get("trace_shift_regret", []), dtype=np.float64)
    return samples


def plot_metric_score_ccdf(out_dir: Path, grid_points: int) -> Path:
    rows = _chlor_report_rows()
    samples = _metric_samples(rows)
    metrics = ["HV", "EU", "Adapt score", "Trace regret"]
    methods = _ordered_methods([method for method in METHOD_ORDER if method in samples])
    x_grid = np.linspace(0.0, 1.0, grid_points)

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 4.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        pooled = np.concatenate(
            [
                values[metric]
                for values in samples.values()
                if metric in values and len(values[metric]) > 0
            ]
        )
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled) == 0:
            continue
        vmin, vmax = float(np.min(pooled)), float(np.max(pooled))
        for method in methods:
            values = samples[method].get(metric, np.zeros(0, dtype=np.float64))
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            if metric == "Trace regret":
                score = 1.0 - _normalize(values, vmin, vmax)
            else:
                score = _normalize(values, vmin, vmax)
            cdf = _empirical_cdf(score, x_grid)
            ccdf = 1.0 - cdf
            lw = 3.8 if method == "dynamic" else 3.0
            ax.plot(x_grid, ccdf, color=METHOD_COLORS[method], linewidth=lw, alpha=0.98, label=METHOD_NAMES[method])
        ax.set_title(metric if metric != "Trace regret" else "Trace regret (inverted)")
        ax.set_xlabel("Normalized performance score")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("P(score >= x)")
    _plot_legend(fig, [axes[0]], methods, y_anchor=-0.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / "metric_score_ccdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_oracle_gap_cdf(out_dir: Path, grid_points: int) -> Path:
    rows = _chlor_report_rows()
    metrics = ["HV", "EU", "adapt_score"]
    labels = {"HV": "HV gap", "EU": "EU gap", "adapt_score": "Adapt-score gap"}
    methods = _ordered_methods([str(row["method"]) for row in rows])
    metric_tables: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        records = []
        for row in rows:
            method = str(row["method"])
            if metric == "adapt_score":
                values = list(row.get("adapt_score_samples", []))
                regime_ids = [
                    int(item["regime_id"])
                    for item in row.get("regime_metric_rows", [])
                ]
            else:
                key = "hv" if metric == "HV" else "eu"
                regime_ids = [int(item["regime_id"]) for item in row.get("regime_metric_rows", [])]
                values = [float(item[key]) for item in row.get("regime_metric_rows", [])]
            for regime_id, value in zip(regime_ids, values):
                records.append({"method": method, "regime_id": int(regime_id), "value": float(value)})
        metric_tables[metric] = pd.DataFrame(records)

    x_grid = np.linspace(0.0, 1.0, grid_points)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        table = metric_tables[metric]
        best_by_regime = table.groupby("regime_id")["value"].max()
        for method in methods:
            sub = table[table["method"] == method].copy()
            if sub.empty:
                continue
            gaps = []
            for row in sub.itertuples(index=False):
                best = float(best_by_regime.loc[int(row.regime_id)])
                denom = max(abs(best), 1e-12)
                gaps.append(max(0.0, (best - float(row.value)) / denom))
            gaps = np.clip(np.asarray(gaps, dtype=np.float64), 0.0, 1.0)
            cdf = _empirical_cdf(gaps, x_grid)
            lw = 3.8 if method == "dynamic" else 3.0
            ax.plot(x_grid, cdf, color=METHOD_COLORS[method], linewidth=lw, alpha=0.98, label=METHOD_NAMES[method])
        ax.set_title(labels[metric])
        ax.set_xlabel("Relative gap to per-regime oracle")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Cumulative probability")
    _plot_legend(fig, [axes[0]], methods, y_anchor=-0.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / "oracle_gap_cdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    outputs = [
        plot_edge_cdf_with_gap(args.cdf_csv, args.out_dir),
        plot_method_best_return_cdf(args.out_dir, args.grid_points),
        plot_metric_score_ccdf(args.out_dir, args.grid_points),
        plot_oracle_gap_cdf(args.out_dir, args.grid_points),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
