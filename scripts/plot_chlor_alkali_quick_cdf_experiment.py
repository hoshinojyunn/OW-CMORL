from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(
        description="Generate chlor_alkali quick CDF experiment plots without touching the main report."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "chlor_alkali_cdf_quick_experiment",
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


def _empirical_cdf(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(x_grid)
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / float(len(values))


def _norm(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(float(vmax) - float(vmin), 1e-12)
    return np.clip((np.asarray(values, dtype=np.float64) - float(vmin)) / span, 0.0, 1.0)


def _ordered_methods(methods: list[str]) -> list[str]:
    rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda method: (rank.get(method, len(METHOD_ORDER)), method))


def _load_selected_sources() -> dict[str, Path]:
    rows = report._dynamic_rows() + report._morl_rows()
    sources: dict[str, Path] = {}
    for row in rows:
        if str(row["env_key"]) != ENV_KEY:
            continue
        method = str(row["method"])
        if method not in METHOD_ORDER:
            continue
        source = Path(str(row["source"]))
        if method == "dynamic":
            shared_dir = source / "shared_regime_returns"
        else:
            shared_dir = source.parent / "shared_regime_returns"
        if shared_dir.exists():
            sources[method] = shared_dir
    missing = [method for method in METHOD_ORDER if method not in sources]
    if missing:
        raise FileNotFoundError(f"missing shared_regime_returns for: {missing}")
    return sources


def _load_solutions_by_method(sources: dict[str, Path]) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for method, shared_dir in sources.items():
        out[method] = {}
        for npz_path in sorted(shared_dir.glob("*.npz")):
            payload = np.load(npz_path, allow_pickle=True)
            if "solutions" not in payload:
                continue
            regime_id = int(np.asarray(payload["regime_id"]).reshape(-1)[0])
            solutions = np.asarray(payload["solutions"], dtype=np.float64)
            if solutions.ndim == 2 and len(solutions) > 0:
                out[method][regime_id] = solutions
    common = sorted(set.intersection(*[set(rows.keys()) for rows in out.values()]))
    return {method: {regime_id: rows[regime_id] for regime_id in common} for method, rows in out.items()}


def _plot_legend(fig: plt.Figure, methods: list[str], y_anchor: float) -> None:
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[method], linewidth=3.2, label=METHOD_NAMES[method])
        for method in methods
    ]
    fig.legend(
        handles=handles,
        labels=[METHOD_NAMES[method] for method in methods],
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=len(methods),
        frameon=False,
    )


def _regime_normalized_objective_records(
    by_method: dict[str, dict[int, np.ndarray]],
    x_grid: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    methods = _ordered_methods(list(by_method))
    regime_ids = sorted(set.intersection(*[set(by_method[method]) for method in methods]))
    obj_num = next(iter(next(iter(by_method.values())).values())).shape[1]
    for obj_idx in range(obj_num):
        for method in methods:
            regime_cdfs = []
            raw_values = []
            for regime_id in regime_ids:
                pooled = np.concatenate([by_method[item][regime_id] for item in methods], axis=0)
                z = _norm(by_method[method][regime_id][:, obj_idx], pooled[:, obj_idx].min(), pooled[:, obj_idx].max())
                regime_cdfs.append(_empirical_cdf(z, x_grid))
                raw_values.extend(z.tolist())
            mean_cdf = np.asarray(regime_cdfs, dtype=np.float64).mean(axis=0)
            for x, y in zip(x_grid, mean_cdf):
                records.append(
                    {
                        "plot": "regime_normalized_objective",
                        "method": method,
                        "objective": f"obj_{obj_idx}",
                        "x": float(x),
                        "cdf": float(y),
                        "n_values": len(raw_values),
                    }
                )
    return pd.DataFrame(records)


def _plot_regime_normalized_objective_cdf(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = _ordered_methods(df["method"].unique().tolist())
    objectives = sorted(df["objective"].unique().tolist())
    fig, axes = plt.subplots(1, len(objectives), figsize=(5.15 * len(objectives), 4.2), sharey=True)
    if len(objectives) == 1:
        axes = [axes]
    for ax, objective in zip(axes, objectives):
        obj_df = df[df["objective"] == objective]
        for method in methods:
            sub = obj_df[obj_df["method"] == method].sort_values("x")
            lw = 3.8 if method == "dynamic" else 3.0
            ax.plot(sub["x"], sub["cdf"], color=METHOD_COLORS[method], linewidth=lw, alpha=0.98)
        ax.set_title(f"Regime-normalized {objective}")
        ax.set_xlabel("Normalized return within each regime")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Cumulative probability")
    _plot_legend(fig, methods, y_anchor=-0.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / "regime_normalized_objective_cdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _report_rows() -> list[dict[str, object]]:
    rows = report._dynamic_rows() + report._morl_rows()
    rows = [
        row
        for row in rows
        if str(row["env_key"]) == ENV_KEY and str(row["method"]) in METHOD_ORDER
    ]
    rows = report._score_regime_front_rows(rows)
    rows = report._attach_adapt_score(rows)
    return rows


def _performance_score_records(rows: list[dict[str, object]], x_grid: np.ndarray) -> pd.DataFrame:
    methods = _ordered_methods([str(row["method"]) for row in rows])
    metrics = {
        "HV": ("hv", True),
        "EU": ("eu", True),
        "Adapt score": ("adapt_score", True),
        "Trace regret": ("trace_shift_regret", False),
    }
    records: list[dict[str, object]] = []

    metric_values: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in rows:
        method = str(row["method"])
        regime_rows = list(row.get("regime_metric_rows", []))
        metric_values["HV"][method] = np.asarray([float(item["hv"]) for item in regime_rows], dtype=np.float64)
        metric_values["EU"][method] = np.asarray([float(item["eu"]) for item in regime_rows], dtype=np.float64)
        metric_values["Adapt score"][method] = np.asarray(row.get("adapt_score_samples", []), dtype=np.float64)
        trace = row.get("trace_samples", {})
        if isinstance(trace, dict):
            metric_values["Trace regret"][method] = np.asarray(trace.get("trace_shift_regret", []), dtype=np.float64)

    for metric_name, (_key, higher_better) in metrics.items():
        pooled = np.concatenate(
            [
                values[np.isfinite(values)]
                for values in metric_values[metric_name].values()
                if len(values) > 0
            ]
        )
        if len(pooled) == 0:
            continue
        vmin, vmax = float(np.min(pooled)), float(np.max(pooled))
        for method in methods:
            values = metric_values[metric_name].get(method, np.zeros(0, dtype=np.float64))
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            score = _norm(values, vmin, vmax)
            if not higher_better:
                score = 1.0 - score
            cdf = _empirical_cdf(score, x_grid)
            for x, y in zip(x_grid, cdf):
                records.append(
                    {
                        "plot": "performance_score",
                        "metric": metric_name,
                        "method": method,
                        "x": float(x),
                        "cdf": float(y),
                        "n_values": len(score),
                    }
                )
    return pd.DataFrame(records)


def _plot_performance_score_cdf(df: pd.DataFrame, out_dir: Path) -> Path:
    methods = _ordered_methods(df["method"].unique().tolist())
    metrics = ["HV", "EU", "Adapt score", "Trace regret"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.35 * len(metrics), 4.2), sharey=True)
    for ax, metric in zip(axes, metrics):
        metric_df = df[df["metric"] == metric]
        for method in methods:
            sub = metric_df[metric_df["method"] == method].sort_values("x")
            if sub.empty:
                continue
            lw = 3.8 if method == "dynamic" else 3.0
            ax.plot(sub["x"], sub["cdf"], color=METHOD_COLORS[method], linewidth=lw, alpha=0.98)
        title = metric if metric != "Trace regret" else "Trace regret (inverted)"
        ax.set_title(title)
        ax.set_xlabel("Normalized performance score")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Cumulative probability")
    _plot_legend(fig, methods, y_anchor=-0.03)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / "performance_score_cdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_audit_summary(
    objective_df: pd.DataFrame,
    score_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    lines = [
        "# chlor_alkali CDF quick experiment audit",
        "",
        "This directory is independent of `EXPERIMENT_RESULTS_ZH_20_REGIME.md`.",
        "It reuses the existing shared-protocol `solutions` arrays and redraws CDFs with diagnostic normalizations.",
        "",
        "## Generated figures",
        "",
        "- `regime_normalized_objective_cdf_chlor_alkali.png`: objective-wise CDF after normalizing within each shared regime.",
        "- `performance_score_cdf_chlor_alkali.png`: CDF over normalized performance scores; for trace regret the score is inverted so larger is better.",
        "",
        "## Objective CDF mean normalized return",
        "",
    ]
    objective_means = []
    for (objective, method), sub in objective_df.groupby(["objective", "method"]):
        # E[X] = integral_0^1 survival(x) dx for a [0,1] score.
        x = sub.sort_values("x")["x"].to_numpy(dtype=np.float64)
        cdf = sub.sort_values("x")["cdf"].to_numpy(dtype=np.float64)
        mean = float(np.trapz(1.0 - cdf, x))
        objective_means.append({"objective": objective, "method": method, "mean_score": mean})
    obj_table = pd.DataFrame(objective_means).pivot(index="objective", columns="method", values="mean_score")
    lines.append(obj_table.reindex(columns=METHOD_ORDER).round(4).to_markdown())
    lines.extend(["", "## Performance-score CDF mean score", ""])
    score_means = []
    for (metric, method), sub in score_df.groupby(["metric", "method"]):
        x = sub.sort_values("x")["x"].to_numpy(dtype=np.float64)
        cdf = sub.sort_values("x")["cdf"].to_numpy(dtype=np.float64)
        mean = float(np.trapz(1.0 - cdf, x))
        score_means.append({"metric": metric, "method": method, "mean_score": mean})
    score_table = pd.DataFrame(score_means).pivot(index="metric", columns="method", values="mean_score")
    lines.append(score_table.reindex(columns=METHOD_ORDER).round(4).to_markdown())
    lines.append("")
    out_path = out_dir / "AUDIT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()
    x_grid = np.linspace(0.0, 1.0, int(args.grid_points))
    sources = _load_selected_sources()
    by_method = _load_solutions_by_method(sources)
    objective_df = _regime_normalized_objective_records(by_method, x_grid)
    score_df = _performance_score_records(_report_rows(), x_grid)
    objective_csv = args.out_dir / "regime_normalized_objective_cdf.csv"
    score_csv = args.out_dir / "performance_score_cdf.csv"
    objective_df.to_csv(objective_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    outputs = [
        _plot_regime_normalized_objective_cdf(objective_df, args.out_dir),
        _plot_performance_score_cdf(score_df, args.out_dir),
        _write_audit_summary(objective_df, score_df, args.out_dir),
        objective_csv,
        score_csv,
    ]
    print(json.dumps({"out_dir": str(args.out_dir), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
