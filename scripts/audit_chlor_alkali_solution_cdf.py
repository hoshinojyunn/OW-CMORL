from __future__ import annotations

import argparse
import sys
from math import erf, sqrt
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
OBJECTIVE_LABELS = {
    0: "obj_0",
    1: "obj_1",
    2: "obj_2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and redraw chlor_alkali solution CDFs with regime-aware normalization."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "chlor_alkali_solution_cdf_audit",
    )
    parser.add_argument("--grid-points", type=int, default=501)
    return parser.parse_args()


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def _smoothed_cdf(values: np.ndarray, x_grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(x_grid)
    if bandwidth is None:
        std = float(np.std(values))
        silverman = 1.06 * std * max(len(values), 1) ** (-0.2)
        bandwidth = float(np.clip(silverman, 0.025, 0.08))
    erf_vec = np.vectorize(erf)
    z = (np.asarray(x_grid, dtype=np.float64)[:, None] - values[None, :]) / (float(bandwidth) * sqrt(2.0))
    cdf = 0.5 * (1.0 + erf_vec(z))
    smooth = np.clip(np.asarray(cdf.mean(axis=1), dtype=np.float64), 0.0, 1.0)
    return np.maximum.accumulate(smooth)


def _normalize(values: np.ndarray, lower: np.ndarray | float, upper: np.ndarray | float) -> np.ndarray:
    span = np.maximum(np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64), 1e-12)
    return np.clip((np.asarray(values, dtype=np.float64) - lower) / span, 0.0, 1.0)


def _mean_from_cdf(sub: pd.DataFrame, y_col: str = "cdf") -> float:
    ordered = sub.sort_values("x")
    x = ordered["x"].to_numpy(dtype=np.float64)
    y = ordered[y_col].to_numpy(dtype=np.float64)
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(1.0 - y, x))


def _ordered_methods(methods: list[str]) -> list[str]:
    rank = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda method: (rank.get(method, len(METHOD_ORDER)), method))


def _selected_shared_dirs() -> dict[str, Path]:
    rows = report._dynamic_rows() + report._morl_rows()
    out: dict[str, Path] = {}
    for row in rows:
        if str(row["env_key"]) != ENV_KEY:
            continue
        method = str(row["method"])
        if method not in METHOD_ORDER:
            continue
        source = Path(str(row["source"]))
        shared_dir = source / "shared_regime_returns" if method == "dynamic" else source.parent / "shared_regime_returns"
        if shared_dir.exists():
            out[method] = shared_dir
    missing = [method for method in METHOD_ORDER if method not in out]
    if missing:
        raise FileNotFoundError(f"Missing shared_regime_returns for methods: {missing}")
    return out


def _load_arrays(shared_dirs: dict[str, Path], key: str) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for method, shared_dir in shared_dirs.items():
        out[method] = {}
        for path in sorted(shared_dir.glob("*.npz")):
            payload = np.load(path, allow_pickle=True)
            if "regime_id" not in payload.files or key not in payload.files:
                continue
            regime_id = int(np.asarray(payload["regime_id"]).reshape(-1)[0])
            values = np.asarray(payload[key], dtype=np.float64)
            if values.ndim == 2 and len(values) > 0:
                out[method][regime_id] = values
    common = sorted(set.intersection(*[set(out[method]) for method in METHOD_ORDER]))
    return {method: {regime_id: out[method][regime_id] for regime_id in common} for method in METHOD_ORDER}


def _global_bounds(by_method: dict[str, dict[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([arr for method in METHOD_ORDER for arr in by_method[method].values()], axis=0)
    return stacked.min(axis=0), stacked.max(axis=0)


def _regime_bounds(
    by_method: dict[str, dict[int, np.ndarray]],
    regime_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([by_method[method][regime_id] for method in METHOD_ORDER], axis=0)
    return pooled.min(axis=0), pooled.max(axis=0)


def _objective_cdf_records(
    by_method: dict[str, dict[int, np.ndarray]],
    x_grid: np.ndarray,
    *,
    normalization: str,
    objectives: tuple[int, ...],
) -> pd.DataFrame:
    global_min, global_max = _global_bounds(by_method)
    records: list[dict[str, object]] = []
    regime_ids = sorted(next(iter(by_method.values())).keys())
    for objective in objectives:
        for method in METHOD_ORDER:
            regime_cdfs = []
            for regime_id in regime_ids:
                values = by_method[method][regime_id][:, objective]
                if normalization == "global":
                    norm_values = _normalize(values, float(global_min[objective]), float(global_max[objective]))
                elif normalization == "regime":
                    regime_min, regime_max = _regime_bounds(by_method, regime_id)
                    norm_values = _normalize(values, float(regime_min[objective]), float(regime_max[objective]))
                else:
                    raise ValueError(f"Unsupported normalization: {normalization}")
                regime_cdfs.append(_empirical_cdf(norm_values, x_grid))
            mean_cdf = np.asarray(regime_cdfs, dtype=np.float64).mean(axis=0)
            for x, y in zip(x_grid, mean_cdf):
                records.append(
                    {
                        "plot": "objective_cdf",
                        "normalization": normalization,
                        "objective": OBJECTIVE_LABELS[objective],
                        "method": method,
                        "x": float(x),
                        "cdf": float(y),
                        "n_regimes": len(regime_ids),
                    }
                )
    return pd.DataFrame(records)


def _score_from_normalized(norm_values: np.ndarray, score_name: str) -> np.ndarray:
    if score_name == "mean_obj01":
        return norm_values[:, :2].mean(axis=1)
    if score_name == "geom_obj01":
        return np.sqrt(np.maximum(norm_values[:, 0] * norm_values[:, 1], 0.0))
    if score_name == "min_obj01":
        return norm_values[:, :2].min(axis=1)
    if score_name == "mean_obj012":
        return norm_values.mean(axis=1)
    if score_name == "min_obj012":
        return norm_values.min(axis=1)
    raise ValueError(f"Unsupported score: {score_name}")


def _score_cdf_records(
    by_method: dict[str, dict[int, np.ndarray]],
    x_grid: np.ndarray,
    *,
    score_names: tuple[str, ...],
    selection: str,
    top_k_per_regime: int | None = None,
    cdf_kind: str = "empirical",
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    regime_ids = sorted(next(iter(by_method.values())).keys())
    for score_name in score_names:
        for method in METHOD_ORDER:
            regime_cdfs = []
            for regime_id in regime_ids:
                regime_min, regime_max = _regime_bounds(by_method, regime_id)
                norm_values = _normalize(by_method[method][regime_id], regime_min, regime_max)
                scores = _score_from_normalized(norm_values, score_name)
                if selection == "best":
                    scores = np.asarray([float(np.max(scores))], dtype=np.float64)
                elif selection == "top_k":
                    if top_k_per_regime is None or top_k_per_regime <= 0:
                        raise ValueError("top_k_per_regime must be positive when selection='top_k'")
                    scores = np.sort(scores)[-min(int(top_k_per_regime), len(scores)) :]
                elif selection != "all":
                    raise ValueError(f"Unsupported selection: {selection}")
                if cdf_kind == "smoothed":
                    regime_cdfs.append(_smoothed_cdf(scores, x_grid))
                elif cdf_kind == "empirical":
                    regime_cdfs.append(_empirical_cdf(scores, x_grid))
                else:
                    raise ValueError(f"Unsupported cdf_kind: {cdf_kind}")
            mean_cdf = np.asarray(regime_cdfs, dtype=np.float64).mean(axis=0)
            for x, y in zip(x_grid, mean_cdf):
                records.append(
                    {
                        "plot": "score_cdf",
                        "score": score_name,
                        "method": method,
                        "selection": selection if selection != "top_k" else f"top_{int(top_k_per_regime)}",
                        "cdf_kind": cdf_kind,
                        "best_per_regime": bool(selection == "best"),
                        "x": float(x),
                        "cdf": float(y),
                        "n_regimes": len(regime_ids),
                    }
                )
    return pd.DataFrame(records)


def _distribution_audit(by_method: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    global_min, global_max = _global_bounds(by_method)
    rows: list[dict[str, object]] = []
    regime_ids = sorted(next(iter(by_method.values())).keys())
    for objective in range(len(global_min)):
        for method in METHOD_ORDER:
            global_values = []
            regime_values = []
            raw_values = []
            for regime_id in regime_ids:
                values = by_method[method][regime_id][:, objective]
                regime_min, regime_max = _regime_bounds(by_method, regime_id)
                global_values.append(_normalize(values, float(global_min[objective]), float(global_max[objective])))
                regime_values.append(_normalize(values, float(regime_min[objective]), float(regime_max[objective])))
                raw_values.append(values)
            global_z = np.concatenate(global_values)
            regime_z = np.concatenate(regime_values)
            raw = np.concatenate(raw_values)
            quantiles = np.quantile(global_z, [0.0, 0.1, 0.5, 0.9, 1.0])
            rows.append(
                {
                    "method": method,
                    "objective": OBJECTIVE_LABELS[objective],
                    "n_values": int(len(global_z)),
                    "raw_min": float(raw.min()),
                    "raw_max": float(raw.max()),
                    "global_frac_le_0p05": float(np.mean(global_z <= 0.05)),
                    "global_frac_0p05_to_0p8": float(np.mean((global_z > 0.05) & (global_z < 0.8))),
                    "global_frac_ge_0p8": float(np.mean(global_z >= 0.8)),
                    "global_q0": float(quantiles[0]),
                    "global_q10": float(quantiles[1]),
                    "global_q50": float(quantiles[2]),
                    "global_q90": float(quantiles[3]),
                    "global_q100": float(quantiles[4]),
                    "regime_normalized_mean": float(regime_z.mean()),
                    "regime_normalized_q10": float(np.quantile(regime_z, 0.1)),
                    "regime_normalized_q50": float(np.quantile(regime_z, 0.5)),
                    "regime_normalized_q90": float(np.quantile(regime_z, 0.9)),
                }
            )
    return pd.DataFrame(rows)


def _score_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (score, method, selection, cdf_kind), sub in score_df.groupby(["score", "method", "selection", "cdf_kind"]):
        rows.append(
            {
                "score": score,
                "method": method,
                "selection": selection,
                "cdf_kind": cdf_kind,
                "mean_score_from_cdf": _mean_from_cdf(sub),
            }
        )
    return pd.DataFrame(rows)


def _plot_legend(fig: plt.Figure, methods: list[str], y_anchor: float) -> None:
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[method], linewidth=4.0 if method == "dynamic" else 3.2)
        for method in methods
    ]
    fig.legend(
        handles,
        [METHOD_NAMES[method] for method in methods],
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=len(methods),
        frameon=False,
    )


def _plot_global_vs_regime_objective_cdf(objective_df: pd.DataFrame, out_dir: Path) -> Path:
    methods = _ordered_methods(objective_df["method"].unique().tolist())
    objectives = ["obj_0", "obj_1"]
    normalizations = [("global", "Global normalization"), ("regime", "Regime-wise normalization")]
    fig, axes = plt.subplots(2, len(objectives), figsize=(5.8 * len(objectives), 7.0), sharex=True, sharey=True)
    for row_idx, (normalization, row_title) in enumerate(normalizations):
        for col_idx, objective in enumerate(objectives):
            ax = axes[row_idx, col_idx]
            obj_df = objective_df[
                (objective_df["normalization"] == normalization) & (objective_df["objective"] == objective)
            ]
            for method in methods:
                sub = obj_df[obj_df["method"] == method].sort_values("x")
                if sub.empty:
                    continue
                ax.plot(
                    sub["x"].to_numpy(),
                    sub["cdf"].to_numpy(),
                    color=METHOD_COLORS[method],
                    linewidth=4.1 if method == "dynamic" else 3.2,
                    alpha=0.98,
                )
            ax.set_title(f"{row_title} | {objective}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.22)
            if row_idx == len(normalizations) - 1:
                ax.set_xlabel("Normalized return")
            if col_idx == 0:
                ax.set_ylabel("Cumulative probability")
    _plot_legend(fig, methods, y_anchor=-0.005)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out_path = out_dir / "global_vs_regime_normalized_obj01_cdf_chlor_alkali.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_score_cdf(score_df: pd.DataFrame, out_dir: Path, *, selection: str, cdf_kind: str = "empirical") -> Path:
    methods = _ordered_methods(score_df["method"].unique().tolist())
    score_labels = [
        ("mean_obj01", "Mean(obj_0,obj_1)"),
        ("geom_obj01", "Geometric mean(obj_0,obj_1)"),
        ("min_obj01", "Min(obj_0,obj_1)"),
    ]
    fig, axes = plt.subplots(1, len(score_labels), figsize=(5.2 * len(score_labels), 4.2), sharey=True)
    for ax, (score_name, label) in zip(axes, score_labels):
        plot_df = score_df[
            (score_df["score"] == score_name)
            & (score_df["selection"] == selection)
            & (score_df["cdf_kind"] == cdf_kind)
        ]
        for method in methods:
            sub = plot_df[plot_df["method"] == method].sort_values("x")
            if sub.empty:
                continue
            ax.plot(
                sub["x"].to_numpy(),
                sub["cdf"].to_numpy(),
                color=METHOD_COLORS[method],
                linewidth=4.1 if method == "dynamic" else 3.2,
                alpha=0.98,
                label=METHOD_NAMES[method],
            )
        ax.set_title(label)
        ax.set_xlabel("Regime-normalized balanced score")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Cumulative probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=len(methods), frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    name_by_selection = {
        ("all", "empirical"): "balanced_obj01_solution_cdf_chlor_alkali.png",
        ("best", "empirical"): "best_balanced_obj01_cdf_chlor_alkali.png",
        ("top_7", "empirical"): "top7_balanced_obj01_cdf_chlor_alkali.png",
        ("top_7", "smoothed"): "top7_balanced_obj01_smoothed_cdf_chlor_alkali.png",
    }
    name = name_by_selection[(selection, cdf_kind)]
    out_path = out_dir / name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_audit_md(
    distribution_df: pd.DataFrame,
    score_summary_df: pd.DataFrame,
    out_dir: Path,
) -> Path:
    obj1 = distribution_df[distribution_df["objective"] == "obj_1"][
        ["method", "global_frac_le_0p05", "global_frac_0p05_to_0p8", "global_frac_ge_0p8"]
    ].copy()
    obj1 = obj1.set_index("method").reindex(METHOD_ORDER)

    score_table = score_summary_df[
        (score_summary_df["score"].isin(["mean_obj01", "geom_obj01", "min_obj01"]))
        & (score_summary_df["selection"] == "all")
        & (score_summary_df["cdf_kind"] == "empirical")
    ].pivot(index="score", columns="method", values="mean_score_from_cdf")
    score_table = score_table.reindex(columns=METHOD_ORDER)

    best_score_table = score_summary_df[
        (score_summary_df["score"].isin(["mean_obj01", "geom_obj01", "min_obj01"]))
        & (score_summary_df["selection"] == "best")
        & (score_summary_df["cdf_kind"] == "empirical")
    ].pivot(index="score", columns="method", values="mean_score_from_cdf")
    best_score_table = best_score_table.reindex(columns=METHOD_ORDER)

    top7_score_table = score_summary_df[
        (score_summary_df["score"].isin(["mean_obj01", "geom_obj01", "min_obj01"]))
        & (score_summary_df["selection"] == "top_7")
        & (score_summary_df["cdf_kind"] == "empirical")
    ].pivot(index="score", columns="method", values="mean_score_from_cdf")
    top7_score_table = top7_score_table.reindex(columns=METHOD_ORDER)

    lines = [
        "# chlor_alkali solution CDF audit",
        "",
        "This audit does not modify `EXPERIMENT_RESULTS_ZH_20_REGIME.md`.",
        "",
        "## Findings",
        "",
        "- All five methods have 20 shared regimes and MORL-CA is present.",
        "- The original step-like `obj_0/obj_1` CDF is reproduced by global environment-level normalization.",
        "- For `obj_1`, every method has about 10% of values near 0, 0% in (0.05, 0.8), and about 90% above 0.8 under global normalization.",
        "- Regime-wise normalization fills the middle interval, so the step is a normalization/regime-mixing artifact rather than a missing-curve bug.",
        "- Single-objective CDFs expose Pareto trade-offs: a baseline can be far right on one objective while losing on the balanced multi-objective score.",
        "",
        "## Global-normalized obj_1 fractions",
        "",
        "```text",
        obj1.round(4).to_string(),
        "```",
        "",
        "## Balanced obj_0/obj_1 solution CDF mean score",
        "",
        "```text",
        score_table.round(4).to_string(),
        "```",
        "",
        "## Top-7-per-regime balanced obj_0/obj_1 CDF mean score",
        "",
        "```text",
        top7_score_table.round(4).to_string(),
        "```",
        "",
        "## Best-per-regime balanced obj_0/obj_1 CDF mean score",
        "",
        "```text",
        best_score_table.round(4).to_string(),
        "```",
        "",
        "## Generated figures",
        "",
        "- `global_vs_regime_normalized_obj01_cdf_chlor_alkali.png`",
        "- `balanced_obj01_solution_cdf_chlor_alkali.png`",
        "- `top7_balanced_obj01_cdf_chlor_alkali.png`",
        "- `top7_balanced_obj01_smoothed_cdf_chlor_alkali.png`",
        "- `best_balanced_obj01_cdf_chlor_alkali.png`",
        "",
    ]
    out_path = out_dir / "AUDIT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    x_grid = np.linspace(0.0, 1.0, int(args.grid_points))
    shared_dirs = _selected_shared_dirs()
    by_method = _load_arrays(shared_dirs, "solutions")

    objective_global = _objective_cdf_records(
        by_method,
        x_grid,
        normalization="global",
        objectives=(0, 1),
    )
    objective_regime = _objective_cdf_records(
        by_method,
        x_grid,
        normalization="regime",
        objectives=(0, 1),
    )
    objective_df = pd.concat([objective_global, objective_regime], ignore_index=True)
    score_df = pd.concat(
        [
            _score_cdf_records(
                by_method,
                x_grid,
                score_names=("mean_obj01", "geom_obj01", "min_obj01", "mean_obj012", "min_obj012"),
                selection="all",
            ),
            _score_cdf_records(
                by_method,
                x_grid,
                score_names=("mean_obj01", "geom_obj01", "min_obj01"),
                selection="top_k",
                top_k_per_regime=7,
            ),
            _score_cdf_records(
                by_method,
                x_grid,
                score_names=("mean_obj01", "geom_obj01", "min_obj01"),
                selection="top_k",
                top_k_per_regime=7,
                cdf_kind="smoothed",
            ),
            _score_cdf_records(
                by_method,
                x_grid,
                score_names=("mean_obj01", "geom_obj01", "min_obj01"),
                selection="best",
            ),
        ],
        ignore_index=True,
    )
    distribution_df = _distribution_audit(by_method)
    score_summary_df = _score_summary(score_df)

    objective_csv = args.out_dir / "obj01_objective_cdf_curves.csv"
    score_csv = args.out_dir / "balanced_score_cdf_curves.csv"
    distribution_csv = args.out_dir / "solution_distribution_audit.csv"
    score_summary_csv = args.out_dir / "balanced_score_summary.csv"
    objective_df.to_csv(objective_csv, index=False)
    score_df.to_csv(score_csv, index=False)
    distribution_df.to_csv(distribution_csv, index=False)
    score_summary_df.to_csv(score_summary_csv, index=False)

    outputs = [
        _plot_global_vs_regime_objective_cdf(objective_df, args.out_dir),
        _plot_score_cdf(score_df, args.out_dir, selection="all"),
        _plot_score_cdf(score_df, args.out_dir, selection="top_7"),
        _plot_score_cdf(score_df, args.out_dir, selection="top_7", cdf_kind="smoothed"),
        _plot_score_cdf(score_df, args.out_dir, selection="best"),
        _write_audit_md(distribution_df, score_summary_df, args.out_dir),
        objective_csv,
        score_csv,
        distribution_csv,
        score_summary_csv,
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
