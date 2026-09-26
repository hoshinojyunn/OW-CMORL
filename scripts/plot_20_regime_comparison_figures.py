from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import generate_20_regime_report as report
from analyze_benchmark_npz_archives import (
    build_best_cdf_curves,
    build_cdf_curves,
    env_objective_bounds,
    plot_cdf_panels,
    plot_cdf_panels_with_prefix,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "comparison_20_regime"
DEFAULT_FACTOR_CSV = PROJECT_ROOT / "analysis" / "major_dynamic_factors_20_regime.csv"

FACTOR_SPECS: dict[str, list[dict[str, str]]] = {
    "building": [
        {
            "column": "outdoor_temp",
            "label": "Outdoor Temperature",
            "axis_label": "Normalized factor value",
            "color": "#d1495b",
        },
        {
            "column": "electricity_price",
            "label": "Electricity Price",
            "axis_label": "Normalized factor value",
            "color": "#2e86ab",
        },
        {
            "column": "target_temp",
            "label": "Target Temperature",
            "axis_label": "Normalized factor value",
            "color": "#3d9970",
        },
    ],
    "evcharging": [
        {
            "column": "deadline_pressure",
            "label": "Deadline Pressure",
            "axis_label": "Normalized factor value",
            "color": "#d1495b",
        },
        {
            "column": "prev_moer",
            "label": "Previous MOER",
            "axis_label": "Normalized factor value",
            "color": "#2e86ab",
        },
        {
            "column": "forecast_peak",
            "label": "Forecast Peak",
            "axis_label": "Normalized factor value",
            "color": "#e09f3e",
        },
    ],
    "cogen": [
        {
            "column": "renewables_ratio",
            "label": "Renewables Ratio",
            "axis_label": "Normalized factor value",
            "color": "#3d9970",
        },
        {
            "column": "target_power",
            "label": "Target Power",
            "axis_label": "Normalized factor value",
            "color": "#d1495b",
        },
        {
            "column": "gas_price",
            "label": "Gas Price",
            "axis_label": "Normalized factor value",
            "color": "#2e86ab",
        },
    ],
    "chlor_alkali": [
        {
            "column": "current_price",
            "label": "Electricity Price",
            "axis_label": "RMB/kWh",
            "color": "#2e86ab",
        },
        {
            "column": "total_current_limit_kA",
            "label": "Total Current Limit",
            "axis_label": "kA",
            "color": "#d1495b",
        },
        {
            "column": "stress_proxy",
            "label": "Stress Index",
            "axis_label": "Factor value",
            "color": "#e09f3e",
        },
        {
            "column": "aging_proxy",
            "label": "Aging Index",
            "axis_label": "Factor value",
            "color": "#3d9970",
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 20-regime CDF comparisons and major dynamic-factor trends for the current official runs."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--factor-csv", type=Path, default=DEFAULT_FACTOR_CSV)
    parser.add_argument("--cdf-grid-points", type=int, default=201)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    return parser.parse_args()


def _selected_comparison_rows() -> list[dict[str, Any]]:
    rows = report._dynamic_rows() + report._morl_rows() + report._scalar_rows()
    selected = []
    for row in rows:
        env_key = str(row["env_key"])
        method = report._normalize_method(row["method"])
        if method not in report.REPORT_METHODS.get(env_key, []):
            continue
        row = dict(row)
        row["method"] = method
        selected.append(row)
    return selected


def _selected_scalar_family_rows() -> list[dict[str, Any]]:
    rows = report._dynamic_rows() + report._morl_rows() + report._scalar_rows(report.SCALAR_SUPPLEMENT_METHODS)
    selected = []
    for row in rows:
        env_key = str(row["env_key"])
        method = report._normalize_method(row["method"])
        if method not in report.SCALAR_FAMILY_METHODS:
            continue
        row = dict(row)
        row["method"] = method
        selected.append(row)
    return selected


def _collect_group_fronts(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], np.ndarray]:
    group_fronts: dict[tuple[str, str, str], np.ndarray] = {}
    for row in rows:
        env_key = str(row["env_key"])
        method = str(row["method"])
        for regime_row in row.get("regime_front_rows", []):
            front = np.asarray(regime_row.get("front", np.zeros((0, 0))), dtype=np.float64)
            if front.ndim != 2 or len(front) == 0:
                continue
            regime = str(regime_row.get("regime", f"regime_{int(regime_row.get('regime_id', -1)):03d}"))
            group_fronts[(env_key, method, regime)] = front
    return group_fronts


def _dynamic_source_map(rows: list[dict[str, Any]]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for row in rows:
        if str(row["method"]) != "dynamic":
            continue
        out[str(row["env_key"])] = Path(str(row["source"]))
    return out


def _shared_regime_meta_df(dynamic_run_dir: Path, env_key: str) -> pd.DataFrame:
    shared_final_path = dynamic_run_dir / "regime_fronts" / "shared_final.json"
    payload = json.loads(shared_final_path.read_text())
    rows: list[dict[str, Any]] = []
    for regime_row in sorted(payload.get("regimes", []), key=lambda item: int(item.get("regime_id", -1))):
        regime_id = int(regime_row.get("regime_id", -1))
        if regime_id < 0:
            continue
        meta = regime_row.get("regime_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        rows.append(
            {
                "env_key": env_key,
                "regime_id": regime_id,
                "regime": str(regime_row.get("regime", f"regime_{regime_id:03d}")),
                **meta,
            }
        )
    return pd.DataFrame(rows).sort_values("regime_id").reset_index(drop=True)


def _chlor_factor_df() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "analysis" / "chlor_alkali_shared_20_regime_params.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    rename_map = {"regime_name": "regime"}
    df = df.rename(columns=rename_map)
    df["env_key"] = "chlor_alkali"
    return df.sort_values("regime_id").reset_index(drop=True)


def _major_factor_frames(dynamic_source_by_env: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for env_key in report.ENV_ORDER:
        if env_key == "chlor_alkali":
            frames[env_key] = _chlor_factor_df()
            continue
        run_dir = dynamic_source_by_env.get(env_key)
        if run_dir is None:
            continue
        frames[env_key] = _shared_regime_meta_df(run_dir, env_key)
    return frames


def _factor_long_df(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for env_key, df in frames.items():
        for spec in FACTOR_SPECS.get(env_key, []):
            column = spec["column"]
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            for regime_id, regime, value in zip(df["regime_id"], df["regime"], values):
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "env_key": env_key,
                        "regime_id": int(regime_id),
                        "regime": str(regime),
                        "factor_key": column,
                        "factor_label": spec["label"],
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _plot_major_factors(env_key: str, df: pd.DataFrame, out_dir: Path) -> None:
    specs = [spec for spec in FACTOR_SPECS.get(env_key, []) if spec["column"] in df.columns]
    if not specs:
        return

    ncols = len(specs)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.6 * ncols, 7.1),
        gridspec_kw={"height_ratios": [2.0, 1.05]},
    )
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    regime_ids = pd.to_numeric(df["regime_id"], errors="coerce").to_numpy(dtype=np.int64)
    tick_ids = regime_ids if len(regime_ids) <= 10 else regime_ids[::2]

    for col_idx, spec in enumerate(specs):
        top_ax = axes[0, col_idx]
        bottom_ax = axes[1, col_idx]
        values = pd.to_numeric(df[spec["column"]], errors="coerce").to_numpy(dtype=np.float64)
        color = spec["color"]

        top_ax.plot(
            regime_ids,
            values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4.8,
        )
        top_ax.set_title(spec["label"])
        top_ax.set_xlabel("Regime ID")
        top_ax.set_ylabel(spec["axis_label"])
        top_ax.set_xticks(tick_ids)
        top_ax.grid(alpha=0.25)

        finite_values = values[np.isfinite(values)]
        bins = min(8, max(4, len(finite_values) // 2))
        bottom_ax.hist(
            finite_values,
            bins=bins,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
        )
        if len(finite_values) > 0:
            bottom_ax.axvline(float(np.mean(finite_values)), color="black", linestyle="--", linewidth=1.1)
        bottom_ax.set_title("Distribution Across 20 Regimes")
        bottom_ax.set_xlabel(spec["axis_label"])
        bottom_ax.set_ylabel("Count")
        bottom_ax.grid(alpha=0.18)

    fig.suptitle(f"{env_key} Major Dynamic Factors over 20 Shared Regimes", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"major_factors_{env_key}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.factor_csv.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    comparison_rows = _selected_comparison_rows()
    if not comparison_rows:
        raise FileNotFoundError("No comparison rows found for the current 20-regime report selection.")

    group_fronts = _collect_group_fronts(comparison_rows)
    if not group_fronts:
        raise FileNotFoundError("No valid Pareto fronts found for CDF plotting.")

    rng = np.random.default_rng(0)
    bounds = env_objective_bounds(group_fronts)
    cdf_df = build_cdf_curves(
        group_fronts,
        bounds,
        cdf_mode="empirical",
        grid_points=args.cdf_grid_points,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    best_cdf_df = build_best_cdf_curves(
        group_fronts,
        bounds,
        grid_points=args.cdf_grid_points,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    cdf_vs_best_df = pd.concat([cdf_df, best_cdf_df], ignore_index=True)

    cdf_df.to_csv(args.out_dir / "cdf_curves.csv", index=False)
    best_cdf_df.to_csv(args.out_dir / "best_cdf_curves.csv", index=False)
    cdf_vs_best_df.to_csv(args.out_dir / "cdf_vs_best_curves.csv", index=False)
    plot_cdf_panels(cdf_df, args.out_dir, highlight_method="dynamic")
    plot_cdf_panels_with_prefix(
        best_cdf_df,
        args.out_dir,
        prefix="best_cdf",
        highlight_method="best_cdf",
    )
    plot_cdf_panels_with_prefix(
        cdf_vs_best_df,
        args.out_dir,
        prefix="cdf_vs_best",
        highlight_method="best_cdf",
    )

    dynamic_source_by_env = _dynamic_source_map(comparison_rows)
    factor_frames = _major_factor_frames(dynamic_source_by_env)
    factor_long_df = _factor_long_df(factor_frames)
    factor_long_df.to_csv(args.factor_csv, index=False)
    for env_key, env_df in factor_frames.items():
        _plot_major_factors(env_key, env_df, args.out_dir)

    selected_sources = pd.DataFrame(
        [
            {
                "env_key": row["env_key"],
                "method": row["method"],
                "source": row["source"],
            }
            for row in comparison_rows
        ]
    ).sort_values(["env_key", "method"])
    selected_sources.to_csv(args.out_dir / "selected_comparison_sources.csv", index=False)

    scalar_family_rows = _selected_scalar_family_rows()
    scalar_sources = pd.DataFrame(
        [
            {
                "env_key": row["env_key"],
                "method": row["method"],
                "source": row["source"],
            }
            for row in scalar_family_rows
        ]
    ).sort_values(["env_key", "method"])
    scalar_sources.to_csv(args.out_dir / "selected_scalar_family_sources.csv", index=False)

    print(f"wrote cdf curves to {args.out_dir}")
    print(f"wrote major factor summary to {args.factor_csv}")


if __name__ == "__main__":
    main()
