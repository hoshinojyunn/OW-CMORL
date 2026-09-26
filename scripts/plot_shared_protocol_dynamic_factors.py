from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACTOR_CSV = PROJECT_ROOT / "analysis" / "major_dynamic_factors_20_regime.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures" / "comparison_20_regime"

SELECTED_FACTORS: dict[str, dict[str, str]] = {
    "building": {
        "factor_key": "electricity_price",
        "title": "Building",
        "ylabel": "Normalized electricity price",
        "color": "#e63946",
    },
    "evcharging": {
        "factor_key": "prev_moer",
        "title": "EV Charging",
        "ylabel": "Normalized previous MOER",
        "color": "#0077b6",
    },
    "cogen": {
        "factor_key": "renewables_ratio",
        "title": "Cogen",
        "ylabel": "Renewables ratio",
        "color": "#2a9d8f",
    },
    "chlor_alkali": {
        "factor_key": "current_price",
        "title": "Chlor-Alkali",
        "ylabel": "Electricity price (RMB/kWh)",
        "color": "#f4a261",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one representative dynamic factor over the 20 shared regimes for each environment."
    )
    parser.add_argument("--factor-csv", type=Path, default=DEFAULT_FACTOR_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _plot_env(ax: plt.Axes, env_df: pd.DataFrame, env_key: str) -> None:
    spec = SELECTED_FACTORS[env_key]
    factor_key = spec["factor_key"]
    sub = env_df[env_df["factor_key"] == factor_key].copy()
    if sub.empty:
        raise ValueError(f"Missing factor {factor_key!r} for environment {env_key!r}.")
    sub = sub.sort_values("regime_id").reset_index(drop=True)

    ax.plot(
        sub["regime_id"],
        sub["value"],
        color=spec["color"],
        linewidth=4.0,
        marker="o",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.3,
    )
    ax.set_title(spec["title"], fontsize=15, fontweight="bold")
    ax.set_xlabel("Regime ID")
    ax.set_ylabel(spec["ylabel"])
    ax.set_xticks(list(range(0, 20, 2)))
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_single_env(env_df: pd.DataFrame, env_key: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    _plot_env(ax, env_df, env_key)
    fig.tight_layout()
    fig.savefig(out_dir / f"dynamic_factor_{env_key}.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def _save_overview(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))
    axes_flat = axes.reshape(-1)
    for ax, env_key in zip(axes_flat, SELECTED_FACTORS):
        env_df = df[df["env_key"] == env_key]
        _plot_env(ax, env_df, env_key)
    fig.suptitle("Representative Dynamic Factors under the Shared 20-Regime Protocol", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "dynamic_factors_overview.png", dpi=280, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.factor_csv.exists():
        raise FileNotFoundError(args.factor_csv)

    sns.set_theme(style="whitegrid", context="talk")
    df = pd.read_csv(args.factor_csv)
    for env_key in SELECTED_FACTORS:
        env_df = df[df["env_key"] == env_key]
        if env_df.empty:
            raise ValueError(f"Missing environment rows for {env_key!r}.")
        _save_single_env(env_df, env_key, args.out_dir)
    _save_overview(df, args.out_dir)


if __name__ == "__main__":
    main()
