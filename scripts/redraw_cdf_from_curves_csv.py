from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ALIASES = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "qpensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "morl_ca": "MORL-CA",
}

METHOD_PLOT_ORDER = [
    "capql",
    "dynamic",
    "pgmorl",
    "q_pensieve",
    "morlca",
]

METHOD_COLORS = {
    "capql": "#1F77B4",
    "dynamic": "#E31A1C",
    "pgmorl": "#2CA02C",
    "q_pensieve": "#FF7F0E",
    "qpensieve": "#FF7F0E",
    "morlca": "#111111",
    "morl_ca": "#111111",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw CDF figures from an existing curves CSV.")
    parser.add_argument(
        "--curves-csv",
        type=Path,
        default=Path("/root/C-MORL/figures/comparison_20_regime/cdf_curves.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/root/C-MORL/figures/comparison_20_regime"),
    )
    parser.add_argument("--envs", nargs="+", default=["chlor_alkali"])
    parser.add_argument("--line-width", type=float, default=3.2)
    parser.add_argument("--highlight-width", type=float, default=3.9)
    parser.add_argument("--highlight-method", type=str, default="dynamic")
    return parser.parse_args()


def _display_name(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))


def _ordered_methods(methods: list[str]) -> list[str]:
    rank = {method: idx for idx, method in enumerate(METHOD_PLOT_ORDER)}
    return sorted(methods, key=lambda method: (rank.get(str(method), len(METHOD_PLOT_ORDER)), str(method)))


def _color(method: str) -> str:
    return METHOD_COLORS.get(str(method), "#444444")


def plot_env(df: pd.DataFrame, env_key: str, out_dir: Path, line_width: float, highlight_width: float, highlight_method: str) -> Path:
    env_df = df[df["env_key"] == env_key].copy()
    if env_df.empty:
        raise FileNotFoundError(f"no rows found for env={env_key}")

    methods = _ordered_methods(env_df["method"].dropna().astype(str).unique().tolist())
    objectives = sorted(env_df["objective"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(1, len(objectives), figsize=(5.4 * len(objectives), 4.4), sharey=True)
    if len(objectives) == 1:
        axes = [axes]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    for ax, objective in zip(axes, objectives):
        obj_df = env_df[env_df["objective"] == objective]
        for method in methods:
            method_df = obj_df[obj_df["method"] == method].sort_values("x")
            if method_df.empty:
                continue
            lw = highlight_width if method == highlight_method else line_width
            ax.plot(
                method_df["x"].to_numpy(),
                method_df["mean_cdf"].to_numpy(),
                label=_display_name(method),
                color=_color(method),
                linewidth=lw,
                alpha=0.98,
            )
        ax.set_title(f"{env_key} | {objective}")
        ax.set_xlabel("Normalized return")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Cumulative probability")
    fig.legend(
        handles=axes[0].lines,
        labels=[line.get_label() for line in axes[0].lines],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=min(5, len(methods)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_path = out_dir / f"cdf_{env_key}.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.curves_csv)
    for env_key in args.envs:
        out_path = plot_env(
            df=df,
            env_key=env_key,
            out_dir=args.out_dir,
            line_width=float(args.line_width),
            highlight_width=float(args.highlight_width),
            highlight_method=str(args.highlight_method),
        )
        print(out_path)


if __name__ == "__main__":
    main()
