from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("HV", True),
    ("EU", True),
    ("SP", False),
    ("cr_hv", True),
    ("irs", True),
    ("trace_shift_regret", False),
    ("trace_recovery_latency", False),
    ("rag_hv", False),
    ("rag_eu", False),
    ("adapt_score", True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--highlight-method", type=str, default="dynamic")
    return parser.parse_args()


def rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [(metric, higher_better) for metric, higher_better in METRICS if metric in df.columns]
    rows = []
    for env_key, sub in df.groupby("env_key"):
        for metric, higher_better in metrics:
            ascending = not higher_better
            ranked = sub[["method", metric]].sort_values(metric, ascending=ascending).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "env_key": env_key,
                        "metric": metric,
                        "method": row["method"],
                        "value": row[metric],
                        "rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def win_frame(df: pd.DataFrame, highlight_method: str) -> pd.DataFrame:
    metrics = [(metric, higher_better) for metric, higher_better in METRICS if metric in df.columns]
    rows = []
    for env_key, sub in df.groupby("env_key"):
        highlight = sub[sub["method"] == highlight_method]
        if highlight.empty:
            continue
        highlight = highlight.iloc[0]
        baselines = sub[sub["type"].astype(str).str.startswith("baseline")]
        for metric, higher_better in metrics:
            if higher_better:
                wins = int((highlight[metric] > baselines[metric]).sum())
            else:
                wins = int((highlight[metric] < baselines[metric]).sum())
            rows.append(
                {
                    "env_key": env_key,
                    "metric": metric,
                    "wins_vs_baselines": wins,
                    "num_baselines": int(len(baselines)),
                }
            )
    return pd.DataFrame(rows)


def make_metric_grid(df: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    metric_subset = [metric for metric in ["HV", "EU", "cr_hv", "irs", "trace_shift_regret", "trace_recovery_latency", "rag_hv", "adapt_score"] if metric in df.columns]
    envs = list(df["env_key"].unique())
    fig, axes = plt.subplots(len(envs), len(metric_subset), figsize=(3.4 * len(metric_subset), 3.2 * len(envs)))
    if len(envs) == 1:
        axes = np.asarray([axes])
    for row_idx, env_key in enumerate(envs):
        sub = df[df["env_key"] == env_key]
        for col_idx, metric in enumerate(metric_subset):
            ax = axes[row_idx, col_idx]
            higher_better = dict(METRICS)[metric]
            ordered = sub.sort_values(metric, ascending=not higher_better)
            x = np.arange(len(ordered))
            colors = ["tab:blue" if method == highlight_method else "tab:gray" for method in ordered["method"]]
            ax.bar(x, ordered[metric], color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(ordered["method"], rotation=50, ha="right", fontsize=8)
            ax.set_title(f"{env_key} | {metric}", fontsize=9)
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def make_rank_heatmap(rank_df: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    sub = rank_df[rank_df["method"] == highlight_method]
    if sub.empty:
        return
    pivot = sub.pivot(index="env_key", columns="metric", values="rank").sort_index()
    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 2.0, 1.2 * len(pivot.index) + 1.6))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank (1 = best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.raw_csv)
    core_metrics = [metric for metric, _ in METRICS if metric in df.columns]
    if df[core_metrics].isna().any().any():
        raise ValueError("NaN detected in raw comparison CSV.")

    ranks = rank_frame(df)
    wins = win_frame(df, args.highlight_method)
    ranks.to_csv(args.out_dir / "metric_ranks.csv", index=False)
    wins.to_csv(args.out_dir / "dynamic_vs_baselines_wins.csv", index=False)

    summary = df.pivot_table(
        index=["env_key", "method", "type"],
        values=core_metrics,
        aggfunc="first",
    )
    summary.to_csv(args.out_dir / "summary_table.csv")

    make_metric_grid(df, args.out_dir / "metric_grid.png", args.highlight_method)
    make_rank_heatmap(ranks, args.out_dir / "dynamic_rank_heatmap.png", args.highlight_method)


if __name__ == "__main__":
    main()
