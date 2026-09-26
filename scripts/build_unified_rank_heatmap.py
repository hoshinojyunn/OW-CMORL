from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_all_benchmarks import (  # noqa: E402
    ENV_ORDER,
    KEY_METRICS,
    metric_rank_frame,
    plot_rank_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multi-environment benchmark CSVs and render a unified OW-CMORL rank heatmap."
    )
    parser.add_argument(
        "--main-raw",
        type=Path,
        default=PROJECT_ROOT / "figures" / "all_benchmarks_v5b" / "comparison_raw.csv",
        help="comparison_raw.csv containing building / evcharging / cogen results",
    )
    parser.add_argument(
        "--chlor-raw",
        type=Path,
        default=PROJECT_ROOT / "figures" / "chlor_bench_long_final" / "comparison_raw.csv",
        help="comparison_raw.csv containing chlor_alkali results",
    )
    parser.add_argument(
        "--highlight-method",
        type=str,
        default="dynamic",
        help="method name used for the unified rank heatmap",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "unified_rank_heatmap_4env",
    )
    return parser.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty benchmark csv: {path}")
    return df


def _dedupe_rows(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["env_key", "method", "type", "seed"]
    existing = [col for col in keep_cols if col in df.columns]
    if not existing:
        return df.copy()
    return df.drop_duplicates(subset=existing, keep="first").reset_index(drop=True)


def _dynamic_rank_table(rank_df: pd.DataFrame, highlight_method: str) -> pd.DataFrame:
    sub = rank_df[rank_df["method"] == highlight_method].copy()
    if sub.empty:
        raise ValueError(f"Method {highlight_method!r} not found in rank frame.")
    pivot = sub.pivot(index="env_key", columns="metric", values="rank").reindex(ENV_ORDER)
    metrics = [metric for metric, _ in KEY_METRICS]
    available = [metric for metric in metrics if metric in pivot.columns]
    return pivot[available]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    main_df = _load_csv(args.main_raw)
    chlor_df = _load_csv(args.chlor_raw)

    chlor_df = chlor_df[chlor_df["env_key"] == "chlor_alkali"].copy()
    merged = pd.concat([main_df, chlor_df], ignore_index=True, sort=False)
    merged = _dedupe_rows(merged)
    merged = merged.sort_values(["env_key", "type", "method", "seed"]).reset_index(drop=True)

    merged.to_csv(args.out_dir / "comparison_raw_4env.csv", index=False)

    rank_df = metric_rank_frame(merged)
    rank_df.to_csv(args.out_dir / "metric_ranks_4env.csv", index=False)

    plot_rank_heatmap(rank_df, args.out_dir / "rank_heatmap_4env.png", args.highlight_method)

    dynamic_rank = _dynamic_rank_table(rank_df, args.highlight_method)
    dynamic_rank.to_csv(args.out_dir / "owcmorl_rank_table_4env.csv")
    (args.out_dir / "owcmorl_rank_table_4env.md").write_text(dynamic_rank.to_markdown())

    print(f"wrote merged csv to {args.out_dir / 'comparison_raw_4env.csv'}")
    print(f"wrote rank csv to {args.out_dir / 'metric_ranks_4env.csv'}")
    print(f"wrote heatmap to {args.out_dir / 'rank_heatmap_4env.png'}")
    print(dynamic_rank.to_string())


if __name__ == "__main__":
    main()
