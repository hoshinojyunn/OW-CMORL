from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE_METRICS = [
    "HV",
    "EU",
    "cr_hv",
    "irs",
    "trace_shift_regret",
    "trace_recovery_latency",
    "trace_recovery_score",
    "rag_hv",
    "rag_eu",
    "adapt_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--wins-csv", type=Path)
    parser.add_argument("--pairwise-summary-csv", type=Path)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--highlight-method", type=str, default="dynamic")
    return parser.parse_args()


def _format_value(value: float) -> str:
    if pd.isna(value):
        return "nan"
    value = float(value)
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def method_alias(method: str) -> str:
    mapping = {
        "dynamic": "OW-CMORL",
        "no_dyn": "OW-CMORL w/o dyn",
        "capql": "CAPQL",
        "q_pensieve": "Q-Pensieve",
        "qpensieve": "Q-Pensieve",
        "pgmorl": "PGMORL",
    }
    return mapping.get(method, method)


def table_markdown(df: pd.DataFrame) -> str:
    header = "| Method | Type | " + " | ".join(CORE_METRICS) + " |\n"
    sep = "|---|---|" + "---|" * len(CORE_METRICS) + "\n"
    rows = []
    for _, row in df.iterrows():
        values = [
            method_alias(str(row["method"])),
            str(row["type"]),
            *[_format_value(row[metric]) for metric in CORE_METRICS],
        ]
        rows.append("| " + " | ".join(values) + " |")
    return header + sep + "\n".join(rows) + "\n"


def summarize_wins(wins_df: pd.DataFrame, highlight_method: str) -> list[str]:
    if wins_df is None or wins_df.empty:
        return ["- No win statistics are available."]
    lines = []
    total = (
        wins_df.groupby("metric")[["wins_vs_baselines", "num_baselines"]]
        .sum()
        .reset_index()
    )
    lines.append(f"- Wins by `{method_alias(highlight_method)}` against baselines:")
    for _, row in total.iterrows():
        lines.append(
            f"- `{row['metric']}`: {int(row['wins_vs_baselines'])} / {int(row['num_baselines'])}"
        )
    return lines


def summarize_pairwise(pairwise_df: pd.DataFrame | None, highlight_method: str) -> list[str]:
    if pairwise_df is None or pairwise_df.empty:
        return ["- No pairwise baseline win rates are available."]
    overall = pairwise_df[pairwise_df["env_key"] == "overall"].copy()
    if overall.empty:
        return ["- No pairwise baseline win rates are available."]
    overall = overall.sort_values("win_rate", ascending=False).reset_index(drop=True)
    top = overall.head(min(5, len(overall)))
    bottom = overall.tail(min(5, len(overall))).sort_values("win_rate", ascending=True)
    lines = [f"- Per-metric win rates for `{method_alias(highlight_method)}` against each baseline:"]
    if not top.empty:
        lines.append("- Strongest pairwise results:")
        for _, row in top.iterrows():
            lines.append(
                f"  - `{method_alias(highlight_method)}` vs `{method_alias(str(row['baseline_method']))}`: `{_format_value(row['win_rate'])}` ({int(row['wins'])}/{int(row['num_metrics'])})"
            )
    if not bottom.empty:
        lines.append("- Weakest pairwise results:")
        for _, row in bottom.iterrows():
            lines.append(
                f"  - `{method_alias(highlight_method)}` vs `{method_alias(str(row['baseline_method']))}`: `{_format_value(row['win_rate'])}` ({int(row['wins'])}/{int(row['num_metrics'])})"
            )
    return lines


def environment_observations(
    raw_df: pd.DataFrame,
    highlight_method: str,
    pairwise_df: pd.DataFrame | None = None,
) -> list[str]:
    lines = []
    for env_key, env_df in raw_df.groupby("env_key"):
        lines.append(f"### {env_key}")
        highlight = env_df[env_df["method"] == highlight_method]
        baselines = env_df[env_df["method"] != highlight_method]
        if highlight.empty:
            lines.append("- No result is available for the highlighted method.")
            lines.append("")
            continue
        row = highlight.iloc[0]
        hv_rank = int((baselines["HV"] > row["HV"]).sum() + 1)
        adapt_rank = int((baselines["adapt_score"] > row["adapt_score"]).sum() + 1)
        lines.append(
            f"- `{method_alias(highlight_method)}` has `HV={_format_value(row['HV'])}` and `adapt_score={_format_value(row['adapt_score'])}` in this environment."
        )
        lines.append(f"- HV rank: {hv_rank}.")
        lines.append(f"- Adaptation-score rank: {adapt_rank}.")
        lines.append(
            f"- `trace_shift_regret={_format_value(row['trace_shift_regret'])}`, `trace_recovery_latency={_format_value(row['trace_recovery_latency'])}`, and `trace_recovery_score={_format_value(row['trace_recovery_score'])}`."
        )
        if pairwise_df is not None and not pairwise_df.empty:
            env_pairwise = pairwise_df[pairwise_df["env_key"] == env_key].copy()
            if not env_pairwise.empty:
                best = env_pairwise.sort_values("win_rate", ascending=False).iloc[0]
                worst = env_pairwise.sort_values("win_rate", ascending=True).iloc[0]
                lines.append(
                    f"- Strongest comparison: `{method_alias(str(best['baseline_method']))}` with per-metric win rate `{_format_value(best['win_rate'])}`."
                )
                lines.append(
                    f"- Weakest comparison: `{method_alias(str(worst['baseline_method']))}` with per-metric win rate `{_format_value(worst['win_rate'])}`."
                )
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    raw_df = pd.read_csv(args.raw_csv)
    wins_df = pd.read_csv(args.wins_csv) if args.wins_csv and args.wins_csv.exists() else None
    pairwise_df = (
        pd.read_csv(args.pairwise_summary_csv)
        if args.pairwise_summary_csv and args.pairwise_summary_csv.exists()
        else None
    )

    lines: list[str] = []
    lines.append("# Long-Run Benchmark Results")
    lines.append("")
    lines.append("This report summarizes aggregated OW-CMORL and baseline results in dynamic SustainGym environments.")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.extend(summarize_wins(wins_df, args.highlight_method))
    lines.append("")
    lines.extend(summarize_pairwise(pairwise_df, args.highlight_method))
    lines.append("")
    lines.append("## 2. Results by Environment")
    lines.append("")

    for env_key, env_df in raw_df.groupby("env_key"):
        sub = env_df[["method", "type", *CORE_METRICS]].copy()
        sub = sub.sort_values(["type", "method"]).reset_index(drop=True)
        lines.append(f"### {env_key}")
        lines.append("")
        lines.append(table_markdown(sub))
        lines.append("")

    lines.append("## 3. Observations by Environment")
    lines.append("")
    lines.extend(environment_observations(raw_df, args.highlight_method, pairwise_df))
    lines.append("")
    lines.append("## 4. Metric Definitions")
    lines.append("")
    lines.append("- `HV`, `EU`, and `SP` measure Pareto solution-set quality.")
    lines.append("- `cr_hv` and `irs` measure stability across regimes.")
    lines.append("- `trace_shift_regret`, `trace_recovery_latency`, and `trace_recovery_score` measure post-shift loss and recovery.")
    lines.append("- `rag_hv`, `rag_eu`, and `adapt_score` measure the gap to the best known frontier in each regime.")
    lines.append("")
    lines.append("## 5. Reproduction")
    lines.append("")
    lines.append("- Run `aggregate_all_benchmarks.py` first to generate `comparison_raw.csv`.")
    lines.append("- Rerun the aggregation and this script to refresh the report.")
    lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
