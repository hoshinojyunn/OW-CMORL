#!/usr/bin/env python3
"""Write a LaTeX table for the controlled million-slot ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT
       
        / "analysis"
        / "owcmorl_h1h2hnsw_million_slot"
        / "summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "owcmorl_h1h2hnsw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary_csv.open(newline="") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    names = {
        "full": "Full + HNSW",
        "h1_current_token_no_encoder": "H1: current token",
        "h2_no_forecast": "H2: forecast $\\lambda=0$",
        "h2_no_history": "H2: history $\\lambda=0$",
        "exact": "Full + exact",
    }
    body = []
    for variant in ("full", "h1_current_token_no_encoder", "h2_no_forecast", "h2_no_history"):
        row = rows[variant]
        body.append(
            "{} & {:.4f} & {:.4f} & {:.4f} & {:.3f} & {:.2f} \\\\".format(
                names[variant],
                float(row["mean_hv"]),
                float(row["mean_eu"]),
                float(row["mean_regret"]),
                float(row["query_p50_ms"]),
                float(row["mean_recovery_latency_steps"]),
            )
        )
    text = """\\begin{table*}[t]
\\centering
\\small
\\caption{Controlled context-key diagnostic. H1 uses the current token only; H2 removes the forecast or history term. Recovery is the number of dynamic steps required after a factor shift.}
\\label{tab:owcmorl-million-slot-ablation}
\\begin{tabular}{lccccc}
\\toprule
Variant & HV $\\uparrow$ & EU $\\uparrow$ & Regret $\\downarrow$ & p50 ms $\\downarrow$ & Recovery $\\downarrow$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "owcmorl_million_slot_ablation_table.tex"
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
