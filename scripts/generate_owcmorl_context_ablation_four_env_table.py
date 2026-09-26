#!/usr/bin/env python3
"""Generate the four-environment context-window ablation table for the paper."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = (
    "full",
    "h1_current_token_no_encoder",
    "h2_no_forecast",
    "h2_no_history",
)
ENVIRONMENTS = (
    ("building", "Building"),
    ("evcharging", "EV charging"),
    ("cogen", "Cogen"),
    ("chlor_alkali", "Chlor-alkali"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT
       
        / "analysis"
        / "owcmorl_context_ablation_four_env"
        / "summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
       
        / "figures"
        / "owcmorl_context_ablation_four_env"
        / "owcmorl_context_ablation_four_env_table.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    values = {(row["environment"], row["variant"]): row for row in records}
    missing = [
        (environment, variant)
        for environment, _ in ENVIRONMENTS
        for variant in VARIANTS
        if (environment, variant) not in values
    ]
    if missing:
        raise ValueError(f"Missing context-ablation summaries: {missing}")

    rows = []
    for environment_id, environment_name in ENVIRONMENTS:
        cells = []
        for variant in VARIANTS:
            row = values[(environment_id, variant)]
            cells.append(
                f"{float(row['mean_hv']):.3f} / {float(row['mean_eu']):.3f} / "
                f"{float(row['mean_recovery_latency_steps']):.2f}"
            )
        rows.append(f"{environment_name} & " + " & ".join(cells) + r" \\")

    document = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Controlled context-key diagnostic across four environment-conditioned routing profiles. Each condition uses a shared bank of 1M slots and 256 dynamic queries. Entries report HV / EU / recovery steps; larger HV and EU and fewer recovery steps are better. H1 uses only the current factor vector, and the two H2 variants remove the forecast or history component.}",
            r"\label{tab:owcmorl-context-ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Environment & Full window & H1: current factor & H2: no forecast & H2: no history \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document)
    print(args.out)


if __name__ == "__main__":
    main()
