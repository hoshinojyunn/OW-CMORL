#!/usr/bin/env python3
"""Write the four-environment HNSW Top-50 scaling table from summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        default=PROJECT_ROOT / "analysis" / "owcmorl_hnsw_256d_four_env" / "summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "owcmorl_hnsw_256d_four_env",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: dict[str, dict[str, dict[str, str]]] = {}
    with args.summary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["environment"], {})[row["backend"]] = row
    body = []
    for key, label in ENVIRONMENTS:
        hnsw, exact = rows[key]["hnsw"], rows[key]["exact"]
        body.append(
            "{} & {:.6f} / {:.6f} & {:.6f} / {:.6f} & {:.3f} & {:.3f} \\\\".format(
                label,
                float(hnsw["mean_hv"]),
                float(exact["mean_hv"]),
                float(hnsw["mean_eu"]),
                float(exact["mean_eu"]),
                float(hnsw["route_batch_seconds"]),
                float(exact["route_batch_seconds"]),
            )
        )
    text = """\\begin{table*}[t]
\\centering
\\small
\\caption{Four-environment-conditioned HNSW scalability diagnostic. Each environment uses a separate synthetic bank of 1M 256-dimensional slot embeddings; each request retrieves the Top-50 snapshots. Times are total routing time for 1,000 requests, including query construction, retrieval, and payload materialization.}
\\label{tab:hnsw-256d-four-env}
\\begin{tabular}{lcccc}
\\toprule
Environment & HV (HNSW / Exact) $\\uparrow$ & EU (HNSW / Exact) $\\uparrow$ & HNSW (s) $\\downarrow$ & Exact (s) $\\downarrow$ \\\\
\\midrule
""" + "\n".join(body) + """
\\bottomrule
\\end{tabular}
\\end{table*}
"""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "owcmorl_hnsw_256d_four_env_table.tex"
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
