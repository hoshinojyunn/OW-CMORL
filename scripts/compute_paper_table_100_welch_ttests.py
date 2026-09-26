#!/usr/bin/env python3
"""Estimate OW-CMORL-vs-baseline Welch tests from Table 3 means and CIs.

The paper table reports each method's mean and two-sided 95% t-interval over
100 operating conditions, but not paired condition-level observations. Exact
paired t-tests are therefore not identifiable from the table alone. This tool
recovers each method's standard error from its reported interval and performs
Welch tests under an explicit independent-condition approximation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("CAPQL", "PGMORL", "Q-Pensieve", "MORL-CA", "OW-CMORL")
ENVIRONMENTS = ("Building", "EV charging", "Cogen", "Chlor-alkali")
METRICS = (
    ("HV", True),
    ("EU", True),
    ("AS", True),
    ("Reg", False),
    ("Lat", False),
)
CELL_RE = re.compile(
    r"(?P<mean>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\\pm"
    r"(?P<half>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-source",
        type=Path,
        default=PROJECT_ROOT / "innovation_paper" / "experiment_section.tex",
    )
    parser.add_argument("--conditions", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "paper_table_100_welch_ttests",
    )
    return parser.parse_args()


def _clean(cell: str) -> str:
    return cell.replace("$", "").replace("\\textbf{", "").replace("}", "").strip()


def _parse_metric(cell: str) -> tuple[float, float]:
    match = CELL_RE.search(_clean(cell))
    if match is None:
        raise ValueError(f"Could not parse mean and CI half-width: {cell}")
    return float(match.group("mean")), float(match.group("half"))


def read_table(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    text = path.read_text()
    marker = "\\label{tab:experiment_shared_100_results}"
    end = text.find(marker)
    start = text.rfind("\\begin{table*}", 0, end)
    if start < 0 or end < 0:
        raise ValueError(f"Cannot locate the 100-condition result table in {path}")
    rows: dict[tuple[str, str], dict[str, float]] = {}
    current_method = ""
    for line in text[start:end].splitlines():
        if " & " not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [part.strip() for part in line.rstrip()[:-2].split("&")]
        if len(cells) != 7:
            continue
        method = _clean(cells[0]) or current_method
        environment = _clean(cells[1])
        if method not in METHODS or environment not in ENVIRONMENTS:
            continue
        current_method = method
        values: dict[str, float] = {}
        for index, (metric, _higher) in enumerate(METRICS, start=2):
            mean, half_width = _parse_metric(cells[index])
            values[f"{metric}_mean"] = mean
            values[f"{metric}_ci95_half_width"] = half_width
        rows[(method, environment)] = values
    expected = {(method, environment) for method in METHODS for environment in ENVIRONMENTS}
    if set(rows) != expected:
        raise ValueError(f"Incomplete table parse. Missing: {sorted(expected - set(rows))}")
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def main() -> None:
    args = parse_args()
    if args.conditions < 2:
        raise ValueError("At least two conditions are required")
    table = read_table(args.table_source)
    n = int(args.conditions)
    interval_t = float(student_t.ppf(0.975, df=n - 1))
    rows: list[dict[str, float | int | str | bool]] = []
    for environment in ENVIRONMENTS:
        ow = table[("OW-CMORL", environment)]
        for baseline in (method for method in METHODS if method != "OW-CMORL"):
            other = table[(baseline, environment)]
            for metric, higher_is_better in METRICS:
                ow_mean = float(ow[f"{metric}_mean"])
                baseline_mean = float(other[f"{metric}_mean"])
                ow_half = float(ow[f"{metric}_ci95_half_width"])
                baseline_half = float(other[f"{metric}_ci95_half_width"])
                ow_se = ow_half / interval_t
                baseline_se = baseline_half / interval_t
                # Positive differences always support OW-CMORL.
                difference = ow_mean - baseline_mean if higher_is_better else baseline_mean - ow_mean
                difference_se = math.sqrt(ow_se**2 + baseline_se**2)
                t_statistic = difference / difference_se if difference_se > 0 else math.copysign(math.inf, difference)
                numerator = (ow_se**2 + baseline_se**2) ** 2
                denominator = (ow_se**4 + baseline_se**4) / (n - 1)
                welch_df = numerator / denominator if denominator > 0 else math.inf
                p_two_sided = float(2.0 * student_t.sf(abs(t_statistic), df=welch_df))
                p_one_sided = float(student_t.sf(t_statistic, df=welch_df))
                rows.append(
                    {
                        "environment": environment,
                        "baseline": baseline,
                        "metric": metric,
                        "conditions_per_method": n,
                        "ow_mean": ow_mean,
                        "baseline_mean": baseline_mean,
                        "difference_positive_supports_ow": difference,
                        "welch_standard_error": difference_se,
                        "t_statistic": t_statistic,
                        "welch_df": welch_df,
                        "p_two_sided": p_two_sided,
                        "p_one_sided_ow_advantage": p_one_sided,
                        "input_assumption": "independent-condition approximation",
                    }
                )
    p_values = [float(row["p_two_sided"]) for row in rows]
    for row, adjusted in zip(rows, holm_adjust(p_values)):
        row["p_holm_two_sided"] = adjusted
        row["significant_holm_in_ow_direction"] = bool(
            float(row["difference_positive_supports_ow"]) > 0.0 and adjusted < float(args.alpha)
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.out_dir / "welch_t_tests_from_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "holm_significant_ow_advantage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row["significant_holm_in_ow_direction"])

    lines = [
        "# 100-Condition Table t-Test Values",
        "",
        "## Scope",
        "",
        "These values are reconstructed from the published 100-condition table means and two-sided 95% t-intervals. The table does not retain condition-level paired observations or their covariance, so an exact paired t-test cannot be reconstructed. The reported statistic is a two-sample Welch t-test under an independent-condition approximation (`n=100` per method); it is not a replacement for a paired seed-level test.",
        "",
        "Positive `t` values favor OW-CMORL: OW-CMORL minus baseline for HV/EU/AS, and baseline minus OW-CMORL for Reg/Lat. `Holm p` controls the family-wise error rate across all 80 comparisons.",
        "",
        "| Environment | Baseline | Metric | Directional difference | Welch t | df | Two-sided p | Holm p | OW significant |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {environment} | {baseline} | {metric} | {difference_positive_supports_ow:.6g} | "
            "{t_statistic:.4f} | {welch_df:.1f} | {p_two_sided:.3g} | {p_holm_two_sided:.3g} | {significant} |".format(
                **row,
                significant="yes" if row["significant_holm_in_ow_direction"] else "no",
            )
        )
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
