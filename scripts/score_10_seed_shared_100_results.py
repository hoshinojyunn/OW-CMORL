#!/usr/bin/env python3
"""Parse ten shared-100 result tables and compute seed-level Holm tests.

The TeX source stores one aggregate mean (over 100 conditions) per seed,
method, and environment.  Those ten seed-level means are the paired samples;
the displayed confidence-interval half widths are retained as source metadata
but are not treated as additional observations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "10_seed_shared_100_results.tex"
DEFAULT_OUT = PROJECT_ROOT / "results_10_condition_ttest_study" / "quick_real_v1" / "analysis"

METHODS = ("CAPQL", "PGMORL", "Q-Pensieve", "MORL-CA", "OW-CMORL")
BASELINES = ("CAPQL", "PGMORL", "Q-Pensieve", "MORL-CA")
ENVIRONMENTS = ("Building", "EV charging", "Cogen", "Chlor-alkali")
ENV_KEYS = {
    "Building": "building",
    "EV charging": "evcharging",
    "Cogen": "cogen",
    "Chlor-alkali": "chlor_alkali",
}
METRICS = ("HV", "EU", "AS", "Reg", "Lat")
HIGHER_BETTER = {"HV": True, "EU": True, "AS": True, "Reg": False, "Lat": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def _strip_latex(value: str) -> str:
    value = value.replace("\\textbf{", "").replace("}", "")
    value = value.replace("$", "").replace("\\mathrm{", "").replace("\\", "")
    return value.strip()


def _numeric_pair(cell: str) -> tuple[float, float]:
    # The cells contain forms such as 1.2e+10$\pm$3.4e+9.  Extracting the
    # numeric pair avoids depending on the particular LaTeX wrappers.
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", cell)
    if len(matches) < 2:
        raise ValueError(f"Could not parse mean and CI half-width from cell: {cell!r}")
    return float(matches[0]), float(matches[1])


def _table_blocks(text: str) -> list[str]:
    blocks = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", text, flags=re.DOTALL)
    if len(blocks) != 10:
        raise ValueError(f"Expected exactly 10 tabular blocks, found {len(blocks)}")
    return blocks


def parse_tables(path: Path) -> pd.DataFrame:
    text = path.read_text()
    records: list[dict[str, Any]] = []
    for seed, block in enumerate(_table_blocks(text), start=1):
        current_method: str | None = None
        table_rows = 0
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("\\toprule", "\\midrule", "\\bottomrule", "Method &")):
                continue
            line = line.rstrip("\\").strip()
            cells = [cell.strip() for cell in line.split("&")]
            if len(cells) != 7:
                continue
            method_cell = _strip_latex(cells[0])
            env_cell = _strip_latex(cells[1])
            if method_cell in METHODS:
                current_method = method_cell
            elif method_cell:
                continue
            if current_method is None or env_cell not in ENVIRONMENTS:
                continue
            metric_values: dict[str, float] = {}
            ci_values: dict[str, float] = {}
            for metric, cell in zip(METRICS, cells[2:]):
                mean, ci_half_width = _numeric_pair(cell)
                metric_values[metric] = mean
                ci_values[f"{metric}_ci95_halfwidth"] = ci_half_width
            records.append(
                {
                    "seed": seed,
                    "environment": env_cell,
                    "environment_key": ENV_KEYS[env_cell],
                    "method": current_method,
                    **metric_values,
                    **ci_values,
                }
            )
            table_rows += 1
        if table_rows != 20:
            raise ValueError(f"Seed {seed}: expected 20 method/environment rows, found {table_rows}")

    frame = pd.DataFrame(records)
    expected = {(seed, env, method) for seed in range(1, 11) for env in ENVIRONMENTS for method in METHODS}
    actual = {(int(row.seed), row.environment, row.method) for row in frame.itertuples(index=False)}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Parsed rows do not match expected grid; missing={missing[:4]}, extra={extra[:4]}")
    return frame.sort_values(["seed", "environment", "method"]).reset_index(drop=True)


def _paired_test(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        raise ValueError("At least two seed-level observations are required")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if std == 0.0:
        t_stat = float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0
        p_two = 0.0 if mean != 0 else 1.0
        p_one = 0.0 if mean > 0 else 1.0 if mean < 0 else 0.5
    else:
        t_stat = mean / (std / math.sqrt(n))
        p_two = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
        p_one = float(student_t.sf(t_stat, df=n - 1))
    return {
        "n": n,
        "df": n - 1,
        "mean_difference": mean,
        "std_difference": std,
        "standardized_effect_dz": mean / std if std > 0 else float("inf") if mean > 0 else float("-inf") if mean < 0 else 0.0,
        "t_statistic": t_stat,
        "p_two_sided": p_two,
        "p_one_sided_directional": p_one,
    }


def _holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6g}"


def compute_tests(frame: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for environment in ENVIRONMENTS:
        env_frame = frame[frame["environment"] == environment].set_index(["seed", "method"])
        for baseline in BASELINES:
            for metric in METRICS:
                ow = env_frame.xs("OW-CMORL", level="method")[metric]
                other = env_frame.xs(baseline, level="method")[metric]
                difference = ow - other if HIGHER_BETTER[metric] else other - ow
                result = _paired_test(difference.to_numpy(dtype=np.float64))
                records.append(
                    {
                        "environment": environment,
                        "environment_key": ENV_KEYS[environment],
                        "baseline": baseline,
                        "metric": metric,
                        "direction": "OW - baseline" if HIGHER_BETTER[metric] else "baseline - OW",
                        **result,
                    }
                )
    tests = pd.DataFrame(records)
    tests["p_holm_two_sided"] = _holm(tests["p_two_sided"].astype(float).tolist())
    tests["p_holm_one_sided_directional"] = _holm(tests["p_one_sided_directional"].astype(float).tolist())
    tests["ow_advantage"] = tests["mean_difference"] > 0.0
    tests["significant_two_sided"] = tests["ow_advantage"] & (tests["p_holm_two_sided"] < float(alpha))
    tests["significant_one_sided_directional"] = tests["ow_advantage"] & (tests["p_holm_one_sided_directional"] < float(alpha))
    return tests


def _write_report(path: Path, frame: pd.DataFrame, tests: pd.DataFrame, source: Path, alpha: float) -> None:
    lines = [
        "# 10-Seed Shared 100-Condition T-Test Report",
        "",
        f"Source: `{source}`. The source contains ten table blocks, interpreted as ten paired random seeds. Each cell is already aggregated over 100 shared operating conditions and is reported as mean $\\pm$ 95% CI half-width.",
        "",
        "The t-test sample is the ten seed-level aggregate means (`n=10`, `df=9`), not the 100 conditions. The displayed CI half-widths are retained as metadata but are not used as extra observations. Tests are paired by seed and compare OW-CMORL with each baseline. Positive directional differences favor OW-CMORL; for Reg and Lat the difference is baseline minus OW-CMORL because lower is better.",
        "",
        "Two-sided Holm p-values are the primary family-wise significance result (`alpha=0.05`). Directional one-sided Holm p-values are also included for the preregistered OW-CMORL-advantage alternative. This analysis uses aggregate values from the TeX table; exact condition-level covariance cannot be recovered from the published summaries.",
        "",
        "## Seed-Level Summary",
        "",
        "| Environment | Method | Metric | Mean across seeds | SD across seeds | n |",
        "|---|---|---|---:|---:|---:|",
    ]
    for environment in ENVIRONMENTS:
        for method in METHODS:
            group = frame[(frame["environment"] == environment) & (frame["method"] == method)]
            if len(group) != 10:
                raise ValueError(f"{environment}/{method}: expected 10 seed rows, found {len(group)}")
            means = group[list(METRICS)].mean()
            stds = group[list(METRICS)].std(ddof=1)
            for metric in METRICS:
                lines.append(
                    f"| {environment} | {method} | {metric} | {_fmt(float(means[metric]))} | {_fmt(float(stds[metric]))} | 10 |"
                )
    lines.extend(
        [
            "",
            "## Holm-Corrected Tests",
            "",
            "`t` is computed on the directional paired difference. `significant_two_sided` is the conservative primary flag; `significant_one_sided_directional` reports the directional alternative separately.",
            "",
            "| Environment | Baseline | Metric | Directional difference | t (df=9) | Two-sided p | Holm two-sided p | Directional one-sided p | Holm one-sided p | Two-sided significant |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for _, row in tests.iterrows():
        lines.append(
            f"| {row['environment']} | {row['baseline']} | {row['metric']} | {_fmt(float(row['mean_difference']))} | {_fmt(float(row['t_statistic']))} | {_fmt(float(row['p_two_sided']))} | {_fmt(float(row['p_holm_two_sided']))} | {_fmt(float(row['p_one_sided_directional']))} | {_fmt(float(row['p_holm_one_sided_directional']))} | {'yes' if row['significant_two_sided'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"There are {int(tests['significant_two_sided'].sum())} two-sided Holm-significant OW-CMORL advantages out of {len(tests)} hypotheses and {int(tests['significant_one_sided_directional'].sum())} directional one-sided Holm-significant advantages.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = parse_tables(args.input)
    tests = compute_tests(frame, alpha=float(args.alpha))
    frame.to_csv(args.out_dir / "10_seed_shared_100_seed_metrics.csv", index=False)
    tests.to_csv(args.out_dir / "10_seed_shared_100_paired_t_tests.csv", index=False)
    manifest = {
        "source": str(args.input),
        "seed_blocks": 10,
        "conditions_per_seed": 100,
        "sample_unit": "seed-level aggregate mean, paired by seed",
        "metrics": list(METRICS),
        "hypotheses": len(tests),
        "alpha": float(args.alpha),
        "primary_correction": "Holm over two-sided paired p-values",
        "secondary_correction": "Holm over directional one-sided paired p-values",
        "limitation": "condition-level observations and cross-method covariance are unavailable in the TeX summaries",
    }
    (args.out_dir / "10_seed_shared_100_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Keep the conventional manifest name aligned with the current report.
    (args.out_dir / "scoring_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(args.out_dir / "REPORT.md", frame, tests, args.input, float(args.alpha))
    print(json.dumps({"out_dir": str(args.out_dir), "seed_rows": len(frame), "tests": len(tests)}, indent=2))


if __name__ == "__main__":
    main()
