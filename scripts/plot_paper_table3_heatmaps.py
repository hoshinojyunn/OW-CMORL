#!/usr/bin/env python3
"""Create metric heatmaps from the current detailed comparison table in the paper.

Each heatmap has methods as rows and environments as columns. Cell annotations
show the exact Table 3 mean, while color encodes within-environment relative
performance so that metrics with different physical units remain comparable.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("CAPQL", "PGMORL", "Q-Pensieve", "MORL-CA", "OW-CMORL")
ENVIRONMENTS = ("Building", "EV charging", "Cogen", "Chlor-alkali")
METRICS = (
    ("hv", "HV", True),
    ("eu", "EU", True),
    ("adaptation_score", "Adaptation score", True),
    ("shift_regret", "Shift regret", False),
    ("recovery_latency", "Recovery latency", False),
)
METHOD_ALIASES = {"Q-Pensieve": "Q-Pensieve", "OW-CMORL": "OW-CMORL"}
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-source",
        type=Path,
        default=PROJECT_ROOT / "innovation_paper" / "experiment_section.tex",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "paper_table3_metric_heatmaps",
    )
    parser.add_argument("--preview-dir", type=Path, default=None)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_cell(cell: str) -> str:
    return cell.replace("\\textbf{", "").replace("}", "").replace("$", "").strip()


def first_number(cell: str) -> float:
    match = NUMBER_RE.search(clean_cell(cell))
    if match is None:
        raise ValueError(f"No numeric Table 3 value found in: {cell}")
    return float(match.group(0))


def extract_table_rows(path: Path) -> list[dict[str, float | str]]:
    text = path.read_text()
    marker = "\\label{tab:experiment_shared_100_results}"
    table_end = text.find(marker)
    if table_end < 0:
        raise ValueError(f"Cannot find {marker} in {path}")
    table_start = text.rfind("\\begin{table*}", 0, table_end)
    if table_start < 0:
        raise ValueError("Cannot locate the start of the detailed comparison table")
    block = text[table_start:table_end]
    rows = []
    current_method = ""
    for line in block.splitlines():
        if " & " not in line or not line.rstrip().endswith("\\\\"):
            continue
        cells = [cell.strip() for cell in line.rstrip()[:-2].split("&")]
        if len(cells) != 7:
            continue
        method_cell = clean_cell(cells[0])
        if method_cell:
            current_method = METHOD_ALIASES.get(method_cell, method_cell)
        environment = clean_cell(cells[1])
        if current_method not in METHODS or environment not in ENVIRONMENTS:
            continue
        rows.append(
            {
                "method": current_method,
                "environment": environment,
                "hv": first_number(cells[2]),
                "eu": first_number(cells[3]),
                "adaptation_score": first_number(cells[4]),
                "shift_regret": first_number(cells[5]),
                "recovery_latency": first_number(cells[6]),
            }
        )
    expected = {(method, environment) for method in METHODS for environment in ENVIRONMENTS}
    actual = {(str(row["method"]), str(row["environment"])) for row in rows}
    if actual != expected:
        raise ValueError(f"Incomplete Table 3 extraction. Missing: {sorted(expected - actual)}")
    return rows


def relative_scores(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low <= np.finfo(float).eps:
        return np.ones_like(values)
    return (values - low) / (high - low) if higher_is_better else (high - values) / (high - low)


def format_value(value: float, metric_key: str) -> str:
    if metric_key in {"hv", "eu", "shift_regret"} and abs(value) >= 1.0e4:
        return f"{value:.2e}"
    if metric_key == "adaptation_score":
        return f"{value:.3f}"
    return f"{value:.2f}"


def save_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    metric_key: str,
    higher_is_better: bool,
    data: dict[tuple[str, str], dict[str, float | str]],
    out_dir: Path,
    preview_dir: Path | None,
) -> None:
    values = np.asarray(
        [[float(data[(method, environment)][metric_key]) for environment in ENVIRONMENTS] for method in METHODS]
    )
    normalized = np.column_stack(
        [relative_scores(values[:, column], higher_is_better) for column in range(values.shape[1])]
    )
    fig, axis = plt.subplots(figsize=(3.45, 2.95))
    fig.subplots_adjust(left=0.27, right=0.92, top=0.98, bottom=0.25)
    image = axis.imshow(normalized, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(np.arange(len(ENVIRONMENTS)), ENVIRONMENTS, rotation=24, ha="right")
    axis.set_yticks(np.arange(len(METHODS)), METHODS)
    for row_index in range(normalized.shape[0]):
        for column_index in range(normalized.shape[1]):
            score = normalized[row_index, column_index]
            text_color = "white" if score >= 0.60 else "black"
            axis.text(
                column_index,
                row_index,
                format_value(values[row_index, column_index], metric_key),
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.1,
            )
    axis.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.05, pad=0.04)
    colorbar.set_label("Within-environment relative score")
    output = out_dir / f"table3_{metric_key}_heatmap.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview_dir is not None:
        fig.savefig(preview_dir / f"table3_{metric_key}_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    rows = extract_table_rows(args.table_source)
    save_csv(args.out_dir / "table3_metric_values.csv", rows)
    data = {(str(row["method"]), str(row["environment"])): row for row in rows}
    for metric_key, _metric_label, higher_is_better in METRICS:
        plot_metric(metric_key, higher_is_better, data, args.out_dir, args.preview_dir)
    for metric_key, metric_label, _higher_is_better in METRICS:
        print(f"{metric_label}: {args.out_dir / f'table3_{metric_key}_heatmap.pdf'}")


if __name__ == "__main__":
    main()
