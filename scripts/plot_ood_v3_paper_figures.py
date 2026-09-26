#!/usr/bin/env python
"""Generate publication figures and a LaTeX table from verified OOD-v3 data.

The script consumes only persisted benchmark artifacts.  It deliberately keeps
the OOD context geometry (100 illustrative condition vectors) separate from
the measured short-horizon OOD benchmark scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROFILE = "ood_v3_site_tariff_trace"
ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
ENV_LABELS = {
    "building": "Building",
    "evcharging": "EV charging",
    "cogen": "Cogen",
    "chlor_alkali": "Chlor-alkali",
}
METHOD_ORDER = ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca", "lcpo"]
METHOD_LABELS = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "lcpo": "LCPO",
}
METHOD_COLORS = {
    "dynamic": "#0072B2",
    "capql": "#D55E00",
    "pgmorl": "#009E73",
    "q_pensieve": "#CC79A7",
    "morlca": "#7F7F7F",
    "lcpo": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / f"{PROFILE}_results_summary.csv",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_trace100_v3",
    )
    parser.add_argument(
        "--cdf-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / PROFILE,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "innovation_paper" / "fig",
    )
    return parser.parse_args()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def _pca2(id_frame: pd.DataFrame, ood_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ignore = {"regime_index", "context_label"}
    fields = [field for field in id_frame.columns if field not in ignore]
    if fields != [field for field in ood_frame.columns if field not in ignore]:
        raise ValueError("ID and OOD context traces have incompatible factor fields.")
    joined = np.vstack((id_frame[fields].to_numpy(float), ood_frame[fields].to_numpy(float)))
    centered = joined - joined.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-12, scale, 1.0)
    _, singular_values, vectors_t = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ vectors_t[:2].T
    total = float(np.sum(singular_values**2))
    explained = (singular_values[:2] ** 2 / total * 100.0) if total > 0.0 else np.zeros(2)
    split = len(id_frame)
    return coordinates[:split], coordinates[split:], explained


def plot_context_geometry(trace_dir: Path, out_dir: Path) -> dict[str, object]:
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.7))
    order_map = None
    for panel, (axis, env_key) in enumerate(zip(axes.ravel(), ENV_ORDER)):
        id_frame = pd.read_csv(trace_dir / f"{env_key}_id_trace.csv")
        ood_frame = pd.read_csv(trace_dir / f"{env_key}_ood_trace.csv")
        id_xy, ood_xy, explained = _pca2(id_frame, ood_frame)
        axis.scatter(id_xy[:, 0], id_xy[:, 1], s=9, color="#A7A7A7", alpha=0.58, linewidths=0, zorder=1)
        order_map = axis.scatter(
            ood_xy[:, 0],
            ood_xy[:, 1],
            s=14,
            c=np.arange(len(ood_xy)),
            cmap="viridis",
            vmin=0,
            vmax=max(1, len(ood_xy) - 1),
            alpha=0.88,
            linewidths=0,
            zorder=2,
        )
        axis.scatter(ood_xy[0, 0], ood_xy[0, 1], s=32, marker="o", color="#0072B2", edgecolors="white", linewidths=0.5, zorder=5)
        axis.scatter(ood_xy[-1, 0], ood_xy[-1, 1], s=35, marker="s", color="#009E73", edgecolors="white", linewidths=0.5, zorder=5)
        axis.text(0.02, 0.97, f"({chr(97 + panel)}) {ENV_LABELS[env_key]}", transform=axis.transAxes, ha="left", va="top")
        axis.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
        axis.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
        axis.tick_params(length=2.5, width=0.6)
    handles = [
        Line2D([0], [0], color="#A7A7A7", marker="o", markersize=3.5, linewidth=0, label="ID conditions"),
        Line2D([0], [0], color="#0072B2", marker="o", markersize=5, linewidth=0, label="OOD start"),
        Line2D([0], [0], color="#009E73", marker="s", markersize=5, linewidth=0, label="OOD end"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.37, -0.01))
    if order_map is not None:
        colorbar = figure.colorbar(order_map, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        colorbar.set_label("OOD condition order")
    figure.subplots_adjust(wspace=0.32, hspace=0.36, bottom=0.16, right=0.92)
    output = out_dir / "ood_v3_context_geometry.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    return {"context_geometry": str(output), "trace_conditions_per_environment": 100}


def _format_score(value: float, is_best: bool) -> str:
    text = f"{value:.3f}"
    return f"\\textbf{{{text}}}" if is_best else text


def write_score_table(frame: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    pivot = frame.pivot(index="method", columns="env_key", values="ood_normalized_score").reindex(index=METHOD_ORDER, columns=ENV_ORDER)
    pivot["overall"] = pivot.mean(axis=1)
    maxima = pivot[ENV_ORDER].max(axis=0)
    overall_max = float(pivot["overall"].max())
    row_break = chr(92) * 2
    lines = [
        "% Generated by scripts/plot_ood_v3_paper_figures.py.",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Building & EV charging & Cogen & Chlor-alkali & Overall " + row_break,
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        row = pivot.loc[method]
        values = [
            _format_score(float(row[env_key]), np.isclose(row[env_key], maxima[env_key]))
            for env_key in ENV_ORDER
        ]
        values.append(_format_score(float(row["overall"]), np.isclose(row["overall"], overall_max)))
        lines.append(f"{METHOD_LABELS[method]} & " + " & ".join(values) + " " + row_break)
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    output = out_dir / "ood_v3_score_table.tex"
    output.write_text("\n".join(lines))
    return output, pivot


def plot_score_summary(frame: pd.DataFrame, pivot: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    figure = plt.figure(figsize=(7.05, 3.15))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.50)
    heat_axis = figure.add_subplot(grid[0, 0])
    bar_axis = figure.add_subplot(grid[0, 1])
    matrix = pivot[ENV_ORDER].T.to_numpy(float)
    image = heat_axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="cividis", aspect="auto")
    heat_axis.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[method] for method in METHOD_ORDER], rotation=35, ha="right")
    heat_axis.set_yticks(np.arange(len(ENV_ORDER)), [ENV_LABELS[env_key] for env_key in ENV_ORDER])
    for row in range(matrix.shape[0]):
        winner = int(np.nanargmax(matrix[row]))
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value < 0.48 else "black"
            heat_axis.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
        heat_axis.add_patch(Rectangle((winner - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="#D55E00", linewidth=1.6))
    heat_axis.set_xlabel("Method")
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Normalized online OOD score")
    ordered = pivot["overall"].sort_values(ascending=True)
    bar_axis.barh(
        [METHOD_LABELS[method] for method in ordered.index],
        ordered.to_numpy(float),
        color=[METHOD_COLORS[method] for method in ordered.index],
        height=0.64,
    )
    bar_axis.set_xlim(0.0, 0.85)
    bar_axis.set_xlabel("Mean score across environments")
    for y_pos, value in enumerate(ordered.to_numpy(float)):
        bar_axis.text(value + 0.015, y_pos, f"{value:.3f}", va="center", fontsize=7)
    bar_axis.grid(axis="x", alpha=0.18, linewidth=0.6)
    output = out_dir / "ood_v3_online_score_summary.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    return {"online_score_summary": str(output), "overall_leader": METHOD_LABELS[str(ordered.index[-1])]}


def plot_cdfs(cdf_dir: Path, out_dir: Path) -> dict[str, object]:
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.65))
    for panel, (axis, env_key) in enumerate(zip(axes.ravel(), ENV_ORDER)):
        path = cdf_dir / f"{PROFILE}_{env_key}_adapt_cdf.csv"
        frame = pd.read_csv(path)
        for method in METHOD_ORDER:
            values = frame[frame["method"] == method]
            if values.empty:
                continue
            axis.step(
                values["regime_adapt_score"],
                values["cdf"],
                where="post",
                linewidth=2.0 if method == "dynamic" else 1.15,
                color=METHOD_COLORS[method],
                alpha=1.0 if method == "dynamic" else 0.86,
            )
        axis.text(0.02, 0.97, f"({chr(97 + panel)}) {ENV_LABELS[env_key]}", transform=axis.transAxes, ha="left", va="top")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(0.0, 1.02)
        axis.set_xlabel("Regime-level adaptation score")
        axis.set_ylabel("Empirical CDF")
        axis.tick_params(length=2.5, width=0.6)
    handles = [Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2.0 if method == "dynamic" else 1.15, label=METHOD_LABELS[method]) for method in METHOD_ORDER]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figure.subplots_adjust(wspace=0.30, hspace=0.36, bottom=0.18)
    output = out_dir / "ood_v3_adaptation_cdf.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)
    return {"adaptation_cdf": str(output)}


def main() -> None:
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_frame = pd.read_csv(args.summary_csv)
    required = {"env_key", "method", "ood_normalized_score"}
    missing = required - set(score_frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    if set(ENV_ORDER) - set(score_frame["env_key"]):
        raise ValueError("The score CSV does not contain all four environments.")
    if set(METHOD_ORDER) - set(score_frame["method"]):
        raise ValueError("The score CSV does not contain all compared methods.")
    table_path, pivot = write_score_table(score_frame, args.out_dir)
    manifest: dict[str, object] = {
        "profile": PROFILE,
        "score_source": str(args.summary_csv),
        "score_interpretation": "method-set-relative online OOD comparative proxy",
        "table": str(table_path),
    }
    manifest.update(plot_context_geometry(args.trace_dir, args.out_dir))
    manifest.update(plot_score_summary(score_frame, pivot, args.out_dir))
    manifest.update(plot_cdfs(args.cdf_dir, args.out_dir))
    (args.out_dir / "ood_v3_paper_figure_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
