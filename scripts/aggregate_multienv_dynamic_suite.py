from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
RUN_RE = re.compile(
    r"(?P<prefix>.+)_(?P<env_key>building|evcharging|cogen|chlor_alkali)_(?P<config>dynamic|static|random|no_dyn_ablation)_seed(?P<seed>\d+)$"
)

from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="suite")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "multi_env_suite",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict[str, object] | None:
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return None

    metrics_path = run_dir / "metrics_history.csv"
    if not metrics_path.exists():
        return None

    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    final = df.iloc[-1].to_dict()
    final.update(
        {
            "env_key": match.group("env_key"),
            "config": match.group("config"),
            "seed": int(match.group("seed")),
            "run_dir": str(run_dir),
        }
    )
    return final


def load_front(run_dir: Path) -> np.ndarray | None:
    front_path = run_dir / "final" / "objs.txt"
    if not front_path.exists():
        return None
    front = np.loadtxt(front_path, delimiter=",")
    if front.ndim == 1:
        front = front[None, :]
    return np.asarray(front, dtype=np.float64)


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def compute_front_metrics(front: np.ndarray, ref_point: np.ndarray) -> dict[str, float]:
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], 0.5)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": eu, "SP": sp}


def plot_metric_panel(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    env_keys = list(df["env_key"].drop_duplicates())
    configs = ["dynamic", "static", "random", "no_dyn_ablation"]
    colors = {
        "dynamic": "tab:blue",
        "static": "tab:orange",
        "random": "tab:red",
        "no_dyn_ablation": "tab:green",
    }

    fig, axes = plt.subplots(1, len(env_keys), figsize=(5 * len(env_keys), 4), squeeze=False)
    for idx, env_key in enumerate(env_keys):
        ax = axes[0, idx]
        sub = df[df["env_key"] == env_key]
        grouped = sub.groupby("config")[metric].agg(["mean", "std"]).reindex(configs)
        x = np.arange(len(grouped))
        ax.bar(
            x,
            grouped["mean"].fillna(0.0).values,
            yerr=grouped["std"].fillna(0.0).values,
            color=[colors[cfg] for cfg in grouped.index],
            capsize=4,
            alpha=0.9,
        )
        ax.set_title(env_key)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, rotation=20)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("HV", "Final HV"),
        ("EU", "Final EU"),
        ("SP", "Final SP"),
        ("cr_hv", "Cross-Regime HV"),
        ("irs", "Instability Robustness"),
        ("trace_recovery_latency", "Recovery Latency"),
    ]
    env_keys = list(df["env_key"].drop_duplicates())
    configs = ["dynamic", "static", "random", "no_dyn_ablation"]
    colors = {
        "dynamic": "tab:blue",
        "static": "tab:orange",
        "random": "tab:red",
        "no_dyn_ablation": "tab:green",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        for env_idx, env_key in enumerate(env_keys):
            sub = df[df["env_key"] == env_key]
            grouped = sub.groupby("config")[metric].agg(["mean", "std"]).reindex(configs)
            x = np.arange(len(grouped)) + 0.06 * env_idx
            width = 0.8 / max(len(env_keys), 1)
            ax.bar(
                x,
                grouped["mean"].fillna(0.0).values,
                yerr=grouped["std"].fillna(0.0).values,
                width=width,
                color=[colors[cfg] for cfg in grouped.index],
                alpha=0.9,
                capsize=3,
                label=env_key if len(env_keys) > 1 else None,
            )
        ax.set_title(title)
        ax.set_xticks(np.arange(len(configs)))
        ax.set_xticklabels(configs, rotation=15)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(len(env_keys), 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fronts_by_env: dict[str, list[np.ndarray]] = {}
    for run_dir in sorted(args.results_root.glob(f"{args.prefix}_*_seed*")):
        row = load_metrics(run_dir)
        front = load_front(run_dir)
        if row is not None and front is not None:
            row["front"] = front
            rows.append(row)
            fronts_by_env.setdefault(row["env_key"], []).append(front)
    if not rows:
        raise FileNotFoundError(f"no matching runs under {args.results_root} with prefix {args.prefix}")

    expanded_rows = []
    for env_key, fronts in fronts_by_env.items():
        ref_point = derive_reference_point(fronts)
        for row in rows:
            if row["env_key"] != env_key:
                continue
            front = row["front"]
            metrics = compute_front_metrics(front, ref_point)
            expanded = {k: v for k, v in row.items() if k != "front"}
            expanded.update(metrics)
            expanded["ref_point"] = json.dumps(ref_point.tolist())
            expanded_rows.append(expanded)

    df = pd.DataFrame(expanded_rows).sort_values(["env_key", "config", "seed"])
    df.to_csv(args.out_dir / "multi_env_metrics_raw.csv", index=False)

    summary = (
        df.groupby(["env_key", "config"])[["HV", "EU", "SP", "cr_hv", "cr_eu", "irs", "trace_shift_regret", "trace_recovery_latency", "trace_recovery_score"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(args.out_dir / "multi_env_metrics_summary.csv", index=False)

    plot_metric_panel(df, "HV", "Final HV", args.out_dir / "hv_by_env.png")
    plot_metric_panel(df, "EU", "Final EU", args.out_dir / "eu_by_env.png")
    plot_metric_panel(df, "SP", "Final SP", args.out_dir / "sp_by_env.png")
    plot_metric_panel(df, "cr_hv", "Cross-Regime HV", args.out_dir / "cr_hv_by_env.png")
    plot_metric_panel(df, "trace_recovery_latency", "Trace Recovery Latency", args.out_dir / "trace_recovery_latency_by_env.png")
    plot_metric_panel(df, "trace_shift_regret", "Trace Shift Regret", args.out_dir / "trace_shift_regret_by_env.png")
    plot_overview(df, args.out_dir / "cogen_suite_overview.png")

    print(df[["env_key", "config", "seed", "HV", "EU", "SP", "cr_hv", "irs", "trace_recovery_latency", "trace_shift_regret"]])
    print(summary.head())


if __name__ == "__main__":
    main()
