from __future__ import annotations

import argparse
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

from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


RUN_NAME_RE = re.compile(r"(?P<prefix>.+)_(?P<config>dynamic|static|random|no_dyn_ablation)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="smoke")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "figures" / "smoke_suite")
    parser.add_argument("--ref-point", nargs="+", type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    return parser.parse_args()


def load_front(run_dir: Path) -> np.ndarray:
    front_path = run_dir / "final" / "objs.txt"
    front = np.loadtxt(front_path, delimiter=",")
    if front.ndim == 1:
        front = front[None, :]
    return front


def compute_metrics(front: np.ndarray, ref_point: np.ndarray, eval_delta_weight: float) -> dict[str, float]:
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], eval_delta_weight)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": float(eu), "SP": float(sp), "points": int(len(front))}


def discover_runs(results_root: Path, prefix: str) -> list[tuple[str, int, Path]]:
    runs = []
    for run_dir in sorted(results_root.glob(f"{prefix}_*_seed*")):
        match = RUN_NAME_RE.match(run_dir.name)
        if match is None:
            continue
        runs.append((match.group("config"), int(match.group("seed")), run_dir))
    return runs


def plot_metric_panel(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby("config")[["HV", "EU", "SP"]]
        .agg(["mean", "std"])
        .sort_index()
    )

    metrics = ["HV", "EU", "SP"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    configs = list(summary.index)
    x = np.arange(len(configs))
    colors = {
        "dynamic": "tab:blue",
        "static": "tab:orange",
        "random": "tab:red",
        "no_dyn_ablation": "tab:green",
    }

    for idx, metric in enumerate(metrics):
        means = summary[(metric, "mean")].values
        stds = summary[(metric, "std")].fillna(0.0).values
        axes[idx].bar(
            x,
            means,
            yerr=stds,
            color=[colors[cfg] for cfg in configs],
            alpha=0.9,
            capsize=4,
        )
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(configs, rotation=20)
        axes[idx].set_title(metric)
    fig.tight_layout()
    fig.savefig(out_dir / "suite_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ref_point = np.asarray(args.ref_point, dtype=np.float64)
    rows = []
    for config_name, seed, run_dir in discover_runs(args.results_root, args.prefix):
        front = load_front(run_dir)
        metrics = compute_metrics(front, ref_point, args.eval_delta_weight)
        metrics["config"] = config_name
        metrics["seed"] = seed
        metrics["run_dir"] = str(run_dir)
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(f"no runs found under {args.results_root} with prefix {args.prefix}")

    df = pd.DataFrame(rows).sort_values(["config", "seed"])
    df.to_csv(args.out_dir / "suite_metrics_raw.csv", index=False)

    summary = df.groupby("config")[["HV", "EU", "SP", "points"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.out_dir / "suite_metrics_summary.csv", index=False)
    plot_metric_panel(df, args.out_dir)
    print(df)
    print(summary)


if __name__ == "__main__":
    main()
