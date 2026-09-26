from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_NAME_RE = re.compile(r"(?P<prefix>.+)_(?P<config>dynamic|static|random|no_dyn_ablation)_seed(?P<seed>\d+)$")
DRIFT_RE = re.compile(r"drift score: ([0-9.]+)")
METRIC_RE = re.compile(
    r"Hyper Volume: ([0-9.]+), Expected Utility: ([0-9.]+), Sparsity: ([0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="smoke")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis" / "shift_correlation")
    return parser.parse_args()


def extract_run_stats(run_dir: Path) -> dict[str, float] | None:
    match = RUN_NAME_RE.match(run_dir.name)
    if match is None:
        return None

    metrics_path = run_dir / "metrics_history.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        if df.empty:
            return None
        last = df.iloc[-1].to_dict()
        drift_scores = [float(v) for v in df["drift_score"].dropna().tolist()] if "drift_score" in df else []
        hv = float(last.get("hv", np.nan))
        eu = float(last.get("eu", np.nan))
        sp = float(last.get("sp", np.nan))
        cr_hv = float(last.get("cr_hv", np.nan))
        irs = float(last.get("irs", np.nan))
        trace_latency = float(last.get("trace_recovery_latency", np.nan))
        trace_regret = float(last.get("trace_shift_regret", np.nan))
    else:
        log_path = run_dir / "log.txt"
        text = log_path.read_text()
        drift_scores = [float(item) for item in DRIFT_RE.findall(text)]
        metrics = METRIC_RE.findall(text)
        if not metrics:
            return None
        hv, eu, sp = map(float, metrics[-1])
        cr_hv = np.nan
        irs = np.nan
        trace_latency = np.nan
        trace_regret = np.nan
    return {
        "config": match.group("config"),
        "seed": int(match.group("seed")),
        "mean_drift": float(np.mean(drift_scores)) if drift_scores else 0.0,
        "max_drift": float(np.max(drift_scores)) if drift_scores else 0.0,
        "HV": hv,
        "EU": eu,
        "SP": sp,
        "CR_HV": cr_hv,
        "IRS": irs,
        "trace_recovery_latency": trace_latency,
        "trace_shift_regret": trace_regret,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(args.results_root.glob(f"{args.prefix}_*_seed*")):
        row = extract_run_stats(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise FileNotFoundError("no matching logs for correlation analysis")

    df = pd.DataFrame(rows).sort_values(["config", "seed"])
    df.to_csv(args.out_dir / "shift_correlation.csv", index=False)

    corr_cols = [c for c in ["mean_drift", "max_drift", "HV", "EU", "SP", "CR_HV", "IRS", "trace_recovery_latency", "trace_shift_regret"] if c in df.columns]
    corr = df[corr_cols].corr()
    corr.to_csv(args.out_dir / "shift_correlation_matrix.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {
        "dynamic": "tab:blue",
        "static": "tab:orange",
        "random": "tab:red",
        "no_dyn_ablation": "tab:green",
    }
    for config_name, group in df.groupby("config"):
        axes[0].scatter(group["mean_drift"], group["HV"], s=50, alpha=0.9, label=config_name, color=colors.get(config_name))
        axes[1].scatter(group["mean_drift"], group["EU"], s=50, alpha=0.9, label=config_name, color=colors.get(config_name))
        axes[2].scatter(group["mean_drift"], group["trace_recovery_latency"], s=50, alpha=0.9, label=config_name, color=colors.get(config_name))
    axes[0].set_xlabel("Mean Drift Score")
    axes[0].set_ylabel("Final HV")
    axes[0].set_title("Drift vs HV")
    axes[1].set_xlabel("Mean Drift Score")
    axes[1].set_ylabel("Final EU")
    axes[1].set_title("Drift vs EU")
    axes[2].set_xlabel("Mean Drift Score")
    axes[2].set_ylabel("Trace Recovery Latency")
    axes[2].set_title("Drift vs Recovery")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "shift_correlation.png", dpi=200)
    plt.close(fig)
    print(df)
    print(corr)


if __name__ == "__main__":
    main()
