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
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


RUN_RE = re.compile(
    r"(?P<prefix>.+)_(?P<env_key>building|evcharging|cogen|chlor_alkali)_(?P<config>dynamic|static|random|no_dyn_ablation)_seed(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "scalarized_baselines",
    )
    parser.add_argument(
        "--dynamic-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--dynamic-prefix",
        type=str,
        default="ctest3",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "baseline_comparison",
    )
    return parser.parse_args()


def load_baseline_rows(env_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result_path in sorted(env_dir.glob("*/*/result.json")):
        data = json.loads(result_path.read_text())
        data["result_path"] = str(result_path)
        rows.append(data)
    return rows


def load_dynamic_fronts(results_root: Path, prefix: str, env_key: str) -> dict[str, np.ndarray]:
    fronts: dict[str, np.ndarray] = {}
    for run_dir in sorted(results_root.glob(f"{prefix}_{env_key}_*_seed*")):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        config = match.group("config")
        front_path = run_dir / "final" / "objs.txt"
        if not front_path.exists():
            continue
        front = np.loadtxt(front_path, delimiter=",")
        if front.ndim == 1:
            front = front[None, :]
        fronts[config] = np.asarray(front, dtype=np.float64)
    return fronts


def load_dynamic_metrics(results_root: Path, prefix: str, env_key: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for run_dir in sorted(results_root.glob(f"{prefix}_{env_key}_*_seed*")):
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        config = match.group("config")
        metrics_path = run_dir / "metrics_history.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        row = df.iloc[-1].to_dict()
        out[config] = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating))}
    return out


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def pareto_front(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def front_metrics(front: np.ndarray, ref_point: np.ndarray) -> dict[str, float]:
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], 0.5)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": float(eu), "SP": float(sp)}


def mean_trace_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_keys = [
        "trace_utility_mean",
        "trace_utility_std",
        "trace_shift_count",
        "trace_shift_regret",
        "trace_recovery_latency",
        "trace_recovery_score",
        "trace_pre_post_gap",
    ]
    out = {}
    for key in metric_keys:
        values = [float(row.get("trace_metrics", {}).get(key, 0.0)) for row in rows]
        out[key] = float(np.mean(values))
    return out


def regime_dispersion(rows: list[dict[str, object]]) -> dict[str, float]:
    per_regime: dict[str, list[np.ndarray]] = {}
    for row in rows:
        for regime_row in row.get("regime_returns", []):
            per_regime.setdefault(regime_row["regime"], []).append(np.asarray(regime_row["objs"], dtype=np.float64))
    if not per_regime:
        return {"cr_hv": 0.0, "cr_eu": 0.0, "cr_sp": 0.0, "irs": 0.0}

    regime_fronts = [pareto_front(np.stack(objs, axis=0)) for objs in per_regime.values()]
    ref_point = derive_reference_point(regime_fronts)
    metrics = [front_metrics(front, ref_point) for front in regime_fronts]
    eu_values = np.asarray([metric["EU"] for metric in metrics], dtype=np.float64)
    return {
        "cr_hv": float(np.mean([metric["HV"] for metric in metrics])),
        "cr_eu": float(np.mean(eu_values)),
        "cr_sp": float(np.mean([metric["SP"] for metric in metrics])),
        "irs": float(1.0 / (1.0 + eu_values.std())),
    }


def plot_bar(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sub = df.sort_values(metric, ascending=(metric in {"trace_shift_regret", "trace_recovery_latency"}))
    x = np.arange(len(sub))
    ax.bar(x, sub[metric], color=["tab:blue" if name == "dynamic" else "tab:gray" for name in sub["method"]])
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["method"], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for env_dir in sorted(path for path in args.baseline_root.iterdir() if path.is_dir()):
        env_key = env_dir.name
        baseline_rows = load_baseline_rows(env_dir)
        dynamic_fronts = load_dynamic_fronts(args.dynamic_root, args.dynamic_prefix, env_key)
        dynamic_metrics = load_dynamic_metrics(args.dynamic_root, args.dynamic_prefix, env_key)
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in baseline_rows:
            grouped.setdefault(str(row["alg"]), []).append(row)

        candidate_fronts: list[np.ndarray] = []
        method_to_front: dict[str, np.ndarray] = {}
        for alg, rows in grouped.items():
            points = np.asarray([row["objs"] for row in rows], dtype=np.float64)
            front = pareto_front(points)
            method_to_front[alg] = front
            candidate_fronts.append(front)
        for config, front in dynamic_fronts.items():
            method_name = "dynamic" if config == "dynamic" else config
            method_to_front[method_name] = front
            candidate_fronts.append(front)
        if not candidate_fronts:
            continue
        ref_point = derive_reference_point(candidate_fronts)

        for alg, rows in grouped.items():
            front = method_to_front[alg]
            record = {"env_key": env_key, "method": alg, "type": "baseline"}
            record.update(front_metrics(front, ref_point))
            record.update(mean_trace_metrics(rows))
            record.update(regime_dispersion(rows))
            record["num_points"] = int(len(front))
            summary_rows.append(record)
        for config, front in dynamic_fronts.items():
            method_name = "dynamic" if config == "dynamic" else config
            record = {"env_key": env_key, "method": method_name, "type": "dynamic"}
            record.update(front_metrics(front, ref_point))
            if config in dynamic_metrics:
                record.update(dynamic_metrics[config])
            record["num_points"] = int(len(front))
            summary_rows.append(record)

    if not summary_rows:
        raise FileNotFoundError("No baseline or dynamic results found for aggregation.")

    df = pd.DataFrame(summary_rows).sort_values(["env_key", "type", "method"])
    core_metrics = ["HV", "EU", "SP", "cr_hv", "irs", "trace_shift_regret", "trace_recovery_latency"]
    missing_mask = df[core_metrics].isna()
    if missing_mask.any().any():
        missing_rows = df.loc[missing_mask.any(axis=1), ["env_key", "method", "type"] + core_metrics]
        missing_rows.to_csv(args.out_dir / "missing_metrics.csv", index=False)
        raise ValueError(f"Missing metrics detected; see {args.out_dir / 'missing_metrics.csv'}")
    df.to_csv(args.out_dir / "baseline_comparison_raw.csv", index=False)

    for env_key, sub in df.groupby("env_key"):
        env_out = args.out_dir / env_key
        env_out.mkdir(parents=True, exist_ok=True)
        sub.to_csv(env_out / "summary.csv", index=False)
        for metric in ["HV", "EU", "SP", "cr_hv", "irs", "trace_shift_regret", "trace_recovery_latency"]:
            if metric in sub.columns:
                plot_bar(sub.fillna(0.0), metric, env_out / f"{metric}.png")

    pivot = df.pivot_table(index=["env_key", "method", "type"], values=["HV", "EU", "SP", "cr_hv", "irs", "trace_shift_regret", "trace_recovery_latency"], aggfunc="mean")
    pivot.to_csv(args.out_dir / "baseline_comparison_table.csv")
    print(df)


if __name__ == "__main__":
    main()
