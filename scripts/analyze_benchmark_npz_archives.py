from __future__ import annotations

import argparse
from collections import defaultdict
from math import erf, sqrt
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from benchmark_npz_utils import (  # noqa: E402
    as_2d,
    bootstrap_ci,
    derive_reference_point,
    empirical_cdf,
    front_metrics,
    load_npz_archive,
    normalize_to_unit,
    pareto_front,
    safe_name,
)


METHOD_ALIASES = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "qpensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "morl_ca": "MORL-CA",
    "acer": "ACER",
    "acktr": "ACKTR",
    "a2c": "A2C",
    "ppo2": "PPO2",
    "trpo_mpi": "TRPO-MPI",
    "ddpg": "DDPG",
    "deepq": "DQN",
    "no_dyn": "OW-CMORL w/o dyn",
    "static": "Static",
    "random": "Random",
}

METHOD_PLOT_ORDER = [
    "capql",
    "dynamic",
    "pgmorl",
    "q_pensieve",
    "morlca",
    "no_dyn",
    "static",
    "random",
    "a2c",
    "acer",
    "acktr",
    "ppo2",
    "trpo_mpi",
    "ddpg",
    "deepq",
]

METHOD_COLORS = {
    "capql": "#1F77B4",
    "dynamic": "#E31A1C",
    "pgmorl": "#2CA02C",
    "q_pensieve": "#FF7F0E",
    "qpensieve": "#FF7F0E",
    "morlca": "#111111",
    "morl_ca": "#111111",
    "no_dyn": "#9467BD",
    "static": "#8C564B",
    "random": "#7F7F7F",
    "a2c": "#17BECF",
    "acer": "#BCBD22",
    "acktr": "#8C6D31",
    "ppo2": "#D62728",
    "trpo_mpi": "#393B79",
    "ddpg": "#637939",
    "deepq": "#AD494A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported benchmark .npz archives.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "archives_npz_canonical_manifest.csv",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=PROJECT_ROOT / "archives_npz_canonical",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "benchmark_npz_analysis_canonical",
    )
    parser.add_argument("--highlight-method", type=str, default="dynamic")
    parser.add_argument("--include-envs", nargs="+")
    parser.add_argument("--exclude-envs", nargs="+", default=[])
    parser.add_argument("--include-methods", nargs="+")
    parser.add_argument("--exclude-methods", nargs="+", default=[])
    parser.add_argument("--cdf-mode", choices=["empirical", "smoothed"], default="empirical")
    parser.add_argument("--require-highlight-method", action="store_true")
    parser.add_argument("--cdf-grid-points", type=int, default=201)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    return parser.parse_args()


def display_method_name(method: str) -> str:
    return METHOD_ALIASES.get(str(method), str(method))


def ordered_methods(methods: list[str]) -> list[str]:
    rank = {method: idx for idx, method in enumerate(METHOD_PLOT_ORDER)}
    return sorted(methods, key=lambda method: (rank.get(str(method), len(METHOD_PLOT_ORDER)), str(method)))


def method_color_map(methods: list[str]) -> dict[str, tuple[float, float, float, float] | str]:
    fallback = plt.cm.tab10(np.linspace(0, 1, max(1, len(methods))))
    color_map: dict[str, tuple[float, float, float, float] | str] = {}
    for idx, method in enumerate(methods):
        color_map[method] = METHOD_COLORS.get(str(method), fallback[idx % len(fallback)])
    return color_map


def bootstrap_mean_ci(values: np.ndarray, *, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    boot = []
    n = len(values)
    for _ in range(max(1, samples)):
        idx = rng.integers(0, n, size=n)
        boot.append(float(np.mean(values[idx])))
    low, high = np.quantile(np.asarray(boot, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def bootstrap_front_metric_ci(
    front: np.ndarray,
    ref_point: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    front = np.asarray(front, dtype=np.float64)
    metric = front_metrics(front, ref_point)
    if len(front) <= 1:
        return {key: (value, value) for key, value in metric.items()}
    boot_values = {key: [] for key in metric}
    n = len(front)
    for _ in range(max(1, samples)):
        idx = rng.integers(0, n, size=n)
        boot_front = pareto_front(front[idx])
        boot_metric = front_metrics(boot_front, ref_point)
        for key, value in boot_metric.items():
            boot_values[key].append(float(value))
    out = {}
    for key, values in boot_values.items():
        low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
        out[key] = (float(low), float(high))
    return out


def bootstrap_vector_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0:
        zeros = np.zeros(values.shape[1] if values.ndim == 2 else 1, dtype=np.float64)
        return zeros, zeros
    if len(values) == 1:
        mean = np.asarray(values[0], dtype=np.float64)
        return mean, mean
    boot = []
    n = len(values)
    for _ in range(max(1, samples)):
        idx = rng.integers(0, n, size=n)
        boot.append(values[idx].mean(axis=0))
    boot = np.asarray(boot, dtype=np.float64)
    low = np.quantile(boot, 0.025, axis=0)
    high = np.quantile(boot, 0.975, axis=0)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def smoothed_cdf(values: np.ndarray, x_grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    x_grid = np.asarray(x_grid, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return np.zeros_like(x_grid, dtype=np.float64)
    if bandwidth is None:
        std = float(np.std(values))
        bandwidth = max(0.04, 1.06 * max(std, 1e-3) * (len(values) ** (-0.2)))
    bandwidth = max(float(bandwidth), 1e-3)
    z = (x_grid[:, None] - values[None, :]) / (bandwidth * sqrt(2.0))
    erf_vec = np.vectorize(erf)
    cdf = 0.5 * (1.0 + erf_vec(z))
    return np.clip(np.asarray(cdf.mean(axis=1), dtype=np.float64), 0.0, 1.0)


def load_archives(manifest_path: Path, archive_root: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, row in manifest.iterrows():
        archive_path = Path(row["archive_path"])
        if not archive_path.is_absolute():
            if archive_path.exists():
                archive_path = archive_path
            else:
                archive_path = archive_root / archive_path
        if not archive_path.exists():
            continue
        solutions, pareto_front_points, meta = load_npz_archive(archive_path)
        rows.append(
            {
                **row.to_dict(),
                "archive_path": str(archive_path),
                "solutions": solutions,
                "pareto_front": pareto_front_points,
                "meta": meta,
            }
        )
    if not rows:
        raise FileNotFoundError("No usable NPZ archives found.")
    return pd.DataFrame(rows)


def filter_archive_df(
    archive_df: pd.DataFrame,
    *,
    include_envs: list[str] | None,
    exclude_envs: list[str] | None,
    include_methods: list[str] | None,
    exclude_methods: list[str] | None,
) -> pd.DataFrame:
    include_env_set = {str(env_key) for env_key in (include_envs or [])}
    exclude_env_set = {str(env_key) for env_key in (exclude_envs or [])}
    include_set = {str(method) for method in (include_methods or [])}
    exclude_set = {str(method) for method in (exclude_methods or [])}
    df = archive_df.copy()
    if include_env_set:
        df = df[df["env_key"].astype(str).isin(include_env_set)]
    if exclude_env_set:
        df = df[~df["env_key"].astype(str).isin(exclude_env_set)]
    if include_set:
        df = df[df["method"].astype(str).isin(include_set)]
    if exclude_set:
        df = df[~df["method"].astype(str).isin(exclude_set)]
    return df.reset_index(drop=True)


def collect_group_fronts(df: pd.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    grouped = {}
    for (env_key, method, regime), sub in df.groupby(["env_key", "method", "regime"]):
        solutions = [np.asarray(arr, dtype=np.float64) for arr in sub["solutions"] if np.asarray(arr).size > 0]
        if not solutions:
            continue
        pooled = np.concatenate([arr if arr.ndim == 2 else arr[None, :] for arr in solutions], axis=0)
        grouped[(env_key, method, regime)] = pareto_front(pooled)
    return grouped


def env_objective_bounds(group_fronts: dict[tuple[str, str, str], np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    env_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for (env_key, _method, _regime), front in group_fronts.items():
        if front.ndim == 2 and len(front) > 0:
            env_points[env_key].append(front)
    for env_key, fronts in env_points.items():
        stacked = np.concatenate(fronts, axis=0)
        bounds[env_key] = (stacked.min(axis=0), stacked.max(axis=0))
    return bounds


def env_reference_points(group_fronts: dict[tuple[str, str, str], np.ndarray]) -> dict[str, np.ndarray]:
    env_points: dict[str, list[np.ndarray]] = defaultdict(list)
    for (env_key, _method, _regime), front in group_fronts.items():
        if front.ndim == 2 and len(front) > 0:
            env_points[env_key].append(front)
    return {env_key: derive_reference_point(fronts) for env_key, fronts in env_points.items()}


def save_regime_metrics(
    group_fronts: dict[tuple[str, str, str], np.ndarray],
    ref_points: dict[str, np.ndarray],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (env_key, method, regime), front in sorted(group_fronts.items()):
        ref_point = ref_points[env_key]
        metric = front_metrics(front, ref_point)
        ci = bootstrap_front_metric_ci(front, ref_point, samples=bootstrap_samples, rng=rng)
        for metric_name in ["HV", "EU", "SP"]:
            mean_value = float(metric[metric_name])
            low, high = ci[metric_name]
            rows.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "regime": regime,
                    "metric": metric_name,
                    "mean": mean_value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_front_points": int(len(front)),
                }
            )
    return pd.DataFrame(rows)


def save_regime_objective_summary(
    group_fronts: dict[tuple[str, str, str], np.ndarray],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (env_key, method, regime), front in sorted(group_fronts.items()):
        front = np.asarray(front, dtype=np.float64)
        if front.ndim != 2 or len(front) == 0:
            continue
        mean = front.mean(axis=0)
        low, high = bootstrap_vector_mean_ci(front, samples=bootstrap_samples, rng=rng)
        for obj_idx in range(front.shape[1]):
            rows.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "regime": regime,
                    "objective": f"obj_{obj_idx}",
                    "mean": float(mean[obj_idx]),
                    "ci_low": float(low[obj_idx]),
                    "ci_high": float(high[obj_idx]),
                    "n_front_points": int(len(front)),
                }
            )
    return pd.DataFrame(rows)


def save_env_metrics(regime_df: pd.DataFrame, bootstrap_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pivot = regime_df.pivot_table(
        index=["env_key", "method", "regime"], columns="metric", values="mean", aggfunc="mean"
    ).reset_index()
    for (env_key, method), sub in pivot.groupby(["env_key", "method"]):
        for metric in ["HV", "EU", "SP"]:
            values = sub[metric].dropna().to_numpy(dtype=np.float64)
            if len(values) == 0:
                continue
            mean_value = float(values.mean())
            low, high = bootstrap_mean_ci(values, samples=bootstrap_samples, rng=rng)
            rows.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "metric": metric,
                    "mean": mean_value,
                    "ci_low": low,
                    "ci_high": high,
                    "n_regimes": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def save_env_objective_summary(
    regime_objective_df: pd.DataFrame,
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for (env_key, method, objective), sub in regime_objective_df.groupby(["env_key", "method", "objective"]):
        values = sub["mean"].dropna().to_numpy(dtype=np.float64)
        if len(values) == 0:
            continue
        mean_value = float(values.mean())
        low, high = bootstrap_mean_ci(values, samples=bootstrap_samples, rng=rng)
        rows.append(
            {
                "env_key": env_key,
                "method": method,
                "objective": objective,
                "mean": mean_value,
                "ci_low": low,
                "ci_high": high,
                "n_regimes": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def build_cdf_curves(
    group_fronts: dict[tuple[str, str, str], np.ndarray],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    cdf_mode: str,
    grid_points: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x_grid = np.linspace(0.0, 1.0, grid_points)
    records = []
    env_methods = defaultdict(lambda: defaultdict(list))
    for (env_key, method, regime), front in group_fronts.items():
        if front.ndim != 2 or len(front) == 0:
            continue
        env_methods[(env_key, method)][regime] = front
    for (env_key, method), regime_map in sorted(env_methods.items()):
        min_vals, max_vals = bounds[env_key]
        for obj_idx in range(min_vals.shape[0]):
            regime_cdfs = []
            for front in regime_map.values():
                values = normalize_to_unit(front[:, obj_idx], float(min_vals[obj_idx]), float(max_vals[obj_idx]))
                if cdf_mode == "smoothed":
                    regime_cdfs.append(smoothed_cdf(values, x_grid))
                else:
                    regime_cdfs.append(empirical_cdf(values, x_grid))
            regime_cdfs = np.asarray(regime_cdfs, dtype=np.float64)
            if len(regime_cdfs) == 0:
                continue
            mean_cdf = regime_cdfs.mean(axis=0)
            boot = []
            if len(regime_cdfs) == 1:
                low = high = mean_cdf
            else:
                for _ in range(max(1, bootstrap_samples)):
                    idx = rng.integers(0, len(regime_cdfs), size=len(regime_cdfs))
                    boot.append(regime_cdfs[idx].mean(axis=0))
                boot = np.asarray(boot, dtype=np.float64)
                low = np.quantile(boot, 0.025, axis=0)
                high = np.quantile(boot, 0.975, axis=0)
            for xi, x in enumerate(x_grid):
                records.append(
                    {
                        "env_key": env_key,
                        "method": method,
                        "objective": f"obj_{obj_idx}",
                        "x": float(x),
                        "mean_cdf": float(mean_cdf[xi]),
                        "ci_low": float(low[xi]),
                        "ci_high": float(high[xi]),
                        "n_regimes": int(len(regime_cdfs)),
                    }
                )
    return pd.DataFrame(records)


def build_best_cdf_curves(
    group_fronts: dict[tuple[str, str, str], np.ndarray],
    bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    grid_points: int,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    x_grid = np.linspace(0.0, 1.0, grid_points)
    records = []
    env_regimes = defaultdict(lambda: defaultdict(list))
    for (env_key, _method, regime), front in group_fronts.items():
        if front.ndim != 2 or len(front) == 0:
            continue
        env_regimes[env_key][regime].append(front)
    for env_key, regime_map in sorted(env_regimes.items()):
        min_vals, max_vals = bounds[env_key]
        regime_reference_fronts: dict[str, np.ndarray] = {}
        for regime, fronts in regime_map.items():
            pooled = np.concatenate(fronts, axis=0)
            regime_reference_fronts[regime] = pareto_front(pooled)
        for obj_idx in range(min_vals.shape[0]):
            regime_best = []
            for regime in sorted(regime_reference_fronts):
                ref_front = regime_reference_fronts[regime]
                if ref_front.ndim != 2 or len(ref_front) == 0:
                    continue
                regime_best.append(float(np.max(ref_front[:, obj_idx])))
            if not regime_best:
                continue
            norm_best = normalize_to_unit(
                np.asarray(regime_best, dtype=np.float64),
                float(min_vals[obj_idx]),
                float(max_vals[obj_idx]),
            )
            mean_cdf = empirical_cdf(norm_best, x_grid)
            if len(norm_best) == 1:
                low = high = mean_cdf
            else:
                boot = []
                for _ in range(max(1, bootstrap_samples)):
                    idx = rng.integers(0, len(norm_best), size=len(norm_best))
                    boot.append(empirical_cdf(norm_best[idx], x_grid))
                boot = np.asarray(boot, dtype=np.float64)
                low = np.quantile(boot, 0.025, axis=0)
                high = np.quantile(boot, 0.975, axis=0)
            for xi, x in enumerate(x_grid):
                records.append(
                    {
                        "env_key": env_key,
                        "method": "best_cdf",
                        "objective": f"obj_{obj_idx}",
                        "x": float(x),
                        "mean_cdf": float(mean_cdf[xi]),
                        "ci_low": float(low[xi]),
                        "ci_high": float(high[xi]),
                        "n_regimes": int(len(norm_best)),
                    }
                )
    return pd.DataFrame(records)


def build_best_reference_cdf(
    cdf_df: pd.DataFrame,
    *,
    reference_method_name: str = "best_cdf",
) -> pd.DataFrame:
    if cdf_df.empty:
        return cdf_df.copy()
    records = []
    key_cols = ["env_key", "objective", "x"]
    for (env_key, objective, x), sub in cdf_df.groupby(key_cols):
        records.append(
            {
                "env_key": env_key,
                "method": reference_method_name,
                "objective": objective,
                "x": float(x),
                "mean_cdf": float(sub["mean_cdf"].min()),
                "ci_low": float(sub["ci_low"].min()),
                "ci_high": float(sub["ci_high"].min()),
                "n_regimes": int(sub["n_regimes"].max()),
            }
        )
    return pd.DataFrame(records)


def plot_cdf_panels(cdf_df: pd.DataFrame, out_dir: Path, highlight_method: str = "dynamic") -> None:
    if cdf_df.empty:
        return
    methods = ordered_methods(cdf_df["method"].unique().tolist())
    color_map = method_color_map(methods)
    for env_key, sub in cdf_df.groupby("env_key"):
        objectives = sorted(sub["objective"].unique().tolist())
        fig, axes = plt.subplots(1, len(objectives), figsize=(5.2 * len(objectives), 4.2), sharey=True)
        if len(objectives) == 1:
            axes = [axes]
        for ax, objective in zip(axes, objectives):
            obj_sub = sub[sub["objective"] == objective]
            for method in methods:
                method_sub = obj_sub[obj_sub["method"] == method].sort_values("x")
                if method_sub.empty:
                    continue
                x = method_sub["x"].to_numpy()
                y = method_sub["mean_cdf"].to_numpy()
                lw = 3.6 if method == highlight_method else 3.0
                alpha = 0.98
                ax.plot(
                    x,
                    y,
                    label=display_method_name(method),
                    color=color_map[method],
                    linewidth=lw,
                    alpha=alpha,
                )
            ax.set_title(f"{env_key} | {objective}")
            ax.set_xlabel("Normalized return")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Cumulative probability")
        fig.legend(
            handles=axes[0].lines,
            labels=[line.get_label() for line in axes[0].lines],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(5, len(methods)),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / f"cdf_{env_key}.png", dpi=260)
        plt.close(fig)


def plot_cdf_panels_with_prefix(
    cdf_df: pd.DataFrame,
    out_dir: Path,
    *,
    prefix: str,
    highlight_method: str = "dynamic",
) -> None:
    if cdf_df.empty:
        return
    methods = ordered_methods(cdf_df["method"].unique().tolist())
    color_map = method_color_map(methods)
    for env_key, sub in cdf_df.groupby("env_key"):
        objectives = sorted(sub["objective"].unique().tolist())
        fig, axes = plt.subplots(1, len(objectives), figsize=(5.2 * len(objectives), 4.2), sharey=True)
        if len(objectives) == 1:
            axes = [axes]
        for ax, objective in zip(axes, objectives):
            obj_sub = sub[sub["objective"] == objective]
            for method in methods:
                method_sub = obj_sub[obj_sub["method"] == method].sort_values("x")
                if method_sub.empty:
                    continue
                x = method_sub["x"].to_numpy()
                y = method_sub["mean_cdf"].to_numpy()
                lw = 3.6 if method == highlight_method else 3.0
                alpha = 0.98
                ax.plot(
                    x,
                    y,
                    label=display_method_name(method),
                    color=color_map[method],
                    linewidth=lw,
                    alpha=alpha,
                )
            ax.set_title(f"{env_key} | {objective}")
            ax.set_xlabel("Normalized best return")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Cumulative probability")
        fig.legend(
            handles=axes[0].lines,
            labels=[line.get_label() for line in axes[0].lines],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(5, len(methods)),
            fontsize=8,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / f"{prefix}_{env_key}.png", dpi=260)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    archive_df = load_archives(args.manifest_path, args.archive_root)
    archive_df = filter_archive_df(
        archive_df,
        include_envs=args.include_envs,
        exclude_envs=args.exclude_envs,
        include_methods=args.include_methods,
        exclude_methods=args.exclude_methods,
    )
    if archive_df.empty:
        raise FileNotFoundError("No archives remain after method filtering.")
    group_fronts = collect_group_fronts(archive_df)
    if not group_fronts:
        raise FileNotFoundError("No grouped fronts found in archives.")
    if args.require_highlight_method:
        envs_with_highlight = {
            env_key
            for (env_key, method, _regime) in group_fronts
            if str(method) == str(args.highlight_method)
        }
        group_fronts = {
            key: front
            for key, front in group_fronts.items()
            if key[0] in envs_with_highlight
        }
        if not group_fronts:
            raise FileNotFoundError("No grouped fronts remain after requiring highlight method.")
    bounds = env_objective_bounds(group_fronts)
    ref_points = env_reference_points(group_fronts)

    regime_df = save_regime_metrics(
        group_fronts,
        ref_points,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    env_df = save_env_metrics(regime_df, bootstrap_samples=args.bootstrap_samples, rng=rng)
    regime_objective_df = save_regime_objective_summary(
        group_fronts,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    env_objective_df = save_env_objective_summary(
        regime_objective_df,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    cdf_df = build_cdf_curves(
        group_fronts,
        bounds,
        cdf_mode=args.cdf_mode,
        grid_points=args.cdf_grid_points,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )
    best_cdf_df = build_best_cdf_curves(
        group_fronts,
        bounds,
        grid_points=args.cdf_grid_points,
        bootstrap_samples=args.bootstrap_samples,
        rng=rng,
    )

    regime_df.to_csv(args.out_dir / "regime_metric_summary.csv", index=False)
    env_df.to_csv(args.out_dir / "env_metric_summary.csv", index=False)
    regime_objective_df.to_csv(args.out_dir / "regime_objective_summary.csv", index=False)
    env_objective_df.to_csv(args.out_dir / "env_objective_summary.csv", index=False)
    cdf_df.to_csv(args.out_dir / "cdf_curves.csv", index=False)
    best_cdf_df.to_csv(args.out_dir / "best_cdf_curves.csv", index=False)
    plot_cdf_panels(cdf_df, args.out_dir, highlight_method=args.highlight_method)
    plot_cdf_panels_with_prefix(
        best_cdf_df,
        args.out_dir,
        prefix="best_cdf",
        highlight_method=args.highlight_method,
    )
    best_ref_cdf_df = pd.concat([cdf_df, best_cdf_df], ignore_index=True)
    best_ref_cdf_df.to_csv(args.out_dir / "cdf_vs_best_curves.csv", index=False)
    plot_cdf_panels_with_prefix(
        best_ref_cdf_df,
        args.out_dir,
        prefix="cdf_vs_best",
        highlight_method="best_cdf",
    )

    print(f"wrote regime summary: {args.out_dir / 'regime_metric_summary.csv'}")
    print(f"wrote env summary: {args.out_dir / 'env_metric_summary.csv'}")
    print(f"wrote regime objective summary: {args.out_dir / 'regime_objective_summary.csv'}")
    print(f"wrote env objective summary: {args.out_dir / 'env_objective_summary.csv'}")
    print(f"wrote cdf curves: {args.out_dir / 'cdf_curves.csv'}")
    print(f"wrote best cdf curves: {args.out_dir / 'best_cdf_curves.csv'}")
    print(f"wrote cdf-vs-best curves: {args.out_dir / 'cdf_vs_best_curves.csv'}")


if __name__ == "__main__":
    main()
