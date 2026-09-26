from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

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
FALLBACK_RUN_RE = re.compile(
    r"(?P<prefix>.+)_(?P<config>dynamic|static|random|no_dyn_ablation)_seed(?P<seed>\d+)$"
)
ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
EXPECTED_SHARED_PROTOCOL_REGIMES = 20
KEY_METRICS = [
    ("HV", True),
    ("EU", True),
    ("adapt_score", True),
    ("trace_recovery_latency", False),
    ("trace_recovery_score", True),
]


def _valid_regime_fronts(regime_fronts: dict[str, np.ndarray] | list[dict[str, Any]] | None) -> bool:
    if isinstance(regime_fronts, dict):
        return len(regime_fronts) >= EXPECTED_SHARED_PROTOCOL_REGIMES
    if isinstance(regime_fronts, list):
        unique_ids = {
            int(row.get("regime_id", -1))
            for row in regime_fronts
            if int(row.get("regime_id", -1)) >= 0
        }
        return len(regime_fronts) >= EXPECTED_SHARED_PROTOCOL_REGIMES and len(unique_ids) >= EXPECTED_SHARED_PROTOCOL_REGIMES
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scalarized-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "scalarized_baselines",
    )
    parser.add_argument(
        "--dynamic-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--dynamic-prefixes", nargs="+", required=True)
    parser.add_argument(
        "--morl-baseline-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "morl_dynamic_baselines",
    )
    parser.add_argument("--morl-suite-names", nargs="+", default=["morl_online_suite"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "all_benchmarks",
    )
    parser.add_argument("--highlight-method", type=str, default="dynamic")
    parser.add_argument("--oracle-tolerance", type=float, default=0.05)
    return parser.parse_args()


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    valid_fronts = [front for front in fronts if front.ndim == 2 and len(front) > 0]
    if not valid_fronts:
        raise ValueError("Cannot derive reference point from empty fronts.")
    stacked = np.concatenate(valid_fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def front_metrics(front: np.ndarray, ref_point: np.ndarray) -> dict[str, float]:
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or len(front) == 0:
        return {"HV": 0.0, "EU": 0.0, "SP": 0.0}
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], 0.5)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": float(eu), "SP": float(sp)}


def mean_trace_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_keys = {
        key
        for row in rows
        for key in row.keys()
        if key.startswith("trace_")
    }
    out = {}
    for key in sorted(metric_keys):
        values = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def regime_front_summary(regime_fronts: dict[str, np.ndarray]) -> dict[str, float]:
    if not regime_fronts:
        return {"cr_hv": 0.0, "cr_eu": 0.0, "cr_sp": 0.0, "irs": 0.0}
    ref_point = derive_reference_point(list(regime_fronts.values()))
    metrics = [front_metrics(front, ref_point) for front in regime_fronts.values()]
    eu_values = np.asarray([metric["EU"] for metric in metrics], dtype=np.float64)
    return {
        "cr_hv": float(np.mean([metric["HV"] for metric in metrics])),
        "cr_eu": float(np.mean([metric["EU"] for metric in metrics])),
        "cr_sp": float(np.mean([metric["SP"] for metric in metrics])),
        "irs": float(1.0 / (1.0 + eu_values.std())),
    }


def normalize_method_name(method: str) -> str:
    if method == "no_dyn_ablation":
        return "no_dyn"
    return method


def normalize_env_key(raw_name: str) -> str | None:
    name = str(raw_name).strip().lower()
    if not name:
        return None
    for env_key in ENV_ORDER:
        if name == env_key or name.startswith(f"{env_key}_"):
            return env_key
    return None


def load_scalarized_methods(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for env_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        env_key = env_dir.name
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for result_path in sorted(env_dir.glob("*/*/result.json")):
            data = json.loads(result_path.read_text())
            if not _valid_regime_fronts(data.get("regime_returns", [])):
                continue
            key = (str(data["alg"]), int(data.get("seed", 0)))
            grouped.setdefault(key, []).append(data)
        for (method, seed), rows in grouped.items():
            front = pareto_front(np.asarray([row["objs"] for row in rows], dtype=np.float64))
            regime_points: dict[str, list[np.ndarray]] = {}
            for row in rows:
                for regime_row in row.get("regime_returns", []):
                    regime_points.setdefault(regime_row["regime"], []).append(
                        np.asarray(regime_row["objs"], dtype=np.float64)
                    )
            regime_fronts = {
                regime: pareto_front(np.stack(points, axis=0))
                for regime, points in regime_points.items()
                if points
            }
            if not _valid_regime_fronts(regime_fronts):
                continue
            records.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "type": "baseline_scalarized",
                    "seed": seed,
                    "front": front,
                    "trace_metrics": mean_trace_metrics(
                        [dict(row.get("trace_metrics", {})) for row in rows]
                    ),
                    "regime_fronts": regime_fronts,
                }
            )
    return records


def latest_regime_front_file(run_dir: Path) -> Path | None:
    regime_dir = run_dir / "regime_fronts"
    if not regime_dir.exists():
        return None
    latest_path = None
    latest_key = (-1, "")
    for path in regime_dir.glob("*.json"):
        data = json.loads(path.read_text())
        key = (int(data.get("iteration", -1)), str(data.get("stage", "")))
        if key > latest_key:
            latest_key = key
            latest_path = path
    return latest_path


def parse_dynamic_run_metadata(run_dir: Path, metrics_df: pd.DataFrame) -> dict[str, Any] | None:
    match = RUN_RE.match(run_dir.name)
    if match is not None:
        return {
            "env_key": match.group("env_key"),
            "config": match.group("config"),
            "seed": int(match.group("seed")),
        }

    fallback = FALLBACK_RUN_RE.match(run_dir.name)
    if fallback is None:
        return None

    env_key = None
    if not metrics_df.empty and "env_name" in metrics_df.columns:
        env_key = normalize_env_key(str(metrics_df.iloc[-1].get("env_name", "")))
    if env_key is None:
        args_path = run_dir / "args.txt"
        if args_path.exists():
            env_match = re.search(r"'--env-name', '([^']+)'", args_path.read_text())
            if env_match is not None:
                env_key = normalize_env_key(env_match.group(1))
    if env_key is None:
        return None

    return {
        "env_key": env_key,
        "config": fallback.group("config"),
        "seed": int(fallback.group("seed")),
    }


def load_dynamic_methods(root: Path, prefixes: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prefix_idx, prefix in enumerate(prefixes):
        candidates = sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and path.name.startswith(prefix) and "_seed" in path.name
        )
        for run_dir in candidates:
            front_path = run_dir / "final" / "objs.txt"
            metrics_path = run_dir / "metrics_history.csv"
            regime_path = latest_regime_front_file(run_dir)
            if not front_path.exists() or not metrics_path.exists() or regime_path is None:
                continue
            front = np.loadtxt(front_path, delimiter=",")
            if front.ndim == 1:
                front = front[None, :]
            front = pareto_front(np.asarray(front, dtype=np.float64))
            metrics_df = pd.read_csv(metrics_path)
            if metrics_df.empty:
                continue
            metadata = parse_dynamic_run_metadata(run_dir, metrics_df)
            if metadata is None:
                continue
            final_metrics = metrics_df.iloc[-1].to_dict()
            regime_payload = json.loads(regime_path.read_text())
            if not _valid_regime_fronts(regime_payload.get("regimes", [])):
                continue
            regime_fronts = {
                regime_row["regime"]: pareto_front(np.asarray(regime_row["front_points"], dtype=np.float64))
                for regime_row in regime_payload.get("regimes", [])
                if regime_row.get("front_points")
            }
            records.append(
                {
                    "env_key": metadata["env_key"],
                    "method": normalize_method_name(metadata["config"]),
                    "type": "method" if metadata["config"] == "dynamic" else "ablation",
                    "seed": int(metadata["seed"]),
                    "front": np.asarray(front, dtype=np.float64),
                    "trace_metrics": {
                        key: float(value)
                        for key, value in final_metrics.items()
                        if key.startswith("trace_")
                    },
                    "regime_fronts": regime_fronts,
                    "extra_metrics": {
                        key: float(value)
                        for key, value in final_metrics.items()
                        if key
                        in {
                            "drift_score",
                            "prediction_error",
                            "context_gap_norm",
                            "online_trace_steps",
                            "online_shift_count",
                        }
                    },
                    "dynamic_prefix": prefix,
                    "source_priority": prefix_idx,
                    "source_name": prefix,
                }
            )
    return records


def load_morl_baseline_methods(root: Path, suite_names: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite_idx, suite_name in enumerate(suite_names):
        suite_root = root / suite_name
        if not suite_root.exists():
            continue
        for summary_path in sorted(suite_root.glob("*/*/seed*/summary.json")):
            data = json.loads(summary_path.read_text())
            if not _valid_regime_fronts(data.get("regime_fronts", [])):
                continue
            regime_fronts = {
                row["regime"]: pareto_front(np.asarray(row["points"], dtype=np.float64))
                for row in data.get("regime_fronts", [])
                if row.get("points")
            }
            records.append(
                {
                    "env_key": str(data["env_key"]),
                    "method": str(data["method"]).lower().replace("-", "_").replace(" ", "_"),
                    "type": "baseline_morl",
                    "seed": int(data.get("seed", 0)),
                    "front": pareto_front(np.asarray(data.get("front_points", []), dtype=np.float64)),
                    "trace_metrics": {
                        key: float(value)
                        for key, value in data.get("trace_metrics", {}).items()
                    },
                    "regime_fronts": regime_fronts,
                    "suite_name": suite_name,
                    "source_priority": suite_idx,
                    "source_name": suite_name,
                }
            )
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, int], int] = {}
    for record in records:
        key = (
            str(record["env_key"]),
            str(record["method"]),
            str(record["type"]),
            int(record["seed"]),
        )
        if key in seen:
            continue
        seen[key] = len(deduped)
        deduped.append(record)
    return deduped


def adaptation_metrics(
    regime_fronts: dict[str, np.ndarray],
    oracle_fronts: dict[str, np.ndarray],
    oracle_tolerance: float,
) -> dict[str, float]:
    if not regime_fronts or not oracle_fronts:
        return {
            "rag_hv": 1.0,
            "rag_eu": 1.0,
            "rwr_hv": 0.0,
            "rwr_eu": 0.0,
            "adapt_score": 0.0,
        }
    hv_gaps = []
    eu_gaps = []
    hv_wins = []
    eu_wins = []
    for regime, oracle_front in oracle_fronts.items():
        method_front = regime_fronts.get(regime)
        if method_front is None or len(method_front) == 0:
            hv_gaps.append(1.0)
            eu_gaps.append(1.0)
            hv_wins.append(0.0)
            eu_wins.append(0.0)
            continue
        ref_point = derive_reference_point([oracle_front, method_front])
        oracle_metric = front_metrics(oracle_front, ref_point)
        method_metric = front_metrics(method_front, ref_point)
        oracle_hv = max(abs(oracle_metric["HV"]), 1e-8)
        oracle_eu = max(abs(oracle_metric["EU"]), 1e-8)
        hv_gap = max(0.0, oracle_metric["HV"] - method_metric["HV"]) / oracle_hv
        eu_gap = max(0.0, oracle_metric["EU"] - method_metric["EU"]) / oracle_eu
        hv_gaps.append(hv_gap)
        eu_gaps.append(eu_gap)
        hv_wins.append(float(hv_gap <= oracle_tolerance))
        eu_wins.append(float(eu_gap <= oracle_tolerance))
    mean_hv_gap = float(np.mean(hv_gaps))
    mean_eu_gap = float(np.mean(eu_gaps))
    return {
        "rag_hv": mean_hv_gap,
        "rag_eu": mean_eu_gap,
        "rwr_hv": float(np.mean(hv_wins)),
        "rwr_eu": float(np.mean(eu_wins)),
        "adapt_score": float(np.clip(1.0 - 0.5 * (mean_hv_gap + mean_eu_gap), 0.0, 1.0)),
    }


def fill_missing_key_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for env_key, sub_idx in df.groupby("env_key").groups.items():
        for metric, higher_better in KEY_METRICS:
            if metric not in df.columns:
                continue
            missing_mask = df.index.isin(sub_idx) & df[metric].isna()
            if not missing_mask.any():
                continue
            observed = df.loc[list(sub_idx), metric].dropna()
            if observed.empty:
                fill_value = 0.0 if higher_better else 1.0
            else:
                fill_value = float(observed.min() if higher_better else observed.max())
            df.loc[missing_mask, metric] = fill_value
    return df


def enrich_records(records: list[dict[str, Any]], oracle_tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        env_records = [record for record in records if record["env_key"] == env_key]
        if not env_records:
            continue
        ref_point = derive_reference_point([record["front"] for record in env_records])
        regime_oracles: dict[str, np.ndarray] = {}
        regime_pool: dict[str, list[np.ndarray]] = {}
        for record in env_records:
            for regime, front in record["regime_fronts"].items():
                if front.ndim == 2 and len(front) > 0:
                    regime_pool.setdefault(regime, []).append(front)
        for regime, fronts in regime_pool.items():
            regime_oracles[regime] = pareto_front(np.concatenate(fronts, axis=0))

        for record in env_records:
            front = record["front"]
            core = front_metrics(front, ref_point)
            regime_summary = regime_front_summary(record["regime_fronts"])
            adapt = adaptation_metrics(record["regime_fronts"], regime_oracles, oracle_tolerance)
            row = {
                "env_key": env_key,
                "method": record["method"],
                "type": record["type"],
                "seed": int(record["seed"]),
                "num_points": int(len(front)),
                **core,
                **regime_summary,
                **adapt,
            }
            row.update(record.get("trace_metrics", {}))
            row.update(record.get("extra_metrics", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def metric_rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env_key, sub in df.groupby("env_key"):
        for metric, higher_better in KEY_METRICS:
            ranked = sub[["method", metric]].sort_values(metric, ascending=not higher_better).reset_index(drop=True)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "env_key": env_key,
                        "metric": metric,
                        "method": row["method"],
                        "value": row[metric],
                        "rank": rank,
                    }
                )
    return pd.DataFrame(rows)


def regime_mean_rank_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    regime_metrics = [("HV", True), ("EU", True), ("SP", False)]
    for env_key in ENV_ORDER:
        env_records = [record for record in records if record["env_key"] == env_key]
        if not env_records:
            continue

        regime_names = sorted(
            {
                str(regime)
                for record in env_records
                for regime in record.get("regime_fronts", {}).keys()
            }
        )
        if not regime_names:
            continue

        method_rank_rows = []
        for regime in regime_names:
            present_fronts = []
            for record in env_records:
                front = record.get("regime_fronts", {}).get(regime)
                if front is not None and getattr(front, "ndim", 0) == 2 and len(front) > 0:
                    present_fronts.append(front)
            if not present_fronts:
                continue
            ref_point = derive_reference_point(present_fronts)
            metric_rows = []
            for record in env_records:
                front = record.get("regime_fronts", {}).get(regime)
                if front is None or getattr(front, "ndim", 0) != 2 or len(front) == 0:
                    metric_rows.append(
                        {
                            "env_key": env_key,
                            "regime": regime,
                            "method": record["method"],
                            "HV": np.nan,
                            "EU": np.nan,
                            "SP": np.nan,
                        }
                    )
                else:
                    metric_rows.append(
                        {
                            "env_key": env_key,
                            "regime": regime,
                            "method": record["method"],
                            **front_metrics(front, ref_point),
                        }
                    )
            regime_df = pd.DataFrame(metric_rows)
            for metric, higher_better in regime_metrics:
                regime_df[f"{metric}_rank"] = regime_df[metric].rank(
                    ascending=not higher_better,
                    method="average",
                    na_option="bottom",
                )
            method_rank_rows.append(regime_df)

        if not method_rank_rows:
            continue

        all_regime_ranks = pd.concat(method_rank_rows, ignore_index=True)
        for method, sub in all_regime_ranks.groupby("method"):
            for metric, _higher_better in regime_metrics:
                rows.append(
                    {
                        "env_key": env_key,
                        "metric": f"mean_rank_{metric.lower()}",
                        "method": method,
                        "value": float(sub[metric].mean(skipna=True)),
                        "rank": float(sub[f"{metric}_rank"].mean(skipna=True)),
                    }
                )
    return pd.DataFrame(rows)


def wins_frame(df: pd.DataFrame, highlight_method: str) -> pd.DataFrame:
    rows = []
    baselines = {"baseline_scalarized", "baseline_morl"}
    for env_key, sub in df.groupby("env_key"):
        highlight = sub[sub["method"] == highlight_method]
        if highlight.empty:
            continue
        highlight_row = highlight.iloc[0]
        base_rows = sub[sub["type"].isin(baselines)]
        for metric, higher_better in KEY_METRICS:
            if higher_better:
                wins = int((highlight_row[metric] > base_rows[metric]).sum())
            else:
                wins = int((highlight_row[metric] < base_rows[metric]).sum())
            rows.append(
                {
                    "env_key": env_key,
                    "metric": metric,
                    "wins_vs_baselines": wins,
                    "num_baselines": int(len(base_rows)),
                }
            )
    return pd.DataFrame(rows)


def pairwise_wins_frame(df: pd.DataFrame, highlight_method: str) -> pd.DataFrame:
    rows = []
    baselines = {"baseline_scalarized", "baseline_morl"}
    for env_key, sub in df.groupby("env_key"):
        highlight = sub[sub["method"] == highlight_method]
        if highlight.empty:
            continue
        highlight_row = highlight.iloc[0]
        base_rows = sub[sub["type"].isin(baselines)]
        for _, base_row in base_rows.iterrows():
            for metric, higher_better in KEY_METRICS:
                highlight_value = float(highlight_row[metric])
                baseline_value = float(base_row[metric])
                win = highlight_value > baseline_value if higher_better else highlight_value < baseline_value
                rows.append(
                    {
                        "env_key": env_key,
                        "baseline_method": str(base_row["method"]),
                        "metric": metric,
                        "highlight_method": highlight_method,
                        "highlight_value": highlight_value,
                        "baseline_value": baseline_value,
                        "highlight_wins": int(win),
                    }
                )
    return pd.DataFrame(rows)


def summarize_pairwise_wins(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_df.empty:
        return pd.DataFrame(
            columns=[
                "env_key",
                "baseline_method",
                "highlight_method",
                "wins",
                "num_metrics",
                "win_rate",
            ]
        )

    grouped = (
        pairwise_df.groupby(["env_key", "baseline_method", "highlight_method"])["highlight_wins"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "wins", "count": "num_metrics"})
    )
    grouped["win_rate"] = grouped["wins"] / grouped["num_metrics"].clip(lower=1)

    overall = (
        pairwise_df.groupby(["baseline_method", "highlight_method"])["highlight_wins"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "wins", "count": "num_metrics"})
    )
    overall["env_key"] = "overall"
    overall["win_rate"] = overall["wins"] / overall["num_metrics"].clip(lower=1)

    return pd.concat([grouped, overall], ignore_index=True, sort=False)[
        ["env_key", "baseline_method", "highlight_method", "wins", "num_metrics", "win_rate"]
    ]


def render_markdown_table(df: pd.DataFrame, metrics: list[str]) -> str:
    header = ["env_key", "method", "type", *metrics]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for _, row in df.iterrows():
        values = [
            str(row["env_key"]),
            str(row["method"]),
            str(row["type"]),
            *[
                f"{float(row[metric]):.4f}" if pd.notna(row[metric]) else "nan"
                for metric in metrics
            ],
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_latex_table(df: pd.DataFrame, metrics: list[str]) -> str:
    cols = "lll" + "c" * len(metrics)
    lines = [
        "\\begin{tabular}{" + cols + "}",
        "\\toprule",
        "Env & Method & Type & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        metric_values = [
            f"{float(row[metric]):.4f}" if pd.notna(row[metric]) else "nan"
            for metric in metrics
        ]
        lines.append(
            f"{row['env_key']} & {row['method']} & {row['type']} & " + " & ".join(metric_values) + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def plot_metric_grid(df: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    metrics = [metric for metric, _ in KEY_METRICS]
    envs = [env for env in ENV_ORDER if env in set(df["env_key"])]
    fig, axes = plt.subplots(len(envs), len(metrics), figsize=(3.0 * len(metrics), 3.0 * len(envs)))
    if len(envs) == 1:
        axes = np.asarray([axes])
    for row_idx, env_key in enumerate(envs):
        sub = df[df["env_key"] == env_key]
        for col_idx, (metric, higher_better) in enumerate(KEY_METRICS):
            ax = axes[row_idx, col_idx]
            ordered = sub.sort_values(metric, ascending=not higher_better)
            colors = [
                "tab:blue" if method == highlight_method else ("tab:green" if row_type == "ablation" else "tab:gray")
                for method, row_type in zip(ordered["method"], ordered["type"])
            ]
            x = np.arange(len(ordered))
            ax.bar(x, ordered[metric], color=colors)
            ax.set_title(f"{env_key} | {metric}", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(ordered["method"], rotation=55, ha="right", fontsize=7)
            ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_rank_heatmap(rank_df: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    sub = rank_df[rank_df["method"] == highlight_method]
    if sub.empty:
        return
    pivot = sub.pivot(index="env_key", columns="metric", values="rank").reindex(ENV_ORDER)
    metric_order = [metric for metric, _higher_better in KEY_METRICS if metric in pivot.columns]
    pivot = pivot[metric_order]
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 2.0, 1.3 * len(pivot.index) + 1.2))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                label = f"{value:.2f}" if abs(float(value) - round(float(value))) > 1e-8 else f"{int(round(float(value)))}"
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank (1 = best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_adaptation_panel(df: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    metrics = [("rag_hv", False), ("rag_eu", False), ("adapt_score", True), ("rwr_hv", True), ("rwr_eu", True)]
    envs = [env for env in ENV_ORDER if env in set(df["env_key"])]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 4.2))
    for ax, (metric, higher_better) in zip(np.atleast_1d(axes), metrics):
        ordered = (
            df[df["type"] != "ablation"]
            .groupby("method")[metric]
            .mean()
            .sort_values(ascending=not higher_better)
        )
        colors = ["tab:blue" if method == highlight_method else "tab:gray" for method in ordered.index]
        ax.bar(np.arange(len(ordered)), ordered.values, color=colors)
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(ordered)))
        ax.set_xticklabels(ordered.index, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def plot_pairwise_winrate(pairwise_summary: pd.DataFrame, out_path: Path, highlight_method: str) -> None:
    overall = pairwise_summary[pairwise_summary["env_key"] == "overall"].copy()
    if overall.empty:
        return
    overall = overall.sort_values("win_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(overall)), 4.2))
    ax.bar(np.arange(len(overall)), overall["win_rate"], color="tab:blue")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"{highlight_method} win rate")
    ax.set_title(f"{highlight_method} vs each baseline")
    ax.set_xticks(np.arange(len(overall)))
    ax.set_xticklabels(overall["baseline_method"], rotation=55, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    for idx, value in enumerate(overall["win_rate"]):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    records.extend(load_scalarized_methods(args.scalarized_root))
    records.extend(load_dynamic_methods(args.dynamic_root, args.dynamic_prefixes))
    records.extend(load_morl_baseline_methods(args.morl_baseline_root, args.morl_suite_names))
    records = dedupe_records(records)
    if not records:
        raise FileNotFoundError("No benchmark records found.")

    raw_df = enrich_records(records, oracle_tolerance=args.oracle_tolerance)
    raw_df = fill_missing_key_metrics(raw_df)
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    raw_df[numeric_cols] = raw_df[numeric_cols].fillna(0.0)
    raw_df = raw_df.sort_values(["env_key", "type", "method", "seed"]).reset_index(drop=True)
    key_cols = [metric for metric, _ in KEY_METRICS]
    if raw_df[key_cols].isna().any().any():
        missing = raw_df.loc[raw_df[key_cols].isna().any(axis=1)]
        missing.to_csv(args.out_dir / "missing_metrics.csv", index=False)
        raise ValueError(f"Missing benchmark metrics detected; see {args.out_dir / 'missing_metrics.csv'}")
    raw_df.to_csv(args.out_dir / "comparison_raw.csv", index=False)

    summary = (
        raw_df.groupby(["env_key", "method", "type"])[key_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(args.out_dir / "comparison_summary.csv", index=False)

    table_metrics = [
        "HV",
        "EU",
        "cr_hv",
        "irs",
        "trace_shift_regret",
        "trace_recovery_latency",
        "trace_recovery_score",
        "rag_hv",
        "rag_eu",
        "adapt_score",
    ]
    table_df = raw_df[["env_key", "method", "type", *table_metrics]].copy()
    table_df.to_csv(args.out_dir / "summary_table.csv", index=False)
    (args.out_dir / "summary_table.md").write_text(render_markdown_table(table_df, table_metrics))
    (args.out_dir / "summary_table.tex").write_text(render_latex_table(table_df, table_metrics))

    rank_df = metric_rank_frame(raw_df)
    rank_df.to_csv(args.out_dir / "metric_ranks.csv", index=False)
    regime_rank_df = regime_mean_rank_frame(records)
    if not regime_rank_df.empty:
        regime_rank_df.to_csv(args.out_dir / "regime_mean_ranks.csv", index=False)
    wins_df = wins_frame(raw_df, args.highlight_method)
    wins_df.to_csv(args.out_dir / "wins_vs_baselines.csv", index=False)
    pairwise_df = pairwise_wins_frame(raw_df, args.highlight_method)
    pairwise_df.to_csv(args.out_dir / "pairwise_wins.csv", index=False)
    pairwise_summary = summarize_pairwise_wins(pairwise_df)
    pairwise_summary.to_csv(args.out_dir / "pairwise_summary.csv", index=False)

    plot_metric_grid(raw_df, args.out_dir / "metric_grid.png", args.highlight_method)
    plot_rank_heatmap(rank_df, args.out_dir / "rank_heatmap.png", args.highlight_method)
    if not regime_rank_df.empty:
        plot_rank_heatmap(regime_rank_df, args.out_dir / "rank_heatmap_regime_mean.png", args.highlight_method)
    plot_adaptation_panel(raw_df, args.out_dir / "adaptation_panel.png", args.highlight_method)
    plot_pairwise_winrate(pairwise_summary, args.out_dir / "pairwise_winrate.png", args.highlight_method)

    print(raw_df[["env_key", "method", "type", *table_metrics]].to_string(index=False))


if __name__ == "__main__":
    main()
