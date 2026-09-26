from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(PROJECT_ROOT))

from src.dynamic_morl.metrics import compute_trace_shift_metrics, summarize_regime_metrics
from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


CONFIGS = ("dynamic", "static", "random", "no_dyn_ablation")


@dataclass
class Expert:
    method: str
    alg: str
    weight_idx: int
    objs: np.ndarray
    trace_metrics: dict[str, float]
    regime_returns: dict[str, np.ndarray]
    trace_segments: list[dict[str, Any]]
    global_eu: float
    mean_regime_eu: float
    std_regime_eu: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen"], required=True)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "scalarized_baselines",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--prefix", type=str, default="bank")
    parser.add_argument("--bank-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--router-resilience-lambda", type=float, default=0.15)
    parser.add_argument("--router-regret-lambda", type=float, default=0.05)
    return parser.parse_args()


def normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0 or np.ptp(values) <= 1e-8:
        return np.zeros_like(values)
    scaled = (values - values.min()) / (np.ptp(values) + 1e-8)
    return 1.0 - scaled if invert else scaled


def crowding_distance(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    num_samples, num_objectives = points.shape
    distance = np.zeros(num_samples, dtype=np.float64)
    if num_samples <= 2:
        distance[:] = 1.0
        return distance
    for obj_idx in range(num_objectives):
        order = np.argsort(points[:, obj_idx])
        ordered = points[order, obj_idx]
        span = ordered[-1] - ordered[0]
        if span <= 1e-8:
            continue
        distance[order[0]] += 1.0
        distance[order[-1]] += 1.0
        distance[order[1:-1]] += (ordered[2:] - ordered[:-2]) / span
    return distance


def pareto_front(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def front_metrics(front: np.ndarray, ref_point: np.ndarray, eval_delta_weight: float) -> dict[str, float]:
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], eval_delta_weight)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"hv": float(hv), "eu": float(eu), "sp": float(sp)}


def load_trace_segments(trace_path: Path) -> list[dict[str, Any]]:
    if not trace_path.exists():
        return []
    df = pd.read_csv(trace_path)
    if df.empty:
        return []
    obj_cols = [col for col in df.columns if col.startswith("obj_")]
    regime_ids = df["regime_id"].to_numpy()
    boundaries = np.where(regime_ids[1:] != regime_ids[:-1])[0] + 1
    cuts = [0, *boundaries.tolist(), len(df)]
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        seg = df.iloc[start:end]
        segments.append(
            {
                "regime": str(seg["regime"].iloc[0]),
                "regime_id": seg["regime_id"].iloc[0],
                "obj": seg[obj_cols].to_numpy(dtype=np.float64),
            }
        )
    return segments


def load_experts(env_root: Path, eval_delta_weight: float) -> list[Expert]:
    experts: list[Expert] = []
    prefs = generate_w_batch_test(3 if env_root.name != "cogen" else 4, eval_delta_weight)
    for result_path in sorted(env_root.glob("*/*/result.json")):
        data = json.loads(result_path.read_text())
        method = f"{data['alg']}_w{result_path.parent.name.split('_w')[-1]}"
        objs = np.asarray(data["objs"], dtype=np.float64)
        regime_return_rows = [
            (str(row["regime"]), np.asarray(row["objs"], dtype=np.float64))
            for row in data.get("regime_returns", [])
        ]
        regime_returns = {regime: values for regime, values in regime_return_rows}
        regime_utils = []
        for reg_obj in regime_returns.values():
            regime_utils.append(float(np.mean(reg_obj @ prefs.T)))
        trace_segments = load_trace_segments(result_path.parent / "dynamic_trace.csv")
        if not trace_segments:
            trace_segments = [
                {
                    "regime": regime,
                    "regime_id": idx,
                    "obj": values[None, :],
                }
                for idx, (regime, values) in enumerate(regime_return_rows)
            ]
        experts.append(
            Expert(
                method=method,
                alg=str(data["alg"]),
                weight_idx=int(result_path.parent.name.split("_w")[-1]),
                objs=objs,
                trace_metrics={k: float(v) for k, v in data.get("trace_metrics", {}).items()},
                regime_returns=regime_returns,
                trace_segments=trace_segments,
                global_eu=float(np.mean(objs @ prefs.T)),
                mean_regime_eu=float(np.mean(regime_utils)) if regime_utils else 0.0,
                std_regime_eu=float(np.std(regime_utils)) if regime_utils else 0.0,
            )
        )
    return experts


def select_bank(experts: list[Expert], bank_size: int, config: str) -> list[Expert]:
    if not experts:
        return []
    points = np.stack([expert.objs for expert in experts], axis=0)
    crowd = normalize(crowding_distance(points))
    global_eu = normalize(np.asarray([expert.global_eu for expert in experts], dtype=np.float64))
    resilience = (
        0.35 * normalize(np.asarray([expert.trace_metrics.get("trace_recovery_score", 0.0) for expert in experts]))
        + 0.25 * normalize(np.asarray([expert.trace_metrics.get("trace_shift_regret", 0.0) for expert in experts]), invert=True)
        + 0.15 * normalize(np.asarray([expert.trace_metrics.get("trace_recovery_latency", 0.0) for expert in experts]), invert=True)
        + 0.15 * normalize(np.asarray([expert.trace_metrics.get("trace_utility_mean", 0.0) for expert in experts]))
        + 0.10 * normalize(np.asarray([expert.trace_metrics.get("trace_pre_post_gap", 0.0) for expert in experts]))
    )
    stability = 0.7 * normalize(np.asarray([expert.mean_regime_eu for expert in experts], dtype=np.float64)) + 0.3 * normalize(
        np.asarray([expert.std_regime_eu for expert in experts], dtype=np.float64), invert=True
    )

    if config == "dynamic":
        score = 0.35 * crowd + 0.25 * global_eu + 0.25 * resilience + 0.15 * stability
    elif config == "no_dyn_ablation":
        score = 0.45 * crowd + 0.35 * global_eu + 0.20 * stability
    elif config == "static":
        score = 0.55 * global_eu + 0.45 * stability
    elif config == "random":
        rng = np.random.default_rng(0)
        score = rng.random(len(experts))
    else:
        raise ValueError(config)

    used: set[int] = set()
    selected: list[Expert] = []
    if config == "dynamic":
        global_front_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
        selected = [experts[idx] for idx in global_front_idx[: min(bank_size, len(global_front_idx))]]
        used = set(global_front_idx.tolist())
        if len(selected) >= bank_size:
            return selected

    order = np.argsort(-score)
    for idx in order:
        if idx in used:
            continue
        selected.append(experts[idx])
        used.add(idx)
        if len(selected) >= min(bank_size, len(experts)):
            break
    return selected


def build_regime_fronts(selected: list[Expert]) -> dict[str, np.ndarray]:
    fronts: dict[str, list[np.ndarray]] = {}
    for expert in selected:
        for regime, objs in expert.regime_returns.items():
            fronts.setdefault(regime, []).append(np.asarray(objs, dtype=np.float64))
    return {regime: np.stack(rows, axis=0) for regime, rows in fronts.items() if rows}


def summarize_selected_bank(selected: list[Expert], eval_delta_weight: float) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    bank_points = np.stack([expert.objs for expert in selected], axis=0)
    bank_front = pareto_front(bank_points)
    regime_fronts = build_regime_fronts(selected)
    all_fronts = [bank_front, *[pareto_front(front) for front in regime_fronts.values()]]
    ref_point = derive_reference_point(all_fronts)
    regime_rows = []
    for regime, front in regime_fronts.items():
        metrics = front_metrics(pareto_front(front), ref_point, eval_delta_weight)
        regime_rows.append(
            {
                "regime": regime,
                "hv": metrics["hv"],
                "eu": metrics["eu"],
                "sp": metrics["sp"],
                "points": int(len(front)),
            }
        )
    summary = front_metrics(bank_front, ref_point, eval_delta_weight)
    summary.update(summarize_regime_metrics(regime_rows))
    summary["ep_size"] = int(len(selected))
    summary["selected_size"] = int(len(selected))
    return bank_front, summary, regime_rows


def reference_schedule(experts: list[Expert]) -> list[str]:
    for expert in experts:
        schedule = [segment["regime"] for segment in expert.trace_segments]
        if schedule:
            return schedule
    return []


def segment_by_regime(expert: Expert) -> dict[str, dict[str, Any]]:
    mapping = {}
    for segment in expert.trace_segments:
        mapping[str(segment["regime"])] = segment
    return mapping


def regime_score(expert: Expert, regime: str, prefs: np.ndarray, resilience_bonus: float, regret_penalty: float) -> float:
    regime_obj = expert.regime_returns.get(regime)
    if regime_obj is None:
        return -1e18
    value = float(np.mean(regime_obj @ prefs.T))
    value += resilience_bonus * float(expert.trace_metrics.get("trace_recovery_score", 0.0))
    value -= regret_penalty * float(expert.trace_metrics.get("trace_shift_regret", 0.0))
    return value


def route_experts(
    selected: list[Expert],
    config: str,
    schedule: list[str],
    eval_delta_weight: float,
    resilience_bonus: float,
    regret_penalty: float,
) -> tuple[dict[str, Expert], dict[str, Any] | None]:
    if not selected or not schedule:
        return {}, None
    prefs = generate_w_batch_test(selected[0].objs.shape[0], eval_delta_weight)
    if config == "static":
        best = max(selected, key=lambda expert: float(np.mean(expert.objs @ prefs.T)))
        mapping = {regime: best for regime in set(schedule)}
    elif config == "random":
        rng = np.random.default_rng(0)
        mapping = {regime: selected[int(rng.integers(len(selected)))] for regime in set(schedule)}
    elif config == "no_dyn_ablation":
        mapping = {
            regime: max(selected, key=lambda expert: regime_score(expert, regime, prefs, 0.0, 0.0))
            for regime in set(schedule)
        }
    else:
        mapping = {
            regime: max(
                selected,
                key=lambda expert: regime_score(
                    expert,
                    regime,
                    prefs,
                    resilience_bonus=resilience_bonus,
                    regret_penalty=regret_penalty,
                ),
            )
            for regime in set(schedule)
        }

    trace = {"obj": [], "regime_id": []}
    for regime in schedule:
        expert = mapping[regime]
        segment = segment_by_regime(expert).get(regime)
        if segment is None:
            continue
        trace["obj"].append(segment["obj"])
        trace["regime_id"].append(np.repeat(segment["regime_id"], len(segment["obj"])))
    if not trace["obj"]:
        return mapping, None
    return mapping, {
        "obj": np.concatenate(trace["obj"], axis=0),
        "regime_id": np.concatenate(trace["regime_id"], axis=0),
    }


def save_outputs(
    out_dir: Path,
    config: str,
    env_name: str,
    bank_front: np.ndarray,
    summary: dict[str, Any],
    regime_rows: list[dict[str, Any]],
    trace_metrics: dict[str, float],
    selected: list[Expert],
    router_mapping: dict[str, Expert],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(final_dir / "objs.txt", bank_front, fmt="%.6f", delimiter=",")

    metrics_row = {
        "stage": "bank",
        "iteration": 0,
        "seed": 0,
        "env_name": env_name,
        "selection_method": f"expert-bank-{config}",
        "hv": summary["hv"],
        "eu": summary["eu"],
        "sp": summary["sp"],
        "drift_score": 0.0,
        "regime_loss": 0.0,
        "ep_size": summary["ep_size"],
        "selected_size": summary["selected_size"],
        "cr_hv": summary["cr_hv"],
        "cr_eu": summary["cr_eu"],
        "cr_sp": summary["cr_sp"],
        "hv_std": summary["hv_std"],
        "eu_std": summary["eu_std"],
        "sp_std": summary["sp_std"],
        "irs": summary["irs"],
        **trace_metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics_history.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out_dir / "regime_metrics.csv", index=False)
    selection_payload = {
        "config": config,
        "selected_experts": [
            {
                "method": expert.method,
                "alg": expert.alg,
                "weight_idx": expert.weight_idx,
                "objs": expert.objs.tolist(),
            }
            for expert in selected
        ],
        "router_mapping": {regime: expert.method for regime, expert in router_mapping.items()},
    }
    (out_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2))


def main() -> None:
    args = parse_args()
    env_root = args.baseline_root / args.env_key
    experts = load_experts(env_root, args.eval_delta_weight)
    if not experts:
        raise FileNotFoundError(f"no experts found in {env_root}")

    env_name = f"{args.env_key}_expert_bank_dynamic"
    schedule = reference_schedule(experts)
    for config in CONFIGS:
        selected = select_bank(experts, args.bank_size, config)
        bank_front, summary, regime_rows = summarize_selected_bank(selected, args.eval_delta_weight)
        router_mapping, routed_trace = route_experts(
            selected,
            config,
            schedule,
            args.eval_delta_weight,
            resilience_bonus=args.router_resilience_lambda,
            regret_penalty=args.router_regret_lambda,
        )
        trace_metrics = compute_trace_shift_metrics(
            routed_trace or {"obj": [], "regime_id": []},
            args.eval_delta_weight,
        )
        out_dir = args.out_root / f"{args.prefix}_{args.env_key}_{config}_seed{args.seed}"
        save_outputs(
            out_dir,
            config,
            env_name,
            bank_front,
            summary,
            regime_rows,
            trace_metrics,
            selected,
            router_mapping,
        )
        print(
            json.dumps(
                {
                    "config": config,
                    "selected": len(selected),
                    "front_points": int(len(bank_front)),
                    "hv": summary["hv"],
                    "eu": summary["eu"],
                    "trace_shift_regret": trace_metrics["trace_shift_regret"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
