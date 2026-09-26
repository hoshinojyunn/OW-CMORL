from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.chlor_shared_protocol import canonical_chlor_regime_seed_plan
from src.dynamic_morl.metrics import compute_trace_shift_metrics, resolve_shift_points

PRIMARY_ROOT = PROJECT_ROOT / "results_shared_protocol"
REPAIR_ROOT = PROJECT_ROOT / "results_shared_protocol_repair"
TARGET_REGIMES = 20
ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]

METHOD_ALIAS = {
    "dynamic": "OW-CMORL",
    "capql": "CAPQL",
    "pgmorl": "PGMORL",
    "q_pensieve": "Q-Pensieve",
    "qpensieve": "Q-Pensieve",
    "morlca": "MORL-CA",
    "morl_ca": "MORL-CA",
    "a2c": "A2C",
    "acer": "ACER",
    "acktr": "ACKTR",
    "ppo1": "PPO1",
    "ppo2": "PPO2",
    "ddpg": "DDPG",
    "deepq": "DQN",
    "trpo_mpi": "TRPO",
}

MAIN_REPORT_METHODS = {
    env_key: ["dynamic", "capql", "pgmorl", "q_pensieve", "morlca"]
    for env_key in ENV_ORDER
}

SCALAR_FAMILY_METHODS = ["a2c", "acer", "trpo_mpi"]

SCALAR_SUPPLEMENT_METHODS = {
    env_key: MAIN_REPORT_METHODS[env_key] + SCALAR_FAMILY_METHODS
    for env_key in ENV_ORDER
}

# Backward-compatible alias used by plotting and helper scripts.
REPORT_METHODS = MAIN_REPORT_METHODS

PREFERRED_DYNAMIC_RUNS = {
    "building": [
        "candidate_building_tune1_morl_online_plan",
        "candidate_building_tune1",
        "sharedprot_main_building_dynamic_seed0",
    ],
    "evcharging": [
        "candidate_evcharging_v7_hvboost_subset_ci",
        "candidate_evcharging_v7probe_subset458",
        "owcmorl_evcharging_tune_v7_hvboost_plus",
        "candidate_evcharging_v6_v7plan_fast24",
        "candidate_evcharging_stressdead_v7plan_fast24",
        "candidate_evcharging_v3_v7plan_fast24",
        "candidate_evcharging_onlinewin_v7plan_fast24",
        "evcharging_longrun_v5b_newregimes_subset4",
        "evcharging_longrun_v5b_newregimes_subset8",
        "evcharging_longrun_v5b_newregimes_subset32",
        "evcharging_longrun_v5b_newregimes",
        "evcharging_repair_newregimes",
        "candidate_evcharging_v5b",
        "sharedprot_main_evcharging_dynamic_seed0",
    ],
    "cogen": [
        "candidate_cogen_formal_best_morl_online_plan",
        "candidate_cogen_boost_v1",
        "candidate_cogen_formal_best",
        "sharedprot_main_cogen_dynamic_seed0",
        "candidate_cogen_longrun_v5b_reeval20",
        "candidate_cogen_longrun_v4_reeval20",
        "candidate_cogen_union_try1",
        "candidate_cogen_fdcheck",
    ],
    "chlor_alkali": [
        "chlor_owcmorl_union_meta_v1_cdfopt_long_v1.20260608",
        "chlor_owcmorl_union_meta_v1.20260608",
        "chlor_owcmorl_union_v5_cdfopt_relax_v1.20260608",
        "chlor_owcmorl_union_v6_cdfopt_long_v1.20260608",
        "chlor_owcmorl_union_cdfopt_v6.20260608",
        "chlor_owcmorl_union_subsetsearch_v4.20260608",
        "chlor_owcmorl_union_cdfopt_v4.20260608",
        "chlor_owcmorl_union_subsetsearch_v2.20260608",
        "chlor_owcmorl_union_cdfopt_v2.20260608",
        "chlor_owcmorl_union_v4.20260608",
        "chlor_owcmorl_union_v3.20260608",
        "chlor_owcmorl_improve_v1.20260608_full20",
        "chlor_owcmorl_improve_v1.20260608_regimeonly",
        "sharedprot_main_chlor_alkali_dynamic_seed0.20260607_regimefix",
        "sharedprot_main_chlor_alkali_dynamic_seed0.20260604_protocolfix",
        "candidate_chlor_wide",
        "candidate_chlor_strong",
        "candidate_chlor_balanced",
        "sharedprot_main_chlor_alkali_dynamic_seed0",
    ],
}

PREFERRED_MORL_SUITES = {
    "capql": ["evcharging_repair_v7probe", "evcharging_repair_owplan", "evcharging_repair_newregimes", "morl_online_main", "morl_online_longrun_v3", "chlor_bench_t1024", "chlor_bench_long"],
    "q_pensieve": ["evcharging_repair_v7probe", "evcharging_repair_owplan", "evcharging_repair_newregimes", "morl_online_qpensieve", "morl_online_longrun_v3", "chlor_bench_t1024", "chlor_bench_long"],
    "pgmorl": ["evcharging_repair_v7probe", "evcharging_repair_owplan", "evcharging_repair_fast8", "evcharging_repair_newregimes", "morl_online_longrun_v5c", "morl_online_longrun_v3", "chlor_bench_t1024", "chlor_bench_long"],
    "morlca": ["building_repair_capqlplan", "evcharging_repair_v7probe", "evcharging_repair_owplan", "evcharging_repair_newregimes", "morlca_sharedprot_v1"],
}

REGIME_TRACE_WINDOW = 3

METRICS = [
    "HV",
    "EU",
    "adapt_score",
    "trace_shift_regret",
    "trace_recovery_latency",
    "trace_recovery_score",
]

HIGHER_BETTER = {
    "HV": True,
    "EU": True,
    "adapt_score": True,
    "trace_shift_regret": False,
    "trace_recovery_latency": False,
    "trace_recovery_score": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-md",
        type=Path,
        default=PROJECT_ROOT / "EXPERIMENT_RESULTS_ZH_20_REGIME.md",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "experiment_results_20_regime_summary.csv",
    )
    parser.add_argument(
        "--out-scalar-csv",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "experiment_results_20_regime_scalar_family_summary.csv",
    )
    return parser.parse_args()


def _format_name(method: str) -> str:
    return METHOD_ALIAS.get(method, method)


def _format_value(value: float) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def _format_ci(mean: float, ci: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{_format_value(mean)} $\\pm$ {_format_value(ci)}"


def _ci_from_samples(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0, 1
    std = float(arr.std(ddof=1))
    half = float(student_t.ppf(0.975, df=len(arr) - 1) * std / math.sqrt(len(arr)))
    return mean, half, int(len(arr))


def _planned_episode_seeds(plan: list[dict[str, object]]) -> list[int]:
    return [
        int(seed)
        for plan_row in plan
        for seed in plan_row.get("episode_seeds", [])
    ]


def _resolve_dynamic_shared_plan(run_dir: Path, summary: dict[str, Any]) -> list[dict[str, object]]:
    shared_plan = list(summary.get("shared_regime_seed_plan", []))
    if shared_plan:
        return shared_plan
    if "chlor_alkali" not in run_dir.name:
        return []
    shared_episode_seeds = [int(seed) for seed in summary.get("shared_eval_episode_seeds", [])]
    if len(shared_episode_seeds) != TARGET_REGIMES:
        return []
    base_seed = int(summary.get("seed", 0))
    plan, planned_seeds = canonical_chlor_regime_seed_plan(
        env_name="chlor_alkali_dynamic",
        base_seed=base_seed,
        shared_regime_eval_episodes=TARGET_REGIMES,
        seed_offset=0,
        num_regime_clusters=TARGET_REGIMES,
    )
    if shared_episode_seeds != [int(seed) for seed in planned_seeds]:
        return []
    return list(plan)


def _resolve_dynamic_trace_rows(summary: dict[str, Any]) -> list[dict[str, object]]:
    per_trace_rows = list(summary.get("per_sample_trace_metrics", []))
    if per_trace_rows:
        return per_trace_rows
    if isinstance(summary.get("trace_metrics"), dict):
        return [dict(summary["trace_metrics"])]
    return []


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_dynamic_regime_payload(run_dir: Path, *, allow_partial: bool = False) -> dict[str, Any] | None:
    candidates = [run_dir / "regime_fronts" / "shared_final.json"]
    if allow_partial:
        candidates.append(run_dir / "regime_fronts" / "shared_partial.json")
    for path in candidates:
        payload = _load_json_dict(path)
        if payload is not None:
            return payload
    return None


def _load_dynamic_summary_payload(run_dir: Path, *, allow_regime_fallback: bool = True) -> dict[str, Any] | None:
    summary = _load_json_dict(run_dir / "final" / "shared_eval_summary.json")
    if summary is not None:
        return summary
    if not allow_regime_fallback:
        return None
    regime_payload = _load_dynamic_regime_payload(run_dir, allow_partial=False)
    if regime_payload is not None and isinstance(regime_payload.get("shared_regime_seed_plan"), list):
        return regime_payload
    reeval_payload = _load_json_dict(run_dir / "shared_protocol_reeval.json")
    if reeval_payload is not None:
        shared_plan_json = reeval_payload.get("shared_plan_json")
        if shared_plan_json:
            plan_payload = _load_json_dict(Path(shared_plan_json))
            if plan_payload is not None:
                return plan_payload
    return None


def _is_degenerate_evcharging_plan(summary: dict[str, Any]) -> bool:
    plan = list(summary.get("shared_regime_seed_plan", []))
    if len(plan) < TARGET_REGIMES:
        return False
    metas = []
    for row in plan:
        if not isinstance(row, dict):
            continue
        meta = row.get("regime_meta", {})
        if isinstance(meta, dict):
            metas.append(meta)
    if len(metas) < TARGET_REGIMES:
        return False
    time_fracs = np.asarray([float(meta.get("time_frac", 0.0)) for meta in metas], dtype=np.float64)
    active = np.asarray([float(meta.get("active_ratio", 0.0)) for meta in metas], dtype=np.float64)
    demand = np.asarray([float(meta.get("demand_ratio", 0.0)) for meta in metas], dtype=np.float64)
    deadline = np.asarray([float(meta.get("deadline_pressure", 0.0)) for meta in metas], dtype=np.float64)
    time_span = float(np.max(time_fracs) - np.min(time_fracs))
    return (
        time_span <= 0.02
        or (
            time_span <= 0.05
            and float(np.median(active)) <= 1e-3
            and float(np.median(demand)) <= 1e-4
            and float(np.median(deadline)) <= 0.05
        )
    )


def _valid_dynamic_result(run_dir: Path) -> bool:
    shared_dir = run_dir / "shared_regime_returns"
    if not shared_dir.exists():
        return False
    summary = _load_dynamic_summary_payload(run_dir, allow_regime_fallback=True)
    payload = _load_dynamic_regime_payload(run_dir, allow_partial=False)
    if summary is None or payload is None:
        return False
    if "evcharging" in run_dir.name and _is_degenerate_evcharging_plan(summary):
        return False
    regimes = list(payload.get("regimes", []))
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in regimes
        if int(row.get("regime_id", -1)) >= 0
    }
    shared_episode_seeds = [int(seed) for seed in summary.get("shared_eval_episode_seeds", [])]
    shared_plan = _resolve_dynamic_shared_plan(run_dir, summary)
    planned_episode_seeds = _planned_episode_seeds(shared_plan)
    return (
        len(regimes) >= TARGET_REGIMES
        and len(unique_ids) >= TARGET_REGIMES
        and len(shared_plan) >= TARGET_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == TARGET_REGIMES
        and len(list(shared_dir.glob("*.json"))) >= TARGET_REGIMES
        and len(list(shared_dir.glob("*.npz"))) >= TARGET_REGIMES
    )


def _valid_morl_summary(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    shared_dir = summary_path.parent / "shared_regime_returns"
    if not shared_dir.exists():
        return False
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return False
    if "evcharging" in str(summary_path) and _is_degenerate_evcharging_plan(data):
        return False
    regime_fronts = list(data.get("regime_fronts", []))
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in regime_fronts
        if int(row.get("regime_id", -1)) >= 0
    }
    shared_episode_seeds = [int(seed) for seed in data.get("shared_eval_episode_seeds", [])]
    shared_plan = list(data.get("shared_regime_seed_plan", []))
    planned_episode_seeds = _planned_episode_seeds(shared_plan)
    seed_match = (
        shared_episode_seeds == planned_episode_seeds
        or (
            len(shared_episode_seeds) >= TARGET_REGIMES
            and len(planned_episode_seeds) == TARGET_REGIMES
            and shared_episode_seeds[:TARGET_REGIMES] == planned_episode_seeds
        )
    )
    return (
        len(regime_fronts) >= TARGET_REGIMES
        and len(unique_ids) >= TARGET_REGIMES
        and len(shared_plan) >= TARGET_REGIMES
        and seed_match
        and len(list(shared_dir.glob("*.json"))) >= TARGET_REGIMES
        and len(list(shared_dir.glob("*.npz"))) >= TARGET_REGIMES
    )


def _valid_scalar_run(result_path: Path) -> bool:
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text())
    except Exception:
        return False
    if "evcharging" in str(result_path) and _is_degenerate_evcharging_plan(payload):
        return False
    regime_rows = list(payload.get("regime_returns", []))
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in regime_rows
        if int(row.get("regime_id", -1)) >= 0
    }
    shared_episode_seeds = [int(seed) for seed in payload.get("shared_eval_episode_seeds", [])]
    shared_plan = list(payload.get("shared_regime_seed_plan", []))
    planned_episode_seeds = _planned_episode_seeds(shared_plan)
    shared_dir = result_path.parent / "shared_regime_returns"
    return (
        len(regime_rows) >= TARGET_REGIMES
        and len(unique_ids) >= TARGET_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == TARGET_REGIMES
        and len(list(shared_dir.glob("*.json"))) >= TARGET_REGIMES
        and len(list(shared_dir.glob("*.npz"))) >= TARGET_REGIMES
    )


def _pareto_front(points: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or len(arr) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-arr, only_non_dominated_front=True)
    return np.asarray(arr[nd_idx], dtype=np.float64)


def _derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    valid_fronts = [front for front in fronts if front.ndim == 2 and len(front) > 0]
    if not valid_fronts:
        raise ValueError("Cannot derive reference point from empty fronts.")
    stacked = np.concatenate(valid_fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def _compute_eu(front: np.ndarray, num_prefs: int = 64) -> float:
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or len(front) == 0:
        return 0.0
    dim = front.shape[1]
    weights = np.random.default_rng(0).dirichlet(np.ones(dim), size=num_prefs)
    utility = front @ weights.T
    return float(np.mean(np.max(utility, axis=0)))


def _front_metrics(front: np.ndarray, ref_point: np.ndarray) -> tuple[float, float]:
    if front.ndim != 2 or len(front) == 0:
        return 0.0, 0.0
    hv = float(Hypervolume(ref_point=-ref_point).do(-front))
    eu = _compute_eu(front)
    return hv, eu


def _prefer_path(primary: Path, repair: Path, validator) -> Path | None:
    if validator(repair):
        return repair
    if validator(primary):
        return primary
    return None


def _normalize_method(method: str) -> str:
    method = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    if method == "qpensieve":
        return "q_pensieve"
    return method


def _trace_metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric in row:
        try:
            return float(row[metric])
        except Exception:
            return None
    trace_metrics = row.get("trace_metrics", {})
    if isinstance(trace_metrics, dict) and metric in trace_metrics:
        try:
            return float(trace_metrics[metric])
        except Exception:
            return None
    return None


def _trace_metric_list_from_rows(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        shift_count = _trace_metric_value(row, "trace_shift_count")
        if shift_count is None or float(shift_count) <= 0.0:
            continue
        value = _trace_metric_value(row, metric)
        if value is not None and np.isfinite(value):
            values.append(value)
    return values


def _shared_plan_signature(payload: dict[str, Any]) -> str | None:
    plan = list(payload.get("shared_regime_seed_plan", []))
    if not plan:
        return None
    signature_rows = []
    for row in plan:
        if not isinstance(row, dict):
            continue
        signature_rows.append(
            {
                "regime_id": int(row.get("regime_id", -1)),
                "regime": str(row.get("regime", "")),
                "episode_seeds": [int(seed) for seed in row.get("episode_seeds", [])],
            }
        )
    if not signature_rows:
        return None
    return json.dumps(signature_rows, separators=(",", ":"), ensure_ascii=True)


def _shared_plan_match_signature(env_key: str, payload: dict[str, Any]) -> str | None:
    if env_key != "building":
        return _shared_plan_signature(payload)
    plan = list(payload.get("shared_regime_seed_plan", []))
    if not plan:
        return None
    signature_rows = []
    for row in plan:
        if not isinstance(row, dict):
            continue
        regime_meta = row.get("regime_meta", {})
        signature_rows.append(
            {
                "regime_id": int(row.get("regime_id", -1)),
                "regime": str(row.get("regime", "")),
                "regime_meta": {
                    str(key): regime_meta[key]
                    for key in sorted(regime_meta)
                }
                if isinstance(regime_meta, dict)
                else {},
            }
        )
    if not signature_rows:
        return None
    return json.dumps(signature_rows, separators=(",", ":"), ensure_ascii=True)


def _morl_trace_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    per_items = list(payload.get("per_policy", payload.get("per_preference", [])))
    if per_items:
        nested = []
        for item in per_items:
            if not isinstance(item, dict):
                continue
            trace_metrics = item.get("trace_metrics")
            if isinstance(trace_metrics, dict):
                nested.append(trace_metrics)
            else:
                nested.append(item)
        return nested
    if isinstance(payload.get("trace_metrics"), dict):
        return [dict(payload["trace_metrics"])]
    return []


def _extract_solution_points(row: dict[str, Any]) -> np.ndarray:
    for key in ("solution_points", "points", "front_points"):
        value = row.get(key)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 2:
                return arr
    return np.zeros((0, 0), dtype=np.float64)


def _extract_front_points(row: dict[str, Any]) -> np.ndarray:
    for key in ("front_points", "points", "solution_points"):
        value = row.get(key)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            return _pareto_front(value)
    return np.zeros((0, 0), dtype=np.float64)


def _regime_front_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        regime_id = int(row.get("regime_id", -1))
        if regime_id < 0:
            continue
        out.append(
            {
                "regime_id": regime_id,
                "regime": str(row.get("regime", f"regime_{regime_id:03d}")),
                "front": _extract_front_points(row),
                "solution_points": _extract_solution_points(row),
            }
        )
    out.sort(key=lambda item: int(item["regime_id"]))
    return out


def _trace_rows_from_regime_solution_points(
    regime_front_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = sorted(
        (
            row
            for row in regime_front_rows
            if np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64).ndim == 2
            and len(np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64)) > 0
        ),
        key=lambda item: int(item.get("regime_id", -1)),
    )
    if len(rows) < TARGET_REGIMES:
        return []
    solution_blocks = [
        np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64)
        for row in rows
    ]
    solution_count = min(block.shape[0] for block in solution_blocks)
    if solution_count <= 0:
        return []

    trace_rows: list[dict[str, Any]] = []
    regime_ids = np.arange(len(solution_blocks), dtype=np.int64)
    for solution_idx in range(solution_count):
        seq = np.stack([block[solution_idx] for block in solution_blocks], axis=0)
        trace_rows.append(
            compute_trace_shift_metrics(
                {
                    "obj": seq,
                    "regime_id": regime_ids,
                },
                eval_delta_weight=0.5,
                recovery_window=REGIME_TRACE_WINDOW,
                use_regime_id=True,
                min_shift_gap=1,
            )
        )
    return trace_rows


def _recovery_event_latency_samples_from_solution_points(
    regime_front_rows: list[dict[str, Any]],
) -> list[float]:
    rows = sorted(
        (
            row
            for row in regime_front_rows
            if np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64).ndim == 2
            and len(np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64)) > 0
        ),
        key=lambda item: int(item.get("regime_id", -1)),
    )
    if len(rows) < TARGET_REGIMES:
        return []

    solution_blocks = [
        np.asarray(row.get("solution_points", np.zeros((0, 0))), dtype=np.float64)
        for row in rows
    ]
    solution_count = min(block.shape[0] for block in solution_blocks)
    if solution_count <= 0:
        return []

    event_latencies: list[float] = []
    regime_ids = np.arange(len(solution_blocks), dtype=np.int64)
    for solution_idx in range(solution_count):
        seq = np.stack([block[solution_idx] for block in solution_blocks], axis=0)
        trace = {
            "obj": seq,
            "regime_id": regime_ids,
        }
        step_utility, shift_points = resolve_shift_points(
            trace,
            eval_delta_weight=0.5,
            recovery_window=REGIME_TRACE_WINDOW,
            use_regime_id=True,
            min_shift_gap=1,
        )
        if len(shift_points) == 0:
            continue
        window = max(1, min(int(REGIME_TRACE_WINDOW), len(step_utility)))
        for shift_idx in shift_points:
            pre = step_utility[max(0, shift_idx - window) : shift_idx]
            post = step_utility[shift_idx : shift_idx + window]
            if len(pre) == 0 or len(post) == 0:
                continue
            baseline = float(np.mean(pre))
            if len(post) < window:
                rolling = np.asarray([float(np.mean(post))], dtype=np.float64)
            else:
                kernel = np.ones(window, dtype=np.float64) / float(window)
                rolling = np.convolve(post, kernel, mode="valid")
            target = baseline * 0.9
            hit = np.where(rolling >= target)[0]
            event_latencies.append(float(hit[0]) if len(hit) else float(len(post)))
    return event_latencies


def _score_ci_from_latency_ci(mean_score: float, mean_latency: float, latency_ci: float) -> float:
    if not (np.isfinite(mean_score) and np.isfinite(mean_latency) and np.isfinite(latency_ci)):
        return float("nan")
    lower_latency = max(0.0, float(mean_latency) - float(latency_ci))
    upper_latency = max(0.0, float(mean_latency) + float(latency_ci))
    score_from_lower = float(1.0 / (1.0 + lower_latency))
    score_from_upper = float(1.0 / (1.0 + upper_latency))
    return max(abs(float(mean_score) - score_from_lower), abs(float(mean_score) - score_from_upper))


def _dynamic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        run_dir = None
        for run_name in PREFERRED_DYNAMIC_RUNS.get(env_key, [f"sharedprot_main_{env_key}_dynamic_seed0"]):
            primary = PRIMARY_ROOT / run_name
            repair = REPAIR_ROOT / run_name
            run_dir = _prefer_path(primary, repair, _valid_dynamic_result)
            if run_dir is not None:
                break
        if run_dir is None:
            continue
        summary = _load_dynamic_summary_payload(run_dir, allow_regime_fallback=True)
        regime_payload = _load_dynamic_regime_payload(run_dir, allow_partial=False)
        if summary is None or regime_payload is None:
            continue
        regime_front_rows = _regime_front_rows(list(regime_payload.get("regimes", [])))
        per_trace_rows = (
            _trace_rows_from_regime_solution_points(regime_front_rows)
            or _resolve_dynamic_trace_rows(summary)
        )
        rows.append(
            {
                "env_key": env_key,
                "method": "dynamic",
                "source": str(run_dir),
                "plan_signature": _shared_plan_match_signature(env_key, summary),
                "regime_front_rows": regime_front_rows,
                "trace_samples": {
                    "trace_shift_regret": _trace_metric_list_from_rows(per_trace_rows, "trace_shift_regret"),
                    "trace_recovery_latency": _trace_metric_list_from_rows(per_trace_rows, "trace_recovery_latency"),
                    "trace_recovery_score": _trace_metric_list_from_rows(per_trace_rows, "trace_recovery_score"),
                },
            }
        )
    return rows


def _find_morl_candidates(method: str, env_key: str) -> list[Path]:
    method_names = [method]
    if method == "q_pensieve":
        method_names.append("qpensieve")
    preferred_suites = PREFERRED_MORL_SUITES.get(method, [])
    paths: list[Path] = []
    for root in [REPAIR_ROOT / "morl_dynamic_baselines", PRIMARY_ROOT / "morl_dynamic_baselines"]:
        if not root.exists():
            continue
        for suite_name in preferred_suites:
            for method_name in method_names:
                candidate = root / suite_name / method_name / env_key / "seed0" / "summary.json"
                if candidate.exists():
                    paths.append(candidate)
    return paths


def _morl_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env_key in ENV_ORDER:
        for method in ["capql", "pgmorl", "q_pensieve", "morlca"]:
            summary_path = None
            for candidate in _find_morl_candidates(method, env_key):
                if _valid_morl_summary(candidate):
                    summary_path = candidate
                    break
            if summary_path is None:
                continue
            payload = json.loads(summary_path.read_text())
            regime_front_rows = _regime_front_rows(list(payload.get("regime_fronts", [])))
            trace_rows = (
                _trace_rows_from_regime_solution_points(regime_front_rows)
                or _morl_trace_rows(payload)
            )
            rows.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "source": str(summary_path),
                    "plan_signature": _shared_plan_match_signature(env_key, payload),
                    "regime_front_rows": regime_front_rows,
                    "trace_samples": {
                        "trace_shift_regret": _trace_metric_list_from_rows(trace_rows, "trace_shift_regret"),
                        "trace_recovery_latency": _trace_metric_list_from_rows(trace_rows, "trace_recovery_latency"),
                        "trace_recovery_score": _trace_metric_list_from_rows(trace_rows, "trace_recovery_score"),
                    },
                }
            )
    return rows


def _scalar_rows(methods_map: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = PRIMARY_ROOT / "scalarized_baselines"
    if not root.exists():
        return rows
    if methods_map is None:
        methods_map = REPORT_METHODS
    for env_key in ENV_ORDER:
        methods = [
            method
            for method in methods_map.get(env_key, [])
            if method not in {"dynamic", "capql", "pgmorl", "q_pensieve"}
        ]
        for method in methods:
            alg_dir = root / env_key / method
            if not alg_dir.exists():
                continue
            run_payloads = []
            for result_path in sorted(alg_dir.glob("*/result.json")):
                if not _valid_scalar_run(result_path):
                    continue
                run_payloads.append(json.loads(result_path.read_text()))
            if not run_payloads:
                continue
            trace_rows = [payload.get("trace_metrics", {}) for payload in run_payloads]
            grouped_regimes: dict[int, dict[str, Any]] = {}
            for payload in run_payloads:
                for regime_row in payload.get("regime_returns", []):
                    regime_id = int(regime_row.get("regime_id", -1))
                    if regime_id < 0:
                        continue
                    bucket = grouped_regimes.setdefault(
                        regime_id,
                        {
                            "regime": regime_row.get("regime"),
                            "regime_id": regime_id,
                            "regime_meta": dict(regime_row.get("regime_meta", {})),
                            "points": [],
                        },
                    )
                    bucket["points"].append(list(regime_row.get("objs", [])))
            regime_fronts = []
            regime_metric_rows = []
            for regime_id in sorted(grouped_regimes):
                bucket = grouped_regimes[regime_id]
                front = _pareto_front(bucket["points"])
                regime_fronts.append(front)
                regime_metric_rows.append(
                    {
                        "regime_id": regime_id,
                        "regime": str(bucket.get("regime", f"regime_{regime_id:03d}")),
                        "front": front,
                    }
                )
            ref_point = _derive_reference_point(regime_fronts) if regime_fronts else np.zeros(0, dtype=np.float64)
            scored_regimes = []
            for item in regime_metric_rows:
                hv, eu = _front_metrics(np.asarray(item["front"], dtype=np.float64), ref_point)
                scored_regimes.append(
                    {
                        "regime_id": int(item["regime_id"]),
                        "regime": str(item["regime"]),
                        "hv": float(hv),
                        "eu": float(eu),
                    }
                )
            rows.append(
                {
                    "env_key": env_key,
                    "method": method,
                    "source": str(alg_dir),
                    "plan_signature": _shared_plan_match_signature(env_key, run_payloads[0]) if run_payloads else None,
                    "regime_front_rows": [
                        {
                            "regime_id": int(item["regime_id"]),
                            "regime": str(item["regime"]),
                            "front": np.asarray(item["front"], dtype=np.float64),
                        }
                        for item in regime_metric_rows
                    ],
                    "trace_samples": {
                        "trace_shift_regret": _trace_metric_list_from_rows(trace_rows, "trace_shift_regret"),
                        "trace_recovery_latency": _trace_metric_list_from_rows(trace_rows, "trace_recovery_latency"),
                        "trace_recovery_score": _trace_metric_list_from_rows(trace_rows, "trace_recovery_score"),
                    },
                }
            )
    return rows


def _align_rows_by_shared_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(str(row["env_key"]), []).append(row)

    kept: list[dict[str, Any]] = []
    for env_key, env_rows in by_env.items():
        signatures = [row.get("plan_signature") for row in env_rows if row.get("plan_signature")]
        if not signatures:
            kept.extend(env_rows)
            continue
        dynamic_sig = next(
            (
                row.get("plan_signature")
                for row in env_rows
                if str(row.get("method")) == "dynamic" and row.get("plan_signature")
            ),
            None,
        )
        if dynamic_sig is not None:
            target_sig = dynamic_sig
        else:
            counts: dict[str, int] = {}
            for sig in signatures:
                counts[str(sig)] = counts.get(str(sig), 0) + 1
            target_sig = max(counts.items(), key=lambda item: item[1])[0]
        kept.extend(
            row
            for row in env_rows
            if row.get("plan_signature") in {None, target_sig}
        )
    return kept


def _score_regime_front_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(str(row["env_key"]), []).append(row)

    for env_rows in by_env.values():
        regime_front_pool: dict[int, list[np.ndarray]] = {}
        for row in env_rows:
            for regime_row in row.get("regime_front_rows", []):
                front = np.asarray(regime_row.get("front", np.zeros((0, 0))), dtype=np.float64)
                if front.ndim == 2 and len(front) > 0:
                    regime_front_pool.setdefault(int(regime_row["regime_id"]), []).append(front)

        regime_ref_points: dict[int, np.ndarray] = {}
        for regime_id, fronts in regime_front_pool.items():
            if fronts:
                regime_ref_points[regime_id] = _derive_reference_point(fronts)

        for row in env_rows:
            regime_metric_rows: list[dict[str, Any]] = []
            for regime_row in row.get("regime_front_rows", []):
                regime_id = int(regime_row["regime_id"])
                front = np.asarray(regime_row.get("front", np.zeros((0, 0))), dtype=np.float64)
                ref_point = regime_ref_points.get(regime_id)
                if ref_point is None:
                    hv, eu = 0.0, 0.0
                else:
                    hv, eu = _front_metrics(front, ref_point)
                regime_metric_rows.append(
                    {
                        "regime_id": regime_id,
                        "regime": str(regime_row["regime"]),
                        "hv": float(hv),
                        "eu": float(eu),
                    }
                )
            regime_metric_rows.sort(key=lambda item: int(item["regime_id"]))
            row["regime_metric_rows"] = regime_metric_rows
            row["hv_samples"] = [float(item["hv"]) for item in regime_metric_rows]
            row["eu_samples"] = [float(item["eu"]) for item in regime_metric_rows]
    return rows


def _attach_adapt_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_env: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_env.setdefault(str(row["env_key"]), []).append(row)
    for env_rows in by_env.values():
        oracle_by_regime: dict[int, dict[str, float]] = {}
        for row in env_rows:
            for metric_row in row.get("regime_metric_rows", []):
                regime_id = int(metric_row["regime_id"])
                bucket = oracle_by_regime.setdefault(
                    regime_id,
                    {"hv": -float("inf"), "eu": -float("inf")},
                )
                hv_value = float(metric_row["hv"])
                eu_value = float(metric_row["eu"])
                if np.isfinite(hv_value):
                    bucket["hv"] = max(float(bucket["hv"]), hv_value)
                if np.isfinite(eu_value):
                    bucket["eu"] = max(float(bucket["eu"]), eu_value)
        for row in env_rows:
            adapt_samples: list[float] = []
            for metric_row in row.get("regime_metric_rows", []):
                regime_id = int(metric_row["regime_id"])
                oracle = oracle_by_regime.get(regime_id, {})
                method_hv = float(metric_row["hv"])
                method_eu = float(metric_row["eu"])
                best_hv = float(oracle.get("hv", float("nan")))
                best_eu = float(oracle.get("eu", float("nan")))
                hv_gap = 1.0
                eu_gap = 1.0
                if np.isfinite(best_hv) and np.isfinite(method_hv):
                    hv_gap = max(0.0, best_hv - method_hv) / max(abs(best_hv), 1e-8)
                if np.isfinite(best_eu) and np.isfinite(method_eu):
                    eu_gap = max(0.0, best_eu - method_eu) / max(abs(best_eu), 1e-8)
                adapt_samples.append(float(np.clip(1.0 - 0.5 * (hv_gap + eu_gap), 0.0, 1.0)))
            row["adapt_score_samples"] = adapt_samples
    return rows


def build_summary(methods_map: dict[str, list[str]] | None = None) -> pd.DataFrame:
    if methods_map is None:
        methods_map = REPORT_METHODS
    rows = _dynamic_rows() + _morl_rows() + _scalar_rows(methods_map)
    rows = [
        row
        for row in rows
        if row["method"] in methods_map.get(str(row["env_key"]), [])
    ]
    rows = _align_rows_by_shared_plan(rows)
    rows = _score_regime_front_rows(rows)
    rows = _attach_adapt_score(rows)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {
            "env_key": row["env_key"],
            "method": row["method"],
            "source": row["source"],
        }
        samples_by_metric = {
            "HV": row["hv_samples"],
            "EU": row["eu_samples"],
            "adapt_score": row.get("adapt_score_samples", []),
            "trace_shift_regret": row["trace_samples"].get("trace_shift_regret", []),
            "trace_recovery_latency": row["trace_samples"].get("trace_recovery_latency", []),
            "trace_recovery_score": row["trace_samples"].get("trace_recovery_score", []),
        }
        for metric, values in samples_by_metric.items():
            mean, ci, n = _ci_from_samples(list(values))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci"] = ci
            record[f"{metric}_n"] = n
        recovery_event_latencies = _recovery_event_latency_samples_from_solution_points(
            row.get("regime_front_rows", [])
        )
        if recovery_event_latencies:
            latency_mean, latency_ci, latency_n = _ci_from_samples(recovery_event_latencies)
            record["trace_recovery_latency_mean"] = latency_mean
            record["trace_recovery_latency_ci"] = latency_ci
            record["trace_recovery_latency_n"] = latency_n
            score_mean = record.get("trace_recovery_score_mean", float("nan"))
            if np.isfinite(score_mean):
                record["trace_recovery_score_ci"] = _score_ci_from_latency_ci(
                    float(score_mean),
                    latency_mean,
                    latency_ci,
                )
                record["trace_recovery_score_n"] = latency_n
        out_rows.append(record)
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df["method"] = df["method"].map(_normalize_method)
    return df


def _best_mask(df: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(df[f"{metric}_mean"], errors="coerce")
    if HIGHER_BETTER[metric]:
        target = values.max()
    else:
        target = values.min()
    return pd.Series(np.isclose(values, target, rtol=1e-10, atol=1e-12), index=df.index)


def _format_cell(df: pd.DataFrame, idx: int, metric: str) -> str:
    text = _format_ci(df.loc[idx, f"{metric}_mean"], df.loc[idx, f"{metric}_ci"])
    if bool(_best_mask(df, metric).iloc[idx]):
        return f"**{text}**"
    return text


def write_report(df: pd.DataFrame, out_md: Path, scalar_df: pd.DataFrame | None = None) -> None:
    lines = [
        "# Unified 20-Regime Shared-Protocol Results",
        "",
        "For each environment and regime, collect the displayed methods' Pareto fronts and derive a shared reference point from their union. Recompute each method's `HV / EU` on that regime, then report the mean and t-based 95% confidence interval over 20 shared regimes.",
        "",
        "The main table covers the dynamic MORL methods `OW-CMORL`, `CAPQL`, `PGMORL`, `Q-Pensieve`, and `MORL-CA`. Scalarized RL appears separately in the supplementary comparison.",
        "",
        "The supplementary scalarized-RL comparison uses a fixed family-coverage rule across all four environments: `A2C` for actor-critic, `ACER` for replay-based off-policy actor-critic, and `TRPO` for trust-region policy gradients.",
        "",
        "Dynamic adaptation metrics are defined as follows:",
        "",
        "- `adapt_score = 1 - (gap_hv + gap_eu) / 2`, where `gap_hv = \\max(0, HV^* - HV) / \\max(|HV^*|, \\varepsilon)` and `gap_eu = \\max(0, EU^* - EU) / \\max(|EU^*|, \\varepsilon)`. Here `HV^*` and `EU^*` are the best values among displayed methods for the same environment and regime. Higher scores mean closer agreement with both benchmarks.",
        "",
        "- `trace_shift_regret = \\mathbb{E}[\\max(\\text{baseline} - \\text{post}, 0)]`, where `baseline` is mean utility before a shift and `post` is observed utility afterward. Lower regret means less utility lost during adaptation.",
        "",
        "- `trace_recovery_latency = \\mathbb{E}[\\tau]`, where `\\tau = \\min\\{ t \\mid \\overline{u}_{post}(t) \\ge 0.9 \\times \\text{baseline} \\}`. It measures the steps needed after a shift to recover `90\\%` of pre-shift utility; lower is better.",
        "",
        "- `trace_recovery_score = 1 / (1 + trace_recovery_latency)` maps recovery speed to `[0,1]`; higher is better.",
        "",
        "The `trace_*` metrics use t-based 95% confidence intervals only for samples with `trace_shift_count > 0`. When shared-regime `solution_points` exist, metrics are rebuilt from per-solution traces. The `trace_recovery_latency` interval uses regime-shift event samples of `\\tau`; the `trace_recovery_score` interval follows by the monotone transform `1 / (1 + \\tau)`. Otherwise, the calculation falls back to OW-CMORL `per_sample_trace_metrics`, MORL baseline `per_policy/per_preference` traces, or per-weight scalarized runs. Missing valid shift samples are shown as `--`.",
        "",
        "`adapt_score` summarizes solution quality across regimes; `trace_shift_regret` captures immediate loss after a shift; `trace_recovery_latency` measures recovery time; and `trace_recovery_score` maps that time to a bounded score.",
        "",
    ]
    for env_key in ENV_ORDER:
        sub = df[df["env_key"] == env_key].copy()
        if sub.empty:
            continue
        allowed = MAIN_REPORT_METHODS.get(env_key, [])
        sub = sub[sub["method"].isin(allowed)].reset_index(drop=True)
        if sub.empty:
            continue
        lines.append(f"## {env_key}")
        lines.append("")
        lines.append("| Method | HV | EU | adapt_score | trace_shift_regret | trace_recovery_latency | trace_recovery_score |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for idx in range(len(sub)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_name(str(sub.loc[idx, "method"])),
                        _format_cell(sub, idx, "HV"),
                        _format_cell(sub, idx, "EU"),
                        _format_cell(sub, idx, "adapt_score"),
                        _format_cell(sub, idx, "trace_shift_regret"),
                        _format_cell(sub, idx, "trace_recovery_latency"),
                        _format_cell(sub, idx, "trace_recovery_score"),
                    ]
                )
                + " |"
            )
        if env_key == "evcharging":
            lines.append("")
            lines.append(
                "For EV charging, `trace_recovery_latency` and `trace_recovery_score` are recomputed from shared-regime `solution_points` when available. The latency interval uses regime-shift event samples of `\\tau`; the score interval follows through `1 / (1 + \\tau)`. Event-level variation keeps these intervals informative even when several trace means coincide."
            )
        lines.append("")
    if scalar_df is not None and not scalar_df.empty:
        lines.append("## Representative Scalarized RL (Supplementary)")
        lines.append("")
        lines.append(
            "Scalarized RL is shown as a supplementary comparison using fixed family coverage: `A2C` for actor-critic, `ACER` for replay-based off-policy actor-critic, and `TRPO` for trust-region policy gradients."
        )
        lines.append("")
        for env_key in ENV_ORDER:
            sub = scalar_df[scalar_df["env_key"] == env_key].copy()
            if sub.empty:
                continue
            allowed = [method for method in SCALAR_FAMILY_METHODS if method in set(sub["method"])]
            sub = sub[sub["method"].isin(allowed)].reset_index(drop=True)
            if sub.empty:
                continue
            lines.append(f"### {env_key}")
            lines.append("")
            lines.append("| Method | HV | EU | adapt_score | trace_shift_regret | trace_recovery_latency | trace_recovery_score |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for idx in range(len(sub)):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _format_name(str(sub.loc[idx, "method"])),
                            _format_cell(sub, idx, "HV"),
                            _format_cell(sub, idx, "EU"),
                            _format_cell(sub, idx, "adapt_score"),
                            _format_cell(sub, idx, "trace_shift_regret"),
                            _format_cell(sub, idx, "trace_recovery_latency"),
                            _format_cell(sub, idx, "trace_recovery_score"),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = build_summary(MAIN_REPORT_METHODS)
    scalar_context_df = build_summary(SCALAR_SUPPLEMENT_METHODS)
    scalar_df = scalar_context_df[scalar_context_df["method"].isin(SCALAR_FAMILY_METHODS)].copy()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_scalar_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    scalar_df.to_csv(args.out_scalar_csv, index=False)
    write_report(df, args.out_md, scalar_df)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_scalar_csv}")


if __name__ == "__main__":
    main()
