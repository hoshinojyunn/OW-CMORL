from __future__ import annotations

import csv
import os
from typing import Callable

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .chlor_alkali_env import (
    DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    chlor_alkali_regime_catalog,
)
from .chlor_shared_protocol import (
    canonical_chlor_regime_rows,
    canonical_chlor_regime_seed_plan,
    chlor_regime_eval_env_kwargs,
)
from .dynamic_building import BUILDING_REGIMES
from .envs import make_env
from .fine_regimes import (
    DEFAULT_FINE_REGIME_CLUSTERS,
    DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    DEFAULT_FINE_REGIME_EPISODES,
    assign_episode_returns_to_catalog,
    assign_episode_returns_to_plan,
    catalog_cache_signature,
    coerce_regime_seed_plan,
    flatten_regime_seed_plan,
    load_or_build_catalog,
    load_or_build_regime_seed_plan,
    selected_episode_seeds_from_plan,
)
from .shared_eval import build_shared_episode_seeds, env_key_from_name
from ..ood_protocol import load_env_config_json
from .sustaingym_wrappers import (
    DEFAULT_BUILDING_WEATHERS,
    DEFAULT_COGEN_RENEWABLES,
    DEFAULT_EV_PERIODS,
)
from .utils import compute_eu, compute_sparsity, generate_w_batch_test
from pymoo.indicators.hv import Hypervolume


def compute_front_metrics(front: np.ndarray, ref_point: np.ndarray, eval_delta_weight: float):
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], eval_delta_weight)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return float(hv), float(eu), float(sp)


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def _step_utility(objs: np.ndarray, eval_delta_weight: float) -> np.ndarray:
    weights = generate_w_batch_test(objs.shape[1], eval_delta_weight)
    return np.mean(objs @ weights.T, axis=1)


def _oracle_shift_points(
    trace: dict[str, np.ndarray],
    expected_len: int,
    min_shift_gap: int,
) -> np.ndarray:
    regime_ids = np.asarray(trace.get("regime_id", []))
    episodes = np.asarray(trace.get("episode", []), dtype=np.int64)

    if len(episodes) == expected_len and len(np.unique(episodes)) > 1:
        shift_points = np.where(episodes[1:] != episodes[:-1])[0] + 1
        if len(regime_ids) == expected_len:
            shift_points = shift_points[regime_ids[shift_points] != regime_ids[shift_points - 1]]
    elif len(regime_ids) == expected_len:
        shift_points = np.where(regime_ids[1:] != regime_ids[:-1])[0] + 1
    else:
        return np.array([], dtype=np.int64)

    if len(shift_points) == 0:
        return np.array([], dtype=np.int64)

    min_gap = max(1, int(min_shift_gap))
    selected = [int(shift_points[0])]
    for shift_idx in shift_points[1:]:
        if int(shift_idx) - selected[-1] >= min_gap:
            selected.append(int(shift_idx))
    return np.asarray(selected, dtype=np.int64)


def _signal_shift_points(
    objs: np.ndarray,
    step_utility: np.ndarray,
    recovery_window: int,
    shift_threshold: float,
    min_shift_gap: int,
) -> np.ndarray:
    if len(step_utility) < max(3, 2 * recovery_window):
        return np.array([], dtype=np.int64)
    utility_scale = float(np.std(step_utility)) + 1e-6
    obj_scale = float(np.mean(np.std(objs, axis=0))) + 1e-6
    candidates = []
    for shift_idx in range(recovery_window, len(step_utility) - recovery_window):
        pre_u = step_utility[shift_idx - recovery_window : shift_idx]
        post_u = step_utility[shift_idx : shift_idx + recovery_window]
        pre_obj = objs[shift_idx - recovery_window : shift_idx]
        post_obj = objs[shift_idx : shift_idx + recovery_window]
        utility_delta = abs(float(np.mean(post_u) - np.mean(pre_u))) / utility_scale
        obj_delta = float(np.linalg.norm(post_obj.mean(axis=0) - pre_obj.mean(axis=0))) / obj_scale
        score = 0.7 * utility_delta + 0.3 * obj_delta
        if score >= shift_threshold:
            candidates.append((score, shift_idx))
    if not candidates:
        return np.array([], dtype=np.int64)
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[int] = []
    for _score, shift_idx in candidates:
        if all(abs(shift_idx - prev_idx) >= min_shift_gap for prev_idx in selected):
            selected.append(int(shift_idx))
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


def resolve_shift_points(
    trace: dict[str, np.ndarray],
    eval_delta_weight: float,
    recovery_window: int = 12,
    use_regime_id: bool = True,
    shift_threshold: float = 0.6,
    min_shift_gap: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    objs = np.asarray(trace.get("obj", []), dtype=np.float64)
    if len(objs) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    step_utility = _step_utility(objs, eval_delta_weight)
    shift_points = (
        _oracle_shift_points(
            trace,
            len(step_utility),
            min_shift_gap=min_shift_gap,
        )
        if use_regime_id
        else np.array([], dtype=np.int64)
    )
    if len(shift_points) == 0:
        window = max(1, min(int(recovery_window), max(1, len(step_utility) // 3)))
        min_gap = max(1, min(int(min_shift_gap), max(1, len(step_utility) // 2)))
        shift_points = _signal_shift_points(
            objs,
            step_utility,
            recovery_window=window,
            shift_threshold=shift_threshold,
            min_shift_gap=min_gap,
    )
    return step_utility, shift_points


def _trace_step_axis(trace: dict[str, np.ndarray], expected_len: int) -> tuple[np.ndarray, bool]:
    steps = np.asarray(trace.get("step", []), dtype=np.int64)
    if len(steps) != expected_len:
        return np.arange(expected_len, dtype=np.int64), False
    if len(steps) > 1 and np.any(steps[1:] < steps[:-1]):
        return np.arange(expected_len, dtype=np.int64), False
    return steps, True


def _recovery_target(baseline: float, recovery_target_ratio: float, eps: float = 1e-8) -> float:
    # Use an absolute-gap interpretation so signed utilities still recover to
    # 90% of the pre-shift level instead of requiring an improvement beyond the
    # baseline when the utility is negative.
    return float(baseline - (1.0 - recovery_target_ratio) * max(abs(baseline), eps))


def _recovery_latency_from_segment(
    post_utility: np.ndarray,
    post_steps: np.ndarray,
    target: float,
    *,
    use_online_steps: bool,
) -> float:
    if len(post_utility) == 0:
        return 0.0

    running_mean = np.cumsum(post_utility, dtype=np.float64) / np.arange(
        1,
        len(post_utility) + 1,
        dtype=np.float64,
    )
    hit = np.where(running_mean >= target)[0]
    hit_idx = int(hit[0]) if len(hit) else int(len(post_utility) - 1)

    if not use_online_steps:
        return float(hit_idx + 1)

    start_step = int(post_steps[0])
    end_step = int(post_steps[hit_idx])
    return float(max(1, end_step - start_step + 1))


def compute_trace_shift_metrics(
    trace: dict[str, np.ndarray],
    eval_delta_weight: float,
    recovery_window: int = 12,
    use_regime_id: bool = True,
    shift_threshold: float = 0.6,
    min_shift_gap: int = 6,
    recovery_target_ratio: float = 0.9,
):
    objs = np.asarray(trace.get("obj", []), dtype=np.float64)
    if len(objs) == 0:
        return {
            "trace_utility_mean": 0.0,
            "trace_utility_std": 0.0,
            "trace_shift_count": 0,
            "trace_shift_regret": 0.0,
            "trace_recovery_latency": 0.0,
            "trace_recovery_score": 1.0,
            "trace_pre_post_gap": 0.0,
        }

    step_utility, shift_points = resolve_shift_points(
        trace,
        eval_delta_weight,
        recovery_window=recovery_window,
        use_regime_id=use_regime_id,
        shift_threshold=shift_threshold,
        min_shift_gap=min_shift_gap,
    )
    if len(shift_points) == 0:
        return {
            "trace_utility_mean": float(step_utility.mean()),
            "trace_utility_std": float(step_utility.std()),
            "trace_shift_count": 0,
            "trace_shift_regret": 0.0,
            "trace_recovery_latency": 0.0,
            "trace_recovery_score": 1.0,
            "trace_pre_post_gap": 0.0,
        }

    regret_values = []
    latency_values = []
    gap_values = []
    window = max(1, min(int(recovery_window), len(step_utility)))
    step_axis, has_online_steps = _trace_step_axis(trace, len(step_utility))

    for shift_pos, shift_idx in enumerate(shift_points):
        pre = step_utility[max(0, shift_idx - window) : shift_idx]
        next_shift_idx = int(shift_points[shift_pos + 1]) if shift_pos + 1 < len(shift_points) else int(len(step_utility))
        post_segment = step_utility[shift_idx:next_shift_idx]
        post_segment_steps = step_axis[shift_idx:next_shift_idx]
        if len(pre) == 0 or len(post_segment) == 0:
            continue
        baseline = float(np.mean(pre))
        post = post_segment[:window]
        post_mean = float(np.mean(post))
        gap_values.append(post_mean - baseline)
        regret_values.append(float(np.mean(np.maximum(baseline - post, 0.0))))
        target = _recovery_target(baseline, recovery_target_ratio)
        latency_values.append(
            _recovery_latency_from_segment(
                post_segment,
                post_segment_steps,
                target,
                use_online_steps=has_online_steps,
            )
        )

    if not regret_values:
        return {
            "trace_utility_mean": float(step_utility.mean()),
            "trace_utility_std": float(step_utility.std()),
            "trace_shift_count": int(len(shift_points)),
            "trace_shift_regret": 0.0,
            "trace_recovery_latency": 0.0,
            "trace_recovery_score": 1.0,
            "trace_pre_post_gap": 0.0,
        }

    mean_latency = float(np.mean(latency_values))
    return {
        "trace_utility_mean": float(step_utility.mean()),
        "trace_utility_std": float(step_utility.std()),
        "trace_shift_count": int(len(shift_points)),
        "trace_shift_regret": float(np.mean(regret_values)),
        "trace_recovery_latency": mean_latency,
        "trace_recovery_score": float(1.0 / (1.0 + mean_latency)),
        "trace_pre_post_gap": float(np.mean(gap_values)),
    }


def compute_trace_objective_gaps(
    trace: dict[str, np.ndarray],
    recovery_window: int = 12,
    eval_delta_weight: float = 0.5,
    use_regime_id: bool = True,
    shift_threshold: float = 0.6,
    min_shift_gap: int = 6,
) -> np.ndarray:
    objs = np.asarray(trace.get("obj", []), dtype=np.float64)
    if objs.ndim != 2 or len(objs) == 0:
        return np.zeros(0, dtype=np.float64)

    _, shift_points = resolve_shift_points(
        trace,
        eval_delta_weight,
        recovery_window=recovery_window,
        use_regime_id=use_regime_id,
        shift_threshold=shift_threshold,
        min_shift_gap=min_shift_gap,
    )
    if len(shift_points) == 0:
        return np.zeros(objs.shape[1], dtype=np.float64)

    window = max(1, min(int(recovery_window), len(objs)))
    gap_rows = []
    for shift_idx in shift_points:
        pre = objs[max(0, shift_idx - window) : shift_idx]
        post = objs[shift_idx : shift_idx + window]
        if len(pre) == 0 or len(post) == 0:
            continue
        gap_rows.append(np.maximum(pre.mean(axis=0) - post.mean(axis=0), 0.0))

    if not gap_rows:
        return np.zeros(objs.shape[1], dtype=np.float64)
    return np.mean(np.stack(gap_rows, axis=0), axis=0)


def _env_key_from_name(env_name: str) -> str:
    if env_name.startswith("building_") or env_name == "sustaingym_building_dynamic":
        return "building"
    if env_name == "evcharging_dynamic":
        return "evcharging"
    if env_name == "cogen_dynamic":
        return "cogen"
    if env_name == "chlor_alkali_dynamic":
        return "chlor_alkali"
    return env_name


def _dynamic_env_kwargs(args) -> dict[str, object]:
    if args.env_name in {"building_3d_dynamic", "building_9d_dynamic"}:
        env_kwargs = {
            "regimes": tuple(BUILDING_REGIMES),
            "schedule": getattr(args, "regime_schedule", "cyclic"),
            "episodes_per_regime": int(getattr(args, "episodes_per_regime", 4)),
        }
    elif args.env_name == "sustaingym_building_dynamic":
        env_kwargs = {
            "weathers": tuple(getattr(args, "sustaingym_building_weathers", ()) or DEFAULT_BUILDING_WEATHERS),
            "schedule": getattr(args, "regime_schedule", "cyclic"),
            "episodes_per_regime": int(getattr(args, "episodes_per_regime", 4)),
        }
    elif args.env_name == "evcharging_dynamic":
        env_kwargs = {
            "site": getattr(args, "ev_site", "caltech"),
            "periods": tuple(getattr(args, "ev_periods", ()) or DEFAULT_EV_PERIODS),
            "schedule": getattr(args, "regime_schedule", "cyclic"),
            "episodes_per_regime": int(getattr(args, "episodes_per_regime", 4)),
            "moer_forecast_steps": int(getattr(args, "ev_moer_forecast_steps", 36)),
            "project_action_in_env": not bool(getattr(args, "ev_disable_projection", False)),
        }
    elif args.env_name == "cogen_dynamic":
        env_kwargs = {
            "renewables": tuple(getattr(args, "cogen_renewables", ()) or DEFAULT_COGEN_RENEWABLES),
            "schedule": getattr(args, "regime_schedule", "cyclic"),
            "episodes_per_regime": int(getattr(args, "episodes_per_regime", 4)),
            "forecast_horizon": int(getattr(args, "cogen_forecast_horizon", 3)),
            "forecast_noise_std": float(getattr(args, "cogen_forecast_noise_std", 0.0)),
        }
    elif args.env_name == "chlor_alkali_dynamic":
        env_kwargs = {
            "dynamic_price": True,
            "aging_dynamics": True,
            "episode_length": int(getattr(args, "chlor_episode_length", 288)),
            "schedule": "random",
            "num_regime_clusters": int(
                getattr(args, "chlor_regime_clusters", DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS)
            ),
        }
    else:
        env_kwargs = {}
    if getattr(args, "env_config_json", ""):
        env_kwargs.update(load_env_config_json(getattr(args, "env_config_json")))
    return env_kwargs


def dynamic_regime_catalog(args):
    env_key = _env_key_from_name(args.env_name)
    env_kwargs = _dynamic_env_kwargs(args)

    def env_factory(factory_seed: int):
        thunk = make_env(
            args.env_name,
            seed=int(getattr(args, "seed", 0)) + int(factory_seed),
            rank=0,
            log_dir=None,
            allow_early_resets=True,
            env_kwargs=env_kwargs,
        )
        return thunk()

    n_clusters = int(
        env_kwargs.get(
            "num_regime_clusters",
            getattr(args, "fine_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS),
        )
    )
    return load_or_build_catalog(
        env_key=env_key,
        env_factory=env_factory,
        n_clusters=n_clusters,
        episodes=int(getattr(args, "fine_regime_catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
        context_steps=int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
        seed=0,
        cache_signature=catalog_cache_signature(
            {
                "env_name": args.env_name,
                "env_kwargs": env_kwargs,
                "n_clusters": int(n_clusters),
                "episodes": int(getattr(args, "fine_regime_catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
                "context_steps": int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
            }
        ),
    )


def shared_regime_seed_plan(args) -> tuple[list[dict[str, object]], list[int]]:
    override_path = getattr(args, "shared_plan_json", None)
    if override_path:
        plan = coerce_regime_seed_plan(override_path)
        return plan, selected_episode_seeds_from_plan(plan)
    if args.env_name == "chlor_alkali_dynamic":
        return canonical_chlor_regime_seed_plan(
            env_name=args.env_name,
            base_seed=int(getattr(args, "seed", 0)),
            shared_regime_eval_episodes=max(
                int(getattr(args, "shared_regime_eval_episodes", 0)),
                int(getattr(args, "trace_eval_samples", 0)),
                int(getattr(args, "eval_num", 0)),
                1,
            ),
            seed_offset=int(getattr(args, "shared_regime_seed_offset", 0)),
            num_regime_clusters=int(
                getattr(args, "chlor_regime_clusters", DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS)
            ),
        )

    catalog = dynamic_regime_catalog(args)
    target_count = int(
        getattr(args, "shared_regime_eval_episodes", 0)
        or len(catalog.labels)
        or getattr(args, "fine_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)
    )
    target_regime_ids = sorted(int(idx) for idx in catalog.labels)[: min(target_count, len(catalog.labels))]
    search_start_seed = build_shared_episode_seeds(
        args.env_name,
        base_seed=int(getattr(args, "seed", 0)),
        num_episodes=1,
        seed_offset=int(getattr(args, "shared_regime_seed_offset", 0)),
    )[0]
    env_kwargs = _dynamic_env_kwargs(args)

    def env_factory(_factory_seed: int):
        thunk = make_env(
            args.env_name,
            seed=int(getattr(args, "seed", 0)),
            rank=0,
            log_dir=None,
            allow_early_resets=True,
            env_kwargs=env_kwargs,
        )
        return thunk()

    plan_cache_signature = catalog_cache_signature(
        {
            "env_name": args.env_name,
            "env_kwargs": env_kwargs,
            "target_regime_ids": target_regime_ids,
            "shared_regime_eval_episodes": int(getattr(args, "shared_regime_eval_episodes", 0)),
            "shared_regime_seed_offset": int(getattr(args, "shared_regime_seed_offset", 0)),
            "search_start_seed": int(search_start_seed),
        }
    )
    plan = load_or_build_regime_seed_plan(
        env_key=env_key_from_name(args.env_name),
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(search_start_seed),
        seeds_per_regime=1,
        context_steps=int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
        max_search_episodes=max(256, 64 * max(1, len(target_regime_ids))),
        target_regime_ids=target_regime_ids,
        cache_signature=plan_cache_signature,
    )
    return plan, selected_episode_seeds_from_plan(plan)


def _regime_rows_from_grouped(
    grouped: dict[str, dict[str, object]],
    *,
    ref_point: np.ndarray,
    eval_delta_weight: float,
) -> list[dict[str, object]]:
    metrics = []
    for regime_name in sorted(grouped):
        bucket = grouped[regime_name]
        solutions = np.asarray(bucket["solutions"], dtype=np.float64)
        if solutions.ndim != 2 or len(solutions) == 0:
            continue
        front = pareto_front(solutions)
        if len(front) == 0:
            continue
        hv, eu, sp = compute_front_metrics(front, ref_point, eval_delta_weight)
        metrics.append(
            {
                "regime": regime_name,
                "regime_id": int(bucket["regime_id"]),
                "regime_meta": dict(bucket["regime_meta"]),
                "hv": hv,
                "eu": eu,
                "sp": sp,
                "points": int(len(front)),
                "num_solutions": int(len(solutions)),
                "front_points": front.tolist(),
                "solution_points": solutions.tolist(),
            }
        )
    return metrics


def regime_specs(args):
    if args.env_name in {"building_3d_dynamic", "building_9d_dynamic"}:
        return [
            {
                "name": regime.name,
                "env_name": args.env_name.replace("_dynamic", "_static"),
                "env_kwargs": {"regimes": (regime,), "episodes_per_regime": 10**9},
            }
            for regime in BUILDING_REGIMES
        ]
    if args.env_name == "sustaingym_building_dynamic":
        return [
            {
                "name": weather,
                "env_name": "sustaingym_building_static",
                "env_kwargs": {"weathers": (weather,), "episodes_per_regime": 10**9},
            }
            for weather in args.sustaingym_building_weathers or DEFAULT_BUILDING_WEATHERS
        ]
    if args.env_name == "evcharging_dynamic":
        return [
            {
                "name": period,
                "env_name": "evcharging_static",
                "env_kwargs": {"periods": (period,), "episodes_per_regime": 10**9, "site": args.ev_site},
            }
            for period in args.ev_periods or DEFAULT_EV_PERIODS
        ]
    if args.env_name == "cogen_dynamic":
        return [
            {
                "name": f"renewables_{magnitude:g}",
                "env_name": "cogen_static",
                "env_kwargs": {
                    "renewables": (magnitude,),
                    "episodes_per_regime": 10**9,
                    "forecast_horizon": args.cogen_forecast_horizon,
                    "forecast_noise_std": args.cogen_forecast_noise_std,
                },
            }
            for magnitude in args.cogen_renewables or DEFAULT_COGEN_RENEWABLES
        ]
    if args.env_name == "chlor_alkali_dynamic":
        rows = canonical_chlor_regime_rows(
            num_regime_clusters=int(
                getattr(args, "chlor_regime_clusters", DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS)
            ),
        )
        return [
            {
                "name": str(row["regime"]),
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row["regime_meta"]),
                "env_name": "chlor_alkali_dynamic",
                "env_kwargs": chlor_regime_eval_env_kwargs(
                    regime_name=str(row["regime"]),
                    regime_id=int(row["regime_id"]),
                    episode_length=int(args.chlor_episode_length),
                    schedule="random",
                    dynamic_price=True,
                    aging_dynamics=True,
                    num_regime_clusters=int(
                        getattr(args, "chlor_regime_clusters", DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS)
                    ),
                ),
            }
            for row in rows
        ]
    return []


def evaluate_regime_metrics(
    args,
    sample_batch,
    evaluation_fn: Callable,
    ref_point: np.ndarray,
    limit: int | None = None,
    episode_seeds: list[int] | None = None,
    incremental_callback: Callable[[list[dict[str, object]]], None] | None = None,
):
    sample_batch = list(sample_batch)
    if limit is not None and limit > 0:
        sample_batch = sample_batch[:limit]
    if not sample_batch:
        return []

    if args.env_name == "chlor_alkali_dynamic":
        rows = []
        shared_plan, _shared_episode_seeds = shared_regime_seed_plan(args)
        regime_by_id = {
            int(spec["regime_id"]): spec
            for spec in regime_specs(args)
        }
        for plan_row in shared_plan:
            regime = regime_by_id.get(int(plan_row["regime_id"]))
            if regime is None:
                continue
            solutions = []
            for sample in sample_batch:
                eval_out = evaluation_fn(
                    args,
                    sample,
                    return_trace=False,
                    env_name=regime["env_name"],
                    env_kwargs=regime["env_kwargs"],
                    episode_seeds=[int(seed) for seed in plan_row.get("episode_seeds", [])],
                )
                objs = eval_out[0] if isinstance(eval_out, tuple) else eval_out
                solutions.append(np.asarray(objs, dtype=np.float64))
            solutions_arr = np.asarray(solutions, dtype=np.float64)
            if solutions_arr.ndim != 2 or len(solutions_arr) == 0:
                continue
            front = pareto_front(solutions_arr)
            if len(front) == 0:
                continue
            hv, eu, sp = compute_front_metrics(front, ref_point, args.eval_delta_weight)
            rows.append(
                {
                    "regime": str(regime["name"]),
                    "regime_id": int(regime.get("regime_id", len(rows))),
                    "regime_meta": dict(regime.get("regime_meta", {})),
                    "hv": hv,
                    "eu": eu,
                    "sp": sp,
                    "points": int(len(front)),
                    "num_solutions": int(len(solutions_arr)),
                    "front_points": front.tolist(),
                    "solution_points": solutions_arr.tolist(),
                }
            )
            if incremental_callback is not None:
                incremental_callback(list(rows))
        return rows

    shared_plan, resolved_episode_seeds = shared_regime_seed_plan(args)
    eval_episode_seeds = (
        [int(seed) for seed in episode_seeds]
        if episode_seeds is not None
        else [int(seed) for seed in resolved_episode_seeds]
    )
    grouped: dict[str, dict[str, object]] = {}

    for sample in sample_batch:
        _objs, trace = evaluation_fn(args, sample, return_trace=True, episode_seeds=eval_episode_seeds)
        if trace is None:
            continue
        assignments = assign_episode_returns_to_plan(
            trace,
            plan=shared_plan,
        )
        for row in assignments:
            bucket = grouped.setdefault(
                row["regime"],
                {
                    "regime": row["regime"],
                    "regime_id": int(row["regime_id"]),
                    "regime_meta": dict(row["regime_meta"]),
                    "solutions": [],
                },
            )
            bucket["solutions"].append(np.asarray(row["return"], dtype=np.float64))
        if incremental_callback is not None:
            incremental_callback(
                _regime_rows_from_grouped(
                    grouped,
                    ref_point=np.asarray(ref_point, dtype=np.float64),
                    eval_delta_weight=args.eval_delta_weight,
                )
            )

    return _regime_rows_from_grouped(
        grouped,
        ref_point=np.asarray(ref_point, dtype=np.float64),
        eval_delta_weight=args.eval_delta_weight,
    )


def summarize_regime_metrics(regime_metric_rows):
    if not regime_metric_rows:
        return {}
    hvs = np.asarray([row["hv"] for row in regime_metric_rows], dtype=np.float64)
    eus = np.asarray([row["eu"] for row in regime_metric_rows], dtype=np.float64)
    sps = np.asarray([row["sp"] for row in regime_metric_rows], dtype=np.float64)
    return {
        "cr_hv": float(hvs.mean()),
        "cr_eu": float(eus.mean()),
        "cr_sp": float(sps.mean()),
        "hv_std": float(hvs.std()),
        "eu_std": float(eus.std()),
        "sp_std": float(sps.std()),
        "irs": float(1.0 / (1.0 + eus.std())),
    }


def summarize_trace_metrics(trace_metric_rows):
    if not trace_metric_rows:
        return {}
    keys = [
        "trace_utility_mean",
        "trace_utility_std",
        "trace_shift_count",
        "trace_shift_regret",
        "trace_recovery_latency",
        "trace_recovery_score",
        "trace_pre_post_gap",
    ]
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in trace_metric_rows], dtype=np.float64)
        summary[key] = float(values.mean())
    return summary


def append_metrics_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
