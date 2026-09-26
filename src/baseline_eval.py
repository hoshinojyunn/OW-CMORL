from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.baseline_envs import (
    BASELINE_ENVS,
    build_dynamic_env,
    default_env_kwargs,
    get_env_spec,
    preference_grid,
)
from src.dynamic_morl.fine_regimes import (
    DEFAULT_FINE_REGIME_CLUSTERS,
    DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    DEFAULT_FINE_REGIME_EPISODES,
    assign_episode_returns_to_catalog,
    assign_episode_returns_to_plan,
    catalog_cache_signature,
    coerce_regime_seed_plan,
    load_or_build_catalog,
    load_or_build_regime_seed_plan,
    selected_episode_seeds_from_plan,
)
from src.dynamic_morl.chlor_shared_protocol import (
    canonical_chlor_regime_rows,
    canonical_chlor_regime_seed_plan,
    chlor_regime_eval_env_kwargs,
)
from src.dynamic_morl.metrics import (
    compute_trace_shift_metrics,
)
from src.dynamic_morl.shared_eval import (
    build_shared_episode_seeds,
    normalize_episode_seeds,
)


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def merge_trace_segments(trace_segments: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    if not trace_segments:
        return None
    obs_rows = []
    obj_rows = []
    context_rows = []
    regime_names = []
    regime_ids = []
    steps = []
    episodes = []
    context_feature_names: list[str] = []
    regime_records = []
    step_offset = 0
    for episode_idx, trace in enumerate(trace_segments):
        obs = np.asarray(trace.get("obs", []), dtype=np.float32)
        obj = np.asarray(trace.get("obj", []), dtype=np.float32)
        if "return" in trace:
            regime_records.append(
                {
                    "eval_id": int(episode_idx),
                    "regime": str(trace.get("episode_regime", "episode")),
                    "return": np.asarray(trace["return"], dtype=np.float32),
                }
            )
        if obs.ndim != 2 or obj.ndim != 2 or len(obs) == 0 or len(obs) != len(obj):
            continue
        context = np.asarray(trace.get("context", []), dtype=np.float32)
        obs_rows.append(obs)
        obj_rows.append(obj)
        if context.ndim == 2 and len(context) == len(obj):
            context_rows.append(context)
        regime_names.extend(list(trace.get("regime", [])))
        regime_ids.extend(list(np.asarray(trace.get("regime_id", []))))
        if not context_feature_names and trace.get("context_feature_names"):
            context_feature_names = list(trace.get("context_feature_names", []))
        local_steps = np.asarray(trace.get("step", np.arange(len(obj))), dtype=np.int64)
        steps.extend((local_steps + step_offset).tolist())
        episodes.extend([episode_idx] * len(obj))
        step_offset += int(local_steps[-1]) + 1 if len(local_steps) else len(obj)
    if not obs_rows:
        return None
    merged = {
        "obs": np.concatenate(obs_rows, axis=0),
        "obj": np.concatenate(obj_rows, axis=0),
        "regime": regime_names,
        "regime_id": np.asarray(regime_ids),
        "step": np.asarray(steps, dtype=np.int64),
        "episode": np.asarray(episodes, dtype=np.int64),
    }
    if context_rows:
        merged["context"] = np.concatenate(context_rows, axis=0)
    if context_feature_names:
        merged["context_feature_names"] = context_feature_names
    if regime_records:
        merged["regime_records"] = regime_records
    return merged


def evaluate_old_gym_policy(
    env_factory: Callable[[], object],
    action_fn: Callable[[np.ndarray], np.ndarray],
    *,
    episodes: int = 1,
    episode_seeds: list[int] | None = None,
    record_trace: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray] | None]:
    returns = []
    trace_segments: list[dict[str, np.ndarray]] = []
    env = env_factory()
    try:
        seeds = episode_seeds if episode_seeds is not None else list(range(max(1, episodes)))
        for episode_idx, episode_seed in enumerate(seeds):
            if hasattr(env, "seed"):
                env.seed(int(episode_seed))
            obs = env.reset()
            done = False
            total_obj = np.zeros(getattr(env, "reward_num", getattr(env, "rwd_dim", 0)), dtype=np.float64)
            trace = {"obs": [], "obj": [], "context": [], "regime": [], "regime_id": [], "step": [], "context_feature_names": []}
            step_idx = 0
            while not done:
                obs_before = np.asarray(obs, dtype=np.float32)
                action = np.asarray(action_fn(obs_before), dtype=np.float32)
                obs, reward, done, info = env.step(action)
                obj = np.asarray(info.get("obj_raw", info.get("obj", reward)), dtype=np.float32)
                total_obj += obj.astype(np.float64)
                if record_trace:
                    context_vec = np.asarray(info.get("context_vector", []), dtype=np.float32).reshape(-1)
                    trace["obs"].append(obs_before)
                    trace["obj"].append(obj)
                    if len(context_vec) > 0:
                        trace["context"].append(context_vec)
                    if not trace["context_feature_names"] and info.get("context_feature_names"):
                        trace["context_feature_names"] = list(info.get("context_feature_names", []))
                    trace["regime"].append(
                        info.get("regime_name")
                        or info.get("period_name")
                        or info.get("weather")
                        or str(info.get("renewables_magnitude", "regime"))
                    )
                    trace["regime_id"].append(
                        info.get("regime_id", info.get("period_id", info.get("renewables_magnitude", 0)))
                    )
                    trace["step"].append(step_idx)
                step_idx += 1
            returns.append(total_obj)
            if record_trace:
                trace_segments.append(
                    {
                        "obs": np.asarray(trace["obs"], dtype=np.float32),
                        "obj": np.asarray(trace["obj"], dtype=np.float32),
                        "context": np.asarray(trace["context"], dtype=np.float32),
                        "regime": trace["regime"],
                        "regime_id": np.asarray(trace["regime_id"]),
                        "step": np.asarray(trace["step"], dtype=np.int64),
                        "context_feature_names": list(trace.get("context_feature_names", [])),
                        "return": total_obj.astype(np.float32),
                        "episode_regime": trace["regime"][0] if trace["regime"] else "episode",
                    }
                )
    finally:
        env.close()
    merged = merge_trace_segments(trace_segments) if record_trace else None
    return np.mean(np.stack(returns, axis=0), axis=0), merged


def _regime_args(env_name: str, obj_num: int, seed: int, env_kwargs: dict[str, object]) -> SimpleNamespace:
    merged = default_env_kwargs(get_env_spec(resolve_env_key_from_name(env_name)).env_key)
    merged.update(env_kwargs)
    return SimpleNamespace(
        env_name=env_name,
        obj_num=obj_num,
        seed=seed,
        ev_site=merged.get("site", "caltech"),
        ev_periods=list(merged.get("periods", ())),
        cogen_renewables=list(merged.get("renewables", ())),
        cogen_forecast_horizon=int(merged.get("forecast_horizon", 3)),
        cogen_forecast_noise_std=float(merged.get("forecast_noise_std", 0.0)),
        chlor_episode_length=int(merged.get("episode_length", 288)),
        chlor_regime_clusters=int(merged.get("num_regime_clusters", 12)),
        sustaingym_building_weathers=[],
    )


def resolve_env_key_from_name(env_name: str) -> str:
    for env_key, spec in BASELINE_ENVS.items():
        if env_name in {spec.dynamic_env_name, spec.static_env_name}:
            return env_key
    raise ValueError(f"Unsupported env name: {env_name}")


def _fine_regime_catalog(env_name: str, seed: int, env_kwargs: dict[str, object]):
    env_key = resolve_env_key_from_name(env_name)
    merged = default_env_kwargs(env_key)
    merged.update(env_kwargs)

    def env_factory(factory_seed: int):
        return build_dynamic_env(
            get_env_spec(env_key).dynamic_env_name,
            seed=seed + int(factory_seed),
            env_kwargs=merged,
        )

    return load_or_build_catalog(
        env_key=env_key,
        env_factory=env_factory,
        n_clusters=int(merged.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
        episodes=int(merged.get("catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
        context_steps=int(merged.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
        seed=0,
        cache_signature=catalog_cache_signature(
            {
                "env_name": get_env_spec(env_key).dynamic_env_name,
                "env_kwargs": merged,
                "episodes": int(merged.get("catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
                "context_steps": int(merged.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
            }
        ),
    )


def _shared_regime_seed_plan(
    env_name: str,
    seed: int,
    env_kwargs: dict[str, object],
    *,
    shared_regime_eval_episodes: int,
    shared_regime_seed_offset: int,
) -> tuple[object, int, list[dict[str, object]], list[int]]:
    catalog = _fine_regime_catalog(env_name, seed, env_kwargs)
    context_steps = int(env_kwargs.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS))
    target_regime_count = int(shared_regime_eval_episodes or len(catalog.labels) or DEFAULT_FINE_REGIME_CLUSTERS)
    target_regime_ids = sorted(int(idx) for idx in catalog.labels)[: min(target_regime_count, len(catalog.labels))]
    start_seed = build_shared_episode_seeds(
        env_name,
        base_seed=seed,
        num_episodes=1,
        seed_offset=shared_regime_seed_offset,
    )[0]

    merged = default_env_kwargs(resolve_env_key_from_name(env_name))
    merged.update(env_kwargs)

    def env_factory(_factory_seed: int):
        return build_dynamic_env(
            env_name,
            seed=seed,
            env_kwargs=merged,
        )

    plan_cache_signature = catalog_cache_signature(
        {
            "env_name": env_name,
            "env_kwargs": merged,
            "target_regime_ids": target_regime_ids,
            "shared_regime_eval_episodes": int(shared_regime_eval_episodes),
            "shared_regime_seed_offset": int(shared_regime_seed_offset),
            "search_start_seed": int(start_seed),
        }
    )
    plan = load_or_build_regime_seed_plan(
        env_key=resolve_env_key_from_name(env_name),
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(start_seed),
        seeds_per_regime=1,
        context_steps=context_steps,
        max_search_episodes=max(256, 64 * max(1, len(target_regime_ids))),
        target_regime_ids=target_regime_ids,
        cache_signature=plan_cache_signature,
    )
    return catalog, context_steps, plan, selected_episode_seeds_from_plan(plan)


def _regime_stem(regime_name: str, regime_id: int) -> str:
    return f"regime_{int(regime_id):03d}_{str(regime_name)}".replace("/", "_").replace(" ", "_")


def _save_regime_front_artifacts(regime_output_dir: Path | None, regime_fronts: dict[str, dict[str, object]]) -> None:
    if regime_output_dir is None:
        return
    regime_output_dir.mkdir(parents=True, exist_ok=True)
    for regime_name, row in sorted(regime_fronts.items()):
        points = np.asarray(row.get("points", []), dtype=np.float64)
        front = pareto_front(points)
        stem = _regime_stem(regime_name, int(row.get("regime_id", -1)))
        payload = {
            "regime": str(regime_name),
            "regime_id": int(row.get("regime_id", -1)),
            "regime_meta": dict(row.get("regime_meta", {})),
            "num_solutions": int(len(points)),
            "front_points": front.tolist(),
            "solution_points": points.tolist(),
        }
        (regime_output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        np.savez_compressed(
            regime_output_dir / f"{stem}.npz",
            solutions=np.asarray(points, dtype=np.float32),
            pareto_front_points=np.asarray(front, dtype=np.float32),
            regime_id=np.asarray([int(row.get("regime_id", -1))], dtype=np.int32),
        )


def _chlor_catalog_rows(env_name: str, seed: int, env_kwargs: dict[str, object]) -> list[dict[str, object]]:
    return canonical_chlor_regime_rows(
        num_regime_clusters=int(env_kwargs.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
        dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
        seed=0,
        train_path=env_kwargs.get("train_path"),
    )


def _evaluate_chlor_policy_regimes(
    *,
    env_name: str,
    seed: int,
    env_kwargs: dict[str, object],
    shared_regime_eval_episodes: int,
    shared_regime_seed_offset: int,
    regime_seed_plan: list[dict[str, object]] | None = None,
    evaluate_episode_fn: Callable[..., tuple[np.ndarray, dict[str, np.ndarray] | None]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if regime_seed_plan is None:
        plan, _flat_seeds = canonical_chlor_regime_seed_plan(
            env_name=env_name,
            base_seed=seed,
            shared_regime_eval_episodes=int(shared_regime_eval_episodes),
            seed_offset=int(shared_regime_seed_offset),
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
            dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
            train_path=env_kwargs.get("train_path"),
        )
    else:
        # A registered shared plan is the authoritative evaluation protocol.
        # Do not merely relabel a random chlor trajectory by its sequence
        # position: each evaluation environment must be restricted to the
        # plan's actual dynamic regime.
        plan = [dict(row) for row in regime_seed_plan]
    rows_by_id = {
        int(row["regime_id"]): row for row in _chlor_catalog_rows(env_name, seed, env_kwargs)
    }
    for plan_row in plan:
        regime_id = int(plan_row["regime_id"])
        catalog_row = rows_by_id.get(regime_id)
        regime_name = str(plan_row.get("regime", catalog_row["regime"] if catalog_row else ""))
        if not regime_name:
            continue
        regime_meta = dict(plan_row.get("regime_meta", catalog_row["regime_meta"] if catalog_row else {}))
        episode_seeds = [int(value) for value in plan_row.get("episode_seeds", [])]
        if not episode_seeds:
            continue
        episode_obj, _trace = evaluate_episode_fn(
            episode_seed=int(episode_seeds[0]),
            env_name=env_name,
            env_kwargs={
                **dict(env_kwargs),
                **chlor_regime_eval_env_kwargs(
                    regime_name=regime_name,
                    regime_id=regime_id,
                    episode_length=int(env_kwargs.get("episode_length", 288)),
                    schedule="random",
                    dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
                    aging_dynamics=bool(env_kwargs.get("aging_dynamics", True)),
                    num_regime_clusters=int(env_kwargs.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
                ),
            },
        )
        rows.append(
            {
                "regime": regime_name,
                "regime_id": regime_id,
                "regime_meta": regime_meta,
                "points": [np.asarray(episode_obj, dtype=np.float64).tolist()],
            }
        )
    return rows


def _evaluate_policy_regimes_from_plan(
    *,
    env_name: str,
    env_kwargs: dict[str, object],
    regime_seed_plan: list[dict[str, object]],
    evaluate_episode_fn: Callable[..., tuple[np.ndarray, dict[str, np.ndarray] | None]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plan_row in regime_seed_plan:
        episode_seeds = [int(seed) for seed in plan_row.get("episode_seeds", [])]
        if not episode_seeds:
            continue
        episode_obj, _trace = evaluate_episode_fn(
            episode_seed=int(episode_seeds[0]),
            env_name=env_name,
            env_kwargs=dict(env_kwargs),
        )
        rows.append(
            {
                "regime": str(plan_row["regime"]),
                "regime_id": int(plan_row["regime_id"]),
                "regime_meta": dict(plan_row.get("regime_meta", {})),
                "points": [np.asarray(episode_obj, dtype=np.float64).tolist()],
            }
        )
    return rows


def _regime_fronts_from_trace(
    trace: dict[str, object] | None,
    *,
    catalog,
    context_steps: int,
) -> list[dict[str, object]]:
    if trace is None:
        return []
    assignments = assign_episode_returns_to_catalog(
        trace,
        catalog=catalog,
        context_steps=context_steps,
    )
    grouped: dict[str, dict[str, object]] = {}
    for row in assignments:
        bucket = grouped.setdefault(
            row["regime"],
            {
                "regime": row["regime"],
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row["regime_meta"]),
                "points": [],
            },
        )
        bucket["points"].append(np.asarray(row["return"], dtype=np.float64).tolist())
    return [grouped[key] for key in sorted(grouped)]


def _regime_fronts_from_trace_plan(
    trace: dict[str, object] | None,
    *,
    regime_seed_plan: list[dict[str, object]],
) -> list[dict[str, object]]:
    if trace is None:
        return []
    assignments = assign_episode_returns_to_plan(
        trace,
        plan=regime_seed_plan,
    )
    grouped: dict[str, dict[str, object]] = {}
    for row in assignments:
        bucket = grouped.setdefault(
            row["regime"],
            {
                "regime": row["regime"],
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row["regime_meta"]),
                "points": [],
            },
        )
        bucket["points"].append(np.asarray(row["return"], dtype=np.float64).tolist())
    return [grouped[key] for key in sorted(grouped)]


def evaluate_conditioned_policy(
    *,
    env_name: str,
    obj_num: int,
    seed: int,
    policy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    preferences: list[list[float]],
    env_kwargs: dict[str, object] | None = None,
    eval_episodes: int = 3,
    trace_eval_episodes: int = 3,
    eval_delta_weight: float = 0.5,
    trace_recovery_window: int = 12,
    shared_regime_eval_episodes: int = 0,
    shared_regime_seed_offset: int = 0,
    shared_regime_seed_plan: list[dict[str, object]] | dict[str, object] | str | Path | None = None,
    regime_output_dir: Path | None = None,
) -> dict[str, object]:
    env_kwargs = dict(env_kwargs or {})
    front_points = []
    regime_fronts: dict[str, dict[str, object]] = {}
    trace_metric_rows = []
    per_preference_rows = []
    override_plan = (
        coerce_regime_seed_plan(shared_regime_seed_plan)
        if shared_regime_seed_plan is not None
        else None
    )
    force_plan_override = override_plan is not None
    if override_plan is not None:
        catalog = _fine_regime_catalog(env_name, seed, env_kwargs)
        context_steps = int(env_kwargs.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS))
        regime_seed_plan = override_plan
        episode_seeds = selected_episode_seeds_from_plan(regime_seed_plan)
    elif resolve_env_key_from_name(env_name) == "chlor_alkali":
        catalog = _fine_regime_catalog(env_name, seed, env_kwargs)
        context_steps = int(env_kwargs.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS))
        eval_run_episodes = max(int(eval_episodes), int(trace_eval_episodes), 1)
        regime_seed_plan, episode_seeds = canonical_chlor_regime_seed_plan(
            env_name=env_name,
            base_seed=seed,
            shared_regime_eval_episodes=shared_regime_eval_episodes or eval_run_episodes,
            seed_offset=shared_regime_seed_offset,
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
            dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
            train_path=env_kwargs.get("train_path"),
        )
    else:
        catalog, context_steps, regime_seed_plan, episode_seeds = _shared_regime_seed_plan(
            env_name,
            seed,
            env_kwargs,
            shared_regime_eval_episodes=shared_regime_eval_episodes,
            shared_regime_seed_offset=shared_regime_seed_offset,
        )

    for pref in preferences:
        pref_array = np.asarray(pref, dtype=np.float32)
        objs, dyn_trace = evaluate_old_gym_policy(
            lambda: build_old_gym_vector_env_from_name(
                env_name,
                seed=seed,
                env_kwargs=env_kwargs,
            ),
            lambda obs, pref_array=pref_array: policy_fn(obs, pref_array),
            episodes=len(episode_seeds),
            episode_seeds=episode_seeds,
            record_trace=True,
        )
        front_points.append(objs.tolist())

        if dyn_trace is not None:
            trace_metrics = compute_trace_shift_metrics(
                dyn_trace,
                eval_delta_weight,
                recovery_window=trace_recovery_window,
            )
            trace_metric_rows.append(trace_metrics)
        else:
            trace_metrics = {}

        if resolve_env_key_from_name(env_name) == "chlor_alkali":
            regime_rows = _evaluate_chlor_policy_regimes(
                env_name=env_name,
                seed=seed,
                env_kwargs=env_kwargs,
                shared_regime_eval_episodes=shared_regime_eval_episodes,
                shared_regime_seed_offset=shared_regime_seed_offset,
                regime_seed_plan=regime_seed_plan if force_plan_override else None,
                evaluate_episode_fn=lambda *, episode_seed, env_name, env_kwargs: evaluate_old_gym_policy(
                    lambda: build_old_gym_vector_env_from_name(
                        env_name,
                        seed=seed,
                        env_kwargs=env_kwargs,
                    ),
                    lambda obs, pref_array=pref_array: policy_fn(obs, pref_array),
                    episodes=1,
                    episode_seeds=[episode_seed],
                    record_trace=False,
                ),
            )
        else:
            regime_rows = _regime_fronts_from_trace_plan(
                dyn_trace,
                regime_seed_plan=regime_seed_plan,
            )
            if not regime_rows:
                regime_rows = _evaluate_policy_regimes_from_plan(
                    env_name=env_name,
                    env_kwargs=env_kwargs,
                    regime_seed_plan=regime_seed_plan,
                    evaluate_episode_fn=lambda *, episode_seed, env_name, env_kwargs: evaluate_old_gym_policy(
                        lambda: build_old_gym_vector_env_from_name(
                            env_name,
                            seed=seed,
                            env_kwargs=env_kwargs,
                        ),
                        lambda obs, pref_array=pref_array: policy_fn(obs, pref_array),
                        episodes=1,
                        episode_seeds=[episode_seed],
                        record_trace=False,
                    ),
                )

        for row in regime_rows:
            bucket = regime_fronts.setdefault(
                row["regime"],
                {
                    "regime": row["regime"],
                    "regime_id": int(row["regime_id"]),
                    "regime_meta": dict(row["regime_meta"]),
                    "points": [],
                },
            )
            bucket["points"].extend(list(row["points"]))
        _save_regime_front_artifacts(regime_output_dir, regime_fronts)

        per_preference_rows.append(
            {
                "preference": pref,
                "objs": objs.tolist(),
                "trace_metrics": trace_metrics,
            }
        )

    return {
        "front_points": front_points,
        "regime_fronts": [regime_fronts[key] for key in sorted(regime_fronts)],
        "trace_metrics": _mean_metric_rows(trace_metric_rows),
        "per_preference": per_preference_rows,
        "shared_eval_episode_seeds": [int(seed_value) for seed_value in episode_seeds],
        "shared_regime_seed_plan": regime_seed_plan,
    }


def evaluate_archive_policies(
    *,
    env_name: str,
    obj_num: int,
    seed: int,
    policy_fns: list[Callable[[np.ndarray], np.ndarray]],
    env_kwargs: dict[str, object] | None = None,
    eval_episodes: int = 3,
    trace_eval_episodes: int = 3,
    eval_delta_weight: float = 0.5,
    trace_recovery_window: int = 12,
    shared_regime_eval_episodes: int = 0,
    shared_regime_seed_offset: int = 0,
    shared_regime_seed_plan: list[dict[str, object]] | dict[str, object] | str | Path | None = None,
    regime_output_dir: Path | None = None,
) -> dict[str, object]:
    env_kwargs = dict(env_kwargs or {})
    front_points = []
    regime_fronts: dict[str, dict[str, object]] = {}
    trace_metric_rows = []
    per_policy_rows = []
    override_plan = (
        coerce_regime_seed_plan(shared_regime_seed_plan)
        if shared_regime_seed_plan is not None
        else None
    )
    force_plan_override = override_plan is not None
    if override_plan is not None:
        catalog = _fine_regime_catalog(env_name, seed, env_kwargs)
        context_steps = int(env_kwargs.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS))
        regime_seed_plan = override_plan
        episode_seeds = selected_episode_seeds_from_plan(regime_seed_plan)
    elif resolve_env_key_from_name(env_name) == "chlor_alkali":
        catalog = _fine_regime_catalog(env_name, seed, env_kwargs)
        context_steps = int(env_kwargs.get("context_catalog_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS))
        eval_run_episodes = max(int(eval_episodes), int(trace_eval_episodes), 1)
        regime_seed_plan, episode_seeds = canonical_chlor_regime_seed_plan(
            env_name=env_name,
            base_seed=seed,
            shared_regime_eval_episodes=shared_regime_eval_episodes or eval_run_episodes,
            seed_offset=shared_regime_seed_offset,
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
            dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
            train_path=env_kwargs.get("train_path"),
        )
    else:
        catalog, context_steps, regime_seed_plan, episode_seeds = _shared_regime_seed_plan(
            env_name,
            seed,
            env_kwargs,
            shared_regime_eval_episodes=shared_regime_eval_episodes,
            shared_regime_seed_offset=shared_regime_seed_offset,
        )

    for policy_idx, policy_fn in enumerate(policy_fns):
        objs, dyn_trace = evaluate_old_gym_policy(
            lambda: build_old_gym_scalar_env_from_name(
                env_name,
                seed=seed,
                env_kwargs=env_kwargs,
            ),
            policy_fn,
            episodes=len(episode_seeds),
            episode_seeds=episode_seeds,
            record_trace=True,
        )
        front_points.append(objs.tolist())

        if dyn_trace is not None:
            trace_metrics = compute_trace_shift_metrics(
                dyn_trace,
                eval_delta_weight,
                recovery_window=trace_recovery_window,
            )
            trace_metric_rows.append(trace_metrics)
        else:
            trace_metrics = {}

        if resolve_env_key_from_name(env_name) == "chlor_alkali":
            regime_rows = _evaluate_chlor_policy_regimes(
                env_name=env_name,
                seed=seed,
                env_kwargs=env_kwargs,
                shared_regime_eval_episodes=shared_regime_eval_episodes,
                shared_regime_seed_offset=shared_regime_seed_offset,
                regime_seed_plan=regime_seed_plan if force_plan_override else None,
                evaluate_episode_fn=lambda *, episode_seed, env_name, env_kwargs: evaluate_old_gym_policy(
                    lambda: build_old_gym_scalar_env_from_name(
                        env_name,
                        seed=seed,
                        env_kwargs=env_kwargs,
                    ),
                    policy_fn,
                    episodes=1,
                    episode_seeds=[episode_seed],
                    record_trace=False,
                ),
            )
        else:
            regime_rows = _regime_fronts_from_trace_plan(
                dyn_trace,
                regime_seed_plan=regime_seed_plan,
            )
            if not regime_rows:
                regime_rows = _evaluate_policy_regimes_from_plan(
                    env_name=env_name,
                    env_kwargs=env_kwargs,
                    regime_seed_plan=regime_seed_plan,
                    evaluate_episode_fn=lambda *, episode_seed, env_name, env_kwargs: evaluate_old_gym_policy(
                        lambda: build_old_gym_scalar_env_from_name(
                            env_name,
                            seed=seed,
                            env_kwargs=env_kwargs,
                        ),
                        policy_fn,
                        episodes=1,
                        episode_seeds=[episode_seed],
                        record_trace=False,
                    ),
                )

        for row in regime_rows:
            bucket = regime_fronts.setdefault(
                row["regime"],
                {
                    "regime": row["regime"],
                    "regime_id": int(row["regime_id"]),
                    "regime_meta": dict(row["regime_meta"]),
                    "points": [],
                },
            )
            bucket["points"].extend(list(row["points"]))
        _save_regime_front_artifacts(regime_output_dir, regime_fronts)

        per_policy_rows.append(
            {
                "policy_id": policy_idx,
                "objs": objs.tolist(),
                "trace_metrics": trace_metrics,
            }
        )

    return {
        "front_points": front_points,
        "regime_fronts": [regime_fronts[key] for key in sorted(regime_fronts)],
        "trace_metrics": _mean_metric_rows(trace_metric_rows),
        "per_policy": per_policy_rows,
        "shared_eval_episode_seeds": [int(seed_value) for seed_value in episode_seeds],
        "shared_regime_seed_plan": regime_seed_plan,
    }


def build_old_gym_vector_env_from_name(
    env_name: str,
    *,
    seed: int = 0,
    env_kwargs: dict[str, object] | None = None,
):
    from src.baseline_envs import OldGymVectorRewardWrapper

    return OldGymVectorRewardWrapper(env_name, seed=seed, env_kwargs=env_kwargs)


def build_old_gym_scalar_env_from_name(
    env_name: str,
    *,
    seed: int = 0,
    env_kwargs: dict[str, object] | None = None,
):
    from src.baseline_envs import OldGymScalarInfoWrapper

    return OldGymScalarInfoWrapper(env_name, seed=seed, env_kwargs=env_kwargs)


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row.keys()})
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(values)) if values else 0.0
    return out


def save_baseline_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))


def save_front_csv(path: Path, front_points: list[list[float]]) -> None:
    if not front_points:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([f"obj_{idx}" for idx in range(len(front_points[0]))])
        writer.writerows(front_points)
