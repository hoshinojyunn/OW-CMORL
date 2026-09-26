from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.baseline_envs import build_dynamic_env, get_env_spec
from src.dynamic_morl.chlor_alkali_env import (
    DEFAULT_CHLOR_ALKALI_TEST,
    DEFAULT_CHLOR_ALKALI_TRAIN,
)
from src.dynamic_morl.fine_regimes import (
    DEFAULT_FINE_REGIME_CLUSTERS,
    DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    DEFAULT_FINE_REGIME_EPISODES,
    catalog_cache_signature,
    load_or_build_catalog,
    load_or_build_regime_seed_plan,
    selected_episode_seeds_from_plan,
)
from src.dynamic_morl.shared_eval import build_shared_episode_seeds


DEFAULT_OOD_PROFILE = "ood_v1"


OOD_EXPERIMENT_PROFILES: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
    "ood_v1": {
        "train": {
            "building": {
                "schedule": "cyclic",
                "episodes_per_regime": 1,
            },
            "evcharging": {
                "site": "caltech",
                "periods": ["Summer 2019", "Spring 2020", "Summer 2021"],
                "schedule": "cyclic",
                "episodes_per_regime": 1,
                "moer_forecast_steps": 36,
                "project_action_in_env": True,
            },
            "cogen": {
                "renewables": [0.0, 10.0, 20.0],
                "schedule": "cyclic",
                "episodes_per_regime": 1,
                "forecast_horizon": 3,
                "forecast_noise_std": 0.0,
            },
            "chlor_alkali": {
                "dataset_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
                "train_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
                "dynamic_price": True,
                "aging_dynamics": True,
                "episode_length": 288,
                "schedule": "random",
                "num_regime_clusters": 20,
            },
        },
        "eval": {
            "building": {
                "regimes": [
                    {
                        "name": "warm_dry_tariff",
                        "weather": "Warm_Dry",
                        "occupancy_scale": 1.22,
                        "carbon_scale": 1.16,
                        "price_scale": 1.20,
                        "target": 22.5,
                    },
                    {
                        "name": "hot_humid_peak",
                        "weather": "Hot_Humid",
                        "occupancy_scale": 1.30,
                        "carbon_scale": 1.12,
                        "price_scale": 1.22,
                        "target": 23.5,
                    },
                    {
                        "name": "cold_dry_tariff",
                        "weather": "Cold_Dry",
                        "occupancy_scale": 1.08,
                        "carbon_scale": 1.22,
                        "price_scale": 1.28,
                        "target": 21.5,
                    },
                    {
                        "name": "very_hot_humid_stress",
                        "weather": "Very_Hot_Humid",
                        "occupancy_scale": 1.10,
                        "carbon_scale": 1.26,
                        "price_scale": 1.30,
                        "target": 24.0,
                    },
                ],
                "schedule": "random",
                "episodes_per_regime": 1,
            },
            "evcharging": {
                "site": "caltech",
                "periods": ["Fall 2019"],
                "schedule": "random",
                "episodes_per_regime": 1,
                "moer_forecast_steps": 36,
                "project_action_in_env": True,
            },
            "cogen": {
                "renewables": [30.0, 45.0, 60.0],
                "schedule": "random",
                "episodes_per_regime": 1,
                "forecast_horizon": 3,
                "forecast_noise_std": 0.08,
            },
            "chlor_alkali": {
                "dataset_path": str(DEFAULT_CHLOR_ALKALI_TEST),
                "train_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
                "dynamic_price": True,
                "aging_dynamics": True,
                "episode_length": 288,
                "schedule": "random",
                "num_regime_clusters": 20,
            },
        },
    }
}

# This profile is fixed before executing its evaluation.  It preserves the
# existing held-out families for three environments and adds a tariff-stress
# trace for chlor-alkali.  The train frame and the fitted surrogate remain on
# normal CA_train pricing; only the evaluation trace is shifted.
OOD_EXPERIMENT_PROFILES["ood_v2_price_trace"] = deepcopy(OOD_EXPERIMENT_PROFILES["ood_v1"])
OOD_EXPERIMENT_PROFILES["ood_v2_price_trace"]["eval"]["chlor_alkali"] = {
    "dataset_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
    "train_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
    "dynamic_price": True,
    "aging_dynamics": True,
    "episode_length": 288,
    "schedule": "random",
    "num_regime_clusters": 10,
    "price_multiplier": 1.20,
    "price_offset": 0.70,
}

# Frozen before any v3 result is generated. This profile changes every v2
# evaluation configuration rather than reusing the v2 OOD traces. The training
# families remain ID-only and are identical to the corresponding v1/v2 splits.
OOD_EXPERIMENT_PROFILES["ood_v3_site_tariff_trace"] = deepcopy(OOD_EXPERIMENT_PROFILES["ood_v1"])
OOD_EXPERIMENT_PROFILES["ood_v3_site_tariff_trace"]["eval"] = {
    "building": {
        "regimes": [
            {
                "name": "warm_dry_high_tariff",
                "weather": "Warm_Dry",
                "occupancy_scale": 1.18,
                "carbon_scale": 1.20,
                "price_scale": 1.36,
                "target": 22.0,
            },
            {
                "name": "hot_humid_tariff_stress",
                "weather": "Hot_Humid",
                "occupancy_scale": 1.28,
                "carbon_scale": 1.18,
                "price_scale": 1.40,
                "target": 23.0,
            },
            {
                "name": "cold_dry_high_carbon",
                "weather": "Cold_Dry",
                "occupancy_scale": 1.12,
                "carbon_scale": 1.28,
                "price_scale": 1.44,
                "target": 21.0,
            },
            {
                "name": "very_hot_humid_tariff_stress",
                "weather": "Very_Hot_Humid",
                "occupancy_scale": 1.16,
                "carbon_scale": 1.24,
                "price_scale": 1.48,
                "target": 24.5,
            },
        ],
        "schedule": "random",
        "episodes_per_regime": 1,
    },
    "evcharging": {
        "site": "jpl",
        "periods": ["Fall 2019"],
        "schedule": "random",
        "episodes_per_regime": 1,
        "moer_forecast_steps": 36,
        "project_action_in_env": True,
    },
    "cogen": {
        "renewables": [35.0, 50.0, 65.0],
        "schedule": "random",
        "episodes_per_regime": 1,
        "forecast_horizon": 3,
        "forecast_noise_std": 0.12,
    },
    "chlor_alkali": {
        "dataset_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
        "train_path": str(DEFAULT_CHLOR_ALKALI_TRAIN),
        "dynamic_price": True,
        "aging_dynamics": True,
        "episode_length": 288,
        "schedule": "random",
        "num_regime_clusters": 10,
        "price_multiplier": 1.28,
        "price_offset": 0.85,
    },
}


def available_ood_profiles() -> tuple[str, ...]:
    return tuple(sorted(OOD_EXPERIMENT_PROFILES))


def resolve_ood_phase_env_config(
    env_key: str,
    *,
    phase: str,
    profile: str = DEFAULT_OOD_PROFILE,
) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(OOD_EXPERIMENT_PROFILES[profile][phase][env_key]))
    except KeyError as exc:
        raise ValueError(
            f"Unsupported OOD profile/env combination: profile={profile}, phase={phase}, env_key={env_key}"
        ) from exc


def resolve_ood_train_env_config(env_key: str, profile: str = DEFAULT_OOD_PROFILE) -> dict[str, Any]:
    return resolve_ood_phase_env_config(env_key, phase="train", profile=profile)


def resolve_ood_eval_env_config(env_key: str, profile: str = DEFAULT_OOD_PROFILE) -> dict[str, Any]:
    return resolve_ood_phase_env_config(env_key, phase="eval", profile=profile)


def load_env_config_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict) and isinstance(payload.get("env_kwargs"), dict):
        payload = payload["env_kwargs"]
    if not isinstance(payload, dict):
        raise ValueError(f"Expected env config JSON object in {path}")
    return dict(payload)


def build_ood_shared_plan(
    env_key: str,
    *,
    seed: int = 0,
    profile: str = DEFAULT_OOD_PROFILE,
    env_kwargs: dict[str, Any] | None = None,
    shared_regime_eval_episodes: int = 20,
    shared_regime_seed_offset: int = 0,
    fine_regime_clusters: int = DEFAULT_FINE_REGIME_CLUSTERS,
    catalog_episodes: int = DEFAULT_FINE_REGIME_EPISODES,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    resolved_env_kwargs = resolve_ood_eval_env_config(env_key, profile=profile)
    if env_kwargs:
        resolved_env_kwargs.update(dict(env_kwargs))
    resolved_env_kwargs.setdefault("catalog_episodes", int(catalog_episodes))
    resolved_env_kwargs.setdefault("context_catalog_steps", int(context_steps))
    resolved_env_kwargs.setdefault("num_regime_clusters", int(fine_regime_clusters))

    spec = get_env_spec(env_key)
    env_name = spec.dynamic_env_name

    def env_factory(factory_seed: int):
        return build_dynamic_env(
            env_name,
            seed=int(seed) + int(factory_seed),
            env_kwargs=resolved_env_kwargs,
        )

    cluster_count = int(resolved_env_kwargs.get("num_regime_clusters", fine_regime_clusters))
    catalog = load_or_build_catalog(
        env_key=env_key,
        env_factory=env_factory,
        n_clusters=cluster_count,
        episodes=int(resolved_env_kwargs.get("catalog_episodes", catalog_episodes)),
        context_steps=int(resolved_env_kwargs.get("context_catalog_steps", context_steps)),
        seed=0,
        cache_signature=catalog_cache_signature(
            {
                "protocol": "ood",
                "profile": profile,
                "env_key": env_key,
                "env_name": env_name,
                "env_kwargs": resolved_env_kwargs,
                "episodes": int(resolved_env_kwargs.get("catalog_episodes", catalog_episodes)),
                "context_steps": int(resolved_env_kwargs.get("context_catalog_steps", context_steps)),
                "clusters": cluster_count,
            }
        ),
    )

    target_regime_count = int(shared_regime_eval_episodes or len(catalog.labels) or cluster_count)
    target_regime_ids = sorted(int(idx) for idx in catalog.labels)[: min(target_regime_count, len(catalog.labels))]
    start_seed = build_shared_episode_seeds(
        env_name,
        base_seed=int(seed),
        num_episodes=1,
        seed_offset=int(shared_regime_seed_offset),
    )[0]
    plan = load_or_build_regime_seed_plan(
        env_key=env_key,
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(start_seed),
        seeds_per_regime=1,
        context_steps=int(resolved_env_kwargs.get("context_catalog_steps", context_steps)),
        max_search_episodes=max(256, 64 * max(1, len(target_regime_ids))),
        target_regime_ids=target_regime_ids,
        cache_signature=catalog_cache_signature(
            {
                "protocol": "ood_plan",
                "profile": profile,
                "env_key": env_key,
                "env_name": env_name,
                "env_kwargs": resolved_env_kwargs,
                "target_regime_ids": target_regime_ids,
                "shared_regime_eval_episodes": int(shared_regime_eval_episodes),
                "shared_regime_seed_offset": int(shared_regime_seed_offset),
                "search_start_seed": int(start_seed),
            }
        ),
    )
    return resolved_env_kwargs, plan, selected_episode_seeds_from_plan(plan)
