from __future__ import annotations

from typing import Iterable


DEFAULT_SHARED_EVAL_EPISODES = 20
ENV_SHARED_EVAL_EPISODES = {
    "building": 20,
    "evcharging": 20,
    "cogen": 20,
    "chlor_alkali": 20,
}
ENV_SHARED_SEED_OFFSETS = {
    "building": 11000,
    "evcharging": 21000,
    "cogen": 31000,
    "chlor_alkali": 41000,
}


def env_key_from_name(env_name: str) -> str:
    name = str(env_name).strip().lower()
    if name.startswith("building_") or name == "sustaingym_building_dynamic":
        return "building"
    if name == "evcharging_dynamic":
        return "evcharging"
    if name == "cogen_dynamic":
        return "cogen"
    if name == "chlor_alkali_dynamic":
        return "chlor_alkali"
    return name


def default_shared_eval_episodes(env_name: str) -> int:
    return int(ENV_SHARED_EVAL_EPISODES.get(env_key_from_name(env_name), DEFAULT_SHARED_EVAL_EPISODES))


def resolve_shared_eval_episodes(
    env_name: str,
    requested_episodes: int | None = None,
    fallback_episodes: int | None = None,
) -> int:
    requested = int(requested_episodes or 0)
    if requested > 0:
        return requested
    fallback = int(fallback_episodes or 0)
    if fallback > 0:
        return fallback
    return default_shared_eval_episodes(env_name)


def build_shared_episode_seeds(
    env_name: str,
    *,
    base_seed: int = 0,
    num_episodes: int | None = None,
    seed_offset: int = 0,
) -> list[int]:
    env_key = env_key_from_name(env_name)
    count = resolve_shared_eval_episodes(env_name, requested_episodes=num_episodes)
    offset = int(ENV_SHARED_SEED_OFFSETS.get(env_key, 51000)) + int(seed_offset)
    start = int(base_seed) + offset
    return [start + idx for idx in range(max(1, count))]


def normalize_episode_seeds(
    env_name: str,
    *,
    base_seed: int = 0,
    episode_seeds: Iterable[int] | None = None,
    num_episodes: int | None = None,
    seed_offset: int = 0,
) -> list[int]:
    if episode_seeds is not None:
        seeds = [int(seed) for seed in episode_seeds]
        if seeds:
            return seeds
    return build_shared_episode_seeds(
        env_name,
        base_seed=base_seed,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
    )
