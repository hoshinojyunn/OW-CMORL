from __future__ import annotations

from typing import Any

from .chlor_alkali_env import (
    DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    chlor_alkali_regime_catalog,
)
from .shared_eval import build_shared_episode_seeds


def canonical_chlor_regime_rows(
    *,
    num_regime_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    dynamic_price: bool = True,
    seed: int = 0,
    train_path: str | None = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "dynamic_price": bool(dynamic_price),
        "seed": int(seed),
        "n_clusters": int(num_regime_clusters),
    }
    if train_path is not None:
        kwargs["train_path"] = train_path
    rows = chlor_alkali_regime_catalog(**kwargs)
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = {
            str(key): value
            for key, value in row.items()
            if key not in {"regime_id", "regime_name", "count"}
        }
        out.append(
            {
                "regime": str(row["regime_name"]),
                "regime_id": int(row["regime_id"]),
                "regime_meta": meta,
            }
        )
    out.sort(key=lambda row: int(row["regime_id"]))
    return out


def canonical_chlor_regime_seed_plan(
    *,
    env_name: str = "chlor_alkali_dynamic",
    base_seed: int = 0,
    shared_regime_eval_episodes: int = 0,
    seed_offset: int = 0,
    num_regime_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    dynamic_price: bool = True,
    train_path: str | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows = canonical_chlor_regime_rows(
        num_regime_clusters=int(num_regime_clusters),
        dynamic_price=bool(dynamic_price),
        seed=0,
        train_path=train_path,
    )
    target_count = int(shared_regime_eval_episodes or len(rows) or num_regime_clusters)
    selected = rows[: min(target_count, len(rows))]
    shared_episode_seeds = build_shared_episode_seeds(
        env_name,
        base_seed=int(base_seed),
        num_episodes=max(1, len(selected)),
        seed_offset=int(seed_offset),
    )[: len(selected)]
    plan: list[dict[str, Any]] = []
    for sequence_index, (row, episode_seed) in enumerate(zip(selected, shared_episode_seeds)):
        plan.append(
            {
                "regime": str(row["regime"]),
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row["regime_meta"]),
                "episode_seeds": [int(episode_seed)],
                "episode_seed_records": [
                    {
                        "episode_seed": int(episode_seed),
                        "sequence_index": int(sequence_index),
                        "probe_index": int(sequence_index),
                    }
                ],
            }
        )
    return plan, [int(seed) for seed in shared_episode_seeds]


def chlor_regime_eval_env_kwargs(
    *,
    regime_name: str,
    regime_id: int | None = None,
    episode_length: int,
    num_regime_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    schedule: str = "random",
    dynamic_price: bool = True,
    aging_dynamics: bool = True,
) -> dict[str, Any]:
    kwargs = {
        "allowed_regimes": (str(regime_name),),
        "episode_length": int(episode_length),
        "schedule": str(schedule),
        "dynamic_price": bool(dynamic_price),
        "aging_dynamics": bool(aging_dynamics),
        "num_regime_clusters": int(num_regime_clusters),
    }
    if regime_id is not None:
        kwargs["allowed_regime_ids"] = (int(regime_id),)
    return kwargs
