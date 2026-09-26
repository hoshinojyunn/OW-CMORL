#!/usr/bin/env python3
"""Build reproducible 100-condition ID shared plans for lightweight LCPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_envs import build_dynamic_env, default_env_kwargs, get_env_spec
from src.dynamic_morl.fine_regimes import (
    catalog_cache_signature,
    load_or_build_catalog,
    load_or_build_regime_seed_plan,
    selected_episode_seeds_from_plan,
)
from src.dynamic_morl.shared_eval import build_shared_episode_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-keys",
        nargs="+",
        default=["building", "evcharging", "cogen", "chlor_alkali"],
        choices=["building", "evcharging", "cogen", "chlor_alkali"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditions", type=int, default=100)
    parser.add_argument("--catalog-episodes", type=int, default=256)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "lcpo_lightweight_100",
    )
    return parser.parse_args()


def build_plan(env_key: str, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    spec = get_env_spec(env_key)
    env_kwargs = dict(default_env_kwargs(env_key))
    env_kwargs.update(
        {
            "schedule": "cyclic",
            "episodes_per_regime": 1,
            "num_regime_clusters": int(args.conditions),
            "catalog_episodes": int(args.catalog_episodes),
            "context_catalog_steps": int(args.context_steps),
        }
    )

    def env_factory(factory_seed: int):
        return build_dynamic_env(
            spec.dynamic_env_name,
            seed=int(args.seed) + int(factory_seed),
            env_kwargs=env_kwargs,
        )

    signature = catalog_cache_signature(
        {
            "protocol": "lcpo_lightweight_shared",
            "env_key": env_key,
            "env_name": spec.dynamic_env_name,
            "env_kwargs": env_kwargs,
            "conditions": int(args.conditions),
            "catalog_episodes": int(args.catalog_episodes),
            "context_steps": int(args.context_steps),
        }
    )
    catalog = load_or_build_catalog(
        env_key=env_key,
        env_factory=env_factory,
        n_clusters=int(args.conditions),
        episodes=int(args.catalog_episodes),
        context_steps=int(args.context_steps),
        seed=int(args.seed),
        cache_signature=signature,
    )
    target_ids = sorted(int(regime_id) for regime_id in catalog.labels)[: int(args.conditions)]
    if len(target_ids) != int(args.conditions):
        raise RuntimeError(f"{env_key}: catalog exposes {len(target_ids)} of {args.conditions} conditions.")
    start_seed = build_shared_episode_seeds(
        spec.dynamic_env_name,
        base_seed=int(args.seed),
        num_episodes=1,
        seed_offset=0,
    )[0]
    plan = load_or_build_regime_seed_plan(
        env_key=env_key,
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(start_seed),
        seeds_per_regime=1,
        context_steps=int(args.context_steps),
        max_search_episodes=max(256, 64 * len(target_ids)),
        target_regime_ids=target_ids,
        cache_signature=signature,
    )
    if len(plan) != int(args.conditions):
        raise RuntimeError(f"{env_key}: generated {len(plan)} conditions, expected {args.conditions}.")
    return env_kwargs, plan


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for env_key in args.env_keys:
        env_kwargs, plan = build_plan(env_key, args)
        payload = {
            "env_key": env_key,
            "seed": int(args.seed),
            "conditions": int(args.conditions),
            "env_kwargs": env_kwargs,
            "shared_regime_seed_plan": plan,
            "shared_eval_episode_seeds": selected_episode_seeds_from_plan(plan),
        }
        path = args.out_dir / f"{env_key}_shared_plan.json"
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"env_key": env_key, "plan_path": str(path), "conditions": len(plan)}))


if __name__ == "__main__":
    main()
