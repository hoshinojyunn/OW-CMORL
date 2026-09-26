from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ood_protocol import (
    DEFAULT_OOD_PROFILE,
    available_ood_profiles,
    build_ood_shared_plan,
    resolve_ood_eval_env_config,
    resolve_ood_train_env_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build shared OOD evaluation plans and env configs.")
    parser.add_argument(
        "--env-keys",
        nargs="+",
        default=["building", "evcharging", "cogen", "chlor_alkali"],
        choices=["building", "evcharging", "cogen", "chlor_alkali"],
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_OOD_PROFILE,
        choices=list(available_ood_profiles()),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=20)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument(
        "--reuse-unchanged-eval-plans-from",
        type=str,
        default=None,
        help="Reuse a prior profile's plan only when the fully resolved evaluation kwargs are identical.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_protocols",
    )
    return parser.parse_args()


def _resolved_eval_kwargs(env_key: str, profile: str, args: argparse.Namespace) -> dict[str, object]:
    env_kwargs = resolve_ood_eval_env_config(env_key, profile=profile)
    env_kwargs.setdefault("catalog_episodes", int(args.fine_regime_catalog_episodes))
    env_kwargs.setdefault("context_catalog_steps", int(args.fine_regime_context_steps))
    env_kwargs.setdefault("num_regime_clusters", int(args.fine_regime_clusters))
    return env_kwargs


def main() -> None:
    args = parse_args()
    for env_key in args.env_keys:
        env_kwargs = _resolved_eval_kwargs(env_key, args.profile, args)
        plan: list[dict[str, object]] | None = None
        episode_seeds: list[int] | None = None
        reused_from: str | None = None
        if args.reuse_unchanged_eval_plans_from:
            source_dir = args.out_dir / args.reuse_unchanged_eval_plans_from / env_key
            source_config_path = source_dir / "eval_env_config.json"
            source_plan_path = source_dir / "shared_plan.json"
            source_seeds_path = source_dir / "episode_seeds.json"
            if source_config_path.exists() and source_plan_path.exists() and source_seeds_path.exists():
                source_config = json.loads(source_config_path.read_text())
                source_kwargs = source_config.get("env_kwargs", {})
                source_plan = json.loads(source_plan_path.read_text()).get("shared_regime_seed_plan", [])
                source_seeds = json.loads(source_seeds_path.read_text()).get("shared_eval_episode_seeds", [])
                if (
                    source_kwargs == env_kwargs
                    and isinstance(source_plan, list)
                    and isinstance(source_seeds, list)
                    and len(source_plan) == int(args.shared_regime_eval_episodes)
                    and len(source_seeds) == int(args.shared_regime_eval_episodes)
                ):
                    plan = source_plan
                    episode_seeds = [int(seed) for seed in source_seeds]
                    reused_from = str(source_dir)
        if plan is None or episode_seeds is None:
            env_kwargs, plan, episode_seeds = build_ood_shared_plan(
                env_key,
                seed=int(args.seed),
                profile=args.profile,
                shared_regime_eval_episodes=int(args.shared_regime_eval_episodes),
                shared_regime_seed_offset=int(args.shared_regime_seed_offset),
                fine_regime_clusters=int(args.fine_regime_clusters),
                catalog_episodes=int(args.fine_regime_catalog_episodes),
                context_steps=int(args.fine_regime_context_steps),
            )
        env_dir = args.out_dir / args.profile / env_key
        env_dir.mkdir(parents=True, exist_ok=True)
        train_env_payload = {
            "profile": args.profile,
            "env_key": env_key,
            "env_kwargs": resolve_ood_train_env_config(env_key, profile=args.profile),
        }
        env_payload = {
            "profile": args.profile,
            "env_key": env_key,
            "env_kwargs": env_kwargs,
            "shared_regime_eval_episodes": int(len(episode_seeds)),
            "shared_regime_seed_offset": int(args.shared_regime_seed_offset),
            "fine_regime_clusters": int(args.fine_regime_clusters),
            "fine_regime_catalog_episodes": int(args.fine_regime_catalog_episodes),
            "fine_regime_context_steps": int(args.fine_regime_context_steps),
            "reused_unchanged_eval_plan_from": reused_from,
        }
        (env_dir / "train_env_config.json").write_text(json.dumps(train_env_payload, indent=2))
        (env_dir / "eval_env_config.json").write_text(json.dumps(env_payload, indent=2))
        (env_dir / "env_config.json").write_text(json.dumps(env_payload, indent=2))
        (env_dir / "shared_plan.json").write_text(
            json.dumps({"shared_regime_seed_plan": plan}, indent=2)
        )
        (env_dir / "episode_seeds.json").write_text(
            json.dumps({"shared_eval_episode_seeds": [int(seed) for seed in episode_seeds]}, indent=2)
        )
        print(
            json.dumps(
                {
                    "env_key": env_key,
                    "profile": args.profile,
                    "plan_path": str(env_dir / "shared_plan.json"),
                    "train_env_config_path": str(env_dir / "train_env_config.json"),
                    "eval_env_config_path": str(env_dir / "eval_env_config.json"),
                    "regimes": len(plan),
                    "reused_unchanged_eval_plan_from": reused_from,
                }
            )
        )


if __name__ == "__main__":
    main()
