#!/usr/bin/env python
"""Run the LCPO baseline on a dynamic MORL environment.

LCPO is a single-objective online RL method.  To obtain a MORL front under the
existing benchmark protocol, this entry point trains one LCPO instance for each
standard scalarization preference, then evaluates those policies together on
the shared 20-regime plan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_envs import get_env_spec, make_scalar_env
from src.baseline_eval import evaluate_archive_policies, save_baseline_summary, save_front_csv
from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan
from src.lcpo_baseline import LCPOTrainer
from src.ood_protocol import load_env_config_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=256, help="Short smoke-training budget per preference.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--action-bins", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--trpo-kl-in", type=float, default=0.1)
    parser.add_argument("--trpo-kl-out", type=float, default=1e-3)
    parser.add_argument("--trpo-damping", type=float, default=0.1)
    parser.add_argument("--trpo-dual", action="store_true")
    parser.add_argument("--cg-iterations", type=int, default=5)
    parser.add_argument("--line-search-backtracks", type=int, default=5)
    parser.add_argument("--ood-capacity", type=int, default=4096)
    parser.add_argument("--ood-recent-window", type=int, default=32)
    parser.add_argument("--ood-threshold", type=float, default=2.0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--trace-eval-episodes", type=int, default=1)
    parser.add_argument("--trace-recovery-window", type=int, default=12)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=20)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--shared-plan-json", type=Path, required=True)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument(
        "--train-env-config-json",
        type=Path,
        default=None,
        help="Optional ID training environment config. Evaluation still uses --env-config-json.",
    )
    parser.add_argument("--env-config-json", type=Path, default=None)
    parser.add_argument("--max-preferences", type=int, default=0, help="0 keeps the complete standard preference grid.")
    parser.add_argument(
        "--preference-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional original preference-grid IDs to train or evaluate; used for parallel evaluation shards.",
    )
    parser.add_argument(
        "--load-policy-dir",
        type=Path,
        default=None,
        help="Load policy_XX.pt checkpoints from this directory and run evaluation without retraining.",
    )
    parser.add_argument("--save-dir", type=Path, required=True)
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _env_kwargs(args: argparse.Namespace, env_config_json: Path | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "episodes_per_regime": int(args.episodes_per_regime),
        "schedule": str(args.regime_schedule),
        "num_regime_clusters": int(args.fine_regime_clusters),
        "catalog_episodes": int(args.fine_regime_catalog_episodes),
        "context_catalog_steps": int(args.fine_regime_context_steps),
    }
    if args.env_key == "chlor_alkali":
        kwargs["episode_length"] = int(args.chlor_episode_length)
    if env_config_json is not None:
        kwargs.update(load_env_config_json(env_config_json))
    return kwargs


def _train_one_preference(
    *,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    preference: list[float],
    preference_id: int,
) -> tuple[LCPOTrainer, dict[str, Any]]:
    local_seed = int(args.seed) + int(preference_id)
    _seed_everything(local_seed)
    env = make_scalar_env(args.env_key, seed=local_seed, env_kwargs=env_kwargs)
    try:
        observation = np.asarray(env.reset(), dtype=np.float32)
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
        context_dim = len(observation)
        trainer = LCPOTrainer(
            obs_dim=len(observation),
            action_low=action_low,
            action_high=action_high,
            context_dim=context_dim,
            action_bins=args.action_bins,
            hidden_sizes=(args.hidden_size, args.hidden_size),
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            trpo_kl_in=args.trpo_kl_in,
            trpo_kl_out=args.trpo_kl_out,
            trpo_damping=args.trpo_damping,
            trpo_dual=args.trpo_dual,
            cg_iterations=args.cg_iterations,
            line_search_backtracks=args.line_search_backtracks,
            ood_capacity=args.ood_capacity,
            ood_recent_window=args.ood_recent_window,
            ood_threshold=args.ood_threshold,
            seed=local_seed,
        )
        current_context = observation.copy()
        transition_count = 0
        update_rows: list[dict[str, Any]] = []
        while transition_count < int(args.total_timesteps):
            states: list[np.ndarray] = []
            next_states: list[np.ndarray] = []
            action_indices: list[np.ndarray] = []
            rewards: list[float] = []
            terminated: list[bool] = []
            truncated: list[bool] = []
            contexts: list[np.ndarray] = []
            target_batch = min(int(args.batch_size), int(args.total_timesteps) - transition_count)
            while len(states) < target_batch:
                indices, action = trainer.sample_action(observation)
                next_observation, _reward, done, info = env.step(action)
                next_observation = np.asarray(next_observation, dtype=np.float32)
                info = dict(info)
                objective = np.asarray(info.get("obj_raw", info.get("obj")), dtype=np.float32)
                states.append(observation.copy())
                next_states.append(next_observation)
                action_indices.append(indices)
                rewards.append(float(np.dot(objective, np.asarray(preference, dtype=np.float32))))
                terminated.append(bool(done and not info.get("TimeLimit.truncated", False)))
                truncated.append(bool(done and info.get("TimeLimit.truncated", False)))
                contexts.append(current_context.copy())
                current_context = next_observation.copy()
                observation = next_observation
                transition_count += 1
                if done:
                    observation = np.asarray(env.reset(), dtype=np.float32)
                    current_context = observation.copy()
            stats = trainer.update(
                observations=np.asarray(states, dtype=np.float32),
                next_observations=np.asarray(next_states, dtype=np.float32),
                action_indices=np.asarray(action_indices, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=bool),
                truncated=np.asarray(truncated, dtype=bool),
                contexts=np.asarray(contexts, dtype=np.float32),
            )
            update_rows.append(
                {
                    "policy_loss": stats.policy_loss,
                    "value_loss": stats.value_loss,
                    "entropy": stats.entropy,
                    "local_kl": stats.local_kl,
                    "ood_kl": stats.ood_kl,
                    "used_lcpo_constraint": stats.used_lcpo_constraint,
                    "ood_batch_size": stats.ood_batch_size,
                    "accepted_step": stats.accepted_step,
                }
            )
        constraint_rows = [row for row in update_rows if row["used_lcpo_constraint"]]
        train_stats = {
            "preference_id": int(preference_id),
            "preference": [float(value) for value in preference],
            "seed": local_seed,
            "timesteps": transition_count,
            "updates": len(update_rows),
            "lcpo_updates": trainer.lcpo_updates,
            "fallback_updates": trainer.fallback_updates,
            "constraint_accept_rate": float(np.mean([row["accepted_step"] for row in constraint_rows])) if constraint_rows else 0.0,
            "mean_ood_batch_size": float(np.mean([row["ood_batch_size"] for row in update_rows])) if update_rows else 0.0,
            "mean_ood_kl": float(np.mean([row["ood_kl"] for row in constraint_rows])) if constraint_rows else 0.0,
            "mean_local_kl": float(np.mean([row["local_kl"] for row in constraint_rows])) if constraint_rows else 0.0,
            "updates_detail": update_rows,
        }
        return trainer, train_stats
    finally:
        env.close()


def _load_one_preference(
    *,
    args: argparse.Namespace,
    env_kwargs: dict[str, Any],
    preference_id: int,
) -> LCPOTrainer:
    """Reconstruct a trainer and restore one completed policy for evaluation."""

    if args.load_policy_dir is None:
        raise ValueError("A checkpoint directory is required when loading an LCPO policy.")
    checkpoint_path = args.load_policy_dir / f"policy_{preference_id:02d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing LCPO checkpoint: {checkpoint_path}")
    local_seed = int(args.seed) + int(preference_id)
    _seed_everything(local_seed)
    env = make_scalar_env(args.env_key, seed=local_seed, env_kwargs=env_kwargs)
    try:
        observation = np.asarray(env.reset(), dtype=np.float32)
        trainer = LCPOTrainer(
            obs_dim=len(observation),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
            context_dim=len(observation),
            action_bins=args.action_bins,
            hidden_sizes=(args.hidden_size, args.hidden_size),
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            trpo_kl_in=args.trpo_kl_in,
            trpo_kl_out=args.trpo_kl_out,
            trpo_damping=args.trpo_damping,
            trpo_dual=args.trpo_dual,
            cg_iterations=args.cg_iterations,
            line_search_backtracks=args.line_search_backtracks,
            ood_capacity=args.ood_capacity,
            ood_recent_window=args.ood_recent_window,
            ood_threshold=args.ood_threshold,
            seed=local_seed,
        )
        trainer.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return trainer
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.save_dir = args.save_dir.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    spec = get_env_spec(args.env_key)
    train_env_config_json = args.train_env_config_json or args.env_config_json
    train_env_kwargs = _env_kwargs(args, train_env_config_json)
    eval_env_kwargs = _env_kwargs(args, args.env_config_json)
    shared_plan = coerce_regime_seed_plan(args.shared_plan_json)
    if len(shared_plan) != int(args.shared_regime_eval_episodes):
        raise ValueError(
            f"Expected {args.shared_regime_eval_episodes} shared regimes, found {len(shared_plan)} in {args.shared_plan_json}."
        )
    preferences = [list(weights) for weights in spec.preference_grid]
    if args.max_preferences > 0:
        preferences = preferences[: int(args.max_preferences)]
    selected_preferences = list(enumerate(preferences))
    if args.preference_ids is not None:
        requested_ids = [int(preference_id) for preference_id in args.preference_ids]
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("--preference-ids must not contain duplicates.")
        invalid_ids = [preference_id for preference_id in requested_ids if not 0 <= preference_id < len(preferences)]
        if invalid_ids:
            raise ValueError(f"Invalid preference IDs for {args.env_key}: {invalid_ids}")
        selected_preferences = [(preference_id, preferences[preference_id]) for preference_id in requested_ids]
    if not selected_preferences:
        raise ValueError("No scalarization preferences selected.")
    shutil.rmtree(args.save_dir / "shared_regime_returns", ignore_errors=True)
    policy_fns = []
    train_stats = []
    for preference_id, preference in selected_preferences:
        print(
            f"[lcpo] env={args.env_key} preference={preference_id + 1}/{len(preferences)} {preference}",
            flush=True,
        )
        if args.load_policy_dir is None:
            trainer, stats = _train_one_preference(
                args=args,
                env_kwargs=train_env_kwargs,
                preference=preference,
                preference_id=preference_id,
            )
            torch.save(trainer.state_dict(), args.save_dir / f"policy_{preference_id:02d}.pt")
            train_stats.append(stats)
        else:
            trainer = _load_one_preference(
                args=args,
                env_kwargs=train_env_kwargs,
                preference_id=preference_id,
            )

        def _policy_fn(observation, trainer=trainer):
            return trainer.deterministic_action(observation)

        policy_fns.append(_policy_fn)
    print(f"[lcpo] evaluating {len(policy_fns)} policies on {len(shared_plan)} shared regimes", flush=True)
    eval_summary = evaluate_archive_policies(
        env_name=spec.dynamic_env_name,
        obj_num=spec.obj_num,
        seed=args.seed,
        policy_fns=policy_fns,
        env_kwargs=eval_env_kwargs,
        eval_episodes=args.eval_episodes,
        trace_eval_episodes=args.trace_eval_episodes,
        eval_delta_weight=args.eval_delta_weight,
        trace_recovery_window=args.trace_recovery_window,
        shared_regime_eval_episodes=args.shared_regime_eval_episodes,
        shared_regime_seed_offset=args.shared_regime_seed_offset,
        shared_regime_seed_plan=shared_plan,
        regime_output_dir=args.save_dir / "shared_regime_returns",
    )
    summary: dict[str, Any] = {
        "method": "LCPO",
        "env_key": args.env_key,
        "env_name": spec.dynamic_env_name,
        "seed": int(args.seed),
        "total_timesteps_per_preference": int(args.total_timesteps),
        "total_training_timesteps": int(args.total_timesteps) * len(selected_preferences),
        "policy_source": "checkpoint" if args.load_policy_dir is not None else "trained",
        "loaded_policy_dir": str(args.load_policy_dir) if args.load_policy_dir is not None else None,
        "obj_num": int(spec.obj_num),
        "preferences": [preference for _preference_id, preference in selected_preferences],
        "preference_ids": [int(preference_id) for preference_id, _preference in selected_preferences],
        "env_kwargs": eval_env_kwargs,
        "train_env_kwargs": train_env_kwargs,
        "train_env_config_json": str(train_env_config_json) if train_env_config_json else None,
        "eval_env_config_json": str(args.env_config_json) if args.env_config_json else None,
        "lcpo_config": {
            "source": "LCPO/windy-gym/agent/core_alg/core_lcpo.py",
            "action_representation": "factorized per-dimension categorical action bins",
            "action_bins": int(args.action_bins),
            "context_source": "online observation (including dynamic factors); no regime label",
            "trpo_kl_in": float(args.trpo_kl_in),
            "trpo_kl_out": float(args.trpo_kl_out),
            "trpo_damping": float(args.trpo_damping),
            "trpo_dual": bool(args.trpo_dual),
            "ood_recent_window": int(args.ood_recent_window),
            "ood_threshold": float(args.ood_threshold),
        },
        "training": train_stats,
        **eval_summary,
    }
    save_baseline_summary(args.save_dir / "summary.json", summary)
    save_front_csv(args.save_dir / "front_points.csv", summary["front_points"])
    (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(
        json.dumps(
            {
                "method": "LCPO",
                "env_key": args.env_key,
                "front_size": len(summary["front_points"]),
                "lcpo_updates": sum(item["lcpo_updates"] for item in train_stats),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
