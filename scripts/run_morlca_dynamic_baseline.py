from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.baseline_envs import get_env_spec, make_vector_env, register_baseline_envs
from src.baseline_eval import evaluate_conditioned_policy, save_baseline_summary, save_front_csv
from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan
from src.ood_protocol import load_env_config_json
from src.morl_ca_baseline import MORLCABaseline, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-init", type=float, default=0.2)
    parser.add_argument("--aow-aux-coef", type=float, default=0.2)
    parser.add_argument("--eval-pref-bias-scale", type=float, default=0.75)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--trace-eval-episodes", type=int, default=3)
    parser.add_argument("--trace-recovery-window", type=int, default=12)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--shared-plan-json", type=Path, default=None)
    parser.add_argument("--env-config-json", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--save-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.save_dir = args.save_dir.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print("[morlca] start")
    register_baseline_envs()
    set_global_seed(args.seed)
    torch.set_num_threads(1)

    spec = get_env_spec(args.env_key)
    env_kwargs = {
        "episodes_per_regime": int(args.episodes_per_regime),
        "schedule": str(args.regime_schedule),
        "num_regime_clusters": int(args.fine_regime_clusters),
        "catalog_episodes": int(args.fine_regime_catalog_episodes),
        "context_catalog_steps": int(args.fine_regime_context_steps),
    }
    if args.env_key == "chlor_alkali":
        env_kwargs["episode_length"] = int(args.chlor_episode_length)
    if args.env_config_json is not None:
        env_kwargs.update(load_env_config_json(args.env_config_json))
    shared_plan = coerce_regime_seed_plan(args.shared_plan_json) if args.shared_plan_json else None

    shutil.rmtree(args.save_dir / "shared_regime_returns", ignore_errors=True)
    print("[morlca] env ready")
    train_env = make_vector_env(args.env_key, seed=args.seed, env_kwargs=env_kwargs)
    model = MORLCABaseline(
        obs_dim=int(train_env.observation_space.shape[0]),
        action_low=train_env.action_space.low,
        action_high=train_env.action_space.high,
        obj_num=int(spec.obj_num),
        hidden_dim=int(args.hidden_size),
        gamma=float(args.gamma),
        tau=float(args.tau),
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        alpha_lr=float(args.alpha_lr),
        alpha_init=float(args.alpha_init),
        batch_size=int(args.batch_size),
        replay_size=max(50000, int(args.total_timesteps * 2)),
        start_steps=min(int(args.start_steps), max(128, int(args.total_timesteps // 5))),
        updates_per_step=int(args.updates_per_step),
        aow_aux_coef=float(args.aow_aux_coef),
        eval_pref_bias_scale=float(args.eval_pref_bias_scale),
        device="cpu",
    )

    model_path = args.save_dir / "model" / "morlca.pt"
    if not args.skip_train:
        train_stats = model.train_on_env(
            train_env,
            total_timesteps=int(args.total_timesteps),
            seed=int(args.seed),
        )
        model.save(model_path)
        (args.save_dir / "train_stats.json").write_text(json.dumps(train_stats.__dict__, indent=2))
    else:
        print("[morlca] loading checkpoints")
        model.load(model_path)
    train_env.close()
    if args.train_only:
        (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print(json.dumps({"method": "MORL-CA", "env_key": args.env_key, "stage": "train_only"}, indent=2))
        return

    print("[morlca] evaluate")
    eval_summary = evaluate_conditioned_policy(
        env_name=spec.dynamic_env_name,
        obj_num=spec.obj_num,
        seed=args.seed,
        policy_fn=lambda obs, pref: model.act(obs, pref=np.asarray(pref, dtype=np.float32), deterministic=True),
        preferences=[list(pref) for pref in spec.preference_grid],
        env_kwargs=env_kwargs,
        eval_episodes=args.eval_episodes,
        trace_eval_episodes=args.trace_eval_episodes,
        eval_delta_weight=args.eval_delta_weight,
        trace_recovery_window=args.trace_recovery_window,
        shared_regime_eval_episodes=args.shared_regime_eval_episodes,
        shared_regime_seed_offset=args.shared_regime_seed_offset,
        shared_regime_seed_plan=shared_plan,
        regime_output_dir=args.save_dir / "shared_regime_returns",
    )
    print("[morlca] save")
    summary = {
        "method": "MORL-CA",
        "env_key": args.env_key,
        "env_name": spec.dynamic_env_name,
        "seed": args.seed,
        "total_timesteps": int(args.total_timesteps),
        "obj_num": spec.obj_num,
        "env_kwargs": env_kwargs,
        "preferences": [list(pref) for pref in spec.preference_grid],
        **eval_summary,
    }
    save_baseline_summary(args.save_dir / "summary.json", summary)
    save_front_csv(args.save_dir / "front_points.csv", summary["front_points"])
    (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    print(json.dumps({"method": summary["method"], "env_key": summary["env_key"], "front_size": len(summary["front_points"])}, indent=2))


if __name__ == "__main__":
    main()
