from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "CAPQL"))

from src.baseline_envs import get_env_spec, make_vector_env, register_baseline_envs
from src.baseline_eval import evaluate_conditioned_policy, save_baseline_summary, save_front_csv
from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan
from src.ood_protocol import load_env_config_json


def install_capql_stubs() -> None:
    if "hydra" not in sys.modules:
        hydra_module = types.ModuleType("hydra")
        sys.modules["hydra"] = hydra_module
    if "omegaconf" not in sys.modules:
        omegaconf_module = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def to_yaml(value):
                return str(value)

        omegaconf_module.OmegaConf = _OmegaConf
        sys.modules["omegaconf"] = omegaconf_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--angle", type=float, default=22.5)
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


def build_config(args: argparse.Namespace, env_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(type="CAPQL", hidden_size=args.hidden_size),
        training=SimpleNamespace(
            eval=False,
            gamma=0.99,
            tau=0.005,
            lr=3e-4,
            alpha=0.2,
            seed=args.seed,
            batch_size=args.batch_size,
            num_steps=int(args.total_timesteps),
            updates_per_step=int(args.updates_per_step),
            start_steps=min(int(args.start_steps), max(128, int(args.total_timesteps // 5))),
            delta=0.1,
            target_update_interval=1,
            replay_size=max(50000, int(args.total_timesteps * 2)),
        ),
        weight_sampler=SimpleNamespace(angle=float(args.angle)),
        gpu=SimpleNamespace(cuda=False),
        name=env_id,
    )


def infer_hidden_size_from_checkpoint(model_dir: Path, default_hidden_size: int) -> int:
    policy_path = model_dir / "policy.pt"
    if not policy_path.exists():
        return int(default_hidden_size)
    state_dict = torch.load(policy_path, map_location="cpu")
    weight = state_dict.get("linear1.weight")
    if weight is None or len(weight.shape) != 2:
        return int(default_hidden_size)
    return int(weight.shape[0])


def main() -> None:
    args = parse_args()
    args.save_dir = args.save_dir.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print("[capql] start")
    register_baseline_envs()
    install_capql_stubs()
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
    print("[capql] env ready")

    from training_pipeline import TrainingProcess

    config = build_config(args, spec.vector_env_id)
    train_env = make_vector_env(args.env_key, seed=args.seed, env_kwargs=env_kwargs)
    print("[capql] trainer init")

    trainer = TrainingProcess(config, train_env)
    trainer.writer.close()
    trainer.dir_checkpoint = str(args.save_dir / "runs" / spec.vector_env_id / f"seed{args.seed}")
    Path(trainer.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    trainer.writer = SummaryWriter(trainer.dir_checkpoint)
    model_dir = args.save_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_train:
        print("[capql] loading checkpoints")
        inferred_hidden_size = infer_hidden_size_from_checkpoint(model_dir, args.hidden_size)
        if inferred_hidden_size != int(args.hidden_size):
            args.hidden_size = inferred_hidden_size
            config = build_config(args, spec.vector_env_id)
            trainer = TrainingProcess(config, train_env)
            trainer.writer.close()
            trainer.dir_checkpoint = str(args.save_dir / "runs" / spec.vector_env_id / f"seed{args.seed}")
            Path(trainer.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            trainer.writer = SummaryWriter(trainer.dir_checkpoint)
    if not args.skip_train:
        trainer()
        torch.save(trainer.agent.policy.state_dict(), model_dir / "policy.pt")
        torch.save(trainer.agent.critic.state_dict(), model_dir / "critic.pt")
    else:
        trainer.agent.policy.load_state_dict(torch.load(model_dir / "policy.pt", map_location="cpu"))
        trainer.agent.critic.load_state_dict(torch.load(model_dir / "critic.pt", map_location="cpu"))
    if args.train_only:
        (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print(json.dumps({"method": "CAPQL", "env_key": args.env_key, "stage": "train_only"}, indent=2))
        return
    print("[capql] evaluate")

    eval_summary = evaluate_conditioned_policy(
        env_name=spec.dynamic_env_name,
        obj_num=spec.obj_num,
        seed=args.seed,
        policy_fn=lambda obs, pref: trainer.agent.select_action(obs, pref, evaluate=True),
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
    print("[capql] save")
    summary = {
        "method": "CAPQL",
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
