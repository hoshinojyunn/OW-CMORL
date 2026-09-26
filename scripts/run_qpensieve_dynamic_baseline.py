from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Q-Pensieve"))

from src.baseline_envs import get_env_spec, make_vector_env, register_baseline_envs
from src.baseline_eval import evaluate_conditioned_policy, save_baseline_summary, save_front_csv
from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan
from src.ood_protocol import load_env_config_json


def install_qpensieve_stubs() -> None:
    if "visdom" not in sys.modules:
        visdom_module = types.ModuleType("visdom")

        class _DummyVisdom:
            def __init__(self, *args, **kwargs):
                pass

            def line(self, *args, **kwargs):
                return None

        visdom_module.Visdom = _DummyVisdom
        sys.modules["visdom"] = visdom_module

    if "rltorch.memory" not in sys.modules:
        rltorch_module = types.ModuleType("rltorch")
        memory_module = types.ModuleType("rltorch.memory")
        network_module = types.ModuleType("rltorch.network")

        class _UnusedMemory:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("Prioritized rltorch memory should not be used in this runner.")

        def _create_linear_network(input_dim, output_dim, hidden_units=(256, 256), initializer="xavier"):
            import torch.nn as nn

            layers = []
            last_dim = int(input_dim)
            for hidden_dim in hidden_units:
                linear = nn.Linear(last_dim, int(hidden_dim))
                if initializer == "xavier":
                    nn.init.xavier_uniform_(linear.weight)
                    nn.init.zeros_(linear.bias)
                layers.extend([linear, nn.ReLU()])
                last_dim = int(hidden_dim)
            head = nn.Linear(last_dim, int(output_dim))
            if initializer == "xavier":
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
            layers.append(head)
            return nn.Sequential(*layers)

        memory_module.MultiStepMemory = _UnusedMemory
        memory_module.PrioritizedMemory = _UnusedMemory
        rltorch_module.memory = memory_module
        network_module.create_linear_network = _create_linear_network
        rltorch_module.network = network_module
        sys.modules["rltorch"] = rltorch_module
        sys.modules["rltorch.memory"] = memory_module
        sys.modules["rltorch.network"] = network_module

    original_np_load = np.load

    def _patched_np_load(file, *args, **kwargs):
        if isinstance(file, str) and file in {"3pref_table.npy", "4pref_table.npy", "5pref_table.npy"}:
            candidate = PROJECT_ROOT / "Q-Pensieve" / file
            if candidate.exists():
                file = str(candidate)
        return original_np_load(file, *args, **kwargs)

    np.load = _patched_np_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--start-steps", type=int, default=5000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--prefer-num", type=int, default=4)
    parser.add_argument("--q-frequency", type=int, default=1000)
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
    print("[qpensieve] start")
    register_baseline_envs()
    install_qpensieve_stubs()
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
    print("[qpensieve] env ready")

    import copy
    from agent import SacAgent
    from utils import soft_update, update_params

    def _patched_calc_policy_loss(self, batch, weights, preference, pref_set):
        states, _, actions, rewards, next_states, dones = batch
        preference_batch = preference.repeat(self.batch_size, 1)
        sampled_action, entropy, _ = self.policy.sample(states, preference_batch)
        entropy = entropy.reshape(-1)
        losses = []
        for critic in [self.critic] + self.Q_memory.sample():
            for pref_entry in pref_set:
                pref_batch = torch.as_tensor(pref_entry, device=self.device).repeat(self.batch_size, 1)
                q1, q2 = critic(states, sampled_action, pref_batch)
                q1 = torch.tensordot(q1, preference, dims=1)
                q2 = torch.tensordot(q2, preference, dims=1)
                q = torch.min(q1, q2)
                losses.append(-q - self.alpha * entropy)
        losses = torch.stack(losses, dim=1)
        policy_loss, _idx = torch.min(losses, 1)
        policy_loss = torch.mean(policy_loss)
        sampled_action, entropy, _ = self.policy.sample(states, preference_batch)
        return policy_loss, entropy

    def _patched_learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.learning_steps % self.q_frequency == 0 and self.learning_steps > 20000:
            self.Q_memory.append(copy.deepcopy(self.critic))

        if self.per:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            weights = 1.0

        preference = torch.as_tensor(self.get_pref(), device=self.device)
        pref_set = [preference]
        for _ in range(self.set_num - 1):
            pref_set.append(torch.as_tensor(self.get_pref(), device=self.device))

        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights, preference, pref_set)
        update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

        policy_loss, entropies = self.calc_policy_loss(batch, weights, preference, pref_set)
        update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        if self.per:
            self.memory.update_priority(indices, errors.cpu().numpy())
        self.QM.update(self.steps, self.cur_p, self.cur_e, self.qmem_p, self.qmem_e)

    SacAgent.calc_policy_loss = _patched_calc_policy_loss
    SacAgent.learn = _patched_learn

    env = make_vector_env(args.env_key, seed=args.seed, env_kwargs=env_kwargs)
    torch.set_num_threads(1)
    agent = SacAgent(
        env=env,
        log_dir=str(args.save_dir),
        num_steps=int(args.total_timesteps),
        batch_size=int(args.batch_size),
        lr=3e-4,
        hidden_units=list(args.hidden_size),
        memory_size=max(50000, int(args.total_timesteps * 2)),
        prefer_num=int(args.prefer_num),
        buf_num=0,
        gamma=0.99,
        tau=0.005,
        entropy_tuning=True,
        ent_coef=0.2,
        multi_step=1,
        per=False,
        grad_clip=None,
        updates_per_step=int(args.updates_per_step),
        start_steps=min(int(args.start_steps), max(128, int(args.total_timesteps // 5))),
        log_interval=10,
        target_update_interval=1,
        eval_interval=max(1000000, int(args.total_timesteps) * 100),
        cuda=False,
        seed=int(args.seed),
        cuda_device=0,
        q_frequency=int(args.q_frequency),
        model_saved_step=max(1, int(args.total_timesteps)),
    )
    if not args.skip_train:
        agent.run()
        agent.save_models("final")
    else:
        print("[qpensieve] loading checkpoints")
        model_dir = args.save_dir / "model"
        agent.policy.load(str(model_dir / "policy_final.pth"))
        agent.critic.load(str(model_dir / "critic_final.pth"))
        agent.critic_target.load(str(model_dir / "critic_target.pth"))
    if args.train_only:
        (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
        print(json.dumps({"method": "Q-Pensieve", "env_key": args.env_key, "stage": "train_only"}, indent=2))
        return

    print("[qpensieve] evaluate")
    eval_summary = evaluate_conditioned_policy(
        env_name=spec.dynamic_env_name,
        obj_num=spec.obj_num,
        seed=args.seed,
        policy_fn=lambda obs, pref: agent.exploit(obs, pref),
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
    print("[qpensieve] save")
    summary = {
        "method": "Q-Pensieve",
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
