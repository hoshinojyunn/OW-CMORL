from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PGMORL_ROOT = PROJECT_ROOT / "PGMORL"
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_envs import get_env_spec, make_scalar_env, register_baseline_envs
from src.baseline_eval import evaluate_archive_policies, save_baseline_summary, save_front_csv
from src.dynamic_morl.fine_regimes import coerce_regime_seed_plan
from src.ood_protocol import load_env_config_json


class Logger:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.logfile.write(data)
        self.logfile.flush()

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--warmup-iter", type=int, default=20)
    parser.add_argument("--update-iter", type=int, default=5)
    parser.add_argument("--delta-weight", type=float, default=0.5)
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
    parser.add_argument("--max-eval-policies", type=int, default=64)
    parser.add_argument("--shared-plan-json", type=Path, default=None)
    parser.add_argument("--env-config-json", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--save-dir", type=Path, required=True)
    return parser.parse_args()


def build_pg_args(args: argparse.Namespace, env_id: str, obj_num: int):
    from arguments import get_parser

    arg_list = [
        "--env-name",
        env_id,
        "--obj-num",
        str(obj_num),
        "--num-env-steps",
        str(int(args.total_timesteps)),
        "--seed",
        str(int(args.seed)),
        "--delta-weight",
        str(float(args.delta_weight)),
        "--warmup-iter",
        str(int(args.warmup_iter)),
        "--update-iter",
        str(int(args.update_iter)),
        "--eval-num",
        "1",
        "--selection-method",
        "prediction-guided",
        "--num-steps",
        str(int(args.num_steps)),
        "--num-processes",
        str(int(args.num_processes)),
        "--ppo-epoch",
        "2",
        "--num-mini-batch",
        "2",
        "--use-linear-lr-decay",
        "--use-gae",
        "--gae-lambda",
        "0.95",
        "--entropy-coef",
        "0",
        "--value-loss-coef",
        "0.5",
        "--use-proper-time-limits",
        "--ob-rms",
        "--obj-rms",
        "--raw",
        "--save-dir",
        str(args.save_dir),
    ]
    parser = get_parser()
    return parser.parse_args(arg_list)


def _select_diverse_subset(points: np.ndarray, max_points: int) -> list[int]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return list(range(len(points)))
    mins = points.min(axis=0, keepdims=True)
    spans = np.maximum(points.max(axis=0, keepdims=True) - mins, 1e-8)
    norm = (points - mins) / spans

    selected: list[int] = []
    for dim in range(norm.shape[1]):
        selected.append(int(np.argmax(norm[:, dim])))
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = [0]

    while len(selected) < min(max_points, len(norm)):
        remaining = [idx for idx in range(len(norm)) if idx not in selected]
        if not remaining:
            break
        distances = []
        for idx in remaining:
            dists = np.linalg.norm(norm[idx] - norm[selected], axis=1)
            distances.append((float(np.min(dists)), idx))
        distances.sort(reverse=True)
        selected.append(int(distances[0][1]))
    return selected[:max_points]


def _select_policy_indices(final_dir: Path, max_eval_policies: int) -> list[int]:
    obj_path = final_dir / "objs.txt"
    if not obj_path.exists():
        return []
    objs = np.loadtxt(obj_path, delimiter=",")
    if objs.ndim == 1:
        objs = objs[None, :]
    nd_idx = NonDominatedSorting().do(-np.asarray(objs, dtype=np.float64), only_non_dominated_front=True)
    nd_idx = [int(idx) for idx in np.atleast_1d(nd_idx).tolist()]
    if len(nd_idx) <= max_eval_policies:
        return nd_idx
    nd_points = np.asarray(objs[nd_idx], dtype=np.float64)
    subset = _select_diverse_subset(nd_points, max_eval_policies)
    return [nd_idx[idx] for idx in subset]


def build_policy_fns(spec, pg_args, max_eval_policies: int) -> list:
    from a2c_ppo_acktr.model import Policy

    env = make_scalar_env(spec.env_key, seed=pg_args.seed)
    obs_shape = env.observation_space.shape
    action_space = env.action_space
    env.close()
    policy_fns = []
    final_dir = Path(pg_args.save_dir) / "final"
    selected_indices = set(_select_policy_indices(final_dir, max(1, int(max_eval_policies))))
    for policy_path in sorted(final_dir.glob("EP_policy_*.pt")):
        idx = int(policy_path.stem.split("_")[-1])
        if selected_indices and idx not in selected_indices:
            continue
        env_params_path = final_dir / f"EP_env_params_{idx}.pkl"
        if not env_params_path.exists():
            continue
        with open(env_params_path, "rb") as fp:
            env_params = pickle.load(fp)
        actor_critic = Policy(
            obs_shape,
            action_space,
            base_kwargs={"layernorm": bool(pg_args.layernorm)},
            obj_num=spec.obj_num,
        )
        actor_critic.to(torch.device("cpu")).double()
        actor_critic.load_state_dict(torch.load(policy_path, map_location="cpu"))
        actor_critic.eval()
        ob_rms = env_params.get("ob_rms")

        def _policy_fn(obs, actor_critic=actor_critic, ob_rms=ob_rms):
            ob = np.asarray(obs, dtype=np.float64)
            if bool(pg_args.ob_rms) and ob_rms is not None:
                ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
            with torch.no_grad():
                _, action, _, _ = actor_critic.act(
                    torch.as_tensor(ob, dtype=torch.float64).unsqueeze(0),
                    None,
                    None,
                    deterministic=True,
                )
            return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)

        policy_fns.append(_policy_fn)
    return policy_fns


def main() -> None:
    args = parse_args()
    args.save_dir = args.save_dir.resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print("[pgmorl] start")
    register_baseline_envs()
    sys.path.insert(1, str(PGMORL_ROOT))
    sys.path.insert(1, str(PGMORL_ROOT / "morl"))
    sys.path.insert(1, str(PGMORL_ROOT / "externals" / "baselines"))
    sys.path.insert(1, str(PGMORL_ROOT / "externals" / "pytorch-a2c-ppo-acktr-gail"))
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
    print("[pgmorl] env ready")

    import morl as pg_morl

    torch.set_default_dtype(torch.float64)
    pg_args = build_pg_args(args, spec.scalar_env_id, spec.obj_num)
    (args.save_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    (args.save_dir / "pg_args.txt").write_text(" ".join(sys.argv[1:]))
    print("[pgmorl] args ready")

    if not args.skip_train:
        with open(args.save_dir / "log.txt", "w") as log_fp:
            logger = Logger(sys.stdout, log_fp)
            old_stdout = sys.stdout
            try:
                sys.stdout = logger
                pg_morl.run(pg_args)
            finally:
                sys.stdout = old_stdout
    if args.train_only:
        print(json.dumps({"method": "PGMORL", "env_key": args.env_key, "stage": "train_only"}, indent=2))
        return

    print("[pgmorl] evaluate")
    policy_fns = build_policy_fns(spec, pg_args, args.max_eval_policies)
    eval_summary = evaluate_archive_policies(
        env_name=spec.dynamic_env_name,
        obj_num=spec.obj_num,
        seed=args.seed,
        policy_fns=policy_fns,
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
    print("[pgmorl] save")
    summary = {
        "method": "PGMORL",
        "env_key": args.env_key,
        "env_name": spec.dynamic_env_name,
        "seed": args.seed,
        "total_timesteps": int(args.total_timesteps),
        "obj_num": spec.obj_num,
        "num_policies": len(policy_fns),
        "env_kwargs": env_kwargs,
        **eval_summary,
    }
    save_baseline_summary(args.save_dir / "summary.json", summary)
    save_front_csv(args.save_dir / "front_points.csv", summary["front_points"])
    print(json.dumps({"method": summary["method"], "env_key": summary["env_key"], "front_size": len(summary["front_points"])}, indent=2))


if __name__ == "__main__":
    main()
