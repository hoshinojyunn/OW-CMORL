from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import types
import tempfile

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "externals" / "baselines"))
sys.path.append(str(PROJECT_ROOT / "sustaingym"))

from src.baseline_envs import build_dynamic_env, default_env_kwargs, resolve_env_key
from src.dynamic_morl.fine_regimes import (
    DEFAULT_FINE_REGIME_CLUSTERS,
    DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    DEFAULT_FINE_REGIME_EPISODES,
    assign_episode_returns_to_catalog,
    build_regime_seed_plan,
    catalog_cache_signature,
    load_or_build_catalog,
    selected_episode_seeds_from_plan,
)
from src.dynamic_morl.chlor_shared_protocol import (
    canonical_chlor_regime_rows,
    canonical_chlor_regime_seed_plan,
    chlor_regime_eval_env_kwargs,
)
from src.dynamic_morl.metrics import compute_trace_shift_metrics
from src.dynamic_morl.shared_eval import build_shared_episode_seeds, normalize_episode_seeds
from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


BASELINE_ALGS = [
    "a2c",
    "acer",
    "acktr",
    "ddpg",
    "deepq",
    "ppo1",
    "ppo2",
    "trpo_mpi",
]
DISCRETE_ALGS = {"acer", "deepq"}
SINGLE_ENV_ALGS = {"deepq", "ppo1", "trpo_mpi"}
NORMALIZED_CONTINUOUS_ALGS = {"ddpg"}


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    stacked = np.concatenate(fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def front_metrics(front: np.ndarray, ref_point: np.ndarray, eval_delta_weight: float) -> dict[str, float]:
    if front.ndim != 2 or len(front) == 0:
        return {"hv": 0.0, "eu": 0.0, "sp": 0.0}
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], eval_delta_weight)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"hv": float(hv), "eu": float(eu), "sp": float(sp)}


def bootstrap_tensorflow() -> None:
    import tensorflow as tf

    tf.compat.v1.disable_v2_behavior()
    v1 = tf.compat.v1
    for attr in [
        "Session",
        "InteractiveSession",
        "get_default_session",
        "get_default_graph",
        "set_random_seed",
        "placeholder",
        "placeholder_with_default",
        "global_variables",
        "global_variables_initializer",
        "local_variables_initializer",
        "variables_initializer",
        "global_variables_initializer",
        "reset_default_graph",
        "ConfigProto",
        "VariableScope",
        "Graph",
        "GraphKeys",
        "summary",
        "train",
        "layers",
        "nn",
        "variable_scope",
        "get_variable",
        "random_uniform_initializer",
        "zeros_initializer",
        "constant",
        "cond",
        "cast",
        "group",
    ]:
        if not hasattr(tf, attr) and hasattr(v1, attr):
            setattr(tf, attr, getattr(v1, attr))
    for attr in ["train", "layers", "nn", "summary"]:
        if hasattr(v1, attr):
            setattr(tf, attr, getattr(v1, attr))
    for name, value in v1.__dict__.items():
        if not name.startswith("_") and not hasattr(tf, name):
            setattr(tf, name, value)
    install_mpi_stub()
    install_tf_contrib_stub(tf)


def install_mpi_stub() -> None:
    if "mpi4py" in sys.modules:
        return

    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Allreduce(self, sendbuf, recvbuf, op=None):
            np.copyto(recvbuf, sendbuf)

        def allreduce(self, value, op=None):
            return value

        def allgather(self, value):
            return [value]

        def Bcast(self, buf, root=0):
            return buf

        def Abort(self, code=1):
            raise RuntimeError(f"MPI Abort called with code {code}")

    mpi_module = types.ModuleType("mpi4py")
    mpi_submodule = types.SimpleNamespace(COMM_WORLD=_FakeComm(), SUM="sum")
    mpi_module.MPI = mpi_submodule
    sys.modules["mpi4py"] = mpi_module


def install_tf_contrib_stub(tf) -> None:
    if "tensorflow.contrib" in sys.modules:
        return

    def flatten(x):
        return tf.compat.v1.layers.flatten(x)

    def fully_connected(inputs, num_outputs, activation_fn=None, **kwargs):
        return tf.compat.v1.layers.dense(inputs, num_outputs, activation=activation_fn, **kwargs)

    def layer_norm(inputs, center=True, scale=True, **kwargs):
        layer = tf.keras.layers.LayerNormalization(center=center, scale=scale, axis=-1, **kwargs)
        return layer(inputs)

    def convolution2d(inputs, num_outputs, kernel_size, stride=1, activation_fn=tf.nn.relu, **kwargs):
        return tf.compat.v1.layers.conv2d(
            inputs,
            filters=num_outputs,
            kernel_size=kernel_size,
            strides=stride,
            activation=activation_fn,
            **kwargs,
        )

    def xavier_initializer():
        return tf.compat.v1.keras.initializers.glorot_uniform()

    def l2_regularizer(scale):
        def regularizer(weights):
            return tf.multiply(scale, tf.nn.l2_loss(weights))

        return regularizer

    def apply_regularization(regularizer, weights_list):
        if not weights_list:
            return tf.constant(0.0, dtype=tf.float32)
        terms = [regularizer(weight) for weight in weights_list]
        return tf.add_n(terms)

    layers_module = types.ModuleType("tensorflow.contrib.layers")
    layers_module.flatten = flatten
    layers_module.fully_connected = fully_connected
    layers_module.layer_norm = layer_norm
    layers_module.convolution2d = convolution2d
    layers_module.xavier_initializer = xavier_initializer
    layers_module.l2_regularizer = l2_regularizer
    layers_module.apply_regularization = apply_regularization

    contrib_module = types.ModuleType("tensorflow.contrib")
    contrib_module.layers = layers_module

    sys.modules["tensorflow.contrib"] = contrib_module
    sys.modules["tensorflow.contrib.layers"] = layers_module
    setattr(tf, "contrib", contrib_module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", required=True, choices=BASELINE_ALGS)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--obj-num", type=int, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--ev-site", type=str, default="caltech")
    parser.add_argument("--ev-periods", nargs="+", default=["Summer 2019", "Spring 2020", "Summer 2021"])
    parser.add_argument("--ev-moer-forecast-steps", type=int, default=36)
    parser.add_argument("--ev-project-action-in-env", type=int, default=1)
    parser.add_argument("--cogen-renewables", nargs="+", type=float, default=[0.0, 10.0, 20.0])
    parser.add_argument("--cogen-forecast-horizon", type=int, default=3)
    parser.add_argument("--cogen-forecast-noise-std", type=float, default=0.0)
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--chlor-regime-clusters", type=int, default=DEFAULT_FINE_REGIME_CLUSTERS)
    parser.add_argument("--macro-levels", type=int, default=5)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--trace-recovery-window", type=int, default=12)
    parser.add_argument("--trace-eval-episodes", type=int, default=3)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--fine-regime-clusters", type=int, default=DEFAULT_FINE_REGIME_CLUSTERS)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=DEFAULT_FINE_REGIME_EPISODES)
    parser.add_argument("--fine-regime-context-steps", type=int, default=DEFAULT_FINE_REGIME_CONTEXT_STEPS)
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def make_base_env(args, env_name: str | None = None, env_kwargs: dict | None = None):
    env_name = env_name or args.env_name
    env_key = resolve_env_key(env_name)
    merged_env_kwargs = default_env_kwargs(env_key)
    merged_env_kwargs.update(dict(env_kwargs or {}))
    return build_dynamic_env(
        env_name,
        seed=args.seed,
        env_kwargs=merged_env_kwargs,
        schedule=str(merged_env_kwargs.get("schedule", args.regime_schedule)),
        episodes_per_regime=int(merged_env_kwargs.get("episodes_per_regime", args.episodes_per_regime)),
    )


def make_action_bank(action_dim: int, levels: int) -> np.ndarray:
    scale = np.linspace(-1.0, 1.0, levels, dtype=np.float32)
    return np.stack([np.full(action_dim, value, dtype=np.float32) for value in scale], axis=0)


def build_wrapped_env(args, env_name: str | None = None, env_kwargs: dict | None = None):
    print("[baseline] importing bench", flush=True)
    from baselines import bench
    print("[baseline] importing generic wrappers", flush=True)
    from src.dynamic_morl.generic_wrappers import (
        DiscreteActionBankWrapper,
        GymCompatibilityWrapper,
        NormalizedActionWrapper,
        ScalarizedRewardWrapper,
    )
    print("[baseline] env helper imports complete", flush=True)

    weights = np.asarray(args.weights, dtype=np.float32)
    print("[baseline] building base env", flush=True)
    env = make_base_env(args, env_name=env_name, env_kwargs=env_kwargs)
    print("[baseline] base env built", flush=True)
    env = GymCompatibilityWrapper(env)
    env = ScalarizedRewardWrapper(env, weights)
    if args.alg in DISCRETE_ALGS:
        print("[baseline] applying discrete wrapper", flush=True)
        if (env_name or args.env_name).startswith("evcharging"):
            from src.dynamic_morl.sustaingym_wrappers import (
                EVChargingAggressionDiscreteWrapper,
            )

            env = EVChargingAggressionDiscreteWrapper(env, levels=args.macro_levels)
        else:
            env = DiscreteActionBankWrapper(
                env,
                make_action_bank(env.action_space.shape[0], args.macro_levels),
            )
        print("[baseline] discrete wrapper applied", flush=True)
    elif args.alg in NORMALIZED_CONTINUOUS_ALGS:
        env = NormalizedActionWrapper(env)
    return bench.Monitor(env, None, allow_early_resets=True)


def make_training_env(
    args,
    env_name: str | None = None,
    env_kwargs: dict | None = None,
    single_env: bool | None = None,
):
    target_env = env_name or args.env_name
    merged_env_kwargs = _catalog_env_kwargs(args, target_env, env_kwargs)
    if single_env is None:
        single_env = args.alg in SINGLE_ENV_ALGS
    if single_env:
        return build_wrapped_env(args, env_name=target_env, env_kwargs=merged_env_kwargs)

    print("[baseline] importing DummyVecEnv", flush=True)
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

    return DummyVecEnv(
        [lambda: build_wrapped_env(args, env_name=target_env, env_kwargs=merged_env_kwargs)]
    )


def reset_training_env(env, *, single_env: bool, seed: int | None = None):
    if single_env:
        if seed is None:
            return env.reset()
        return env.reset(seed=int(seed))

    inner_env = env.envs[0]
    if seed is None:
        return env.reset()
    obs = inner_env.reset(seed=int(seed))
    env._save_obs(0, obs)
    return env._obs_from_buf()


def train_model(args, env):
    from baselines import logger

    logger.configure(str(args.save_dir))
    if args.alg == "a2c":
        from baselines.a2c.a2c import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=args.total_timesteps, nsteps=8, log_interval=10)
    if args.alg == "acer":
        from baselines.acer.acer import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=args.total_timesteps, nsteps=8, log_interval=10)
    if args.alg == "acktr":
        from baselines.acktr.acktr import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=args.total_timesteps, nsteps=8, log_interval=10, nprocs=1, is_async=False)
    if args.alg == "ddpg":
        from baselines.ddpg.ddpg import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=args.total_timesteps, nb_epoch_cycles=2, nb_rollout_steps=8, nb_train_steps=8, nb_eval_steps=8)
    if args.alg == "deepq":
        from baselines.deepq.deepq import learn

        return learn(env, "mlp", seed=args.seed, total_timesteps=args.total_timesteps, print_freq=20, learning_starts=32, buffer_size=2000)
    if args.alg == "ppo1":
        from baselines.ppo1.mlp_policy import MlpPolicy
        from baselines.ppo1.pposgd_simple import learn

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2)

        actor_batch = min(32, max(8, args.total_timesteps))
        return learn(
            env,
            policy_fn,
            max_iters=max(1, int(np.ceil(args.total_timesteps / actor_batch))),
            timesteps_per_actorbatch=actor_batch,
            clip_param=0.2,
            entcoeff=0.0,
            optim_epochs=2,
            optim_stepsize=3e-4,
            optim_batchsize=32,
            gamma=0.99,
            lam=0.95,
            schedule="constant",
        )
    if args.alg == "ppo2":
        from baselines.ppo2.ppo2 import learn

        return learn(
            network="mlp",
            env=env,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            nsteps=8,
            nminibatches=1,
            noptepochs=2,
            log_interval=10,
        )
    if args.alg == "trpo_mpi":
        from baselines.trpo_mpi.trpo_mpi import learn

        batch_size = min(32, max(8, args.total_timesteps))
        return learn(
            network="mlp",
            env=env,
            total_timesteps=0,
            seed=args.seed,
            max_iters=max(1, int(np.ceil(args.total_timesteps / batch_size))),
            timesteps_per_batch=batch_size,
            max_kl=0.01,
            cg_iters=5,
            vf_iters=2,
        )
    raise ValueError(args.alg)


def _model_path(args) -> Path:
    return args.save_dir / "model"


def save_model(args, model) -> None:
    model_path = _model_path(args)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.alg == "ppo1":
        from baselines.common import tf_util as U

        U.save_state(str(model_path))
        return
    if hasattr(model, "save"):
        model.save(str(model_path))
        return
    if args.alg == "ddpg":
        from baselines.common import tf_util

        tf_util.save_variables(str(model_path))
        return


def load_model(args, env):
    from baselines import logger

    logger.configure(str(args.save_dir))
    model_path = _model_path(args)
    if not model_path.exists():
        raise FileNotFoundError(f"missing saved model: {model_path}")

    if args.alg == "a2c":
        from baselines.a2c.a2c import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=0, nsteps=8, log_interval=10, load_path=str(model_path))
    if args.alg == "acer":
        from baselines.acer.acer import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=0, nsteps=8, log_interval=10, load_path=str(model_path))
    if args.alg == "acktr":
        from baselines.acktr.acktr import learn

        return learn("mlp", env, seed=args.seed, total_timesteps=0, nsteps=8, log_interval=10, nprocs=1, is_async=False, load_path=str(model_path))
    if args.alg == "deepq":
        from baselines.deepq.deepq import learn

        return learn(env, "mlp", seed=args.seed, total_timesteps=0, print_freq=20, learning_starts=32, buffer_size=2000, load_path=str(model_path))
    if args.alg == "ddpg":
        env_for_train = env if not hasattr(env, "num_envs") else env
        from baselines.ddpg.ddpg import learn
        from baselines.common import tf_util

        model = learn(
            "mlp",
            env_for_train,
            seed=args.seed,
            total_timesteps=0,
            nb_epoch_cycles=2,
            nb_rollout_steps=8,
            nb_train_steps=8,
            nb_eval_steps=8,
        )
        tf_util.load_variables(str(model_path))
        return model
    if args.alg == "ppo1":
        from baselines import logger
        from baselines.common import tf_util as U
        from baselines.ppo1.mlp_policy import MlpPolicy

        logger.configure(str(args.save_dir))

        def policy_fn(name, ob_space, ac_space):
            return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=2)

        U.get_session()
        model = policy_fn("pi", env.observation_space, env.action_space)
        U.load_state(str(model_path))
        return model
    if args.alg == "ppo2":
        from baselines.ppo2.ppo2 import learn

        return learn(
            network="mlp",
            env=env,
            total_timesteps=0,
            seed=args.seed,
            nsteps=8,
            nminibatches=1,
            noptepochs=2,
            log_interval=10,
            load_path=str(model_path),
        )
    if args.alg == "trpo_mpi":
        from baselines.trpo_mpi.trpo_mpi import learn

        batch_size = 8
        return learn(
            network="mlp",
            env=env,
            total_timesteps=0,
            seed=args.seed,
            max_iters=0,
            timesteps_per_batch=batch_size,
            max_kl=0.01,
            cg_iters=5,
            vf_iters=2,
            load_path=str(model_path),
        )
    raise ValueError(f"skip-train is not supported for alg={args.alg}")


def model_action(model, alg: str, obs):
    if alg == "ddpg":
        action, _, _, _ = model.step(obs, apply_noise=False, compute_Q=False)
        return action
    if alg == "deepq":
        batched_obs = obs[None, ...] if np.asarray(obs).ndim == 1 else obs
        action = model.step(batched_obs, update_eps=0.0)[0]
        return int(np.asarray(action).reshape(-1)[0])
    if alg == "ppo1":
        action, _value = model.act(False, obs)
        return action
    if alg == "trpo_mpi":
        action, _, _, _ = model.step(obs[None, ...] if np.asarray(obs).ndim == 1 else obs)
        return np.asarray(action)[0]
    actions, _, _, _ = model.step(obs)
    return actions


def _trace_info(info: dict) -> tuple[str, int | float]:
    regime_name = (
        info.get("regime_name")
        or info.get("period_name")
        or info.get("weather")
        or str(info.get("renewables_magnitude", "regime"))
    )
    regime_id = info.get("regime_id", info.get("period_id", info.get("renewables_magnitude", 0)))
    return str(regime_name), regime_id


def _catalog_env_kwargs(args, env_name: str | None = None, env_kwargs: dict | None = None) -> dict[str, object]:
    target_env = env_name or args.env_name
    merged = default_env_kwargs(resolve_env_key(target_env))
    merged.update(dict(env_kwargs or {}))
    if target_env.startswith("building_"):
        merged.setdefault("episodes_per_regime", args.episodes_per_regime)
        merged.setdefault("schedule", args.regime_schedule)
    elif target_env == "evcharging_dynamic":
        merged.setdefault("site", args.ev_site)
        merged.setdefault("periods", tuple(args.ev_periods))
        merged.setdefault("moer_forecast_steps", int(args.ev_moer_forecast_steps))
        merged.setdefault("project_action_in_env", bool(int(args.ev_project_action_in_env)))
        merged.setdefault("episodes_per_regime", args.episodes_per_regime)
        merged.setdefault("schedule", args.regime_schedule)
    elif target_env == "cogen_dynamic":
        merged.setdefault("renewables", tuple(args.cogen_renewables))
        merged.setdefault("forecast_horizon", args.cogen_forecast_horizon)
        merged.setdefault("forecast_noise_std", args.cogen_forecast_noise_std)
        merged.setdefault("episodes_per_regime", args.episodes_per_regime)
        merged.setdefault("schedule", args.regime_schedule)
    elif target_env == "chlor_alkali_dynamic":
        merged["dynamic_price"] = bool(merged.get("dynamic_price", True))
        merged["aging_dynamics"] = bool(merged.get("aging_dynamics", True))
        merged["episode_length"] = int(args.chlor_episode_length)
        merged["num_regime_clusters"] = int(args.chlor_regime_clusters)
        merged["schedule"] = str(merged.get("schedule", "random"))
    return merged


def _dynamic_regime_catalog(args, env_name: str | None = None, env_kwargs: dict | None = None):
    target_env = env_name or args.env_name
    merged_env_kwargs = _catalog_env_kwargs(args, target_env, env_kwargs)
    n_clusters = int(
        merged_env_kwargs.get(
            "num_regime_clusters",
            getattr(args, "fine_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS),
        )
    )

    def env_factory(factory_seed: int):
        return build_dynamic_env(
            target_env,
            seed=args.seed + int(factory_seed),
            env_kwargs=merged_env_kwargs,
            schedule=str(merged_env_kwargs.get("schedule", args.regime_schedule)),
            episodes_per_regime=int(merged_env_kwargs.get("episodes_per_regime", args.episodes_per_regime)),
        )

    return load_or_build_catalog(
        env_key=resolve_env_key(target_env),
        env_factory=env_factory,
        n_clusters=n_clusters,
        episodes=int(getattr(args, "fine_regime_catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
        context_steps=int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
        seed=0,
        cache_signature=catalog_cache_signature(
            {
                "env_name": target_env,
                "env_kwargs": merged_env_kwargs,
                "episodes": int(getattr(args, "fine_regime_catalog_episodes", DEFAULT_FINE_REGIME_EPISODES)),
                "context_steps": int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
            }
        ),
    )


def _group_regime_returns_from_trace(args, trace: dict[str, np.ndarray] | None) -> list[dict[str, object]]:
    if trace is None:
        return []
    catalog = _dynamic_regime_catalog(args)
    assignments = assign_episode_returns_to_catalog(
        trace,
        catalog=catalog,
        context_steps=int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
    )
    rows = []
    for row in assignments:
        rows.append(
            {
                "episode": int(row["episode"]),
                "regime": row["regime"],
                "regime_id": int(row["regime_id"]),
                "regime_meta": dict(row["regime_meta"]),
                "context_vector": np.asarray(row["context_vector"], dtype=np.float64).tolist(),
                "objs": np.asarray(row["return"], dtype=np.float64).tolist(),
            }
        )
    return rows


def _group_regime_returns_from_trace_plan(
    trace: dict[str, np.ndarray] | None,
    shared_regime_seed_plan: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if trace is None or not shared_regime_seed_plan:
        return []
    regime_records = list(trace.get("regime_records", []))
    if not regime_records:
        return []
    rows = []
    for plan_idx, plan_row in enumerate(shared_regime_seed_plan):
        if plan_idx >= len(regime_records):
            break
        record = regime_records[plan_idx]
        rows.append(
            {
                "episode": int(record.get("eval_id", plan_idx)),
                "regime": str(plan_row["regime"]),
                "regime_id": int(plan_row["regime_id"]),
                "regime_meta": dict(plan_row.get("regime_meta", {})),
                "context_vector": [],
                "objs": np.asarray(record["return"], dtype=np.float64).tolist(),
            }
        )
    return rows


def _catalog_regime_rows(args) -> list[dict[str, object]]:
    if args.env_name == "chlor_alkali_dynamic":
        return canonical_chlor_regime_rows(
            num_regime_clusters=int(getattr(args, "chlor_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
        )
    catalog = _dynamic_regime_catalog(args)
    rows = []
    for stable_idx in sorted(catalog.labels):
        rows.append(
            {
                "regime": str(catalog.labels[stable_idx]),
                "regime_id": int(stable_idx),
                "regime_meta": dict(catalog.regime_meta(stable_idx)),
            }
        )
    return rows


def _shared_regime_seed_plan(args) -> tuple[list[dict[str, object]], list[int]]:
    if args.env_name == "chlor_alkali_dynamic":
        return canonical_chlor_regime_seed_plan(
            env_name=args.env_name,
            base_seed=int(args.seed),
            shared_regime_eval_episodes=max(
                int(args.shared_regime_eval_episodes),
                int(args.trace_eval_episodes),
                int(args.eval_episodes),
                1,
            ),
            seed_offset=int(args.shared_regime_seed_offset),
            num_regime_clusters=int(getattr(args, "chlor_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
        )

    catalog = _dynamic_regime_catalog(args)
    target_count = int(args.shared_regime_eval_episodes or len(catalog.labels) or args.fine_regime_clusters)
    target_regime_ids = sorted(int(idx) for idx in catalog.labels)[: min(target_count, len(catalog.labels))]
    start_seed = build_shared_episode_seeds(
        args.env_name,
        base_seed=args.seed,
        num_episodes=1,
        seed_offset=int(args.shared_regime_seed_offset),
    )[0]
    env_kwargs = _catalog_env_kwargs(args)

    def env_factory(_factory_seed: int):
        return build_dynamic_env(
            args.env_name,
            seed=args.seed,
            env_kwargs=env_kwargs,
            schedule=str(env_kwargs.get("schedule", args.regime_schedule)),
            episodes_per_regime=int(env_kwargs.get("episodes_per_regime", args.episodes_per_regime)),
        )

    plan = build_regime_seed_plan(
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(start_seed),
        seeds_per_regime=1,
        context_steps=int(getattr(args, "fine_regime_context_steps", DEFAULT_FINE_REGIME_CONTEXT_STEPS)),
        max_search_episodes=max(256, 64 * max(1, len(target_regime_ids))),
        target_regime_ids=target_regime_ids,
    )
    return plan, selected_episode_seeds_from_plan(plan)


def evaluate_episode(
    args,
    model,
    *,
    episode_seed: int | None = None,
    env_name: str | None = None,
    env_kwargs: dict | None = None,
    record_trace: bool = False,
):
    eval_args = argparse.Namespace(**vars(args))
    if episode_seed is not None:
        eval_args.seed = int(episode_seed)
    single_env = args.alg in SINGLE_ENV_ALGS
    env = make_training_env(
        eval_args,
        env_name=env_name,
        env_kwargs=env_kwargs,
        single_env=single_env,
    )
    try:
        return evaluate_episode_with_env(
            args,
            model,
            env=env,
            single_env=single_env,
            episode_seed=episode_seed,
            record_trace=record_trace,
        )
    finally:
        env.close()


def evaluate_episode_with_env(
    args,
    model,
    *,
    env,
    single_env: bool,
    episode_seed: int | None = None,
    record_trace: bool = False,
):
    episode_obj = np.zeros(args.obj_num, dtype=np.float64)
    trace = {"obs": [], "obj": [], "regime": [], "regime_id": [], "step": []}
    trace["episode"] = []

    obs = reset_training_env(env, single_env=single_env, seed=episode_seed)
    if single_env:
        done = False
        step_idx = 0
        while not done:
            obs_before = np.asarray(obs, dtype=np.float32)
            action = model_action(model, args.alg, obs)
            obs, _reward, done, info = env.step(action)
            obj_raw = np.asarray(info["obj_raw"], dtype=np.float64)
            episode_obj += obj_raw
            if record_trace:
                regime_name, regime_id = _trace_info(info)
                trace["obs"].append(obs_before)
                trace["obj"].append(obj_raw.astype(np.float32))
                trace["regime"].append(regime_name)
                trace["regime_id"].append(regime_id)
                trace["step"].append(step_idx)
                trace["episode"].append(0)
            step_idx += 1
    else:
        done = np.array([False])
        step_idx = 0
        while not done[0]:
            obs_before = np.asarray(obs[0], dtype=np.float32)
            action = model_action(model, args.alg, obs)
            obs, _reward, done, infos = env.step(action)
            info = infos[0]
            obj_raw = np.asarray(info["obj_raw"], dtype=np.float64)
            episode_obj += obj_raw
            if record_trace:
                regime_name, regime_id = _trace_info(info)
                trace["obs"].append(obs_before)
                trace["obj"].append(obj_raw.astype(np.float32))
                trace["regime"].append(regime_name)
                trace["regime_id"].append(regime_id)
                trace["step"].append(step_idx)
                trace["episode"].append(0)
            step_idx += 1
    if record_trace:
        trace["obs"] = np.asarray(trace["obs"], dtype=np.float32)
        trace["obj"] = np.asarray(trace["obj"], dtype=np.float32)
        trace["regime_id"] = np.asarray(trace["regime_id"])
        trace["step"] = np.asarray(trace["step"], dtype=np.int64)
        trace["episode"] = np.asarray(trace["episode"], dtype=np.int64)
        return episode_obj, trace
    return episode_obj, None


def evaluate_model(
    args,
    model,
    *,
    env_name: str | None = None,
    env_kwargs: dict | None = None,
    return_trace: bool = False,
    episode_seeds: list[int] | None = None,
):
    total_obj = np.zeros(args.obj_num, dtype=np.float64)
    saved_trace = None
    seeds = episode_seeds if episode_seeds is not None else list(range(max(1, args.eval_episodes)))
    eval_args = argparse.Namespace(**vars(args))
    single_env = args.alg in SINGLE_ENV_ALGS
    env = make_training_env(
        eval_args,
        env_name=env_name,
        env_kwargs=env_kwargs,
        single_env=single_env,
    )
    try:
        for eval_idx, episode_seed in enumerate(seeds):
            episode_obj, trace = evaluate_episode_with_env(
                args,
                model,
                env=env,
                single_env=single_env,
                episode_seed=episode_seed,
                record_trace=return_trace and eval_idx == 0,
            )
            total_obj += episode_obj
            if trace is not None:
                saved_trace = trace
    finally:
        env.close()
    total_obj /= max(len(seeds), 1)
    if return_trace:
        return total_obj, saved_trace
    return total_obj


def evaluate_dynamic_trace(args, model, *, episode_seeds: list[int] | None = None) -> dict[str, np.ndarray] | None:
    seeds = episode_seeds if episode_seeds is not None else normalize_episode_seeds(
        args.env_name,
        base_seed=args.seed,
        num_episodes=max(int(args.trace_eval_episodes), 1),
        seed_offset=int(args.shared_regime_seed_offset),
    )
    if len(seeds) <= 0:
        return None
    trace = {
        "obs": [],
        "obj": [],
        "context": [],
        "regime": [],
        "regime_id": [],
        "step": [],
        "episode": [],
        "context_feature_names": [],
    }
    regime_records = []
    global_step = 0
    single_env = args.alg in SINGLE_ENV_ALGS
    env = make_training_env(args, single_env=single_env)
    try:
        for episode_idx, episode_seed in enumerate(seeds):
            obs = reset_training_env(env, single_env=single_env, seed=int(episode_seed))
            episode_obj = np.zeros(args.obj_num, dtype=np.float64)
            episode_regime = None
            if single_env:
                done = False
                while not done:
                    obs_before = np.asarray(obs, dtype=np.float32)
                    action = model_action(model, args.alg, obs)
                    obs, _reward, done, info = env.step(action)
                    obj_raw = np.asarray(info["obj_raw"], dtype=np.float32)
                    regime_name, regime_id = _trace_info(info)
                    episode_obj += obj_raw.astype(np.float64)
                    episode_regime = regime_name if episode_regime is None else episode_regime
                    context_vec = np.asarray(info.get("context_vector", []), dtype=np.float32).reshape(-1)
                    trace["obs"].append(obs_before)
                    trace["obj"].append(obj_raw)
                    if len(context_vec) > 0:
                        trace["context"].append(context_vec)
                    if not trace["context_feature_names"] and info.get("context_feature_names"):
                        trace["context_feature_names"] = list(info.get("context_feature_names", []))
                    trace["regime"].append(regime_name)
                    trace["regime_id"].append(regime_id)
                    trace["step"].append(global_step)
                    trace["episode"].append(episode_idx)
                    global_step += 1
            else:
                done = np.array([False])
                while not done[0]:
                    obs_before = np.asarray(obs[0], dtype=np.float32)
                    action = model_action(model, args.alg, obs)
                    obs, _reward, done, infos = env.step(action)
                    info = infos[0]
                    obj_raw = np.asarray(info["obj_raw"], dtype=np.float32)
                    regime_name, regime_id = _trace_info(info)
                    episode_obj += obj_raw.astype(np.float64)
                    episode_regime = regime_name if episode_regime is None else episode_regime
                    context_vec = np.asarray(info.get("context_vector", []), dtype=np.float32).reshape(-1)
                    trace["obs"].append(obs_before)
                    trace["obj"].append(obj_raw)
                    if len(context_vec) > 0:
                        trace["context"].append(context_vec)
                    if not trace["context_feature_names"] and info.get("context_feature_names"):
                        trace["context_feature_names"] = list(info.get("context_feature_names", []))
                    trace["regime"].append(regime_name)
                    trace["regime_id"].append(regime_id)
                    trace["step"].append(global_step)
                    trace["episode"].append(episode_idx)
                    global_step += 1

            regime_records.append(
                {
                    "eval_id": int(episode_idx),
                    "regime": str(episode_regime or "episode"),
                    "return": episode_obj.astype(np.float32),
                }
            )
    finally:
        env.close()
    if len(trace["obj"]) == 0:
        return None
    trace["obs"] = np.asarray(trace["obs"], dtype=np.float32)
    trace["obj"] = np.asarray(trace["obj"], dtype=np.float32)
    trace["context"] = np.asarray(trace["context"], dtype=np.float32)
    trace["regime_id"] = np.asarray(trace["regime_id"])
    trace["step"] = np.asarray(trace["step"], dtype=np.int64)
    trace["episode"] = np.asarray(trace["episode"], dtype=np.int64)
    trace["regime_records"] = regime_records
    return trace


def _save_partial_regime_result(
    args,
    objs: np.ndarray,
    regime_rows: list[dict[str, object]],
    trace_metrics: dict[str, float],
    shared_episode_seeds: list[int],
    shared_regime_seed_plan: list[dict[str, object]] | None = None,
) -> None:
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_incremental_regime_artifacts(args, regime_rows)
    (args.save_dir / "result.json").write_text(
        json.dumps(
            {
                "alg": args.alg,
                "env_name": args.env_name,
                "seed": args.seed,
                "weights": args.weights,
                "total_timesteps": args.total_timesteps,
                "objs": objs.tolist(),
                "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                "shared_regime_seed_plan": list(shared_regime_seed_plan or []),
                "trace_metrics": trace_metrics,
                "regime_returns": regime_rows,
            },
            indent=2,
        )
    )
    save_regime_csv(args.save_dir / "regime_returns.csv", regime_rows)


def _partial_rows_from_trace_plan(
    trace: dict[str, np.ndarray] | None,
    shared_regime_seed_plan: list[dict[str, object]] | None,
    upto: int,
) -> list[dict[str, object]]:
    if trace is None or not shared_regime_seed_plan:
        return []
    regime_records = list(trace.get("regime_records", []))
    rows: list[dict[str, object]] = []
    limit = min(int(upto), len(shared_regime_seed_plan), len(regime_records))
    for plan_idx in range(limit):
        plan_row = shared_regime_seed_plan[plan_idx]
        record = regime_records[plan_idx]
        rows.append(
            {
                "episode": int(record.get("eval_id", plan_idx)),
                "regime": str(plan_row["regime"]),
                "regime_id": int(plan_row["regime_id"]),
                "regime_meta": dict(plan_row.get("regime_meta", {})),
                "context_vector": [],
                "objs": np.asarray(record["return"], dtype=np.float64).tolist(),
            }
        )
    return rows


def evaluate_regimes(
    args,
    model,
    dynamic_trace: dict[str, np.ndarray] | None = None,
    *,
    objs: np.ndarray | None = None,
    trace_metrics: dict[str, float] | None = None,
    shared_episode_seeds: list[int] | None = None,
    shared_regime_seed_plan: list[dict[str, object]] | None = None,
    existing_regime_rows: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    existing_regime_rows = list(existing_regime_rows or [])
    if args.env_name == "chlor_alkali_dynamic":
        rows: list[dict[str, object]] = list(existing_regime_rows)
        completed_ids = {
            int(row.get("regime_id", -1))
            for row in rows
            if int(row.get("regime_id", -1)) >= 0
        }
        plan, _shared_episode_seeds = _shared_regime_seed_plan(args)
        rows_by_id = {
            int(row["regime_id"]): row for row in _catalog_regime_rows(args)
        }
        for plan_row in plan:
            if int(plan_row.get("regime_id", -1)) in completed_ids:
                continue
            catalog_row = rows_by_id.get(int(plan_row["regime_id"]))
            if catalog_row is None:
                continue
            episode_obj, regime_trace = evaluate_episode(
                args,
                model,
                episode_seed=int(plan_row["episode_seeds"][0]),
                env_kwargs=chlor_regime_eval_env_kwargs(
                    regime_name=str(catalog_row["regime"]),
                    regime_id=int(catalog_row["regime_id"]),
                    episode_length=int(args.chlor_episode_length),
                    schedule="random",
                    dynamic_price=True,
                    aging_dynamics=True,
                    num_regime_clusters=int(getattr(args, "chlor_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
                ),
                record_trace=True,
            )
            rows.append(
                {
                    "episode": int(len(rows)),
                    "regime": str(catalog_row["regime"]),
                    "regime_id": int(catalog_row["regime_id"]),
                    "regime_meta": dict(catalog_row["regime_meta"]),
                    "context_vector": [],
                    "objs": np.asarray(episode_obj, dtype=np.float64).tolist(),
                }
            )
            completed_ids.add(int(catalog_row["regime_id"]))
            if objs is not None and trace_metrics is not None and shared_episode_seeds is not None:
                _save_partial_regime_result(
                    args,
                    objs,
                    rows,
                    trace_metrics,
                    shared_episode_seeds,
                    shared_regime_seed_plan=shared_regime_seed_plan,
                )
        return rows
    dynamic_trace = dynamic_trace if dynamic_trace is not None else evaluate_dynamic_trace(
        args,
        model,
        episode_seeds=shared_episode_seeds,
    )
    rows = _group_regime_returns_from_trace_plan(dynamic_trace, shared_regime_seed_plan)
    if rows:
        if existing_regime_rows:
            existing_by_id = {
                int(row.get("regime_id", -1)): row
                for row in existing_regime_rows
                if int(row.get("regime_id", -1)) >= 0
            }
            merged_rows = []
            for row in rows:
                regime_id = int(row.get("regime_id", -1))
                merged_rows.append(existing_by_id.get(regime_id, row))
            rows = merged_rows
        if objs is not None and trace_metrics is not None and shared_episode_seeds is not None:
            start_upto = max(1, len(existing_regime_rows) + 1)
            for upto in range(start_upto, len(rows) + 1):
                partial_rows = rows[:upto]
                _save_partial_regime_result(
                    args,
                    objs,
                    partial_rows,
                    trace_metrics,
                    shared_episode_seeds,
                    shared_regime_seed_plan=shared_regime_seed_plan,
                )
        return rows
    return _group_regime_returns_from_trace(args, dynamic_trace)


def save_trace_csv(path: Path, trace: dict[str, np.ndarray] | None) -> None:
    if trace is None or len(trace.get("obj", [])) == 0:
        return
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "step",
                "episode",
                "regime",
                "regime_id",
                *[f"obj_{idx}" for idx in range(trace["obj"].shape[1])],
            ]
        )
        for idx in range(len(trace["obj"])):
            writer.writerow(
                [
                    int(trace["step"][idx]),
                    int(np.asarray(trace.get("episode", np.zeros(len(trace["obj"]), dtype=np.int64)))[idx]),
                    trace["regime"][idx],
                    trace["regime_id"][idx],
                    *trace["obj"][idx].tolist(),
                ]
            )


def save_regime_csv(path: Path, regime_rows: list[dict[str, object]]) -> None:
    if not regime_rows:
        return
    obj_num = len(regime_rows[0]["objs"])
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["episode", "regime", "regime_id", "regime_meta_json", *[f"obj_{idx}" for idx in range(obj_num)]])
        for row in regime_rows:
            writer.writerow(
                [
                    row.get("episode", -1),
                    row["regime"],
                    row.get("regime_id", -1),
                    json.dumps(row.get("regime_meta", {}), ensure_ascii=False),
                    *row["objs"],
                ]
            )


def _load_existing_partial_result(
    args,
) -> tuple[list[dict[str, object]], list[int], list[dict[str, object]]] | None:
    result_path = args.save_dir / "result.json"
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text())
    except Exception:
        return None
    regime_rows = list(payload.get("regime_returns", []))
    shared_episode_seeds = [int(seed) for seed in payload.get("shared_eval_episode_seeds", [])]
    shared_regime_seed_plan = list(payload.get("shared_regime_seed_plan", []))
    if (
        args.env_name == "chlor_alkali_dynamic"
        and regime_rows
        and not shared_regime_seed_plan
        and len(shared_episode_seeds) >= int(getattr(args, "fine_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS))
    ):
        canonical_plan, canonical_seeds = canonical_chlor_regime_seed_plan(
            env_name=args.env_name,
            base_seed=int(args.seed),
            shared_regime_eval_episodes=int(getattr(args, "fine_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
            seed_offset=int(args.shared_regime_seed_offset),
            num_regime_clusters=int(getattr(args, "chlor_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS)),
        )
        canonical_by_id = {
            int(row["regime_id"]): str(row["regime"])
            for row in canonical_chlor_regime_rows(
                num_regime_clusters=int(getattr(args, "chlor_regime_clusters", DEFAULT_FINE_REGIME_CLUSTERS))
            )
        }
        compatible_rows = []
        for row in regime_rows:
            regime_id = int(row.get("regime_id", -1))
            regime_name = str(row.get("regime", ""))
            if regime_id < 0:
                continue
            canonical_name = canonical_by_id.get(regime_id)
            if canonical_name is None or canonical_name != regime_name:
                continue
            compatible_rows.append(row)
        if compatible_rows:
            compatible_rows.sort(key=lambda row: int(row.get("regime_id", -1)))
            regime_rows = compatible_rows
            shared_regime_seed_plan = canonical_plan
            shared_episode_seeds = canonical_seeds
    if not regime_rows or not shared_episode_seeds or not shared_regime_seed_plan:
        return None
    return regime_rows, shared_episode_seeds, shared_regime_seed_plan


def _prune_partial_artifacts(args, keep_regime_rows: list[dict[str, object]]) -> None:
    keep_stems = {_regime_stem(row) for row in keep_regime_rows}
    regime_dir = args.save_dir / "shared_regime_returns"
    if regime_dir.exists():
        for path in regime_dir.iterdir():
            if path.suffix not in {".json", ".npz"}:
                continue
            stem = path.stem
            if stem not in keep_stems:
                path.unlink()
    save_regime_csv(args.save_dir / "regime_returns.csv", keep_regime_rows)


def _regime_stem(row: dict[str, object]) -> str:
    regime_id = int(row.get("regime_id", -1))
    regime_name = str(row.get("regime", "regime"))
    return f"regime_{regime_id:03d}_{regime_name}".replace("/", "_").replace(" ", "_")


def save_incremental_regime_artifacts(args, regime_rows: list[dict[str, object]]) -> None:
    if not regime_rows:
        return
    regime_dir = args.save_dir / "shared_regime_returns"
    regime_dir.mkdir(parents=True, exist_ok=True)
    for row in regime_rows:
        stem = _regime_stem(row)
        payload = {
            "alg": args.alg,
            "env_name": args.env_name,
            "seed": args.seed,
            "weights": list(args.weights),
            "episode": int(row.get("episode", -1)),
            "regime": str(row.get("regime", "regime")),
            "regime_id": int(row.get("regime_id", -1)),
            "regime_meta": dict(row.get("regime_meta", {})),
            "context_vector": list(row.get("context_vector", [])),
            "objs": list(row.get("objs", [])),
        }
        (regime_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(
            regime_dir / f"{stem}.npz",
            objs=np.asarray(row.get("objs", []), dtype=np.float32),
            context_vector=np.asarray(row.get("context_vector", []), dtype=np.float32),
            regime_id=np.asarray([int(row.get("regime_id", -1))], dtype=np.int32),
        )


def save_result(
    args,
    objs: np.ndarray,
    trace: dict[str, np.ndarray] | None,
    dynamic_trace: dict[str, np.ndarray] | None,
    regime_rows: list[dict[str, object]],
    trace_metrics: dict[str, float],
    shared_episode_seeds: list[int],
    shared_regime_seed_plan: list[dict[str, object]] | None = None,
):
    args.save_dir.mkdir(parents=True, exist_ok=True)
    save_incremental_regime_artifacts(args, regime_rows)
    (args.save_dir / "result.json").write_text(
        json.dumps(
            {
                "alg": args.alg,
                "env_name": args.env_name,
                "seed": args.seed,
                "weights": args.weights,
                "total_timesteps": args.total_timesteps,
                "objs": objs.tolist(),
                "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                "shared_regime_seed_plan": list(shared_regime_seed_plan or []),
                "trace_metrics": trace_metrics,
                "regime_returns": regime_rows,
            },
            indent=2,
        )
    )
    with open(args.save_dir / "result.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["alg", "env_name", "seed", "weights", "objs", "trace_metrics"])
        writer.writerow([args.alg, args.env_name, args.seed, json.dumps(args.weights), json.dumps(objs.tolist()), json.dumps(trace_metrics)])
    save_trace_csv(args.save_dir / "trace.csv", trace)
    save_trace_csv(args.save_dir / "dynamic_trace.csv", dynamic_trace)
    save_regime_csv(args.save_dir / "regime_returns.csv", regime_rows)


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.save_dir / "model"
    existing_regime_rows: list[dict[str, object]] = []
    existing_shared_episode_seeds: list[int] | None = None
    existing_shared_regime_seed_plan: list[dict[str, object]] | None = None
    existing_partial = _load_existing_partial_result(args)
    if existing_partial is None:
        shutil.rmtree(args.save_dir / "shared_regime_returns", ignore_errors=True)
        for stale_name in ["result.json", "result.csv", "regime_returns.csv", "trace.csv", "dynamic_trace.csv"]:
            stale_path = args.save_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
    else:
        partial_rows, partial_seeds, partial_plan = existing_partial
        existing_regime_rows = list(partial_rows)
        existing_shared_episode_seeds = list(partial_seeds)
        existing_shared_regime_seed_plan = list(partial_plan)
        _prune_partial_artifacts(args, existing_regime_rows)
    if not args.skip_train and model_path.exists():
        args.skip_train = True
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    print(f"[baseline] start alg={args.alg} env={args.env_name} seed={args.seed}", flush=True)
    env = make_training_env(args)
    print("[baseline] training env built", flush=True)
    bootstrap_tensorflow()
    print("[baseline] tf bootstrapped", flush=True)
    if args.skip_train:
        model = load_model(args, env)
        print("[baseline] model loaded", flush=True)
    else:
        model = train_model(args, env)
        save_model(args, model)
        print("[baseline] training complete", flush=True)
    if existing_shared_regime_seed_plan and existing_shared_episode_seeds:
        shared_regime_seed_plan = existing_shared_regime_seed_plan
        shared_episode_seeds = existing_shared_episode_seeds
    else:
        shared_regime_seed_plan, shared_episode_seeds = _shared_regime_seed_plan(args)
    objs, trace = evaluate_model(args, model, return_trace=True, episode_seeds=shared_episode_seeds)
    dynamic_trace = evaluate_dynamic_trace(args, model, episode_seeds=shared_episode_seeds)
    trace_metrics = compute_trace_shift_metrics(
        dynamic_trace or trace or {"obj": [], "regime_id": []},
        args.eval_delta_weight,
        recovery_window=args.trace_recovery_window,
    )
    regime_rows = evaluate_regimes(
        args,
        model,
        dynamic_trace=dynamic_trace,
        objs=objs,
        trace_metrics=trace_metrics,
        shared_episode_seeds=shared_episode_seeds,
        shared_regime_seed_plan=shared_regime_seed_plan,
        existing_regime_rows=existing_regime_rows,
    )
    print("[baseline] evaluation complete", flush=True)
    save_result(
        args,
        objs,
        trace,
        dynamic_trace,
        regime_rows,
        trace_metrics,
        shared_episode_seeds,
        shared_regime_seed_plan=shared_regime_seed_plan,
    )
    print({"alg": args.alg, "env_name": args.env_name, "seed": args.seed, "objs": objs.tolist(), "trace_metrics": trace_metrics})


if __name__ == "__main__":
    main()
