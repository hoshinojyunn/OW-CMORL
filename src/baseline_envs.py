from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gym
import numpy as np

from src.dynamic_morl.chlor_alkali_env import ChlorAlkaliEnv
from src.dynamic_morl.dynamic_building import BUILDING_REGIMES, DynamicBuildingEnv
from src.dynamic_morl.sustaingym_wrappers import (
    DEFAULT_COGEN_RENEWABLES,
    DEFAULT_EV_PERIODS,
    DynamicCogenEnv,
    DynamicEVChargingEnv,
)


@dataclass(frozen=True)
class BaselineEnvSpec:
    env_key: str
    dynamic_env_name: str
    static_env_name: str
    obj_num: int
    vector_env_id: str
    scalar_env_id: str
    preference_grid: tuple[tuple[float, ...], ...]


BASELINE_ENVS: dict[str, BaselineEnvSpec] = {
    "building": BaselineEnvSpec(
        env_key="building",
        dynamic_env_name="building_3d_dynamic",
        static_env_name="building_3d_static",
        obj_num=3,
        vector_env_id="DynBuildingM-v0",
        scalar_env_id="DynBuildingPG-v0",
        preference_grid=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    ),
    "evcharging": BaselineEnvSpec(
        env_key="evcharging",
        dynamic_env_name="evcharging_dynamic",
        static_env_name="evcharging_static",
        obj_num=3,
        vector_env_id="DynEVChargingM-v0",
        scalar_env_id="DynEVChargingPG-v0",
        preference_grid=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    ),
    "cogen": BaselineEnvSpec(
        env_key="cogen",
        dynamic_env_name="cogen_dynamic",
        static_env_name="cogen_static",
        obj_num=4,
        vector_env_id="DynCogenM-v0",
        scalar_env_id="DynCogenPG-v0",
        preference_grid=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0, 0.0),
            (0.5, 0.0, 0.5, 0.0),
            (0.5, 0.0, 0.0, 0.5),
            (0.0, 0.5, 0.5, 0.0),
            (0.0, 0.5, 0.0, 0.5),
            (0.0, 0.0, 0.5, 0.5),
            (0.25, 0.25, 0.25, 0.25),
        ),
    ),
    "chlor_alkali": BaselineEnvSpec(
        env_key="chlor_alkali",
        dynamic_env_name="chlor_alkali_dynamic",
        static_env_name="chlor_alkali_static",
        obj_num=3,
        vector_env_id="DynChlorAlkaliM-v0",
        scalar_env_id="DynChlorAlkaliPG-v0",
        preference_grid=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    ),
}


PGMORL_OBJ_OFFSETS: dict[str, np.ndarray] = {
    "building": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
    "evcharging": np.asarray([1.0, 2.0, 0.01], dtype=np.float32),
    "cogen": np.asarray([20000.0, 1000.0, 3500000.0, 12000000.0], dtype=np.float32),
    "chlor_alkali": np.asarray([600.0, 120.0, 3000.0], dtype=np.float32),
}


PGMORL_OBJ_SCALES: dict[str, np.ndarray] = {
    "building": np.asarray([20000.0, 20000.0, 20000.0], dtype=np.float32),
    "evcharging": np.asarray([20.0, 2.0, 0.01], dtype=np.float32),
    "cogen": np.asarray([20000.0, 1000.0, 3500000.0, 12000000.0], dtype=np.float32),
    "chlor_alkali": np.asarray([600.0, 120.0, 3000.0], dtype=np.float32),
}


def get_env_spec(env_key: str) -> BaselineEnvSpec:
    try:
        return BASELINE_ENVS[env_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported env key: {env_key}") from exc


def preference_grid(env_key: str) -> list[list[float]]:
    spec = get_env_spec(env_key)
    return [list(weights) for weights in spec.preference_grid]


def default_env_kwargs(env_key: str) -> dict[str, Any]:
    if env_key == "building":
        return {"regimes": tuple(BUILDING_REGIMES)}
    if env_key == "evcharging":
        return {"site": "caltech", "periods": tuple(DEFAULT_EV_PERIODS)}
    if env_key == "cogen":
        return {
            "renewables": tuple(DEFAULT_COGEN_RENEWABLES),
            "forecast_horizon": 3,
            "forecast_noise_std": 0.0,
        }
    if env_key == "chlor_alkali":
        return {
            "dynamic_price": True,
            "aging_dynamics": True,
            "episode_length": 288,
            "num_regime_clusters": 20,
        }
    raise ValueError(f"Unsupported env key: {env_key}")


def resolve_env_key(env_name: str) -> str:
    for env_key, spec in BASELINE_ENVS.items():
        if env_name in {spec.dynamic_env_name, spec.static_env_name}:
            return env_key
    raise ValueError(f"Unsupported env name: {env_name}")


def build_dynamic_env(
    env_name: str,
    *,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    schedule: str = "cyclic",
    episodes_per_regime: int = 1,
):
    env_kwargs = {**default_env_kwargs(resolve_env_key(env_name)), **dict(env_kwargs or {})}
    if env_name == "building_3d_dynamic":
        return DynamicBuildingEnv(
            dimension="3d",
            regimes=tuple(env_kwargs.get("regimes", BUILDING_REGIMES)),
            schedule=env_kwargs.get("schedule", schedule),
            episodes_per_regime=int(env_kwargs.get("episodes_per_regime", episodes_per_regime)),
            seed=seed,
        )
    if env_name == "building_3d_static":
        return DynamicBuildingEnv(
            dimension="3d",
            regimes=tuple(env_kwargs.get("regimes", (BUILDING_REGIMES[0],))),
            schedule=env_kwargs.get("schedule", "cyclic"),
            episodes_per_regime=int(env_kwargs.get("episodes_per_regime", 10**9)),
            seed=seed,
        )
    if env_name == "evcharging_dynamic":
        return DynamicEVChargingEnv(
            site=env_kwargs.get("site", "caltech"),
            periods=tuple(env_kwargs.get("periods", DEFAULT_EV_PERIODS)),
            schedule=env_kwargs.get("schedule", schedule),
            episodes_per_period=int(env_kwargs.get("episodes_per_regime", episodes_per_regime)),
            moer_forecast_steps=int(env_kwargs.get("moer_forecast_steps", 36)),
            project_action_in_env=bool(env_kwargs.get("project_action_in_env", True)),
            seed=seed,
        )
    if env_name == "evcharging_static":
        periods = tuple(env_kwargs.get("periods", DEFAULT_EV_PERIODS))
        return DynamicEVChargingEnv(
            site=env_kwargs.get("site", "caltech"),
            periods=(periods[0],),
            schedule=env_kwargs.get("schedule", "cyclic"),
            episodes_per_period=int(env_kwargs.get("episodes_per_regime", 10**9)),
            moer_forecast_steps=int(env_kwargs.get("moer_forecast_steps", 36)),
            project_action_in_env=bool(env_kwargs.get("project_action_in_env", True)),
            seed=seed,
        )
    if env_name == "cogen_dynamic":
        return DynamicCogenEnv(
            renewables_magnitudes=tuple(env_kwargs.get("renewables", DEFAULT_COGEN_RENEWABLES)),
            schedule=env_kwargs.get("schedule", schedule),
            episodes_per_regime=int(env_kwargs.get("episodes_per_regime", episodes_per_regime)),
            forecast_horizon=int(env_kwargs.get("forecast_horizon", 3)),
            forecast_noise_std=float(env_kwargs.get("forecast_noise_std", 0.0)),
            seed=seed,
        )
    if env_name == "cogen_static":
        renewables = tuple(env_kwargs.get("renewables", DEFAULT_COGEN_RENEWABLES))
        return DynamicCogenEnv(
            renewables_magnitudes=(renewables[0],),
            schedule=env_kwargs.get("schedule", "cyclic"),
            episodes_per_regime=int(env_kwargs.get("episodes_per_regime", 10**9)),
            forecast_horizon=int(env_kwargs.get("forecast_horizon", 3)),
            forecast_noise_std=float(env_kwargs.get("forecast_noise_std", 0.0)),
            seed=seed,
        )
    if env_name == "chlor_alkali_dynamic":
        return ChlorAlkaliEnv(
            dataset_path=env_kwargs.get("dataset_path"),
            train_path=env_kwargs.get("train_path"),
            dynamic_price=bool(env_kwargs.get("dynamic_price", True)),
            aging_dynamics=bool(env_kwargs.get("aging_dynamics", True)),
            episode_length=int(env_kwargs.get("episode_length", 288)),
            schedule=env_kwargs.get("schedule", schedule),
            allowed_regimes=tuple(env_kwargs.get("allowed_regimes", ())),
            allowed_regime_ids=tuple(env_kwargs.get("allowed_regime_ids", ())),
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", 12)),
            price_multiplier=float(env_kwargs.get("price_multiplier", 1.0)),
            price_offset=float(env_kwargs.get("price_offset", 0.0)),
            surrogate_seed=int(env_kwargs.get("surrogate_seed", 0)),
            seed=seed,
        )
    if env_name == "chlor_alkali_static":
        return ChlorAlkaliEnv(
            dataset_path=env_kwargs.get("dataset_path"),
            train_path=env_kwargs.get("train_path"),
            dynamic_price=bool(env_kwargs.get("dynamic_price", False)),
            aging_dynamics=bool(env_kwargs.get("aging_dynamics", False)),
            episode_length=int(env_kwargs.get("episode_length", 288)),
            schedule=env_kwargs.get("schedule", "sequential"),
            allowed_regimes=tuple(env_kwargs.get("allowed_regimes", ())),
            allowed_regime_ids=tuple(env_kwargs.get("allowed_regime_ids", ())),
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", 12)),
            price_multiplier=float(env_kwargs.get("price_multiplier", 1.0)),
            price_offset=float(env_kwargs.get("price_offset", 0.0)),
            surrogate_seed=int(env_kwargs.get("surrogate_seed", 0)),
            seed=seed,
        )
    raise ValueError(f"Unsupported env name: {env_name}")


class _OldGymDynamicWrapper(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        env_name: str,
        *,
        seed: int = 0,
        env_kwargs: dict[str, Any] | None = None,
        reward_mode: str = "vector",
        obj_mode: str = "raw",
    ):
        assert reward_mode in {"vector", "scalar"}
        assert obj_mode in {"raw", "pgmorl"}
        self.env_name = env_name
        self._seed = int(seed)
        self._pending_seed = int(seed)
        self.env_kwargs = dict(env_kwargs or {})
        self.reward_mode = reward_mode
        self.obj_mode = obj_mode
        self._env = build_dynamic_env(env_name, seed=self._seed, env_kwargs=self.env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        env_key = resolve_env_key(env_name)
        self.env_key = env_key
        self.reward_num = get_env_spec(env_key).obj_num
        self.rwd_dim = self.reward_num
        self._max_episode_steps = int(getattr(self._env, "_max_episode_steps", 1000))
        self.max_episode_steps = self._max_episode_steps

    def _transform_obj(self, obj: np.ndarray) -> np.ndarray:
        obj = np.asarray(obj, dtype=np.float32).reshape(-1)
        if self.obj_mode != "pgmorl":
            return obj
        offset = PGMORL_OBJ_OFFSETS[self.env_key]
        scale = PGMORL_OBJ_SCALES[self.env_key]
        transformed = (obj + offset) / np.maximum(scale, 1e-6)
        return np.maximum(transformed, 1e-3).astype(np.float32)

    def set_params(self, _env_params=None):
        return None

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._seed = int(seed)
            self._pending_seed = int(seed)
        return [self._seed]

    def reset(self):
        if self._pending_seed is None:
            result = self._env.reset()
        else:
            result = self._env.reset(seed=self._pending_seed)
            self._pending_seed = None
        obs = result[0] if isinstance(result, tuple) else result
        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, obj, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        obj_raw = np.asarray(info.get("obj_raw", info.get("reward_vector", obj)), dtype=np.float32)
        obj = self._transform_obj(obj_raw)
        info = dict(info)
        info["obj"] = obj.copy()
        info["obj_raw"] = obj_raw.copy()
        if self.reward_mode == "vector":
            reward = obj.copy()
        else:
            reward = 0.0
        return np.asarray(obs, dtype=np.float32), reward, done, info

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        if hasattr(self._env, "render"):
            return self._env.render()
        return None


class OldGymVectorRewardWrapper(_OldGymDynamicWrapper):
    def __init__(
        self,
        env_name: str,
        *,
        seed: int = 0,
        env_kwargs: dict[str, Any] | None = None,
        obj_mode: str = "raw",
    ):
        super().__init__(env_name, seed=seed, env_kwargs=env_kwargs, reward_mode="vector", obj_mode=obj_mode)


class OldGymScalarInfoWrapper(_OldGymDynamicWrapper):
    def __init__(
        self,
        env_name: str,
        *,
        seed: int = 0,
        env_kwargs: dict[str, Any] | None = None,
        obj_mode: str = "raw",
    ):
        super().__init__(env_name, seed=seed, env_kwargs=env_kwargs, reward_mode="scalar", obj_mode=obj_mode)


def make_vector_env(env_key: str, *, seed: int = 0, env_kwargs: dict[str, Any] | None = None):
    spec = get_env_spec(env_key)
    return OldGymVectorRewardWrapper(spec.dynamic_env_name, seed=seed, env_kwargs=env_kwargs)


def make_scalar_env(
    env_key: str,
    *,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
    obj_mode: str = "raw",
):
    spec = get_env_spec(env_key)
    return OldGymScalarInfoWrapper(spec.dynamic_env_name, seed=seed, env_kwargs=env_kwargs, obj_mode=obj_mode)


def register_baseline_envs() -> None:
    for env_key, spec in BASELINE_ENVS.items():
        vector_kwargs = {
            "env_name": spec.dynamic_env_name,
            "seed": 0,
            "env_kwargs": default_env_kwargs(env_key),
        }
        scalar_kwargs = {
            "env_name": spec.dynamic_env_name,
            "seed": 0,
            "env_kwargs": default_env_kwargs(env_key),
            "obj_mode": "pgmorl",
        }
        if spec.vector_env_id not in gym.envs.registry.env_specs:
            gym.envs.register(
                id=spec.vector_env_id,
                entry_point="src.baseline_envs:OldGymVectorRewardWrapper",
                kwargs=vector_kwargs,
                max_episode_steps=1000,
            )
        if spec.scalar_env_id not in gym.envs.registry.env_specs:
            gym.envs.register(
                id=spec.scalar_env_id,
                entry_point="src.baseline_envs:OldGymScalarInfoWrapper",
                kwargs=scalar_kwargs,
                max_episode_steps=1000,
            )
