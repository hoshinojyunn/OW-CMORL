from __future__ import annotations

from collections import OrderedDict
from typing import Any

import gym
import numpy as np
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "sustaingym"))

from sustaingym.envs.cogen import CogenEnv
from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator

from .dynamic_building import BUILDING_REGIMES, DynamicBuildingEnv


DEFAULT_EV_PERIODS = ("Summer 2019", "Spring 2020", "Summer 2021")
DEFAULT_COGEN_RENEWABLES = (0.0, 10.0, 20.0)
DEFAULT_BUILDING_WEATHERS = ("Hot_Dry", "Warm_Marine", "Mixed_Marine")
CANONICAL_EV_CHARGER_SLOTS = 54
EV_CONTEXT_FEATURES = (
    "time_frac",
    "active_ratio",
    "demand_ratio",
    "deadline_pressure",
    "prev_moer",
    "forecast_mean",
    "forecast_trend",
    "forecast_peak",
)
COGEN_CONTEXT_FEATURES = (
    "time_frac",
    "renewables_ratio",
    "ambient_temp",
    "ambient_temp_trend",
    "ambient_humidity",
    "target_power",
    "target_power_trend",
    "target_steam",
    "energy_price",
    "gas_price",
)


def _flatten_space(space) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(space, "spaces"):
        lows = []
        highs = []
        for subspace in space.spaces.values():
            low, high = _flatten_space(subspace)
            lows.append(low)
            highs.append(high)
        return np.concatenate(lows, axis=0), np.concatenate(highs, axis=0)

    if hasattr(space, "n") and not hasattr(space, "low"):
        start = getattr(space, "start", 0)
        return (
            np.asarray([start], dtype=np.float32),
            np.asarray([start + space.n - 1], dtype=np.float32),
        )

    return (
        np.asarray(space.low, dtype=np.float32).reshape(-1),
        np.asarray(space.high, dtype=np.float32).reshape(-1),
    )


def _flatten_value(value) -> np.ndarray:
    if isinstance(value, dict):
        parts = [_flatten_value(v) for v in value.values()]
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _unit_scale(value: float, low: float, high: float) -> float:
    span = max(float(high - low), 1e-6)
    return float(np.clip((float(value) - float(low)) / span, 0.0, 1.0))


def _trend_scale(start: float, end: float, magnitude: float) -> float:
    magnitude = max(float(magnitude), 1e-6)
    return float(np.clip(0.5 + 0.5 * (float(end) - float(start)) / magnitude, 0.0, 1.0))


class DynamicEVChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        site: str = "caltech",
        periods: tuple[str, ...] = DEFAULT_EV_PERIODS,
        schedule: str = "cyclic",
        episodes_per_period: int = 4,
        moer_forecast_steps: int = 36,
        project_action_in_env: bool = True,
        seed: int | None = None,
    ):
        assert schedule in {"cyclic", "random"}
        self.site = site
        self.periods = tuple(periods)
        self.schedule = schedule
        self.episodes_per_period = episodes_per_period
        self.moer_forecast_steps = moer_forecast_steps
        self.project_action_in_env = project_action_in_env
        self.rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.current_period_index = 0
        self._elapsed_steps = 0
        self._prev_breakdown = {"profit": 0.0, "carbon_cost": 0.0, "excess_charge": 0.0}
        self.last_obs_dict: dict[str, np.ndarray] | None = None
        self._env_cache: dict[int, EVChargingEnv] = {}
        self._env = self._get_or_create_env(self.current_period_index)
        obs_low, obs_high = self._canonical_obs_bounds(self._env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(CANONICAL_EV_CHARGER_SLOTS, dtype=np.float32),
            high=np.ones(CANONICAL_EV_CHARGER_SLOTS, dtype=np.float32),
            dtype=np.float32,
        )
        self._max_episode_steps = self._env.max_timestep

    @property
    def current_period(self) -> str:
        return self.periods[self.current_period_index]

    def _make_env(self, period: str) -> EVChargingEnv:
        generator = RealTraceGenerator(self.site, period)
        env = EVChargingEnv(
            generator,
            moer_forecast_steps=self.moer_forecast_steps,
            project_action_in_env=self.project_action_in_env,
        )
        self._max_episode_steps = env.max_timestep
        return env

    def _get_or_create_env(self, period_index: int) -> EVChargingEnv:
        env = self._env_cache.get(period_index)
        if env is None:
            env = self._make_env(self.periods[period_index])
            self._env_cache[period_index] = env
        self._max_episode_steps = env.max_timestep
        return env

    def _select_period_index(self) -> int:
        if self.schedule == "random":
            return int(self.rng.integers(len(self.periods)))
        bucket = self.episode_count // self.episodes_per_period
        return bucket % len(self.periods)

    @staticmethod
    def _pad_charger_slots(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(values) >= CANONICAL_EV_CHARGER_SLOTS:
            return values[:CANONICAL_EV_CHARGER_SLOTS]
        return np.pad(values, (0, CANONICAL_EV_CHARGER_SLOTS - len(values)))

    def _canonical_obs_bounds(self, observation_space) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(observation_space, "spaces"):
            return _flatten_space(observation_space)
        lows = []
        highs = []
        for key, subspace in observation_space.spaces.items():
            low, high = _flatten_space(subspace)
            if key in {"est_departures", "demands"}:
                low = self._pad_charger_slots(low)
                high = self._pad_charger_slots(high)
            lows.append(low)
            highs.append(high)
        return np.concatenate(lows, axis=0), np.concatenate(highs, axis=0)

    def _flatten_obs(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        self.last_obs_dict = obs
        parts = []
        for key, value in obs.items():
            values = np.asarray(value, dtype=np.float32).reshape(-1)
            if key in {"est_departures", "demands"}:
                values = self._pad_charger_slots(values)
            parts.append(values)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _context_payload(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        demands = np.asarray(obs["demands"], dtype=np.float32)
        departures = np.asarray(obs["est_departures"], dtype=np.float32)
        forecast = np.asarray(obs["forecasted_moer"], dtype=np.float32)
        prev_moer = float(np.asarray(obs["prev_moer"], dtype=np.float32).reshape(-1)[0])
        active_mask = demands > 1e-6
        active_ratio = float(active_mask.mean()) if len(active_mask) > 0 else 0.0
        max_demand = max(float(self._env.data_generator.requested_energy_cap), 1e-6)
        demand_ratio = float(
            np.clip(demands.sum() / max(max_demand * len(demands), 1e-6), 0.0, 1.0)
        )
        if np.any(active_mask):
            mean_departure = float(np.mean(np.maximum(departures[active_mask], 0.0)))
        else:
            mean_departure = float(self._env.max_timestep)
        deadline_pressure = float(
            np.clip(1.0 - mean_departure / max(float(self._env.max_timestep), 1.0), 0.0, 1.0)
        )
        forecast_mean = float(np.mean(forecast)) if len(forecast) > 0 else prev_moer
        forecast_peak = float(np.max(forecast)) if len(forecast) > 0 else prev_moer
        forecast_trend = (
            _trend_scale(float(forecast[0]), float(forecast[-1]), 1.0)
            if len(forecast) > 0
            else 0.5
        )
        vector = np.asarray(
            [
                float(np.asarray(obs["timestep"], dtype=np.float32).reshape(-1)[0]),
                active_ratio,
                demand_ratio,
                deadline_pressure,
                float(np.clip(prev_moer, 0.0, 1.0)),
                float(np.clip(forecast_mean, 0.0, 1.0)),
                forecast_trend,
                float(np.clip(forecast_peak, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return {
            "context_source": "environment_window",
            "context_label": self.current_period,
            "context_feature_names": list(EV_CONTEXT_FEATURES),
            "context_vector": vector,
            "context_dict": {
                "time_frac": float(vector[0]),
                "active_ratio": active_ratio,
                "demand_ratio": demand_ratio,
                "deadline_pressure": deadline_pressure,
                "prev_moer": prev_moer,
                "forecast_mean": forecast_mean,
                "forecast_trend": forecast_trend,
                "forecast_peak": forecast_peak,
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        next_period_index = self._select_period_index()
        if next_period_index != self.current_period_index or self.episode_count == 0:
            self.current_period_index = next_period_index
            self._env = self._get_or_create_env(self.current_period_index)

        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_breakdown = {"profit": 0.0, "carbon_cost": 0.0, "excess_charge": 0.0}
        self.episode_count += 1
        self._elapsed_steps = 0
        info = dict(info)
        info["period_id"] = self.current_period_index
        info["period_name"] = self.current_period
        info.update(self._context_payload(obs))
        return self._flatten_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, -1.0, 1.0)
        action = 0.5 * (action + 1.0)
        native_action_size = int(np.prod(self._env.action_space.shape))
        obs, _reward, terminated, truncated, info = self._env.step(action[:native_action_size])
        breakdown = dict(info.get("reward_breakdown", {}))
        profit = float(breakdown.get("profit", 0.0) - self._prev_breakdown["profit"])
        carbon = float(
            breakdown.get("carbon_cost", 0.0) - self._prev_breakdown["carbon_cost"]
        )
        excess = float(
            breakdown.get("excess_charge", 0.0) - self._prev_breakdown["excess_charge"]
        )
        self._prev_breakdown = {
            "profit": float(breakdown.get("profit", 0.0)),
            "carbon_cost": float(breakdown.get("carbon_cost", 0.0)),
            "excess_charge": float(breakdown.get("excess_charge", 0.0)),
        }
        self._elapsed_steps += 1
        obj = np.asarray([profit, -carbon, -excess], dtype=np.float32)
        info = dict(info)
        info["period_id"] = self.current_period_index
        info["period_name"] = self.current_period
        info["reward_vector"] = obj
        info.update(self._context_payload(obs))
        return self._flatten_obs(obs), obj, bool(terminated), bool(truncated), info

    def close(self):
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()

    def render(self):
        return None


class DynamicCogenEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        renewables_magnitudes: tuple[float, ...] = DEFAULT_COGEN_RENEWABLES,
        schedule: str = "cyclic",
        episodes_per_regime: int = 4,
        forecast_horizon: int = 3,
        forecast_noise_std: float = 0.0,
        seed: int | None = None,
    ):
        assert schedule in {"cyclic", "random"}
        self.renewables_magnitudes = tuple(renewables_magnitudes)
        self.schedule = schedule
        self.episodes_per_regime = episodes_per_regime
        self.forecast_horizon = forecast_horizon
        self.forecast_noise_std = forecast_noise_std
        self.rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.current_regime_index = 0
        self._elapsed_steps = 0
        self._env_cache: dict[int, CogenEnv] = {}
        self._env = self._get_or_create_env(self.current_regime_index)
        self._action_keys = list(self._env.action_space.spaces.keys())
        self.observation_space = gym.spaces.Box(
            low=_flatten_space(self._env.observation_space)[0],
            high=_flatten_space(self._env.observation_space)[1],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(len(self._action_keys), dtype=np.float32),
            high=np.ones(len(self._action_keys), dtype=np.float32),
            dtype=np.float32,
        )
        self._max_episode_steps = self._env.timesteps_per_day

    @property
    def current_regime(self) -> float:
        return self.renewables_magnitudes[self.current_regime_index]

    def _make_env(self, renewables_magnitude: float) -> CogenEnv:
        env = CogenEnv(
            renewables_magnitude=renewables_magnitude,
            forecast_horizon=self.forecast_horizon,
            forecast_noise_std=self.forecast_noise_std,
        )
        self._max_episode_steps = env.timesteps_per_day
        return env

    def _get_or_create_env(self, regime_index: int) -> CogenEnv:
        env = self._env_cache.get(regime_index)
        if env is None:
            env = self._make_env(self.renewables_magnitudes[regime_index])
            self._env_cache[regime_index] = env
        self._max_episode_steps = env.timesteps_per_day
        return env

    def _select_regime_index(self) -> int:
        if self.schedule == "random":
            return int(self.rng.integers(len(self.renewables_magnitudes)))
        bucket = self.episode_count // self.episodes_per_regime
        return bucket % len(self.renewables_magnitudes)

    def _flatten_obs(self, obs: dict[str, Any]) -> np.ndarray:
        return _flatten_value(obs)

    def _context_payload(self, obs: dict[str, Any]) -> dict[str, Any]:
        tamb = np.asarray(obs["TAMB"], dtype=np.float32)
        rh = np.asarray(obs["RHAMB"], dtype=np.float32)
        target_power = np.asarray(obs["Target_Power"], dtype=np.float32)
        target_steam = np.asarray(obs["Target_Steam"], dtype=np.float32)
        energy_price = np.asarray(obs["Energy_Price"], dtype=np.float32)
        gas_price = np.asarray(obs["Gas_Price"], dtype=np.float32)
        renewables_scale = max(max(map(abs, self.renewables_magnitudes)), 1.0)
        vector = np.asarray(
            [
                float(np.asarray(obs["Time"], dtype=np.float32).reshape(-1)[0]),
                float(np.clip(self.current_regime / renewables_scale, 0.0, 1.0)),
                _unit_scale(float(tamb[0]), 32.0, 115.0),
                _trend_scale(float(tamb[0]), float(tamb[min(len(tamb) - 1, 1)]), 20.0),
                _unit_scale(float(rh[0]), 0.0, 1.0),
                _unit_scale(float(target_power[0]), 0.0, 700.0),
                _trend_scale(
                    float(target_power[0]),
                    float(target_power[min(len(target_power) - 1, 1)]),
                    700.0,
                ),
                _unit_scale(float(target_steam[0]), 0.0, 1300.0),
                _unit_scale(float(energy_price[0]), 0.0, 1500.0),
                _unit_scale(float(gas_price[0]), 0.0, 7.0),
            ],
            dtype=np.float32,
        )
        return {
            "context_source": "environment_window",
            "context_label": f"renewables_{self.current_regime:g}",
            "context_feature_names": list(COGEN_CONTEXT_FEATURES),
            "context_vector": vector,
            "context_dict": {
                "time_frac": float(vector[0]),
                "renewables_magnitude": float(self.current_regime),
                "ambient_temp": float(tamb[0]),
                "ambient_temp_trend": float(vector[3]),
                "ambient_humidity": float(rh[0]),
                "target_power": float(target_power[0]),
                "target_power_trend": float(vector[6]),
                "target_steam": float(target_steam[0]),
                "energy_price": float(energy_price[0]),
                "gas_price": float(gas_price[0]),
            },
        }

    def _map_action(self, action: np.ndarray) -> OrderedDict[str, Any]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        mapped = OrderedDict()
        for idx, key in enumerate(self._action_keys):
            subspace = self._env.action_space.spaces[key]
            value = float(np.clip(action[idx], -1.0, 1.0))
            if hasattr(subspace, "n") and not hasattr(subspace, "low"):
                start = getattr(subspace, "start", 0)
                discrete_idx = int(round((value + 1.0) * 0.5 * (subspace.n - 1)))
                mapped[key] = int(start + discrete_idx)
            else:
                low = float(np.asarray(subspace.low).reshape(-1)[0])
                high = float(np.asarray(subspace.high).reshape(-1)[0])
                mapped_value = low + 0.5 * (value + 1.0) * (high - low)
                mapped[key] = np.asarray([mapped_value], dtype=np.float32)
        return mapped

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        next_regime_index = self._select_regime_index()
        if next_regime_index != self.current_regime_index or self.episode_count == 0:
            self.current_regime_index = next_regime_index
            self._env = self._get_or_create_env(self.current_regime_index)

        obs, info = self._env.reset(seed=seed, options=options)
        self.episode_count += 1
        self._elapsed_steps = 0
        info = dict(info)
        info["renewables_magnitude"] = self.current_regime
        info["regime_id"] = self.current_regime_index
        info.update(self._context_payload(obs))
        return self._flatten_obs(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        mapped_action = self._map_action(action)
        obs, _reward, terminated, truncated, info = self._env.step(mapped_action)
        fuel = float(sum(info["fuel_costs"].values()))
        ramp = float(sum(info["ramp_costs"].values()))
        dyn = float(sum(info["dyn_cv_costs"].values()))
        non_delivery = float(info["non_delivery_cost"])
        self._elapsed_steps += 1
        obj = np.asarray([-fuel, -ramp, -non_delivery, -dyn], dtype=np.float32)
        info = dict(info)
        info["renewables_magnitude"] = self.current_regime
        info["regime_id"] = self.current_regime_index
        info["reward_vector"] = obj
        info.update(self._context_payload(obs))
        return self._flatten_obs(obs), obj, bool(terminated), bool(truncated), info

    def close(self):
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()

    def render(self):
        return None


class DynamicSustainBuildingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        weathers: tuple[str, ...] = DEFAULT_BUILDING_WEATHERS,
        schedule: str = "cyclic",
        episodes_per_regime: int = 4,
        seed: int | None = None,
    ):
        assert schedule in {"cyclic", "random"}
        self.weathers = tuple(weathers)
        self.schedule = schedule
        self.episodes_per_regime = episodes_per_regime
        self.rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.current_regime_index = 0
        self._elapsed_steps = 0
        self._prev_breakdown = {"comfort_level": 0.0, "power_consumption": 0.0}
        self._env_cache: dict[int, DynamicBuildingEnv] = {}
        self._env = self._get_or_create_env(self.current_regime_index)
        self.observation_space = gym.spaces.Box(
            low=np.asarray(self._env.observation_space.low, dtype=np.float32),
            high=np.asarray(self._env.observation_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(self._env.action_space.low, dtype=np.float32),
            high=np.asarray(self._env.action_space.high, dtype=np.float32),
            dtype=np.float32,
        )
        self._max_episode_steps = getattr(self._env, "episode_len", 288)

    @property
    def current_weather(self) -> str:
        return self.weathers[self.current_regime_index]

    def _make_env(self, weather: str) -> DynamicBuildingEnv:
        regime_index = next(
            (
                idx
                for idx, regime in enumerate(BUILDING_REGIMES)
                if regime.weather == weather
            ),
            self.current_regime_index % len(BUILDING_REGIMES),
        )
        env = DynamicBuildingEnv(
            dimension="3d",
            regimes=(BUILDING_REGIMES[regime_index],),
            schedule="cyclic",
            episodes_per_regime=10**9,
        )
        self._max_episode_steps = getattr(env, "_max_episode_steps", 480)
        return env

    def _get_or_create_env(self, regime_index: int) -> DynamicBuildingEnv:
        env = self._env_cache.get(regime_index)
        if env is None:
            env = self._make_env(self.weathers[regime_index])
            self._env_cache[regime_index] = env
        self._max_episode_steps = getattr(env, "_max_episode_steps", 480)
        return env

    def _select_regime_index(self) -> int:
        if self.schedule == "random":
            return int(self.rng.integers(len(self.weathers)))
        bucket = self.episode_count // self.episodes_per_regime
        return bucket % len(self.weathers)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        next_regime_index = self._select_regime_index()
        if next_regime_index != self.current_regime_index or self.episode_count == 0:
            self.current_regime_index = next_regime_index
            self._env = self._get_or_create_env(self.current_regime_index)

        obs, info = self._env.reset(seed=seed, options=options)
        self._prev_breakdown = {"comfort_level": 0.0, "power_consumption": 0.0}
        self.episode_count += 1
        self._elapsed_steps = 0
        info = dict(info)
        info["weather"] = self.current_weather
        info["regime_id"] = self.current_regime_index
        return np.asarray(obs, dtype=np.float32), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        obs, obj, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32)
        )
        obj = np.asarray(obj, dtype=np.float32).reshape(-1)
        self._elapsed_steps += 1
        info = dict(info)
        info["weather"] = self.current_weather
        info["regime_id"] = self.current_regime_index
        info["reward_vector"] = obj
        return np.asarray(obs, dtype=np.float32), obj, bool(terminated), bool(truncated), info

    def close(self):
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()

    def render(self):
        return None


class ScalarizedRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, weights: np.ndarray):
        super().__init__(env)
        self.weights = np.asarray(weights, dtype=np.float32)

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, obj, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info = dict(info)
        info["obj"] = np.asarray(obj, dtype=np.float32)
        info["obj_raw"] = np.asarray(obj, dtype=np.float32)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True
        reward = float(np.dot(info["obj_raw"], self.weights))
        return obs, reward, done, info


class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert hasattr(env.action_space, "low") and hasattr(env.action_space, "high")
        self._low = np.asarray(env.action_space.low, dtype=np.float32)
        self._high = np.asarray(env.action_space.high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(self._low),
            high=np.ones_like(self._high),
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        return self._low + 0.5 * (action + 1.0) * (self._high - self._low)


class DiscreteActionBankWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_bank: np.ndarray):
        super().__init__(env)
        self.action_bank = np.asarray(action_bank, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.action_bank))

    def step(self, action):
        return self.env.step(self.action_bank[int(action)])


class EVChargingAggressionDiscreteWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, levels: int = 7):
        super().__init__(env)
        self.levels = np.linspace(0.0, 1.0, levels, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(levels)

    def _continuous_action(self, action: int) -> np.ndarray:
        base_env = self.unwrapped
        if getattr(base_env, "last_obs_dict", None) is None:
            shape = base_env.action_space.shape
            return np.zeros(shape, dtype=np.float32)
        alpha = float(self.levels[int(action)])
        demands = np.asarray(base_env.last_obs_dict["demands"], dtype=np.float32)
        max_rate = (
            base_env._env.A_PERS_TO_KWH * base_env._env.ACTION_SCALE_FACTOR
        )
        normalized_max = np.clip(demands / max_rate, 0.0, 1.0)
        return alpha * normalized_max

    def step(self, action):
        return self.env.step(self._continuous_action(int(action)))
