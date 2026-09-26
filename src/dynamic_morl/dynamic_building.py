from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from environments.building.env_building import BuildingEnv_3d, BuildingEnv_9d
from environments.building.utils_building import ParameterGenerator


@dataclass(frozen=True)
class BuildingRegime:
    name: str
    weather: str
    occupancy_scale: float
    carbon_scale: float
    price_scale: float
    target: float


def coerce_building_regime(value: BuildingRegime | Mapping[str, Any]) -> BuildingRegime:
    if isinstance(value, BuildingRegime):
        return value
    if isinstance(value, Mapping):
        payload = {key: value[key] for key in asdict(BUILDING_REGIMES[0]).keys() if key in value}
        return BuildingRegime(
            name=str(payload["name"]),
            weather=str(payload["weather"]),
            occupancy_scale=float(payload["occupancy_scale"]),
            carbon_scale=float(payload["carbon_scale"]),
            price_scale=float(payload["price_scale"]),
            target=float(payload["target"]),
        )
    raise TypeError(f"Unsupported building regime payload: {type(value)!r}")


def coerce_building_regimes(
    values: tuple[BuildingRegime, ...] | list[BuildingRegime | Mapping[str, Any]] | None,
) -> tuple[BuildingRegime, ...]:
    if values is None:
        return BUILDING_REGIMES
    return tuple(coerce_building_regime(value) for value in values)


BUILDING_REGIMES: tuple[BuildingRegime, ...] = (
    BuildingRegime(
        name="coastal_mild",
        weather="Warm_Marine",
        occupancy_scale=0.85,
        carbon_scale=0.95,
        price_scale=0.90,
        target=20.0,
    ),
    BuildingRegime(
        name="desert_hot",
        weather="Hot_Dry",
        occupancy_scale=1.15,
        carbon_scale=1.05,
        price_scale=1.10,
        target=21.0,
    ),
    BuildingRegime(
        name="mixed_marine",
        weather="Mixed_Marine",
        occupancy_scale=1.00,
        carbon_scale=1.00,
        price_scale=1.00,
        target=20.0,
    ),
    BuildingRegime(
        name="very_cold",
        weather="Very_Cold",
        occupancy_scale=1.25,
        carbon_scale=1.12,
        price_scale=1.18,
        target=22.0,
    ),
)

BUILDING_CONTEXT_FEATURES: tuple[str, ...] = (
    "time_frac",
    "occupancy_scale",
    "carbon_scale",
    "price_scale",
    "target_temp",
    "outdoor_temp",
    "solar_irradiance",
    "ground_temp",
    "occupancy_level",
    "carbon_intensity",
    "electricity_price",
)


def _unit_scale(value: float, low: float, high: float) -> float:
    span = max(float(high - low), 1e-6)
    return float(np.clip((float(value) - float(low)) / span, 0.0, 1.0))


def make_building_parameters(
    regime: BuildingRegime,
    building: str = "OfficeLarge",
    location: str = "ElPaso",
    time_resolution: int = 3600,
) -> dict[str, Any]:
    activity_schedule = np.ones(1024, dtype=np.float32) * 120.0 * regime.occupancy_scale
    params = ParameterGenerator(
        Building=building,
        Weather=regime.weather,
        Location=location,
        time_reso=time_resolution,
        target=regime.target,
        activity_sch=activity_schedule,
    )
    params["CarbonIntensity"] = np.asarray(params["CarbonIntensity"], dtype=np.float32) * regime.carbon_scale
    params["ElectricityPrice"] = np.asarray(params["ElectricityPrice"], dtype=np.float32) * regime.price_scale
    return params


def summarize_regime(regime: BuildingRegime, horizon: int = 168) -> dict[str, np.ndarray]:
    params = make_building_parameters(regime)
    data = {
        "out_temp": np.asarray(params["OutTemp"][:horizon], dtype=np.float32),
        "ghi": np.asarray(params["ghi"][:horizon], dtype=np.float32),
        "ground_temp": np.asarray(params["GroundTemp"][:horizon], dtype=np.float32),
        "occupancy": np.asarray(params["Occupancy"][:horizon], dtype=np.float32),
        "carbon": np.asarray(params["CarbonIntensity"][:horizon], dtype=np.float32),
        "price": np.asarray(params["ElectricityPrice"][:horizon], dtype=np.float32),
    }
    return data



class DynamicBuildingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dimension: str = "3d",
        regimes: tuple[BuildingRegime, ...] = BUILDING_REGIMES,
        schedule: str = "cyclic",
        episodes_per_regime: int = 4,
        seed: int | None = None,
    ):
        assert dimension in {"3d", "9d"}
        assert schedule in {"cyclic", "random"}
        self.dimension = dimension
        self.regimes = coerce_building_regimes(regimes)
        self.schedule = schedule
        self.episodes_per_regime = episodes_per_regime
        self.rng = np.random.default_rng(seed)

        self.episode_count = 0
        self.current_regime_index = 0
        self._elapsed_steps = 0
        self._max_episode_steps = 480

        self._env = self._make_env(self.regimes[self.current_regime_index])
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def current_regime(self) -> BuildingRegime:
        return self.regimes[self.current_regime_index]

    def _make_env(self, regime: BuildingRegime):
        params = make_building_parameters(regime)
        env_cls = BuildingEnv_3d if self.dimension == "3d" else BuildingEnv_9d
        env = env_cls(params)
        self._max_episode_steps = getattr(env, "_max_episode_steps", 480)
        return env

    def _select_regime_index(self) -> int:
        if self.schedule == "random":
            return int(self.rng.integers(len(self.regimes)))
        bucket = self.episode_count // self.episodes_per_regime
        return bucket % len(self.regimes)

    def _context_payload(self, step_idx: int) -> dict[str, Any]:
        env = self._env
        regime = self.current_regime
        max_idx = max(0, len(env.OutTemp) - 1)
        step_idx = int(np.clip(step_idx, 0, max_idx))
        carbon_series = getattr(env, "CarbonIntensity_Gauss", env.CarbonIntensity)
        price_series = getattr(env, "ElectricityPrice_Gauss", env.ElectricityPrice)
        occupancy_series = env.Occupancy

        out_temp = float(env.OutTemp[step_idx])
        ghi = float(env.ghi[step_idx])
        ground_temp = float(env.GroundTemp[step_idx])
        occupancy = float(occupancy_series[step_idx])
        carbon = float(carbon_series[step_idx])
        price = float(price_series[step_idx])
        target = float(np.mean(np.asarray(env.target, dtype=np.float32)))

        occupancy_denom = max(float(np.max(occupancy_series)), 1.0)
        carbon_denom = max(float(np.max(env.CarbonIntensity)) * 1.25, 1e-6)
        price_denom = max(float(np.max(env.ElectricityPrice)) * 1.25, 1e-6)
        vector = np.asarray(
            [
                _unit_scale(step_idx - int(getattr(env, "random", 0)), 0.0, env._max_episode_steps),
                float(regime.occupancy_scale) / 1.5,
                float(regime.carbon_scale) / 1.5,
                float(regime.price_scale) / 1.5,
                _unit_scale(target, 18.0, 24.0),
                _unit_scale(out_temp, -20.0, 50.0),
                _unit_scale(ghi, 0.0, 1000.0),
                _unit_scale(ground_temp, -10.0, 40.0),
                float(np.clip(occupancy / occupancy_denom, 0.0, 1.0)),
                float(np.clip(carbon / carbon_denom, 0.0, 1.0)),
                float(np.clip(price / price_denom, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return {
            "context_source": "environment_window",
            "context_label": regime.name,
            "context_feature_names": list(BUILDING_CONTEXT_FEATURES),
            "context_vector": vector,
            "context_dict": {
                "time_frac": float(vector[0]),
                "occupancy_scale": float(regime.occupancy_scale),
                "carbon_scale": float(regime.carbon_scale),
                "price_scale": float(regime.price_scale),
                "target_temp": target,
                "outdoor_temp": out_temp,
                "solar_irradiance": ghi,
                "ground_temp": ground_temp,
                "occupancy_level": occupancy,
                "carbon_intensity": carbon,
                "electricity_price": price,
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        next_regime_index = self._select_regime_index()
        if next_regime_index != self.current_regime_index or self.episode_count == 0:
            if self._env is not None:
                self._env.close()
            self.current_regime_index = next_regime_index
            self._env = self._make_env(self.current_regime)
            self.action_space = self._env.action_space
            self.observation_space = self._env.observation_space

        obs = self._env.reset(seed=seed, options=options)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}

        self.episode_count += 1
        self._elapsed_steps = 0
        info = dict(info)
        info["regime_id"] = self.current_regime_index
        info["regime_name"] = self.current_regime.name
        info.update(self._context_payload(int(getattr(self._env, "random", 0))))
        return np.asarray(obs, dtype=np.float32), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        step_idx = int(getattr(self._env, "_elapsed_steps", 0) + getattr(self._env, "random", 0))
        step_out = self._env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated = bool(done)
            truncated = False

        self._elapsed_steps = getattr(self._env, "_elapsed_steps", self._elapsed_steps + 1)
        info = dict(info)
        info["regime_id"] = self.current_regime_index
        info["regime_name"] = self.current_regime.name
        info.update(self._context_payload(step_idx))
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        if self._env is not None:
            self._env.close()

    def render(self):
        return None
