from __future__ import annotations

import os
import sys

import gym
import numpy as np
import torch
from gym.spaces.box import Box

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(BASE_DIR, "externals", "baselines"))

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from .chlor_alkali_env import ChlorAlkaliEnv
from .dynamic_building import BUILDING_REGIMES, DynamicBuildingEnv
from .sustaingym_wrappers import (
    DEFAULT_BUILDING_WEATHERS,
    DEFAULT_COGEN_RENEWABLES,
    DEFAULT_EV_PERIODS,
    DynamicCogenEnv,
    DynamicEVChargingEnv,
    DynamicSustainBuildingEnv,
)


def make_env(env_id, seed, rank, log_dir, allow_early_resets, env_kwargs=None):
    env_kwargs = dict(env_kwargs or {})

    def _thunk():
        if env_id == "building_3d_static":
            env = DynamicBuildingEnv(
                dimension="3d",
                regimes=(BUILDING_REGIMES[0],),
                schedule="cyclic",
                episodes_per_regime=10**9,
                seed=seed + rank,
            )
        elif env_id == "building_3d_dynamic":
            env = DynamicBuildingEnv(
                dimension="3d",
                regimes=tuple(env_kwargs.get("regimes", BUILDING_REGIMES)),
                schedule=env_kwargs.get("schedule", "cyclic"),
                episodes_per_regime=env_kwargs.get("episodes_per_regime", 4),
                seed=seed + rank,
            )
        elif env_id == "building_9d_static":
            env = DynamicBuildingEnv(
                dimension="9d",
                regimes=(BUILDING_REGIMES[0],),
                schedule="cyclic",
                episodes_per_regime=10**9,
                seed=seed + rank,
            )
        elif env_id == "building_9d_dynamic":
            env = DynamicBuildingEnv(
                dimension="9d",
                regimes=tuple(env_kwargs.get("regimes", BUILDING_REGIMES)),
                schedule=env_kwargs.get("schedule", "cyclic"),
                episodes_per_regime=env_kwargs.get("episodes_per_regime", 4),
                seed=seed + rank,
            )
        elif env_id == "sustaingym_building_static":
            env = DynamicSustainBuildingEnv(
                weathers=(env_kwargs.get("weathers", DEFAULT_BUILDING_WEATHERS)[0],),
                schedule="cyclic",
                episodes_per_regime=10**9,
                seed=seed + rank,
            )
        elif env_id == "sustaingym_building_dynamic":
            env = DynamicSustainBuildingEnv(
                weathers=tuple(env_kwargs.get("weathers", DEFAULT_BUILDING_WEATHERS)),
                schedule=env_kwargs.get("schedule", "cyclic"),
                episodes_per_regime=env_kwargs.get("episodes_per_regime", 4),
                seed=seed + rank,
            )
        elif env_id == "evcharging_static":
            env = DynamicEVChargingEnv(
                site=env_kwargs.get("site", "caltech"),
                periods=(env_kwargs.get("periods", DEFAULT_EV_PERIODS)[0],),
                schedule="cyclic",
                episodes_per_period=10**9,
                moer_forecast_steps=env_kwargs.get("moer_forecast_steps", 36),
                project_action_in_env=env_kwargs.get("project_action_in_env", True),
                seed=seed + rank,
            )
        elif env_id == "evcharging_dynamic":
            env = DynamicEVChargingEnv(
                site=env_kwargs.get("site", "caltech"),
                periods=tuple(env_kwargs.get("periods", DEFAULT_EV_PERIODS)),
                schedule=env_kwargs.get("schedule", "cyclic"),
                episodes_per_period=env_kwargs.get("episodes_per_regime", 4),
                moer_forecast_steps=env_kwargs.get("moer_forecast_steps", 36),
                project_action_in_env=env_kwargs.get("project_action_in_env", True),
                seed=seed + rank,
            )
        elif env_id == "cogen_static":
            env = DynamicCogenEnv(
                renewables_magnitudes=(env_kwargs.get("renewables", DEFAULT_COGEN_RENEWABLES)[0],),
                schedule="cyclic",
                episodes_per_regime=10**9,
                forecast_horizon=env_kwargs.get("forecast_horizon", 3),
                forecast_noise_std=env_kwargs.get("forecast_noise_std", 0.0),
                seed=seed + rank,
            )
        elif env_id == "cogen_dynamic":
            env = DynamicCogenEnv(
                renewables_magnitudes=tuple(env_kwargs.get("renewables", DEFAULT_COGEN_RENEWABLES)),
                schedule=env_kwargs.get("schedule", "cyclic"),
                episodes_per_regime=env_kwargs.get("episodes_per_regime", 4),
                forecast_horizon=env_kwargs.get("forecast_horizon", 3),
                forecast_noise_std=env_kwargs.get("forecast_noise_std", 0.0),
                seed=seed + rank,
            )
        elif env_id == "chlor_alkali_dynamic":
            env = ChlorAlkaliEnv(
                dataset_path=env_kwargs.get("dataset_path"),
                train_path=env_kwargs.get("train_path"),
                dynamic_price=env_kwargs.get("dynamic_price", True),
                aging_dynamics=env_kwargs.get("aging_dynamics", True),
                episode_length=env_kwargs.get("episode_length", 288),
                schedule=env_kwargs.get("schedule", "random"),
                allowed_regimes=tuple(env_kwargs.get("allowed_regimes", ())),
                allowed_regime_ids=tuple(env_kwargs.get("allowed_regime_ids", ())),
                num_regime_clusters=int(env_kwargs.get("num_regime_clusters", 12)),
                price_multiplier=float(env_kwargs.get("price_multiplier", 1.0)),
                price_offset=float(env_kwargs.get("price_offset", 0.0)),
                surrogate_seed=int(env_kwargs.get("surrogate_seed", 0)),
                seed=seed + rank,
            )
        elif env_id == "chlor_alkali_static":
            env = ChlorAlkaliEnv(
                dataset_path=env_kwargs.get("dataset_path"),
                train_path=env_kwargs.get("train_path"),
                dynamic_price=env_kwargs.get("dynamic_price", False),
                aging_dynamics=env_kwargs.get("aging_dynamics", False),
                episode_length=env_kwargs.get("episode_length", 288),
                schedule="sequential",
                allowed_regimes=tuple(env_kwargs.get("allowed_regimes", ())),
                allowed_regime_ids=tuple(env_kwargs.get("allowed_regime_ids", ())),
                num_regime_clusters=int(env_kwargs.get("num_regime_clusters", 12)),
                price_multiplier=float(env_kwargs.get("price_multiplier", 1.0)),
                price_offset=float(env_kwargs.get("price_offset", 0.0)),
                surrogate_seed=int(env_kwargs.get("surrogate_seed", 0)),
                seed=seed + rank,
            )
        else:
            raise ValueError(f"Unsupported innovation env: {env_id}")

        env.seed = seed + rank
        env = TimeLimitMask(env, reset_seed=seed + rank)

        env = bench.Monitor(
            env,
            os.path.join(log_dir, str(rank)) if log_dir is not None else None,
            allow_early_resets=allow_early_resets,
        )

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    num_processes,
    gamma,
    log_dir,
    device,
    allow_early_resets,
    env_kwargs=None,
    num_frame_stack=None,
    obj_rms=False,
    ob_rms=False,
):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, env_kwargs)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context="fork")
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False, obj_rms=obj_rms, ob=ob_rms)
        else:
            envs = VecNormalize(envs, gamma=gamma, obj_rms=obj_rms, ob=ob_rms)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


class TimeLimitMask(gym.Wrapper):
    def __init__(self, env=None, reset_seed=None):
        super().__init__(env)
        self.seed = reset_seed

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True
        info["obj"] = rew
        info["obj_raw"] = rew
        rew = 0.0
        return obs, rew, done, info

    def reset(self, **kwargs):
        if "seed" not in kwargs and self.seed is not None:
            kwargs["seed"] = self.seed
        obs, _info = self.env.reset(**kwargs)
        return obs


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        super().__init__(env)
        assert len(op) == 3
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [
                obs_shape[self.op[0]],
                obs_shape[self.op[1]],
                obs_shape[self.op[2]],
            ],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device
        self.dtype = torch.get_default_dtype()

    def reset(self):
        obs = self.venv.reset()
        return torch.from_numpy(obs).to(device=self.device, dtype=self.dtype)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        self.venv.step_async(actions.cpu().numpy())

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).to(device=self.device, dtype=self.dtype)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).to(dtype=self.dtype)
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device("cpu")
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0 :]
        for i, new in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
