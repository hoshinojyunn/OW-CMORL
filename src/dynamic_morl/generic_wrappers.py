from __future__ import annotations

import gym
import numpy as np


def _to_gym_space(space):
    if isinstance(space, gym.Space):
        return space
    if hasattr(space, "n"):
        return gym.spaces.Discrete(int(space.n))
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)
    raise TypeError(f"Unsupported space type: {type(space)!r}")


class GymCompatibilityWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = _to_gym_space(env.observation_space)
        self.action_space = _to_gym_space(env.action_space)


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
        base_env = getattr(self.env, "unwrapped", self.env)
        if done and getattr(base_env, "_max_episode_steps", None) == getattr(base_env, "_elapsed_steps", None):
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
