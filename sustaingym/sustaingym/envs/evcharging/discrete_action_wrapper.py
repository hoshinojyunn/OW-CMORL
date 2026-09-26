from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """Discretize each EVSE's normalized pilot signal into a small action set."""

    def __init__(self, env: gym.Env, levels: int = 5):
        super().__init__(env)
        if levels < 2:
            raise ValueError("levels must be at least 2")
        self.levels = np.linspace(0.0, 1.0, levels, dtype=np.float32)
        shape = int(np.prod(env.action_space.shape))
        self.action_space = gym.spaces.MultiDiscrete(np.full(shape, levels, dtype=np.int64))

    def action(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        return self.levels[np.clip(action, 0, len(self.levels) - 1)]
