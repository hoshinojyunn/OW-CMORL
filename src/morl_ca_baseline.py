from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class AOWNet(nn.Module):
    def __init__(self, obs_dim: int, obj_num: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, obj_num)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(obs), dim=-1)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, obj_num: int, hidden_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.net = _mlp(obs_dim + obj_num, hidden_dim, action_dim * 2)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(torch.cat([obs, weights], dim=-1)).chunk(2, dim=-1)
        return mean, torch.clamp(log_std, -20.0, 2.0)


class VectorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, obj_num: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, hidden_dim, obj_num)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, obj_num: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, obj_num), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = np.asarray(reward, dtype=np.float32)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.dones[idx] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "obs": self.obs.copy(),
            "next_obs": self.next_obs.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
            "size": int(self.size),
            "ptr": int(self.ptr),
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        if int(payload["capacity"]) != self.capacity:
            raise ValueError("MORL-CA replay capacity does not match the checkpoint.")
        for key, target in (
            ("obs", self.obs),
            ("next_obs", self.next_obs),
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
        ):
            source = np.asarray(payload[key], dtype=np.float32)
            if source.shape != target.shape:
                raise ValueError(f"MORL-CA replay {key} shape does not match the checkpoint.")
            target[...] = source
        self.size = int(payload["size"])
        self.ptr = int(payload["ptr"])
        if not (0 <= self.size <= self.capacity and 0 <= self.ptr < self.capacity):
            raise ValueError("MORL-CA replay state is invalid in the checkpoint.")


@dataclass
class MORLCATrainStats:
    total_steps: int = 0
    episodes: int = 0
    updates: int = 0
    mean_episode_return: float = 0.0


class MORLCABaseline:
    def __init__(
        self,
        *,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        obj_num: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        alpha_init: float = 0.2,
        batch_size: int = 256,
        replay_size: int = 100000,
        start_steps: int = 4096,
        updates_per_step: int = 1,
        aow_aux_coef: float = 0.2,
        eval_pref_bias_scale: float = 0.75,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device).view(1, -1)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device).view(1, -1)
        self.action_dim = int(self.action_low.shape[-1])
        self.obj_num = int(obj_num)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.start_steps = int(start_steps)
        self.updates_per_step = int(updates_per_step)
        self.aow_aux_coef = float(aow_aux_coef)
        self.eval_pref_bias_scale = float(eval_pref_bias_scale)
        self.target_entropy = -float(self.action_dim)

        self.aow_net = AOWNet(self.obs_dim, self.obj_num, hidden_dim).to(self.device)
        self.actor = GaussianActor(self.obs_dim, self.action_dim, self.obj_num, hidden_dim).to(self.device)
        self.critic1 = VectorCritic(self.obs_dim, self.action_dim, self.obj_num, hidden_dim).to(self.device)
        self.critic2 = VectorCritic(self.obs_dim, self.action_dim, self.obj_num, hidden_dim).to(self.device)
        self.critic1_target = VectorCritic(self.obs_dim, self.action_dim, self.obj_num, hidden_dim).to(self.device)
        self.critic2_target = VectorCritic(self.obs_dim, self.action_dim, self.obj_num, hidden_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(
            list(self.aow_net.parameters()) + list(self.actor.parameters()),
            lr=float(actor_lr),
        )
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=float(critic_lr))
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=float(critic_lr))
        self.log_alpha = torch.tensor(np.log(max(float(alpha_init), 1e-6)), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=float(alpha_lr))

        self.replay = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            obj_num=self.obj_num,
            capacity=replay_size,
        )
        self.replay_restored = False

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _blend_weights(
        self,
        base_weights: torch.Tensor,
        pref: torch.Tensor | None,
        *,
        bias_scale: float,
    ) -> torch.Tensor:
        if pref is None:
            return base_weights
        pref = torch.clamp(pref, min=1e-6)
        pref = pref / pref.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = torch.log(base_weights.clamp_min(1e-6)) + float(bias_scale) * torch.log(pref)
        return torch.softmax(logits, dim=-1)

    def _scale_action(self, tanh_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + 0.5 * (tanh_action + 1.0) * (self.action_high - self.action_low)

    def _sample_action_tensor(
        self,
        obs: torch.Tensor,
        *,
        pref: torch.Tensor | None = None,
        deterministic: bool = False,
        bias_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_weights = self.aow_net(obs)
        weights = self._blend_weights(base_weights, pref, bias_scale=bias_scale)
        mean, log_std = self.actor(obs, weights)
        if deterministic:
            pre_tanh = mean
        else:
            std = log_std.exp()
            pre_tanh = mean + std * torch.randn_like(mean)
        tanh_action = torch.tanh(pre_tanh)
        action = self._scale_action(tanh_action)
        if deterministic:
            log_prob = torch.zeros((obs.shape[0], 1), dtype=torch.float32, device=obs.device)
        else:
            std = log_std.exp()
            normal_log_prob = -0.5 * (((pre_tanh - mean) / (std + 1e-6)) ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
            normal_log_prob = normal_log_prob.sum(dim=-1, keepdim=True)
            squash_correction = torch.log(1.0 - tanh_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = normal_log_prob - squash_correction
        return action, log_prob, base_weights, weights

    def act(
        self,
        obs: np.ndarray,
        *,
        pref: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).view(1, -1)
        pref_t = None
        if pref is not None:
            pref_t = torch.as_tensor(np.asarray(pref, dtype=np.float32), device=self.device).view(1, -1)
        with torch.no_grad():
            action, _log_prob, _base, _weights = self._sample_action_tensor(
                obs_t,
                pref=pref_t,
                deterministic=deterministic,
                bias_scale=self.eval_pref_bias_scale if pref_t is not None else 0.0,
            )
        return np.asarray(action.squeeze(0).cpu().numpy(), dtype=np.float32)

    def update(self) -> dict[str, float]:
        obs, actions, rewards, next_obs, dones = self.replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions, next_log_prob, _next_base, _next_weights = self._sample_action_tensor(next_obs)
            next_q = torch.min(
                self.critic1_target(next_obs, next_actions),
                self.critic2_target(next_obs, next_actions),
            )
            target_q = rewards + self.gamma * (1.0 - dones) * (next_q - self.alpha.detach() * next_log_prob)

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=5.0)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=5.0)
        self.critic2_optim.step()

        new_actions, log_prob, base_weights, weights = self._sample_action_tensor(obs)
        q_pi = torch.min(self.critic1(obs, new_actions), self.critic2(obs, new_actions))
        weighted_q = (weights * q_pi).sum(dim=-1, keepdim=True)

        q_shifted = q_pi.detach() - q_pi.detach().mean(dim=-1, keepdim=True)
        aow_target = torch.softmax(q_shifted, dim=-1)
        aow_loss = F.mse_loss(base_weights, aow_target)
        actor_loss = (self.alpha.detach() * log_prob - weighted_q).mean() + self.aow_aux_coef * aow_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.aow_net.parameters()) + list(self.actor.parameters()),
            max_norm=5.0,
        )
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        with torch.no_grad():
            for src, dst in ((self.critic1, self.critic1_target), (self.critic2, self.critic2_target)):
                for src_param, dst_param in zip(src.parameters(), dst.parameters()):
                    dst_param.data.mul_(1.0 - self.tau)
                    dst_param.data.add_(self.tau * src_param.data)

        return {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "weighted_q": float(weighted_q.mean().item()),
        }

    def train_on_env(self, env, *, total_timesteps: int, seed: int = 0) -> MORLCATrainStats:
        stats = MORLCATrainStats(total_steps=0, episodes=0, updates=0, mean_episode_return=0.0)
        episode_returns: list[float] = []
        obs = env.reset()
        episode_return = np.zeros(self.obj_num, dtype=np.float64)

        while stats.total_steps < int(total_timesteps):
            if stats.total_steps < self.start_steps:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = self.act(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            reward_vec = np.asarray(info.get("obj_raw", info.get("reward_vector", reward)), dtype=np.float32).reshape(-1)
            self.replay.add(obs, action, reward_vec, next_obs, bool(done))

            obs = np.asarray(next_obs, dtype=np.float32)
            episode_return += reward_vec.astype(np.float64)
            stats.total_steps += 1

            if self.replay.size >= self.batch_size:
                for _ in range(self.updates_per_step):
                    self.update()
                    stats.updates += 1

            if done:
                episode_returns.append(float(np.mean(episode_return)))
                stats.episodes += 1
                stats.mean_episode_return = float(np.mean(episode_returns))
                obs = env.reset()
                episode_return = np.zeros(self.obj_num, dtype=np.float64)

        return stats

    def state_dict(self) -> dict[str, object]:
        return {
            "checkpoint_schema": 2,
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "obj_num": int(self.obj_num),
            "action_low": self.action_low.detach().cpu().numpy().copy(),
            "action_high": self.action_high.detach().cpu().numpy().copy(),
            "aow_net": self.aow_net.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic1_optim": self.critic1_optim.state_dict(),
            "critic2_optim": self.critic2_optim.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optim": self.alpha_optim.state_dict(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        for key, expected in (("obs_dim", self.obs_dim), ("action_dim", self.action_dim), ("obj_num", self.obj_num)):
            if key in payload and int(payload[key]) != expected:
                raise ValueError(f"MORL-CA {key} does not match the checkpoint.")
        if "action_low" in payload and not np.allclose(np.asarray(payload["action_low"]), self.action_low.cpu().numpy()):
            raise ValueError("MORL-CA action lower bounds do not match the checkpoint.")
        if "action_high" in payload and not np.allclose(np.asarray(payload["action_high"]), self.action_high.cpu().numpy()):
            raise ValueError("MORL-CA action upper bounds do not match the checkpoint.")
        self.aow_net.load_state_dict(payload["aow_net"])
        self.actor.load_state_dict(payload["actor"])
        self.critic1.load_state_dict(payload["critic1"])
        self.critic2.load_state_dict(payload["critic2"])
        self.critic1_target.load_state_dict(payload["critic1_target"])
        self.critic2_target.load_state_dict(payload["critic2_target"])
        self.actor_optim.load_state_dict(payload["actor_optim"])
        self.critic1_optim.load_state_dict(payload["critic1_optim"])
        self.critic2_optim.load_state_dict(payload["critic2_optim"])
        self.log_alpha = payload["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_optim.param_groups[0]["lr"])
        self.alpha_optim.load_state_dict(payload["alpha_optim"])
        replay_payload = payload.get("replay")
        if replay_payload is not None:
            self.replay.load_state_dict(dict(replay_payload))
            self.replay_restored = True
        else:
            self.replay_restored = False

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        self.load_state_dict(payload)
