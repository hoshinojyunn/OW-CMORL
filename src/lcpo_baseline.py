"""LCPO components adapted to the dynamic MORL benchmark.

The implementation follows the locally constrained policy update in the
official LCPO repository (``LCPO/windy-gym/agent/core_alg/core_lcpo.py``).
LCPO's public continuous-control implementation represents each continuous
action coordinate as a categorical distribution over action bins.  This module
keeps that representation and its local/OOD KL-constrained TRPO update, while
making the environment interface independent of the Windy-Gym benchmark.

This adapter deliberately uses only online observations (which include the
environment's dynamic factors).  Regime IDs and evaluation plans are never
provided to the learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


def _mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_sizes:
        layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(output_dim)))
    return nn.Sequential(*layers)


class CategoricalActionPolicy(nn.Module):
    """Factorized categorical policy used by the original continuous LCPO code."""

    def __init__(self, obs_dim: int, action_dim: int, action_bins: int, hidden_sizes: Iterable[int]):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_bins = int(action_bins)
        self.network = _mlp(obs_dim, hidden_sizes, self.action_dim * self.action_bins)

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations).view(-1, self.action_dim, self.action_bins)

    def distribution(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.log_softmax(self.logits(observations), dim=-1)
        return log_probs, log_probs.exp()

    def sample(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.logits(observations)
        categorical = torch.distributions.Categorical(logits=logits)
        return categorical.sample()

    def mode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logits(observations).argmax(dim=-1)

    def log_prob_entropy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, probs = self.distribution(observations)
        action_log_prob = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
        entropy = -(probs * log_probs).sum(dim=(-1, -2))
        return action_log_prob, entropy, log_probs, probs


def _flat_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in module.parameters():
            count = parameter.numel()
            parameter.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count


def _flat_grad(
    outputs: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        outputs,
        tuple(parameters),
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )
    return torch.cat([grad.contiguous().reshape(-1) for grad in grads])


def _conjugate_gradient(
    avp,
    vector: torch.Tensor,
    *,
    iterations: int,
    residual_tolerance: float = 1e-10,
) -> torch.Tensor:
    """Conjugate-gradient solver used by TRPO/LCPO."""
    solution = torch.zeros_like(vector)
    residual = vector.clone()
    direction = residual.clone()
    residual_dot = torch.dot(residual, residual)
    for _ in range(max(1, int(iterations))):
        avp_direction = avp(direction)
        denominator = torch.dot(direction, avp_direction)
        if not torch.isfinite(denominator) or denominator.abs() < 1e-12:
            break
        alpha = residual_dot / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * avp_direction
        new_residual_dot = torch.dot(residual, residual)
        if not torch.isfinite(new_residual_dot) or new_residual_dot < residual_tolerance:
            break
        beta = new_residual_dot / torch.clamp(residual_dot, min=1e-12)
        direction = residual + beta * direction
        residual_dot = new_residual_dot
    return solution


def _solve_quadratic(a: float, b: float, c: float) -> tuple[float, float] | None:
    discriminant = b * b - 4.0 * a * c
    if abs(a) < 1e-12 or discriminant < 0.0:
        return None
    root = float(np.sqrt(discriminant))
    return ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))


@dataclass
class LCPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    local_kl: float
    ood_kl: float
    used_lcpo_constraint: bool
    ood_batch_size: int
    accepted_step: bool


class ContextOODReservoir:
    """Reservoir sampler with LCPO's recent-context OOD selection rule.

    The official implementation stores historical observations and asks the
    environment whether they are distant from a recent context window.  This
    adapter uses the observed state as the context representation, retaining
    its associated observation as the KL-constraint sample.  This matches the
    source repository's full-observation distance option and avoids access to
    a regime label.
    """

    def __init__(self, obs_dim: int, context_dim: int, capacity: int, recent_window: int, seed: int):
        self.capacity = max(1, int(capacity))
        self.recent_window = max(1, int(recent_window))
        self.rng = np.random.RandomState(int(seed))
        self.states = np.zeros((self.capacity, int(obs_dim)), dtype=np.float32)
        self.contexts = np.zeros((self.capacity, int(context_dim)), dtype=np.float32)
        self.recent_contexts = np.zeros((self.recent_window, int(context_dim)), dtype=np.float32)
        self.size = 0
        self.total_seen = 0
        self.recent_size = 0
        self.recent_cursor = 0

    def add(self, states: np.ndarray, contexts: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.float32)
        contexts = np.asarray(contexts, dtype=np.float32)
        if states.ndim != 2 or contexts.ndim != 2 or len(states) != len(contexts):
            raise ValueError("LCPO reservoir expects equally sized 2-D state and context batches.")
        for state, context in zip(states, contexts):
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                index = int(self.rng.randint(0, self.total_seen + 1))
                if index >= self.capacity:
                    index = -1
            if index >= 0:
                self.states[index] = state
                self.contexts[index] = context
            self.recent_contexts[self.recent_cursor] = context
            self.recent_cursor = (self.recent_cursor + 1) % self.recent_window
            self.recent_size = min(self.recent_size + 1, self.recent_window)
            self.total_seen += 1

    def sample_ood(self, batch_size: int, threshold: float) -> np.ndarray:
        if self.size == 0 or self.recent_size < 2:
            return np.zeros((0, self.states.shape[1]), dtype=np.float32)
        recent = self.recent_contexts[: self.recent_size]
        centre = recent.mean(axis=0)
        # Per-feature normalization makes the published L2 context test usable
        # across the differently scaled four benchmark environments.
        scale = np.maximum(recent.std(axis=0), 5e-2)
        normalized_distance = ((self.contexts[: self.size] - centre) / scale) ** 2
        mask = normalized_distance.mean(axis=1) > float(threshold)
        candidates = np.flatnonzero(mask)
        if len(candidates) == 0:
            return np.zeros((0, self.states.shape[1]), dtype=np.float32)
        chosen = self.rng.choice(candidates, size=int(batch_size), replace=len(candidates) < int(batch_size))
        return self.states[chosen].copy()

    def state_dict(self) -> dict[str, object]:
        """Return the entire reservoir because it affects LCPO's next update."""

        return {
            "capacity": int(self.capacity),
            "recent_window": int(self.recent_window),
            "states": self.states.copy(),
            "contexts": self.contexts.copy(),
            "recent_contexts": self.recent_contexts.copy(),
            "size": int(self.size),
            "total_seen": int(self.total_seen),
            "recent_size": int(self.recent_size),
            "recent_cursor": int(self.recent_cursor),
            "rng_state": self.rng.get_state(),
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        if int(payload["capacity"]) != self.capacity or int(payload["recent_window"]) != self.recent_window:
            raise ValueError("LCPO reservoir shape does not match the checkpoint.")
        for key, target in (
            ("states", self.states),
            ("contexts", self.contexts),
            ("recent_contexts", self.recent_contexts),
        ):
            source = np.asarray(payload[key], dtype=np.float32)
            if source.shape != target.shape:
                raise ValueError(f"LCPO reservoir {key} shape does not match the checkpoint.")
            target[...] = source
        self.size = int(payload["size"])
        self.total_seen = int(payload["total_seen"])
        self.recent_size = int(payload["recent_size"])
        self.recent_cursor = int(payload["recent_cursor"])
        if not (0 <= self.size <= self.capacity and 0 <= self.recent_size <= self.recent_window):
            raise ValueError("LCPO reservoir sizes are invalid in the checkpoint.")
        if not (0 <= self.recent_cursor < self.recent_window and self.total_seen >= self.size):
            raise ValueError("LCPO reservoir cursor is invalid in the checkpoint.")
        self.rng.set_state(payload["rng_state"])


class LCPOTrainer:
    """Online LCPO trainer for a scalarized preference of a MORL environment."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        context_dim: int,
        action_bins: int = 3,
        hidden_sizes: Iterable[int] = (64, 64),
        learning_rate: float = 4e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.02,
        trpo_kl_in: float = 0.1,
        trpo_kl_out: float = 1e-3,
        trpo_damping: float = 0.1,
        trpo_dual: bool = False,
        cg_iterations: int = 5,
        line_search_backtracks: int = 5,
        ood_capacity: int = 4096,
        ood_recent_window: int = 32,
        ood_threshold: float = 2.0,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.entropy_coef = float(entropy_coef)
        self.trpo_kl_in = float(trpo_kl_in)
        self.trpo_kl_out = float(trpo_kl_out)
        self.trpo_damping = float(trpo_damping)
        self.trpo_dual = bool(trpo_dual)
        self.cg_iterations = int(cg_iterations)
        self.line_search_backtracks = int(line_search_backtracks)
        self.ood_threshold = float(ood_threshold)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.action_dim = int(len(self.action_low))
        self.action_bins = int(action_bins)
        if self.action_bins < 2:
            raise ValueError("LCPO requires at least two action bins.")
        self.policy = CategoricalActionPolicy(obs_dim, self.action_dim, self.action_bins, hidden_sizes).to(self.device)
        self.value = _mlp(obs_dim, hidden_sizes, 1).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        self.ood_reservoir = ContextOODReservoir(
            obs_dim=obs_dim,
            context_dim=context_dim,
            capacity=ood_capacity,
            recent_window=ood_recent_window,
            seed=seed,
        )
        self.reward_var = 1.0
        self.reward_count = 1e-4
        self.lcpo_updates = 0
        self.fallback_updates = 0
        self.context_reservoir_restored = False

    def action_from_indices(self, action_indices: np.ndarray) -> np.ndarray:
        fraction = np.asarray(action_indices, dtype=np.float32) / float(self.action_bins - 1)
        return self.action_low + fraction * (self.action_high - self.action_low)

    def sample_action(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            indices = self.policy.sample(obs).squeeze(0).cpu().numpy().astype(np.int64)
        return indices, self.action_from_indices(indices)

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            indices = self.policy.mode(obs).squeeze(0).cpu().numpy().astype(np.int64)
        return self.action_from_indices(indices)

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float32)
        discounted = np.zeros_like(rewards)
        carry = 0.0
        for index in range(len(rewards)):
            carry = self.gamma * carry + float(rewards[index])
            discounted[index] = carry
        batch_var = float(np.var(discounted))
        batch_count = float(len(discounted))
        delta = float(np.mean(discounted))
        total = self.reward_count + batch_count
        # Only a scale is needed for LCPO's reward normalization.  Keeping a
        # scalar running variance mirrors the source implementation.
        self.reward_var = (
            self.reward_var * self.reward_count + batch_var * batch_count + delta * delta * self.reward_count * batch_count / total
        ) / total
        self.reward_count = total
        return rewards / float(np.sqrt(max(self.reward_var, 1e-6) + 1.0))

    def _returns_and_advantages(
        self,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        returns = np.zeros(len(rewards), dtype=np.float32)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_return = float(next_values[-1])
        last_advantage = 0.0
        for index in reversed(range(len(rewards))):
            if bool(truncated[index]):
                last_return = float(next_values[index])
                last_advantage = 0.0
            continuation = 0.0 if bool(terminated[index]) else 1.0
            returns[index] = last_return = float(rewards[index]) + self.gamma * last_return * continuation
            td_error = float(rewards[index]) + self.gamma * float(next_values[index]) * continuation - float(values[index])
            last_advantage = td_error + self.gamma * self.gae_lambda * last_advantage * continuation
            advantages[index] = last_advantage
        return returns, advantages

    def _a2c_update(self, observations: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor) -> tuple[float, float]:
        action_log_prob, entropy, _log_probs, _probs = self.policy.log_prob_entropy(observations, actions)
        loss = -(action_log_prob * advantages).mean() - self.entropy_coef * entropy.mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_optimizer.step()
        return float(loss.detach().cpu()), float(entropy.mean().detach().cpu())

    def _trpo_lcpo_update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        ood_observations: torch.Tensor,
    ) -> tuple[float, float, float, float, bool]:
        with torch.no_grad():
            old_global_log, old_global_prob = self.policy.distribution(ood_observations)
            _old_local_action_log, _old_entropy, old_local_log, old_local_prob = self.policy.log_prob_entropy(
                observations, actions
            )

        def surrogate_loss() -> torch.Tensor:
            action_log_prob, entropy, _log_prob, _prob = self.policy.log_prob_entropy(observations, actions)
            old_action_log_prob = old_local_log.gather(-1, actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            ratio = torch.exp(action_log_prob - old_action_log_prob)
            return -(advantages * ratio).mean() - self.entropy_coef * entropy.mean()

        def local_kl() -> torch.Tensor:
            new_log, _new_prob = self.policy.distribution(observations)
            return (old_local_prob * (old_local_log - new_log)).sum(dim=(-1, -2))

        def ood_kl() -> torch.Tensor:
            new_log, _new_prob = self.policy.distribution(ood_observations)
            return (old_global_prob * (old_global_log - new_log)).sum(dim=(-1, -2))

        loss = surrogate_loss()
        gradient = _flat_grad(loss, self.policy.parameters()).detach()
        if not torch.isfinite(gradient).all() or gradient.norm() < 1e-12:
            return float(loss.detach().cpu()), 0.0, 0.0, 0.0, False

        def fisher_vector_product(kl_function, vector: torch.Tensor) -> torch.Tensor:
            kl_value = kl_function().mean()
            first_grad = _flat_grad(kl_value, self.policy.parameters(), create_graph=True)
            directional = torch.dot(first_grad, vector)
            second_grad = _flat_grad(directional, self.policy.parameters()).detach()
            return second_grad + self.trpo_damping * vector

        step_out = _conjugate_gradient(
            lambda vector: fisher_vector_product(ood_kl, vector),
            -gradient,
            iterations=self.cg_iterations,
        )
        value_out = float((-gradient * step_out).sum().detach().cpu())
        if not np.isfinite(value_out) or value_out <= 1e-12:
            return float(loss.detach().cpu()), 0.0, 0.0, 0.0, False

        if self.trpo_dual:
            step_in = _conjugate_gradient(
                lambda vector: fisher_vector_product(local_kl, vector),
                -gradient,
                iterations=self.cg_iterations,
            )
            value_in = float((-gradient * step_in).sum().detach().cpu())
            a_in = np.array(
                [
                    0.5 * value_in,
                    value_out,
                    0.5 * float((step_out * fisher_vector_product(ood_kl, step_out)).sum().detach().cpu()),
                ]
            )
            a_out = np.array(
                [
                    0.5 * float((step_in * fisher_vector_product(local_kl, step_out)).sum().detach().cpu()),
                    value_in,
                    0.5 * value_out,
                ]
            )
            difference = a_in / max(self.trpo_kl_in, 1e-12) - a_out / max(self.trpo_kl_out, 1e-12)
            roots = _solve_quadratic(float(difference[0]), float(difference[1]), float(difference[2]))
            if roots is None or max(roots) <= 0.0:
                out_scale = np.sqrt(2.0 * self.trpo_kl_out / value_out)
                full_step = float(out_scale) * step_out
            else:
                slope = max(roots)
                out_scale = np.sqrt(
                    self.trpo_kl_in / max(a_in[0] * slope * slope + a_in[1] * slope + a_in[2], 1e-12)
                )
                full_step = float(out_scale * slope) * step_in + float(out_scale) * step_out
        else:
            full_step = float(np.sqrt(2.0 * self.trpo_kl_out / value_out)) * step_out

        old_parameters = _flat_params(self.policy)
        expected_improvement = float((-gradient * full_step).sum().detach().cpu())
        old_loss = float(loss.detach().cpu())
        accepted = False
        for fraction in 0.5 ** np.arange(max(1, self.line_search_backtracks)):
            _set_flat_params(self.policy, old_parameters + float(fraction) * full_step)
            with torch.no_grad():
                new_loss = float(surrogate_loss().cpu())
                new_local_kl = float(local_kl().mean().cpu())
                new_ood_kl = float(ood_kl().mean().cpu())
            actual_improvement = old_loss - new_loss
            expected = expected_improvement * float(fraction)
            ratio = actual_improvement / max(abs(expected), 1e-12)
            if (
                np.isfinite(new_loss)
                and ratio > 0.1
                and actual_improvement > 0.0
                and new_local_kl <= self.trpo_kl_in
                and new_ood_kl <= self.trpo_kl_out
            ):
                accepted = True
                break
        if not accepted:
            _set_flat_params(self.policy, old_parameters)
        with torch.no_grad():
            _action_log_prob, entropy, _log_prob, _prob = self.policy.log_prob_entropy(observations, actions)
            final_local_kl = float(local_kl().mean().cpu())
            final_ood_kl = float(ood_kl().mean().cpu())
        return old_loss, float(entropy.mean().cpu()), final_local_kl, final_ood_kl, accepted

    def update(
        self,
        *,
        observations: np.ndarray,
        next_observations: np.ndarray,
        action_indices: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        contexts: np.ndarray,
    ) -> LCPOUpdateStats:
        observations = np.asarray(observations, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        action_indices = np.asarray(action_indices, dtype=np.int64)
        normalized_rewards = self._normalize_rewards(np.asarray(rewards, dtype=np.float32))
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        self.ood_reservoir.add(observations, np.asarray(contexts, dtype=np.float32))
        ood_observations = self.ood_reservoir.sample_ood(len(observations), self.ood_threshold)

        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action_indices, dtype=torch.long, device=self.device)
        values_tensor = self.value(obs_tensor).squeeze(-1)
        with torch.no_grad():
            next_values = self.value(next_obs_tensor).squeeze(-1).cpu().numpy()
            values = values_tensor.detach().cpu().numpy()
        returns, advantages = self._returns_and_advantages(
            normalized_rewards,
            terminated,
            truncated,
            values,
            next_values,
        )
        return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        used_lcpo = len(ood_observations) > 0
        if used_lcpo:
            ood_tensor = torch.as_tensor(ood_observations, dtype=torch.float32, device=self.device)
            policy_loss, entropy, local_kl, ood_kl, accepted = self._trpo_lcpo_update(
                obs_tensor, action_tensor, advantage_tensor, ood_tensor
            )
            self.lcpo_updates += 1
        else:
            policy_loss, entropy = self._a2c_update(obs_tensor, action_tensor, advantage_tensor)
            local_kl = 0.0
            ood_kl = 0.0
            accepted = True
            self.fallback_updates += 1

        value_loss = torch.nn.functional.mse_loss(values_tensor, return_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
        self.value_optimizer.step()
        return LCPOUpdateStats(
            policy_loss=float(policy_loss),
            value_loss=float(value_loss.detach().cpu()),
            entropy=float(entropy),
            local_kl=float(local_kl),
            ood_kl=float(ood_kl),
            used_lcpo_constraint=used_lcpo,
            ood_batch_size=int(len(ood_observations)),
            accepted_step=bool(accepted),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "checkpoint_schema": 2,
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "reward_var": float(self.reward_var),
            "reward_count": float(self.reward_count),
            "lcpo_updates": int(self.lcpo_updates),
            "fallback_updates": int(self.fallback_updates),
            "action_low": self.action_low.copy(),
            "action_high": self.action_high.copy(),
            "action_bins": int(self.action_bins),
            "ood_reservoir": self.ood_reservoir.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        """Restore all checkpointed LCPO learner state.

        Version-1 checkpoints predate reservoir persistence.  They are valid
        for a new OOD phase but explicitly start that volatile context cache
        empty; callers can expose ``context_reservoir_restored`` in provenance.
        """

        for key in ("policy", "value", "policy_optimizer", "value_optimizer", "reward_var", "reward_count", "lcpo_updates", "fallback_updates", "action_low", "action_high", "action_bins"):
            if key not in payload:
                raise KeyError(f"LCPO checkpoint is missing {key}.")
        action_low = np.asarray(payload["action_low"], dtype=np.float32).reshape(-1)
        action_high = np.asarray(payload["action_high"], dtype=np.float32).reshape(-1)
        if int(payload["action_bins"]) != self.action_bins:
            raise ValueError("LCPO action-bin count does not match the checkpoint.")
        if action_low.shape != self.action_low.shape or action_high.shape != self.action_high.shape:
            raise ValueError("LCPO action bounds shape does not match the checkpoint.")
        if not (np.allclose(action_low, self.action_low) and np.allclose(action_high, self.action_high)):
            raise ValueError("LCPO action bounds do not match the checkpoint.")
        self.policy.load_state_dict(payload["policy"])
        self.value.load_state_dict(payload["value"])
        self.policy_optimizer.load_state_dict(payload["policy_optimizer"])
        self.value_optimizer.load_state_dict(payload["value_optimizer"])
        self.reward_var = float(payload["reward_var"])
        self.reward_count = float(payload["reward_count"])
        self.lcpo_updates = int(payload["lcpo_updates"])
        self.fallback_updates = int(payload["fallback_updates"])
        reservoir_payload = payload.get("ood_reservoir")
        if reservoir_payload is not None:
            self.ood_reservoir.load_state_dict(dict(reservoir_payload))
            self.context_reservoir_restored = True
        else:
            self.context_reservoir_restored = False
