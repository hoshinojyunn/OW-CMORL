"""Adapter boundary for causal short-budget OOD experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.short_ood_protocol import stable_state_hash


REQUIRED_CHECKPOINTS: dict[str, tuple[str, ...]] = {
    "dynamic": ("final/EP_policy_0.pt",),
    "capql": ("model/policy.pt", "model/critic.pt"),
    "qpensieve": ("model/policy_final.pth", "model/critic_final.pth", "model/critic_target.pth"),
    "pgmorl": ("final/EP_policy_0.pt",),
    "morlca": ("model/morlca.pt",),
    "lcpo": ("policy_00.pt",),
}


@dataclass(frozen=True)
class CheckpointCapability:
    """Checkpoint availability separated from verified native continuation."""

    method: str
    checkpoint_dir: Path
    required: tuple[str, ...]
    missing: tuple[str, ...]
    resume_implemented: bool = False

    @property
    def checkpoint_complete(self) -> bool:
        return not self.missing

    @property
    def supported(self) -> bool:
        return self.checkpoint_complete and self.resume_implemented

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "checkpoint_dir": str(self.checkpoint_dir),
            "required": list(self.required),
            "missing": list(self.missing),
            "checkpoint_complete": self.checkpoint_complete,
            "resume_implemented": self.resume_implemented,
            "supported": self.supported,
        }


def probe_checkpoint(
    method: str,
    checkpoint_dir: str | Path,
    *,
    resume_implemented: bool = False,
) -> CheckpointCapability:
    """Report required model artifacts without inferring unsupported continuation."""

    normalized_method = str(method).lower().replace("q_pensieve", "qpensieve")
    if normalized_method not in REQUIRED_CHECKPOINTS:
        raise ValueError(f"Unsupported method for checkpoint probing: {method}")
    root = Path(checkpoint_dir)
    required = REQUIRED_CHECKPOINTS[normalized_method]
    missing = tuple(path for path in required if not (root / path).is_file())
    return CheckpointCapability(
        method=normalized_method,
        checkpoint_dir=root,
        required=required,
        missing=missing,
        resume_implemented=bool(resume_implemented),
    )


class AdaptationAdapter(ABC):
    """Minimal stateful interface consumed by the protocol runner."""

    method: str

    @abstractmethod
    def state_payload(self) -> Mapping[str, Any]:
        """Return a deterministic fingerprint of all mutable learner state."""

    def state_hash(self) -> str:
        return stable_state_hash(self.state_payload())

    @abstractmethod
    def action(self, observation: Any, preference: Any = None) -> Any:
        """Produce one deployment action without changing learner state."""

    @abstractmethod
    def adapt(self, env: Any, transitions: int) -> Mapping[str, float]:
        """Execute the requested causal local-refinement budget increment.

        The returned mapping must expose both ``updates`` and the actual OOD
        ``transitions`` consumed by the native learner.
        """

    def archive_size(self) -> int | None:
        return None

    def close(self) -> None:
        return None


def assert_unchanged(before: str, after: str) -> None:
    """Fail closed when a purported zero-shot evaluation mutates state."""

    if before != after:
        raise RuntimeError("zero-shot evaluation mutated adapter state")


class MORLCAAdapter(AdaptationAdapter):
    """Native MORL-CA continuation with a single shared OOD transition budget."""

    method = "morlca"

    def __init__(self, model: Any):
        self.model = model
        # A resumed deployed policy must not spend the short OOD budget on the
        # exploration-only warm-up branch used during training from scratch.
        self.model.start_steps = 0

    def state_payload(self) -> Mapping[str, Any]:
        return {
            "method": self.method,
            "model": self.model.state_dict(),
            "replay_restored": bool(getattr(self.model, "replay_restored", False)),
        }

    def action(self, observation: Any, preference: Any = None) -> np.ndarray:
        pref = None if preference is None else np.asarray(preference, dtype=np.float32)
        return self.model.act(np.asarray(observation, dtype=np.float32), pref=pref, deterministic=True)

    def adapt(self, env: Any, transitions: int) -> Mapping[str, float]:
        requested = int(transitions)
        if requested < 0:
            raise ValueError("transitions must be nonnegative")
        if requested == 0:
            return {"transitions": 0.0, "updates": 0.0}
        stats = self.model.train_on_env(env, total_timesteps=requested)
        if int(stats.total_steps) != requested:
            raise RuntimeError("MORL-CA did not consume the requested OOD transition budget")
        return {"transitions": float(stats.total_steps), "updates": float(stats.updates)}

    def archive_size(self) -> int | None:
        return 1


class LCPOAdapter(AdaptationAdapter):
    """LCPO scalarization archive with one method-level OOD interaction budget."""

    method = "lcpo"

    def __init__(self, trainers: Sequence[Any], *, preferences: Sequence[Sequence[float]]):
        self.trainers = list(trainers)
        self.preferences = [tuple(float(value) for value in preference) for preference in preferences]
        if not self.trainers or len(self.trainers) != len(self.preferences):
            raise ValueError("LCPO requires one restored trainer for every scalarization preference.")

    def state_payload(self) -> Mapping[str, Any]:
        return {
            "method": self.method,
            "preferences": self.preferences,
            "trainers": [trainer.state_dict() for trainer in self.trainers],
            "context_reservoir_restored": [
                bool(getattr(trainer, "context_reservoir_restored", False)) for trainer in self.trainers
            ],
        }

    def _preference_index(self, preference: Any) -> int:
        if preference is None:
            return 0
        requested = np.asarray(preference, dtype=np.float32).reshape(-1)
        for index, candidate in enumerate(self.preferences):
            if np.allclose(requested, np.asarray(candidate, dtype=np.float32), atol=1e-7, rtol=0.0):
                return index
        raise ValueError("The requested preference is not represented by this LCPO archive.")

    def action(self, observation: Any, preference: Any = None) -> np.ndarray:
        trainer = self.trainers[self._preference_index(preference)]
        return trainer.deterministic_action(np.asarray(observation, dtype=np.float32))

    @staticmethod
    def _budget_split(transitions: int, count: int) -> list[int]:
        quotient, remainder = divmod(int(transitions), int(count))
        return [quotient + int(index < remainder) for index in range(count)]

    def adapt(self, env: Any, transitions: int) -> Mapping[str, float]:
        requested = int(transitions)
        if requested < 0:
            raise ValueError("transitions must be nonnegative")
        if requested == 0:
            return {"transitions": 0.0, "updates": 0.0}

        actual_transitions = 0
        updates = 0
        for trainer, preference, allocation in zip(
            self.trainers,
            self.preferences,
            self._budget_split(requested, len(self.trainers)),
        ):
            if allocation == 0:
                continue
            observation = np.asarray(env.reset(), dtype=np.float32)
            current_context = observation.copy()
            states: list[np.ndarray] = []
            next_states: list[np.ndarray] = []
            action_indices: list[np.ndarray] = []
            rewards: list[float] = []
            terminated: list[bool] = []
            truncated: list[bool] = []
            contexts: list[np.ndarray] = []
            for _ in range(allocation):
                indices, action = trainer.sample_action(observation)
                next_observation, reward, done, info = env.step(action)
                next_observation = np.asarray(next_observation, dtype=np.float32)
                info = dict(info)
                objective = np.asarray(info.get("obj_raw", info.get("obj", reward)), dtype=np.float32).reshape(-1)
                states.append(observation.copy())
                next_states.append(next_observation)
                action_indices.append(np.asarray(indices, dtype=np.int64))
                rewards.append(float(np.dot(objective, np.asarray(preference, dtype=np.float32))))
                terminated.append(bool(done and not info.get("TimeLimit.truncated", False)))
                truncated.append(bool(done and info.get("TimeLimit.truncated", False)))
                contexts.append(current_context.copy())
                observation = next_observation
                current_context = next_observation.copy()
                actual_transitions += 1
                if done:
                    observation = np.asarray(env.reset(), dtype=np.float32)
                    current_context = observation.copy()
            trainer.update(
                observations=np.asarray(states, dtype=np.float32),
                next_observations=np.asarray(next_states, dtype=np.float32),
                action_indices=np.asarray(action_indices, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=bool),
                truncated=np.asarray(truncated, dtype=bool),
                contexts=np.asarray(contexts, dtype=np.float32),
            )
            updates += 1
        if actual_transitions != requested:
            raise RuntimeError("LCPO did not consume the requested OOD transition budget")
        return {"transitions": float(actual_transitions), "updates": float(updates)}

    def archive_size(self) -> int | None:
        return len(self.trainers)
