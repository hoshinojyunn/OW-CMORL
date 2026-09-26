"""Portable OW-CMORL runtime checkpoints for causal OOD continuation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import torch

from .expert_bank import ContextExpertBank


RUNTIME_SCHEMA_VERSION = 1


def _register_serialized_runtime_imports() -> None:
    """Expose the PPO modules referenced by historic serialized samples."""

    project_root = Path(__file__).resolve().parents[2]
    for dependency_root in (
        project_root / "externals" / "baselines",
        project_root / "externals" / "pytorch-a2c-ppo-acktr-gail",
    ):
        text = str(dependency_root)
        if text not in sys.path:
            sys.path.append(text)


def save_owcmorl_runtime_checkpoint(
    path: str | Path,
    *,
    ep: Any,
    expert_bank: ContextExpertBank | None,
    regime_model: Any,
    context_memory: Any,
    online_trace_window: Any,
    prev_target_embedding: Any,
    current_drift_score: float,
    current_context_state: Any,
    selected_history: Any,
    selected_batch: Any,
    final_samples: Any,
    iteration: int,
    args_signature: dict[str, Any],
) -> None:
    """Persist all mutable OW-CMORL state except the rebuildable HNSW graph."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "ep": deepcopy(ep),
        "expert_bank_state": None if expert_bank is None else expert_bank.state_dict(),
        "regime_model": deepcopy(regime_model),
        "context_memory": deepcopy(context_memory),
        "online_trace_window": deepcopy(online_trace_window),
        "prev_target_embedding": deepcopy(prev_target_embedding),
        "current_drift_score": float(current_drift_score),
        "current_context_state": deepcopy(current_context_state),
        "selected_history": deepcopy(selected_history),
        "selected_batch": deepcopy(selected_batch),
        "final_samples": deepcopy(final_samples),
        "iteration": int(iteration),
        "args_signature": dict(args_signature),
    }
    torch.save(payload, destination)


def load_owcmorl_runtime_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load semantic OW state and mark its ANN index for lazy reconstruction."""

    _register_serialized_runtime_imports()
    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError("Unsupported or incomplete OW-CMORL runtime checkpoint.")
    required = (
        "ep",
        "expert_bank_state",
        "regime_model",
        "context_memory",
        "online_trace_window",
        "selected_history",
        "selected_batch",
        "final_samples",
        "iteration",
        "args_signature",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"OW-CMORL runtime checkpoint is missing: {', '.join(missing)}")
    payload["expert_bank"] = (
        None
        if payload["expert_bank_state"] is None
        else ContextExpertBank.from_state_dict(payload["expert_bank_state"])
    )
    return payload
