"""Small, auditable primitives for causal short-budget OOD experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ShortOODManifest:
    """Protocol fields that must be fixed before a short OOD run starts."""

    budgets: tuple[int, ...] = (0, 64, 128, 256)
    trajectory_seeds: tuple[int, ...] = (101, 202, 303)
    profile: str = "ood_v3_site_tariff_trace"

    def __post_init__(self) -> None:
        budgets = tuple(int(value) for value in self.budgets)
        seeds = tuple(int(value) for value in self.trajectory_seeds)
        if not budgets or budgets[0] != 0:
            raise ValueError("budgets must include 0 as the first entry")
        if any(value < 0 for value in budgets):
            raise ValueError("budgets must be nonnegative")
        if tuple(sorted(set(budgets))) != budgets:
            raise ValueError("budgets must be strictly monotonic")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("trajectory_seeds must be unique and nonempty")
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "trajectory_seeds", seeds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "budgets": list(self.budgets),
            "trajectory_seeds": list(self.trajectory_seeds),
            "profile": self.profile,
        }


class BudgetLedger:
    """Records cumulative causal OOD interaction budgets."""

    def __init__(self, budgets: Iterable[int]):
        self._budgets = tuple(int(value) for value in budgets)
        if not self._budgets or self._budgets[0] != 0:
            raise ValueError("budgets must include 0 as the first entry")
        if tuple(sorted(set(self._budgets))) != self._budgets:
            raise ValueError("budgets must be strictly monotonic")
        self._last: int | None = None

    @property
    def current(self) -> int | None:
        return self._last

    def record(self, cumulative_budget: int) -> None:
        budget = int(cumulative_budget)
        if budget not in self._budgets:
            raise ValueError(f"budget {budget} is not declared by the manifest")
        if self._last is not None and budget <= self._last:
            raise ValueError("cumulative budgets must be strictly monotonic")
        self._last = budget

    def remaining_to(self, target_budget: int) -> int:
        target = int(target_budget)
        if target not in self._budgets:
            raise ValueError(f"budget {target} is not declared by the manifest")
        if self._last is None:
            raise ValueError("record the current budget before requesting an increment")
        if target <= self._last:
            raise ValueError("target budget must be strictly monotonic")
        return target - self._last


def relative_switch_loss(id_pre: float, ood_zero: float, eps: float = 1e-8) -> float:
    """Return the nonnegative relative loss induced by the ID-to-OOD switch."""

    reference = float(id_pre)
    observed = float(ood_zero)
    if not (math.isfinite(reference) and math.isfinite(observed)):
        return float("nan")
    return max(0.0, reference - observed) / max(abs(reference), float(eps))


def id_return_retention(id_pre: float, id_return: float, eps: float = 1e-8) -> float:
    """Return post-adaptation ID retention, guarding signed near-zero metrics."""

    reference = float(id_pre)
    observed = float(id_return)
    if not (math.isfinite(reference) and math.isfinite(observed)) or abs(reference) < float(eps):
        return float("nan")
    return observed / reference


def stable_state_hash(payload: Any) -> str:
    """Hash a nested learner fingerprint without truncating tensor contents."""

    def canonicalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return {
                    "__ndarray_object__": canonicalize(value.tolist()),
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
            contiguous = np.ascontiguousarray(value)
            return {
                "__ndarray__": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            }
        if isinstance(value, np.generic):
            return canonicalize(value.item())
        # Avoid importing torch in the protocol module.  A detached tensor is
        # converted to NumPy before hashing so PyTorch's shortened repr cannot
        # conceal a changed parameter.
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return canonicalize(value.detach().cpu().contiguous().numpy())
        if isinstance(value, dict):
            return {str(key): canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [canonicalize(item) for item in value]
        if isinstance(value, bytes):
            return {"__bytes__": hashlib.sha256(value).hexdigest(), "length": len(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"__repr__": str(value), "__type__": f"{type(value).__module__}.{type(value).__qualname__}"}

    encoded = json.dumps(canonicalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
