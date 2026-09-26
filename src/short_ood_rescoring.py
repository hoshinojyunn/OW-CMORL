"""Trace-level normalized hypervolume rescoring for short OOD event fronts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

import numpy as np

from src.dynamic_morl.metrics import compute_front_metrics


@dataclass(frozen=True)
class TraceHVEnvelope:
    """A fixed objective envelope and reference for one environment trace."""

    lower: np.ndarray
    upper: np.ndarray
    active_dimensions: np.ndarray
    reference: np.ndarray

    def normalize(self, front: np.ndarray) -> np.ndarray:
        values = np.asarray(front, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.lower):
            raise ValueError("front dimensionality is incompatible with the trace HV envelope")
        span = self.upper[self.active_dimensions] - self.lower[self.active_dimensions]
        return (values[:, self.active_dimensions] - self.lower[self.active_dimensions]) / span

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "active_dimensions": self.active_dimensions.astype(int).tolist(),
            "reference": self.reference.tolist(),
        }


def _front_from_row(row: Mapping[str, Any]) -> np.ndarray:
    raw = row["front_points"]
    values = json.loads(raw) if isinstance(raw, str) else raw
    front = np.asarray(values, dtype=np.float64)
    if front.ndim != 2 or front.shape[0] == 0 or front.shape[1] == 0:
        raise ValueError("front_points must contain a nonempty two-dimensional front")
    if not np.all(np.isfinite(front)):
        raise ValueError("front_points contains a non-finite objective value")
    return front


def derive_trace_hv_envelope(
    rows: Iterable[Mapping[str, Any]],
    *,
    margin: float = 0.1,
    span_epsilon: float = 1e-12,
) -> TraceHVEnvelope:
    """Build one fixed normalized-HV envelope from every predeclared phase front."""

    fronts = [_front_from_row(row) for row in rows]
    if not fronts:
        raise ValueError("cannot derive a trace HV envelope from no event rows")
    dimensions = {front.shape[1] for front in fronts}
    if len(dimensions) != 1:
        raise ValueError("all fronts for one environment must have the same objective dimension")
    stacked = np.vstack(fronts)
    lower = stacked.min(axis=0)
    upper = stacked.max(axis=0)
    active = np.flatnonzero((upper - lower) > float(span_epsilon))
    if active.size == 0:
        raise ValueError("all objectives are constant across the trace; normalized HV is undefined")
    return TraceHVEnvelope(
        lower=lower,
        upper=upper,
        active_dimensions=active,
        reference=np.full(active.size, -abs(float(margin)), dtype=np.float64),
    )


def trace_normalized_hv(front: np.ndarray, envelope: TraceHVEnvelope) -> float:
    """Evaluate one front under a fixed trace-level normalized HV reference."""

    normalized = envelope.normalize(front)
    hv, _eu, _sparsity = compute_front_metrics(
        normalized,
        envelope.reference,
        eval_delta_weight=0.25,
    )
    return float(hv)


def rescore_event_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, TraceHVEnvelope]]:
    """Copy event rows and replace HV with per-environment trace-normalized HV."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["env_key"]), []).append(row)
    envelopes = {env_key: derive_trace_hv_envelope(group) for env_key, group in grouped.items()}
    rescored: list[dict[str, Any]] = []
    for env_key, group in grouped.items():
        envelope = envelopes[env_key]
        for row in group:
            copy = dict(row)
            copy["raw_HV"] = str(row["HV"])
            copy["HV"] = trace_normalized_hv(_front_from_row(row), envelope)
            copy["hv_metric"] = "trace_normalized_hv"
            rescored.append(copy)
    return rescored, envelopes
