from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .temporal import cosine_affinity


@dataclass
class ContextSnapshot:
    iteration: int
    current_embedding: np.ndarray
    forecast_embedding: np.ndarray
    target_embedding: np.ndarray
    gap_vector: np.ndarray
    utility_mean: float
    drift_score: float
    prediction_error: float


class OnlineContextMemory:
    def __init__(
        self,
        max_size: int = 8,
        history_lambda: float = 0.35,
        forecast_lambda: float = 0.55,
        nearest_k: int = 3,
    ):
        self.max_size = max(1, int(max_size))
        self.history_lambda = float(np.clip(history_lambda, 0.0, 0.95))
        self.forecast_lambda = float(np.clip(forecast_lambda, 0.0, 1.0))
        self.nearest_k = max(1, int(nearest_k))
        self.snapshots: deque[ContextSnapshot] = deque(maxlen=self.max_size)

    def __len__(self) -> int:
        return len(self.snapshots)

    def update(self, snapshot: ContextSnapshot) -> None:
        self.snapshots.append(snapshot)

    def reference_embedding(self) -> np.ndarray | None:
        if not self.snapshots:
            return None
        weights = np.linspace(1.0, 2.0, num=len(self.snapshots), dtype=np.float64)
        embeddings = np.stack([snapshot.target_embedding for snapshot in self.snapshots], axis=0)
        weights /= weights.sum()
        return np.average(embeddings, axis=0, weights=weights)

    def nearest_gap_vector(self, query_embedding: np.ndarray) -> np.ndarray | None:
        if not self.snapshots:
            return None
        scored = []
        for snapshot in self.snapshots:
            affinity = cosine_affinity(snapshot.target_embedding, query_embedding)
            scored.append((affinity, snapshot.gap_vector))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: min(self.nearest_k, len(scored))]
        affinities = np.asarray([max(0.0, affinity) for affinity, _gap in top], dtype=np.float64)
        if np.allclose(affinities.sum(), 0.0):
            affinities = np.ones(len(top), dtype=np.float64)
        affinities /= affinities.sum()
        gaps = np.stack([gap for _affinity, gap in top], axis=0)
        return np.average(gaps, axis=0, weights=affinities)

    def build_target_embedding(
        self,
        current_embedding: np.ndarray,
        forecast_embedding: np.ndarray,
    ) -> np.ndarray:
        target = (
            (1.0 - self.forecast_lambda) * np.asarray(current_embedding, dtype=np.float64)
            + self.forecast_lambda * np.asarray(forecast_embedding, dtype=np.float64)
        )
        history = self.reference_embedding()
        if history is not None:
            target = (1.0 - self.history_lambda) * target + self.history_lambda * history
        return np.asarray(target, dtype=np.float64)


class OnlineTraceWindow:
    def __init__(self, max_steps: int = 96):
        self.max_steps = max(1, int(max_steps))
        self.context_steps: deque[np.ndarray] = deque(maxlen=self.max_steps)
        self.context_feature_names: list[str] = []

    def __len__(self) -> int:
        return len(self.context_steps)

    def append(
        self,
        context_batch: np.ndarray,
        context_feature_names: list[str] | None = None,
    ) -> None:
        context_batch = np.asarray(context_batch, dtype=np.float32)
        if context_batch.ndim != 2:
            return
        if context_feature_names and not self.context_feature_names:
            self.context_feature_names = list(context_feature_names)
        for context in context_batch:
            self.context_steps.append(np.asarray(context, dtype=np.float32))

    def append_aggregated_traces(
        self,
        traces,
        tail_steps: int = 8,
        sample_mode: str = "tail",
        saliency_alpha: float = 0.25,
    ) -> int:
        segment = aggregate_trace_segment(
            traces,
            tail_steps=tail_steps,
            sample_mode=sample_mode,
            saliency_alpha=saliency_alpha,
        )
        if segment is None:
            return 0
        self.append(
            segment["context"],
            context_feature_names=segment.get("context_feature_names"),
        )
        return int(len(segment["context"]))

    def as_trace(self) -> dict[str, np.ndarray]:
        if not self.context_steps:
            return {
                "context": np.zeros((0, 0), dtype=np.float32),
                "step": np.zeros(0, dtype=np.int64),
                "episode": np.zeros(0, dtype=np.int64),
                "context_feature_names": list(self.context_feature_names),
            }
        context = np.stack(list(self.context_steps), axis=0).astype(np.float32)
        steps = np.arange(len(context), dtype=np.int64)
        return {
            "context": context,
            "step": steps,
            "episode": np.zeros(len(context), dtype=np.int64),
            "context_feature_names": list(self.context_feature_names),
        }


def aggregate_trace_segment(
    traces,
    tail_steps: int = 8,
    sample_mode: str = "tail",
    saliency_alpha: float = 0.25,
) -> dict[str, np.ndarray] | None:
    valid_traces = []
    for trace in traces:
        if trace is None:
            continue
        context = np.asarray(trace.get("context", []), dtype=np.float32)
        if context.ndim != 2 or len(context) == 0:
            continue
        valid_traces.append(
            (
                context,
                list(trace.get("context_feature_names", [])),
            )
        )
    if not valid_traces:
        return None

    tail_steps = max(1, int(tail_steps))
    sample_mode = str(sample_mode)
    context_segments = []
    context_feature_names = valid_traces[0][1]
    for context, feature_names in valid_traces:
        sample_idx = select_trace_indices(
            context,
            sample_size=tail_steps,
            sample_mode=sample_mode,
            saliency_alpha=saliency_alpha,
        )
        context_tail = context[sample_idx]
        if len(context_tail) < tail_steps:
            context_pad = np.repeat(context_tail[:1], tail_steps - len(context_tail), axis=0)
            context_tail = np.concatenate([context_pad, context_tail], axis=0)
        context_segments.append(context_tail)
        if feature_names and not context_feature_names:
            context_feature_names = feature_names

    context_mean = np.mean(np.stack(context_segments, axis=0), axis=0).astype(np.float32)
    return {
        "context": context_mean,
        "context_feature_names": list(context_feature_names),
    }


def select_trace_indices(
    context: np.ndarray,
    sample_size: int,
    sample_mode: str = "tail",
    saliency_alpha: float = 0.25,
) -> np.ndarray:
    length = len(context)
    if length == 0:
        return np.zeros(0, dtype=np.int64)
    sample_size = max(1, min(int(sample_size), length))
    if sample_mode == "tail":
        return np.arange(length - sample_size, length, dtype=np.int64)
    if sample_mode == "uniform":
        return np.linspace(0, length - 1, num=sample_size, dtype=np.int64)
    if sample_mode != "salient_mix":
        raise ValueError(f"Unsupported context sample mode: {sample_mode}")

    salient_k = max(1, sample_size // 2)
    uniform_k = max(0, sample_size - salient_k)
    if length > 1:
        context_delta = np.zeros(length, dtype=np.float64)
        context_delta[1:] = np.linalg.norm(context[1:] - context[:-1], axis=1)
    else:
        context_delta = np.zeros(length, dtype=np.float64)
    context_center = np.mean(context, axis=0, keepdims=True)
    context_level = np.linalg.norm(context - context_center, axis=1)
    saliency = context_level + float(saliency_alpha) * context_delta

    salient_idx = np.argsort(-saliency)[:salient_k]
    if uniform_k > 0:
        uniform_idx = np.linspace(0, length - 1, num=uniform_k, dtype=np.int64)
        merged = np.concatenate([salient_idx, uniform_idx], axis=0)
    else:
        merged = salient_idx

    merged = np.unique(np.clip(merged, 0, length - 1))
    if len(merged) < sample_size:
        recent_idx = np.arange(length - sample_size, length, dtype=np.int64)
        merged = np.unique(np.concatenate([merged, recent_idx], axis=0))
    merged = np.sort(merged)
    if len(merged) > sample_size:
        ranked = sorted(
            ((saliency[idx], int(idx)) for idx in merged),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )[:sample_size]
        merged = np.sort(np.asarray([idx for _score, idx in ranked], dtype=np.int64))
    return merged.astype(np.int64)


def detect_context_shift_points(
    context: np.ndarray,
    *,
    window: int = 6,
    shift_threshold: float = 0.6,
    min_shift_gap: int = 6,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 2 or len(context) < 3:
        return np.zeros(0, dtype=np.int64)

    window = max(1, min(int(window), max(1, len(context) // 3)))
    if len(context) < 2 * window + 1:
        return np.zeros(0, dtype=np.int64)

    feature_scale = float(np.mean(np.std(context, axis=0))) + 1e-6
    candidates = []
    for shift_idx in range(window, len(context) - window):
        pre = context[shift_idx - window : shift_idx]
        post = context[shift_idx : shift_idx + window]
        delta = float(np.linalg.norm(post.mean(axis=0) - pre.mean(axis=0))) / feature_scale
        if delta >= shift_threshold:
            candidates.append((delta, shift_idx))

    if not candidates:
        return np.zeros(0, dtype=np.int64)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[int] = []
    for _score, shift_idx in candidates:
        if all(abs(shift_idx - prev_idx) >= min_shift_gap for prev_idx in selected):
            selected.append(int(shift_idx))
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


def context_volatility(context: np.ndarray) -> float:
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 2 or len(context) < 2:
        return 0.0
    diff = np.linalg.norm(context[1:] - context[:-1], axis=1)
    scale = float(np.mean(np.std(context, axis=0))) + 1e-6
    return float(np.mean(diff) / scale)
