from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .temporal import cosine_affinity


def _bounded_sample_score(
    sample,
    *,
    recovery_weight: float = 1.0,
    regret_weight: float = 0.45,
    latency_weight: float = 0.25,
    gap_weight: float = 0.10,
) -> float:
    trace_metrics = sample.metadata.get("trace_metrics", {})
    recovery = float(trace_metrics.get("trace_recovery_score", 0.0))
    regret = abs(float(trace_metrics.get("trace_shift_regret", 0.0)))
    latency = max(0.0, float(trace_metrics.get("trace_recovery_latency", 0.0)))
    gap = float(trace_metrics.get("trace_pre_post_gap", 0.0))
    return float(
        float(recovery_weight) * recovery
        + float(regret_weight) / (1.0 + regret)
        + float(latency_weight) / (1.0 + latency)
        + float(gap_weight) * np.clip(gap, -1.0, 1.0)
    )


def _sample_uid(sample) -> str:
    metadata = sample.metadata
    uid = metadata.get("sample_uid")
    if uid is None:
        uid = f"sample_{id(sample)}"
        metadata["sample_uid"] = uid
    return str(uid)


@dataclass
class ExpertSlot:
    prototype: np.ndarray
    samples: list = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def mean_score(self) -> float:
        if not self.scores:
            return 0.0
        return float(np.mean(self.scores))


class ContextExpertBank:
    def __init__(
        self,
        max_slots: int = 6,
        experts_per_slot: int = 3,
        merge_threshold: float = 0.82,
        recovery_weight: float = 1.0,
        regret_weight: float = 0.45,
        latency_weight: float = 0.25,
        gap_weight: float = 0.10,
        retrieval_backend: Literal["hnsw", "exact"] = "hnsw",
        hnsw_ef_search: int = 32,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 100,
    ):
        self.max_slots = max(1, int(max_slots))
        self.experts_per_slot = max(1, int(experts_per_slot))
        self.merge_threshold = float(np.clip(merge_threshold, -1.0, 1.0))
        self.recovery_weight = float(recovery_weight)
        self.regret_weight = float(regret_weight)
        self.latency_weight = float(latency_weight)
        self.gap_weight = float(gap_weight)
        if retrieval_backend not in {"hnsw", "exact"}:
            raise ValueError(f"unsupported expert-bank backend: {retrieval_backend!r}")
        self.retrieval_backend = retrieval_backend
        self.hnsw_ef_search = max(1, int(hnsw_ef_search))
        self.hnsw_m = max(2, int(hnsw_m))
        self.hnsw_ef_construction = max(2, int(hnsw_ef_construction))
        self.slots: list[ExpertSlot] = []
        self._insert_count = 0
        self._hnsw_index = None
        self._hnsw_labels: list[int] = []
        self._hnsw_dirty = True
        self._resolved_backend = "exact" if retrieval_backend == "exact" else "hnsw"

    def __len__(self) -> int:
        return sum(len(slot.samples) for slot in self.slots)

    def _locate_slot(self, embedding: np.ndarray) -> int | None:
        if not self.slots:
            return None
        affinities = [cosine_affinity(slot.prototype, embedding) for slot in self.slots]
        best_idx = int(np.argmax(affinities))
        if affinities[best_idx] >= self.merge_threshold:
            return best_idx
        return None

    def _evict_slot(self) -> int:
        if not self.slots:
            return 0
        slot_scores = []
        for idx, slot in enumerate(self.slots):
            slot_scores.append((slot.mean_score(), len(slot.samples), idx))
        slot_scores.sort(key=lambda row: (row[0], row[1]))
        return int(slot_scores[0][2])

    def _insert_into_slot(self, slot: ExpertSlot, sample, embedding: np.ndarray, score: float) -> None:
        sample_copy = deepcopy(sample)
        sample_copy.metadata["sample_uid"] = _sample_uid(sample_copy)
        sample_copy.metadata["bank_score"] = float(score)
        sample_copy.metadata["bank_insert_idx"] = int(self._insert_count)
        self._insert_count += 1

        uid = _sample_uid(sample_copy)
        for idx, existing in enumerate(slot.samples):
            if _sample_uid(existing) == uid:
                if score > slot.scores[idx]:
                    slot.samples[idx] = sample_copy
                    slot.scores[idx] = float(score)
                slot.prototype = 0.5 * slot.prototype + 0.5 * np.asarray(embedding, dtype=np.float64)
                self._hnsw_dirty = True
                return

        slot.samples.append(sample_copy)
        slot.scores.append(float(score))
        slot.prototype = 0.8 * slot.prototype + 0.2 * np.asarray(embedding, dtype=np.float64)

        if len(slot.samples) > self.experts_per_slot:
            ranked = sorted(
                zip(slot.samples, slot.scores),
                key=lambda row: (row[1], row[0].metadata.get("bank_insert_idx", 0)),
                reverse=True,
            )[: self.experts_per_slot]
            slot.samples = [sample_row for sample_row, _score in ranked]
            slot.scores = [float(score_row) for _sample_row, score_row in ranked]
        self._hnsw_dirty = True

    def add(self, sample) -> None:
        embedding = sample.metadata.get("context_embedding")
        if embedding is None:
            return
        embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if embedding.size == 0:
            return
        score = _bounded_sample_score(
            sample,
            recovery_weight=self.recovery_weight,
            regret_weight=self.regret_weight,
            latency_weight=self.latency_weight,
            gap_weight=self.gap_weight,
        )
        slot_idx = self._locate_slot(embedding)
        if slot_idx is None:
            if len(self.slots) >= self.max_slots:
                slot_idx = self._evict_slot()
                self.slots[slot_idx] = ExpertSlot(prototype=embedding.copy())
            else:
                self.slots.append(ExpertSlot(prototype=embedding.copy()))
                slot_idx = len(self.slots) - 1
        self._insert_into_slot(self.slots[slot_idx], sample, embedding, score)

    def update_batch(self, sample_batch) -> None:
        for sample in sample_batch:
            self.add(sample)

    @property
    def resolved_retrieval_backend(self) -> str:
        """Return the backend actually used (HNSW can fall back if optional deps are absent)."""
        return self._resolved_backend

    def _rebuild_hnsw(self) -> bool:
        """Build a small cosine HNSW index over current slot prototypes when needed."""
        if not self._hnsw_dirty:
            return self._hnsw_index is not None
        self._hnsw_index = None
        self._hnsw_labels = []
        self._hnsw_dirty = False
        if self.retrieval_backend != "hnsw" or not self.slots:
            self._resolved_backend = "exact"
            return False
        try:
            import hnswlib
        except ImportError:
            self._resolved_backend = "exact"
            return False

        vectors = np.stack(
            [np.asarray(slot.prototype, dtype=np.float32).reshape(-1) for slot in self.slots], axis=0
        )
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            self._resolved_backend = "exact"
            return False
        index = hnswlib.Index(space="cosine", dim=int(vectors.shape[1]))
        index.init_index(
            max_elements=len(vectors),
            ef_construction=self.hnsw_ef_construction,
            M=self.hnsw_m,
            random_seed=0,
        )
        labels = np.arange(len(vectors), dtype=np.int64)
        index.add_items(vectors, labels)
        index.set_ef(max(self.hnsw_ef_search, 1))
        self._hnsw_index = index
        self._hnsw_labels = labels.tolist()
        self._resolved_backend = "hnsw"
        return True

    def _rank_slots_exact(self, query_embedding: np.ndarray) -> list[tuple[float, float, int]]:
        scored_slots = []
        for idx, slot in enumerate(self.slots):
            affinity = cosine_affinity(slot.prototype, query_embedding)
            weighted = affinity * (0.5 + 0.5 * np.clip(slot.mean_score(), 0.0, 1.5))
            scored_slots.append((weighted, affinity, idx))
        scored_slots.sort(reverse=True)
        return scored_slots

    def _rank_slots(self, query_embedding: np.ndarray, top_slots: int) -> list[tuple[float, float, int]]:
        if not self._rebuild_hnsw():
            return self._rank_slots_exact(query_embedding)
        assert self._hnsw_index is not None
        count = min(max(1, int(top_slots)), len(self.slots))
        try:
            labels, _distances = self._hnsw_index.knn_query(
                np.asarray(query_embedding, dtype=np.float32).reshape(1, -1), k=count
            )
        except RuntimeError:
            # A malformed query should retain the historical exact-search behavior.
            self._resolved_backend = "exact"
            return self._rank_slots_exact(query_embedding)

        scored_slots = []
        for slot_idx in labels[0].tolist():
            slot = self.slots[int(slot_idx)]
            affinity = cosine_affinity(slot.prototype, query_embedding)
            weighted = affinity * (0.5 + 0.5 * np.clip(slot.mean_score(), 0.0, 1.5))
            scored_slots.append((weighted, affinity, int(slot_idx)))
        scored_slots.sort(reverse=True)
        return scored_slots

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        top_slots: int = 3,
        top_per_slot: int | None = None,
    ) -> list:
        if not self.slots:
            return []
        query_embedding = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
        if query_embedding.size == 0:
            return []
        top_slots = max(1, int(top_slots))
        top_per_slot = self.experts_per_slot if top_per_slot is None else max(1, int(top_per_slot))
        scored_slots = self._rank_slots(query_embedding, top_slots)
        selected = []
        for _weighted, affinity, slot_idx in scored_slots[:top_slots]:
            slot = self.slots[slot_idx]
            ranked = sorted(
                zip(slot.samples, slot.scores),
                key=lambda row: (row[1], row[0].metadata.get("bank_insert_idx", 0)),
                reverse=True,
            )[:top_per_slot]
            for sample, score in ranked:
                sample_copy = deepcopy(sample)
                sample_copy.metadata["bank_query_affinity"] = float(affinity)
                sample_copy.metadata["bank_score"] = float(score)
                selected.append(sample_copy)
        return selected

    def query_global(self, *, top_k: int = 1) -> list:
        """Return high-scoring bank samples without reading a context query.

        This deliberately supports the no-context ablation: selection uses the
        existing bank score only and cannot specialize on the current regime.
        """
        ranked = []
        for slot in self.slots:
            for sample, score in zip(slot.samples, slot.scores):
                ranked.append((float(score), int(sample.metadata.get("bank_insert_idx", 0)), sample))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = []
        for score, _insert_idx, sample in ranked[: max(1, int(top_k))]:
            sample_copy = deepcopy(sample)
            sample_copy.metadata["bank_score"] = float(score)
            sample_copy.metadata["bank_query_affinity"] = 0.0
            selected.append(sample_copy)
        return selected

    def export_samples(self) -> list:
        samples = []
        for slot in self.slots:
            for sample in slot.samples:
                samples.append(deepcopy(sample))
        return samples

    def state_dict(self) -> dict:
        """Serialize semantic expert state, excluding the rebuildable ANN graph."""

        return {
            "schema_version": 1,
            "max_slots": int(self.max_slots),
            "experts_per_slot": int(self.experts_per_slot),
            "merge_threshold": float(self.merge_threshold),
            "recovery_weight": float(self.recovery_weight),
            "regret_weight": float(self.regret_weight),
            "latency_weight": float(self.latency_weight),
            "gap_weight": float(self.gap_weight),
            "retrieval_backend": self.retrieval_backend,
            "hnsw_ef_search": int(self.hnsw_ef_search),
            "hnsw_m": int(self.hnsw_m),
            "hnsw_ef_construction": int(self.hnsw_ef_construction),
            "slots": deepcopy(self.slots),
            "insert_count": int(self._insert_count),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "ContextExpertBank":
        """Restore semantic slots and lazily rebuild HNSW on the first query."""

        required = (
            "max_slots",
            "experts_per_slot",
            "merge_threshold",
            "recovery_weight",
            "regret_weight",
            "latency_weight",
            "gap_weight",
            "retrieval_backend",
            "hnsw_ef_search",
            "hnsw_m",
            "hnsw_ef_construction",
            "slots",
            "insert_count",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Expert-bank checkpoint is missing: {', '.join(missing)}")
        bank = cls(
            max_slots=int(payload["max_slots"]),
            experts_per_slot=int(payload["experts_per_slot"]),
            merge_threshold=float(payload["merge_threshold"]),
            recovery_weight=float(payload["recovery_weight"]),
            regret_weight=float(payload["regret_weight"]),
            latency_weight=float(payload["latency_weight"]),
            gap_weight=float(payload["gap_weight"]),
            retrieval_backend=str(payload["retrieval_backend"]),
            hnsw_ef_search=int(payload["hnsw_ef_search"]),
            hnsw_m=int(payload["hnsw_m"]),
            hnsw_ef_construction=int(payload["hnsw_ef_construction"]),
        )
        slots = deepcopy(list(payload["slots"]))
        if len(slots) > bank.max_slots or any(not isinstance(slot, ExpertSlot) for slot in slots):
            raise ValueError("Expert-bank checkpoint contains invalid slots.")
        for slot in slots:
            if len(slot.samples) != len(slot.scores) or len(slot.samples) > bank.experts_per_slot:
                raise ValueError("Expert-bank checkpoint contains an invalid expert slot.")
            slot.prototype = np.asarray(slot.prototype, dtype=np.float64).reshape(-1)
            if slot.prototype.size == 0:
                raise ValueError("Expert-bank checkpoint contains an empty slot prototype.")
        bank.slots = slots
        bank._insert_count = int(payload["insert_count"])
        bank._hnsw_index = None
        bank._hnsw_labels = []
        bank._hnsw_dirty = True
        bank._resolved_backend = "exact" if bank.retrieval_backend == "exact" else "hnsw"
        return bank
