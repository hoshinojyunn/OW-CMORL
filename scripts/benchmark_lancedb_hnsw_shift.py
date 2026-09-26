#!/usr/bin/env python3
"""Benchmark LanceDB-persisted HNSW and brute-force retrieval over shifted queries.

The benchmark intentionally isolates the retrieval back end from MORL quality
metrics.  It writes 100,000 normalized 256-dimensional vectors to LanceDB,
builds an HNSW index over that persisted vector set, and evaluates both
indexed and exact Top-50 searches over the same deterministic sequence of 100
shifted query vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_ablation_small" / "lancedb_hnsw_100k",
    )
    parser.add_argument("--num-vectors", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--num-shifts", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--hnsw-ef", type=int, default=64)
    return parser.parse_args()


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.float32(1e-12))
    return values / norms


def vector_table(vectors: np.ndarray):
    import pyarrow as pa

    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(flat, list_size=int(vectors.shape[1]))
    return pa.table(
        {
            "id": pa.array(np.arange(len(vectors), dtype=np.int64)),
            "vector": vector_array,
        }
    )


class HNSWIndex:
    """Cosine HNSW index over the exact vectors written to LanceDB.

    LanceDB 0.6 (the Python 3.8-compatible version in ``morl-pareto``) only
    exposes IVF-PQ through ``create_index``.  It is deliberately not used as
    a substitute for HNSW.  LanceDB remains the persistent vector store while
    hnswlib supplies the requested HNSW graph over the same rows.
    """

    def __init__(self, vectors: np.ndarray, *, ef_search: int) -> None:
        import hnswlib

        self.index = hnswlib.Index(space="cosine", dim=int(vectors.shape[1]))
        self.index.init_index(
            max_elements=len(vectors),
            ef_construction=max(200, int(ef_search)),
            M=16,
            random_seed=0,
        )
        self.index.set_num_threads(1)
        self.index.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
        self.index.set_ef(max(int(ef_search), 50))

    def query(self, query: np.ndarray, top_k: int) -> np.ndarray:
        labels, _distances = self.index.knn_query(query.reshape(1, -1), k=int(top_k))
        return labels[0].astype(np.int64)


def main() -> None:
    args = parse_args()
    if args.num_vectors < args.top_k:
        raise ValueError("num_vectors must be at least top_k")
    if args.dimensions <= 0 or args.num_shifts <= 0 or args.top_k <= 0:
        raise ValueError("dimensions, num_shifts, and top_k must be positive")

    try:
        import lancedb
    except ImportError as exc:
        raise SystemExit("LanceDB is required. Install it with Python 3.9+ before running this benchmark.") from exc

    args.out_dir = args.out_dir.resolve()
    db_dir = args.out_dir / "lancedb"
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    db_dir.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    vectors = normalized_rows(rng.standard_normal((args.num_vectors, args.dimensions), dtype=np.float32))
    db = lancedb.connect(str(db_dir))
    table = db.create_table("embeddings", data=vector_table(vectors), mode="overwrite")

    index_start = time.perf_counter_ns()
    hnsw = HNSWIndex(vectors, ef_search=args.hnsw_ef)
    index_build_ms = (time.perf_counter_ns() - index_start) / 1e6
    hnsw.index.save_index(str(args.out_dir / "hnsw_cosine_100k.bin"))

    query = normalized_rows(rng.standard_normal((1, args.dimensions), dtype=np.float32))[0]
    shifts: list[np.ndarray] = []
    for _ in range(args.num_shifts):
        # A drifted context rather than independent samples: every shift carries
        # most of the previous state and injects a newly observed perturbation.
        innovation = rng.standard_normal(args.dimensions, dtype=np.float32)
        query = normalized_rows((0.85 * query + 0.15 * innovation).reshape(1, -1))[0]
        shifts.append(query.copy())

    # Execute unmeasured warmups to avoid timing first-use allocations.
    hnsw.query(shifts[0], max(args.top_k, args.hnsw_ef))
    np.argpartition(-(vectors @ shifts[0]), args.top_k - 1)[: args.top_k]

    per_shift: list[dict[str, Any]] = []
    hnsw_candidate_count = min(len(vectors), max(int(args.top_k), int(args.hnsw_ef)))
    for shift_id, shifted_query in enumerate(shifts):
        hnsw_start = time.perf_counter_ns()
        hnsw_candidates = hnsw.query(shifted_query, hnsw_candidate_count)
        # This explicit cosine re-rank is the observable embedding-comparison
        # budget for the approximate path: ef_search candidates, then Top-K.
        hnsw_scores = vectors[hnsw_candidates] @ shifted_query
        hnsw_ids = hnsw_candidates[np.argpartition(-hnsw_scores, args.top_k - 1)[: args.top_k]]
        hnsw_ms = (time.perf_counter_ns() - hnsw_start) / 1e6

        exact_start = time.perf_counter_ns()
        similarities = vectors @ shifted_query
        exact_ids = np.argpartition(-similarities, args.top_k - 1)[: args.top_k]
        exact_ms = (time.perf_counter_ns() - exact_start) / 1e6

        if len(hnsw_ids) != args.top_k or len(exact_ids) != args.top_k:
            raise RuntimeError(f"Top-{args.top_k} retrieval failed at shift {shift_id}")
        hnsw_id_set = set(hnsw_ids.tolist())
        exact_id_set = set(exact_ids.tolist())
        exact_best_similarity = float(similarities[exact_ids].max())
        hnsw_best_similarity = float(similarities[hnsw_ids].max())
        per_shift.append(
            {
                "shift": shift_id + 1,
                "hnsw_ms": hnsw_ms,
                "exact_ms": exact_ms,
                "top50_overlap": len(hnsw_id_set & exact_id_set),
                "hnsw_candidate_embedding_comparisons": int(hnsw_candidate_count),
                "exact_embedding_comparisons": int(len(vectors)),
                "exact_best_similarity": exact_best_similarity,
                "hnsw_best_similarity": hnsw_best_similarity,
                "hnsw_shift_regret": exact_best_similarity - hnsw_best_similarity,
            }
        )

    hnsw_ms = np.asarray([row["hnsw_ms"] for row in per_shift], dtype=np.float64)
    exact_ms = np.asarray([row["exact_ms"] for row in per_shift], dtype=np.float64)
    overlap = np.asarray([row["top50_overlap"] for row in per_shift], dtype=np.float64)
    hnsw_regret = np.asarray([row["hnsw_shift_regret"] for row in per_shift], dtype=np.float64)
    payload = {
        "benchmark": "lancedb_hnsw_shift_retrieval",
        "backend": {
            "storage_library": "lancedb",
            "version": getattr(lancedb, "__version__", "unknown"),
            "index_library": "hnswlib",
            "index_type": "HNSW",
            "metric": "cosine",
            "lancedb_native_hnsw_available": False,
            "storage_table": "embeddings",
        },
        "protocol": {
            "num_vectors": int(args.num_vectors),
            "dimensions": int(args.dimensions),
            "num_shifts": int(args.num_shifts),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
            "shift_rule": "q_t = normalize(0.85*q_(t-1) + 0.15*epsilon_t)",
            "warmup_queries_excluded": 2,
        },
        "index_build_ms": index_build_ms,
        "hnsw": {
            "steps": int(args.num_shifts),
            "candidate_embedding_comparisons_per_step": int(hnsw_candidate_count),
            "candidate_embedding_comparisons_total": int(hnsw_candidate_count * args.num_shifts),
            "total_query_ms": float(hnsw_ms.sum()),
            "mean_recovery_latency_ms": float(hnsw_ms.mean()),
            "median_recovery_latency_ms": float(np.median(hnsw_ms)),
            "p95_recovery_latency_ms": float(np.percentile(hnsw_ms, 95)),
            "mean_shift_regret": float(hnsw_regret.mean()),
            "p95_shift_regret": float(np.percentile(hnsw_regret, 95)),
        },
        "exact": {
            "steps": int(args.num_shifts),
            "candidate_embedding_comparisons_per_step": int(len(vectors)),
            "candidate_embedding_comparisons_total": int(len(vectors) * args.num_shifts),
            "total_query_ms": float(exact_ms.sum()),
            "mean_recovery_latency_ms": float(exact_ms.mean()),
            "median_recovery_latency_ms": float(np.median(exact_ms)),
            "p95_recovery_latency_ms": float(np.percentile(exact_ms, 95)),
            "mean_shift_regret": 0.0,
            "p95_shift_regret": 0.0,
        },
        "speedup_exact_over_hnsw": float(exact_ms.sum() / max(hnsw_ms.sum(), 1e-12)),
        "candidate_comparison_reduction_exact_over_hnsw": float(len(vectors) / hnsw_candidate_count),
        "mean_top50_overlap": float(overlap.mean()),
        "per_shift": per_shift,
    }
    (args.out_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
