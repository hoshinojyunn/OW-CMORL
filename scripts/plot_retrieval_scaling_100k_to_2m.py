#!/usr/bin/env python3
"""Measure and plot HNSW versus exact retrieval from 100k to 2M embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time

# Keep exact full scans and HNSW queries on one CPU thread per request.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import hnswlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_ablation_small" / "retrieval_scaling_100k_to_2m",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=PROJECT_ROOT / "figures" / "owcmorl_ablation_small" / "retrieval_latency_scaling_100k_to_2m.png",
    )
    parser.add_argument("--start-vectors", type=int, default=100_000)
    parser.add_argument("--stop-vectors", type=int, default=2_000_000)
    parser.add_argument("--step-vectors", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--hnsw-ef", type=int, default=64)
    parser.add_argument("--index-threads", type=int, default=4)
    parser.add_argument("--queries-per-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), np.float32(1e-12))
    return values / norms


def query_hnsw(
    index: hnswlib.Index,
    vectors: np.memmap,
    query: np.ndarray,
    *,
    candidate_count: int,
    top_k: int,
) -> np.ndarray:
    labels, _distances = index.knn_query(query.reshape(1, -1), k=candidate_count)
    candidates = labels[0].astype(np.int64)
    scores = vectors[candidates] @ query
    return candidates[np.argpartition(-scores, top_k - 1)[:top_k]]


def query_exact(vectors: np.memmap, count: int, query: np.ndarray, top_k: int) -> np.ndarray:
    scores = vectors[:count] @ query
    return np.argpartition(-scores, top_k - 1)[:top_k]


def build_vectors(path: Path, *, count: int, dimensions: int, seed: int, chunk_size: int) -> np.memmap:
    rng = np.random.default_rng(seed)
    vectors = np.memmap(path, mode="w+", dtype=np.float32, shape=(count, dimensions))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        values = rng.standard_normal((stop - start, dimensions), dtype=np.float32)
        vectors[start:stop] = normalize_rows(values)
    vectors.flush()
    return vectors


def benchmark(args: argparse.Namespace) -> list[dict[str, float | int]]:
    counts = list(range(args.start_vectors, args.stop_vectors + 1, args.step_vectors))
    if not counts or counts[-1] != args.stop_vectors:
        raise ValueError("stop-vectors must be reachable from start-vectors by step-vectors")
    if args.start_vectors < args.top_k or args.dimensions <= 0 or args.queries_per_size <= 0:
        raise ValueError("invalid benchmark dimensions")

    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    vector_path = args.out_dir / "embeddings_2m_256d.f32"
    vectors = build_vectors(
        vector_path,
        count=args.stop_vectors,
        dimensions=args.dimensions,
        seed=args.seed,
        chunk_size=args.step_vectors,
    )

    query_rng = np.random.default_rng(args.seed + 1)
    queries = normalize_rows(query_rng.standard_normal((args.queries_per_size, args.dimensions), dtype=np.float32))
    index = hnswlib.Index(space="cosine", dim=args.dimensions)
    index.init_index(
        max_elements=args.stop_vectors,
        ef_construction=max(200, args.hnsw_ef),
        M=16,
        random_seed=args.seed,
    )
    index.set_num_threads(max(1, int(args.index_threads)))
    index.set_ef(max(args.hnsw_ef, args.top_k))

    rows: list[dict[str, float | int]] = []
    previous_count = 0
    for count in counts:
        index.set_num_threads(max(1, int(args.index_threads)))
        build_start = time.perf_counter_ns()
        index.add_items(vectors[previous_count:count], np.arange(previous_count, count, dtype=np.int64))
        incremental_build_ms = (time.perf_counter_ns() - build_start) / 1e6
        previous_count = count

        candidate_count = min(count, max(args.top_k, args.hnsw_ef))
        index.set_num_threads(1)
        query_hnsw(index, vectors, queries[0], candidate_count=candidate_count, top_k=args.top_k)
        query_exact(vectors, count, queries[0], args.top_k)

        hnsw_times: list[float] = []
        exact_times: list[float] = []
        overlap: list[int] = []
        for query in queries:
            started = time.perf_counter_ns()
            hnsw_ids = query_hnsw(index, vectors, query, candidate_count=candidate_count, top_k=args.top_k)
            hnsw_times.append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            exact_ids = query_exact(vectors, count, query, args.top_k)
            exact_times.append((time.perf_counter_ns() - started) / 1e6)
            overlap.append(len(set(hnsw_ids.tolist()) & set(exact_ids.tolist())))

        hnsw_values = np.asarray(hnsw_times, dtype=np.float64)
        exact_values = np.asarray(exact_times, dtype=np.float64)
        rows.append(
            {
                "embedding_count": count,
                "hnsw_candidate_embedding_comparisons": candidate_count,
                "exact_embedding_comparisons": count,
                "hnsw_median_ms": float(np.median(hnsw_values)),
                "hnsw_mean_ms": float(hnsw_values.mean()),
                "hnsw_p95_ms": float(np.percentile(hnsw_values, 95)),
                "exact_median_ms": float(np.median(exact_values)),
                "exact_mean_ms": float(exact_values.mean()),
                "exact_p95_ms": float(np.percentile(exact_values, 95)),
                "exact_over_hnsw_mean_speedup": float(exact_values.mean() / max(hnsw_values.mean(), 1e-12)),
                "mean_top50_overlap": float(np.mean(overlap)),
                "incremental_index_build_ms": incremental_build_ms,
            }
        )
        print(f"{count}: hnsw={hnsw_values.mean():.3f} ms, exact={exact_values.mean():.3f} ms", flush=True)
    index.save_index(str(args.out_dir / "hnsw_scaling_2m.bin"))
    return rows


def write_outputs(args: argparse.Namespace, rows: list[dict[str, float | int]]) -> None:
    csv_path = args.out_dir / "retrieval_scaling.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    args.figure_path = args.figure_path.resolve()
    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray([int(row["embedding_count"]) for row in rows], dtype=np.int64)
    hnsw = np.asarray([float(row["hnsw_mean_ms"]) for row in rows], dtype=np.float64)
    exact = np.asarray([float(row["exact_mean_ms"]) for row in rows], dtype=np.float64)
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(counts, hnsw, marker="o", markersize=3.5, linewidth=1.8, color="#007a78", label="HNSW")
    axis.plot(counts, exact, marker="s", markersize=3.2, linewidth=1.8, color="#c34a36", label="Exact scan")
    axis.set_xlabel("Number of embeddings")
    axis.set_ylabel("Mean retrieval latency per shift (s)")
    axis.set_xlim(args.start_vectors, args.stop_vectors)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.28, linewidth=0.6)
    axis.legend(frameon=False, loc="upper left")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    fig.savefig(args.figure_path, dpi=220)
    plt.close(fig)

    summary = {
        "benchmark": "hnsw_exact_retrieval_scaling",
        "protocol": {
            "embedding_counts": [int(row["embedding_count"]) for row in rows],
            "dimensions": args.dimensions,
            "top_k": args.top_k,
            "hnsw_ef_search": args.hnsw_ef,
            "queries_per_size": args.queries_per_size,
            "seed": args.seed,
            "timing_statistic": "mean latency over identical query set after one warmup",
        },
        "rows": rows,
        "artifacts": {"csv": str(csv_path), "figure": str(args.figure_path)},
    }
    (args.out_dir / "retrieval_scaling_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    rows = benchmark(args)
    write_outputs(args, rows)


if __name__ == "__main__":
    main()
