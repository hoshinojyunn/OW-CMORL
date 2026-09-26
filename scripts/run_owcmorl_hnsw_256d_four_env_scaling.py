#!/usr/bin/env python3
"""Four-environment conditioned 1M-slot HNSW scaling diagnostic.

This script is a controlled scalability study, not a replacement for
end-to-end OW-CMORL retraining.  Each environment uses its native
dynamic-factor dimensionality and objective dimensionality to generate a
distinct factor-conditioned embedding distribution.  The generated
256-dimensional slot keys, query keys, and candidate solution profiles are
then used to compare HNSW Top-K retrieval with exact Top-K retrieval under
the same bank and the same query stream.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import hnswlib
import numpy as np
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    factor_dim: int
    objective_dim: int
    clusters: int
    seed_offset: int


ENVIRONMENTS = (
    EnvironmentSpec("building", factor_dim=11, objective_dim=3, clusters=10, seed_offset=11),
    EnvironmentSpec("evcharging", factor_dim=8, objective_dim=3, clusters=12, seed_offset=23),
    EnvironmentSpec("cogen", factor_dim=10, objective_dim=4, clusters=14, seed_offset=37),
    EnvironmentSpec("chlor_alkali", factor_dim=6, objective_dim=3, clusters=9, seed_offset=53),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=1_000_000)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--queries-per-env", type=int, default=32)
    parser.add_argument("--timing-queries-per-env", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--hnsw-m", type=int, default=8)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-ef-construction", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--env-keys",
        nargs="+",
        choices=[spec.name for spec in ENVIRONMENTS],
        default=[spec.name for spec in ENVIRONMENTS],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_hnsw_256d_four_env",
    )
    return parser.parse_args()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_environment_bank(
    spec: EnvironmentSpec,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a 256-D bank driven by the environment's factor dimension."""
    rng = np.random.default_rng(args.seed + spec.seed_offset)
    centers = rng.normal(0.0, 0.75, size=(spec.clusters, spec.factor_dim)).astype(np.float32)
    slot_cluster = rng.integers(0, spec.clusters, size=args.slots, endpoint=False)
    factors = (
        centers[slot_cluster]
        + rng.normal(0.0, 0.22, size=(args.slots, spec.factor_dim)).astype(np.float32)
    ).astype(np.float32)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(spec.factor_dim),
        size=(spec.factor_dim, args.embedding_dim),
    ).astype(np.float32)
    embeddings = np.empty((args.slots, args.embedding_dim), dtype=np.float32)
    for start in range(0, args.slots, args.chunk_size):
        stop = min(start + args.chunk_size, args.slots)
        projected = factors[start:stop] @ projection
        projected += rng.normal(0.0, 0.028, size=projected.shape).astype(np.float32)
        embeddings[start:stop] = _normalize_rows(projected)

    # Each slot carries a multi-objective candidate profile. It is queried
    # only after the Top-50 snapshot identifiers are retrieved.
    profiles = rng.uniform(
        0.35,
        1.0,
        size=(args.slots, spec.objective_dim),
    ).astype(np.float32)
    utility_weights = rng.dirichlet(
        np.ones(spec.objective_dim, dtype=np.float64),
        size=51,
    ).astype(np.float64)
    return centers, factors, embeddings, profiles, utility_weights, projection


def _make_queries(
    spec: EnvironmentSpec,
    centers: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    query_cluster = rng.integers(0, spec.clusters, size=count, endpoint=False)
    factors = (
        centers[query_cluster]
        + rng.normal(0.0, 0.15, size=(count, spec.factor_dim)).astype(np.float32)
    ).astype(np.float32)
    return factors


def _query_embedding(factor: np.ndarray, projection: np.ndarray) -> np.ndarray:
    vector = np.asarray(factor, dtype=np.float32).reshape(1, -1) @ projection
    return _normalize_rows(vector)[0]


def _exact_topk(embeddings: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    scores = embeddings @ query
    indices = np.argpartition(-scores, top_k - 1)[:top_k]
    return indices[np.argsort(-scores[indices], kind="stable")]


def _solution_metrics(
    indices: np.ndarray,
    *,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    utility_weights: np.ndarray,
) -> tuple[float, float, float]:
    # The controlled candidate quality is monotone in contextual affinity.
    # Exact Top-K therefore upper-bounds HNSW Top-K under this diagnostic.
    match = np.clip((embeddings[indices] @ query_embedding + 1.0) / 2.0, 0.0, 1.0)
    match = np.sort(match.astype(np.float64))[::-1]
    ranks = (np.arange(len(match), dtype=np.float64) + 0.5) / len(match)
    template = np.column_stack(
        [0.55 + 0.45 * np.cos(2.0 * np.pi * (ranks + objective / utility_weights.shape[1])) ** 2
         for objective in range(utility_weights.shape[1])]
    )
    objectives = match[:, None] * template
    hv = float(Hypervolume(ref_point=np.zeros(objectives.shape[1], dtype=np.float64)).do(-objectives))
    utility = np.max(objectives @ utility_weights.T, axis=0)
    eu = float(np.mean(utility))
    regret = float(1.0 - np.max(match))
    return hv, eu, regret


def _route_only(
    *,
    hnsw: hnswlib.Index | None,
    embeddings: np.ndarray,
    profiles: np.ndarray,
    projection: np.ndarray,
    query_factors: np.ndarray,
    top_k: int,
) -> list[float]:
    """Measure query construction, Top-K retrieval, and payload materialization."""
    times = []
    for factor in query_factors:
        start = time.perf_counter()
        query = _query_embedding(factor, projection)
        if hnsw is None:
            indices = _exact_topk(embeddings, query, top_k)
        else:
            labels, _distances = hnsw.knn_query(query.reshape(1, -1), k=top_k)
            indices = labels[0]
        payload = profiles[indices].copy()
        if payload.shape != (top_k, profiles.shape[1]):
            raise RuntimeError("Top-K candidate materialization returned an unexpected shape")
        times.append(1000.0 * (time.perf_counter() - start))
    return times


def _run_backend(
    *,
    backend: str,
    hnsw: hnswlib.Index | None,
    embeddings: np.ndarray,
    projection: np.ndarray,
    query_factors: np.ndarray,
    slot_factors: np.ndarray,
    profiles: np.ndarray,
    utility_weights: np.ndarray,
    top_k: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for query_id, factor in enumerate(query_factors):
        start = time.perf_counter()
        query = _query_embedding(factor, projection)
        if hnsw is None:
            indices = _exact_topk(embeddings, query, top_k)
        else:
            labels, _distances = hnsw.knn_query(query.reshape(1, -1), k=top_k)
            indices = labels[0]
        # Materialize the Top-50 candidate payload as part of routing time.
        candidate_profiles = profiles[indices].copy()
        if candidate_profiles.shape != (top_k, profiles.shape[1]):
            raise RuntimeError("Top-K candidate materialization returned an unexpected shape")
        route_ms = 1000.0 * (time.perf_counter() - start)
        hv, eu, regret = _solution_metrics(
            indices,
            embeddings=embeddings,
            query_embedding=query,
            utility_weights=utility_weights,
        )
        rows.append(
            {
                "query_id": int(query_id),
                "backend": backend,
                "hv": hv,
                "eu": eu,
                "regret": regret,
                "route_ms": route_ms,
            }
        )
    return rows


def _summarize(
    spec: EnvironmentSpec,
    backend: str,
    rows: list[dict[str, float]],
    *,
    args: argparse.Namespace,
    index_build_seconds: float,
    timing_ms: list[float],
) -> dict[str, object]:
    return {
        "environment": spec.name,
        "backend": backend,
        "mean_hv": float(np.mean([row["hv"] for row in rows])),
        "mean_eu": float(np.mean([row["eu"] for row in rows])),
        "mean_regret": float(np.mean([row["regret"] for row in rows])),
        "route_p50_ms": float(np.percentile(timing_ms, 50)),
        "route_p95_ms": float(np.percentile(timing_ms, 95)),
        "route_batch_seconds": float(np.sum(timing_ms) / 1000.0),
        "index_build_seconds": float(index_build_seconds),
        "slots": int(args.slots),
        "embedding_dim": int(args.embedding_dim),
        "top_k": int(args.top_k),
        "queries": int(args.queries_per_env),
        "timing_queries": int(args.timing_queries_per_env),
        "factor_dim": int(spec.factor_dim),
        "objective_dim": int(spec.objective_dim),
    }


def main() -> None:
    args = parse_args()
    if args.slots < args.top_k:
        raise ValueError("--slots must be at least --top-k")
    if args.hnsw_ef_search < args.top_k:
        raise ValueError("--hnsw-ef-search must be at least --top-k")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    selected = [spec for spec in ENVIRONMENTS if spec.name in args.env_keys]
    for spec in selected:
        centers, slot_factors, embeddings, profiles, utility_weights, projection = _make_environment_bank(
            spec,
            args,
        )
        query_rng = np.random.default_rng(args.seed + 1000 + spec.seed_offset)
        query_factors = _make_queries(
            spec,
            centers,
            args.queries_per_env,
            query_rng,
        )
        timing_factors = _make_queries(
            spec,
            centers,
            args.timing_queries_per_env,
            np.random.default_rng(args.seed + 2000 + spec.seed_offset),
        )

        build_start = time.perf_counter()
        hnsw = hnswlib.Index(space="cosine", dim=args.embedding_dim)
        hnsw.init_index(
            max_elements=args.slots,
            ef_construction=args.hnsw_ef_construction,
            M=args.hnsw_m,
            random_seed=args.seed + spec.seed_offset,
        )
        hnsw.set_num_threads(1)
        hnsw.add_items(embeddings, np.arange(args.slots, dtype=np.int64))
        hnsw.set_ef(args.hnsw_ef_search)
        build_seconds = time.perf_counter() - build_start

        hnsw_rows = _run_backend(
            backend="hnsw",
            hnsw=hnsw,
            embeddings=embeddings,
            projection=projection,
            query_factors=query_factors,
            slot_factors=slot_factors,
            profiles=profiles,
            utility_weights=utility_weights,
            top_k=args.top_k,
        )
        exact_rows = _run_backend(
            backend="exact",
            hnsw=None,
            embeddings=embeddings,
            projection=projection,
            query_factors=query_factors,
            slot_factors=slot_factors,
            profiles=profiles,
            utility_weights=utility_weights,
            top_k=args.top_k,
        )
        hnsw_timing = _route_only(
            hnsw=hnsw,
            embeddings=embeddings,
            profiles=profiles,
            projection=projection,
            query_factors=timing_factors,
            top_k=args.top_k,
        )
        exact_timing = _route_only(
            hnsw=None,
            embeddings=embeddings,
            profiles=profiles,
            projection=projection,
            query_factors=timing_factors,
            top_k=args.top_k,
        )
        for rows in (hnsw_rows, exact_rows):
            for row in rows:
                all_rows.append({"environment": spec.name, **row})
        summaries.append(
            _summarize(
                spec,
                "hnsw",
                hnsw_rows,
                args=args,
                index_build_seconds=build_seconds,
                timing_ms=hnsw_timing,
            )
        )
        summaries.append(
            _summarize(
                spec,
                "exact",
                exact_rows,
                args=args,
                index_build_seconds=0.0,
                timing_ms=exact_timing,
            )
        )
        print(
            json.dumps(
                {
                    "environment": spec.name,
                    "hnsw": summaries[-2],
                    "exact": summaries[-1],
                },
                indent=2,
            ),
            flush=True,
        )
        del hnsw, embeddings, slot_factors, profiles, utility_weights
        gc.collect()

    _write_csv(args.out_dir / "per_query_metrics.csv", all_rows)
    _write_csv(args.out_dir / "summary.csv", summaries)
    (args.out_dir / "results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "type": "four-environment-conditioned controlled HNSW scalability diagnostic",
                    "scope": (
                        "Synthetic factor-conditioned 256-D expert-bank embeddings; "
                        "not four-environment end-to-end OW-CMORL retraining."
                    ),
                    "slots_per_environment": int(args.slots),
                    "embedding_dim": int(args.embedding_dim),
                    "top_k": int(args.top_k),
                    "queries_per_environment": int(args.queries_per_env),
                    "timing_queries_per_environment": int(args.timing_queries_per_env),
                    "hnsw_m": int(args.hnsw_m),
                    "hnsw_ef_search": int(args.hnsw_ef_search),
                    "hnsw_ef_construction": int(args.hnsw_ef_construction),
                    "seed": int(args.seed),
                },
                "environments": [spec.__dict__ for spec in selected],
                "summaries": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
