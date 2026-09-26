#!/usr/bin/env python3
"""Small, reproducible ablations for OW-CMORL retrieval components.

The script deliberately uses the saved scalarized-policy snapshots and their
per-regime rollouts.  It is a replay/re-inference study, not a replacement for
the 20-regime end-to-end table in the paper.  This keeps the ablations small
enough to run on CPU while preserving real policy returns and dynamic traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import hnswlib
import numpy as np
import torch
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(PROJECT_ROOT))

from src.dynamic_morl.metrics import compute_trace_shift_metrics
from src.dynamic_morl.online_context import ContextSnapshot, OnlineContextMemory
from src.dynamic_morl.temporal import RegimeAttention, cosine_affinity


@dataclass(frozen=True)
class PolicySnapshot:
    policy_id: str
    algorithm: str
    returns: dict[str, np.ndarray]
    segments: dict[str, np.ndarray]
    quality: float


@dataclass(frozen=True)
class BankEntry:
    policy: PolicySnapshot
    anchor_regime: str
    embedding: np.ndarray
    score: float


class ExactRetriever:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = _normalize_rows(embeddings)

    def query(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        scores = self.embeddings @ _normalize_vector(query_embedding)
        return np.argsort(-scores, kind="stable")[: min(int(k), len(scores))]


class HNSWRetriever:
    def __init__(self, embeddings: np.ndarray, *, ef_search: int, m: int):
        vectors = _normalize_rows(embeddings).astype(np.float32)
        self.index = hnswlib.Index(space="cosine", dim=int(vectors.shape[1]))
        self.index.init_index(
            max_elements=len(vectors),
            ef_construction=max(16, int(ef_search)),
            M=max(4, int(m)),
            random_seed=0,
        )
        self.index.set_num_threads(1)
        self.index.add_items(vectors, np.arange(len(vectors), dtype=np.int64))
        self.index.set_ef(max(1, int(ef_search)))

    def query(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        k = min(int(k), self.index.get_current_count())
        labels, _distances = self.index.knn_query(
            _normalize_vector(query_embedding).astype(np.float32).reshape(1, -1), k=k
        )
        return labels[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-keys", nargs="+", default=["building", "evcharging", "cogen"])
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=PROJECT_ROOT / "results_fine_regime" / "scalarized_baselines",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_ablation_small",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--context-fit-steps", type=int, default=32)
    parser.add_argument("--retrieval-k", type=int, default=8)
    parser.add_argument("--hnsw-ef-search", type=int, default=32)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument(
        "--single-algorithm",
        type=str,
        default="ppo1",
        help="one fixed training/inference algorithm used by the expert-bank ablation",
    )
    parser.add_argument("--latency-replicas", type=int, default=16)
    parser.add_argument("--latency-query-repeats", type=int, default=1500)
    return parser.parse_args()


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix / norms


def _load_segments(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    grouped: dict[str, list[np.ndarray]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            obj_cols = sorted(key for key in row if key.startswith("obj_"))
            if not obj_cols:
                continue
            grouped.setdefault(str(row["regime"]), []).append(
                np.asarray([float(row[key]) for key in obj_cols], dtype=np.float64)
            )
    return {regime: np.stack(values, axis=0) for regime, values in grouped.items() if values}


def _rank_quality(records: list[tuple[str, str, dict[str, np.ndarray], dict[str, np.ndarray], dict]]) -> list[PolicySnapshot]:
    all_vectors = np.concatenate(
        [np.stack(list(returns.values()), axis=0) for _pid, _alg, returns, _segments, _raw in records], axis=0
    )
    lower = all_vectors.min(axis=0)
    span = np.maximum(all_vectors.max(axis=0) - lower, 1e-12)
    snapshots = []
    for policy_id, algorithm, returns, segments, raw in records:
        normalized = [(value - lower) / span for value in returns.values()]
        utility = float(np.mean(np.stack(normalized, axis=0)))
        trace = raw.get("trace_metrics", {})
        utility += 0.08 * float(trace.get("trace_recovery_score", 0.0))
        utility -= 0.02 * float(trace.get("trace_shift_regret", 0.0)) / (1.0 + abs(float(trace.get("trace_shift_regret", 0.0))))
        snapshots.append(PolicySnapshot(policy_id, algorithm, returns, segments, utility))
    return sorted(snapshots, key=lambda item: (-item.quality, item.policy_id))


def load_snapshots(snapshot_root: Path, env_key: str) -> list[PolicySnapshot]:
    records = []
    for result_path in sorted((snapshot_root / env_key).glob("*/*/result.json")):
        raw = json.loads(result_path.read_text())
        returns = {
            str(row["regime"]): np.asarray(row["objs"], dtype=np.float64)
            for row in raw.get("regime_returns", [])
        }
        if not returns:
            continue
        policy_id = f"{raw['alg']}/{result_path.parent.name}"
        records.append((policy_id, str(raw["alg"]), returns, _load_segments(result_path.parent / "dynamic_trace.csv"), raw))
    if not records:
        raise FileNotFoundError(f"no result.json snapshots found for {env_key}: {snapshot_root / env_key}")
    return _rank_quality(records)


def context_vectors(
    env_key: str,
    regimes: list[str],
    *,
    snapshot_root: Path,
) -> dict[str, np.ndarray]:
    # Fine-regime runs persist the exact dynamic-factor vector with every
    # regime return. Prefer those values whenever they are available so all
    # four environments use the operating conditions that produced the saved
    # policy evaluations.
    persisted_catalog: dict[str, np.ndarray] = {}
    for result_path in sorted((snapshot_root / env_key).glob("*/*/result.json")):
        payload = json.loads(result_path.read_text())
        persisted = {
            str(row["regime"]): np.asarray(row["context_vector"], dtype=np.float32)
            for row in payload.get("regime_returns", [])
            if row.get("context_vector") is not None
        }
        for regime, vector in persisted.items():
            persisted_catalog.setdefault(regime, vector)
    if all(regime in persisted_catalog for regime in regimes):
        return {regime: persisted_catalog[regime] for regime in regimes}

    if env_key == "building":
        factors = {
            "coastal_mild": [0.85, 0.95, 0.90, 20.0 / 24.0],
            "desert_hot": [1.15, 1.05, 1.10, 21.0 / 24.0],
            "mixed_marine": [1.00, 1.00, 1.00, 20.0 / 24.0],
            "very_cold": [1.25, 1.12, 1.18, 22.0 / 24.0],
        }
        return {regime: np.asarray(factors[regime], dtype=np.float32) for regime in regimes}
    if env_key == "cogen":
        values = np.asarray([float(regime.rsplit("_", maxsplit=1)[-1]) for regime in regimes], dtype=np.float32)
        scale = max(float(np.max(np.abs(values))), 1.0)
        return {
            regime: np.asarray([value / scale], dtype=np.float32)
            for regime, value in zip(regimes, values)
        }
    if env_key == "evcharging":
        # The period is the exogenous dynamic factor available to the wrapper.
        return {
            regime: np.eye(len(regimes), dtype=np.float32)[idx]
            for idx, regime in enumerate(regimes)
        }
    raise ValueError(f"unsupported ablation environment: {env_key}")


def cyclic_window(regimes: list[str], position: int, width: int) -> list[str]:
    return [regimes[(position + offset) % len(regimes)] for offset in range(-width + 1, 1)]


@dataclass(frozen=True)
class ContextRepresentations:
    full: dict[str, np.ndarray]
    no_forecast: dict[str, np.ndarray]
    no_history: dict[str, np.ndarray]
    current_token: dict[str, np.ndarray]
    encoder_ms_per_query: float
    current_token_ms_per_query: float


def _build_target_embeddings(
    regimes: list[str],
    current: dict[str, np.ndarray],
    forecast: dict[str, np.ndarray],
    *,
    forecast_lambda: float,
    history_lambda: float,
) -> dict[str, np.ndarray]:
    """Match OnlineContextMemory's target-key construction for one trace pass."""
    memory = OnlineContextMemory(
        max_size=max(1, len(regimes)),
        history_lambda=history_lambda,
        forecast_lambda=forecast_lambda,
        nearest_k=min(3, max(1, len(regimes))),
    )
    targets = {}
    for iteration, regime in enumerate(regimes):
        current_embedding = np.asarray(current[regime], dtype=np.float64)
        forecast_embedding = np.asarray(forecast[regime], dtype=np.float64)
        target = memory.build_target_embedding(current_embedding, forecast_embedding)
        targets[regime] = _normalize_vector(target)
        memory.update(
            ContextSnapshot(
                iteration=iteration,
                current_embedding=current_embedding,
                forecast_embedding=forecast_embedding,
                target_embedding=target,
                gap_vector=np.zeros(1, dtype=np.float64),
                utility_mean=0.0,
                drift_score=0.0,
                prediction_error=0.0,
            )
        )
    return targets


def build_context_representations(
    env_key: str,
    regimes: list[str],
    *,
    snapshot_root: Path,
    window: int,
    fit_steps: int,
    seed: int,
) -> ContextRepresentations:
    torch.manual_seed(seed)
    vectors = context_vectors(env_key, regimes, snapshot_root=snapshot_root)
    traces = [
        {"context": np.stack([vectors[item] for item in cyclic_window(regimes, idx, window)], axis=0)}
        for idx in range(len(regimes))
    ]
    model = RegimeAttention(input_dim=len(next(iter(vectors.values()))), hidden_dim=32, nhead=4, window=window)
    model.fit(traces, steps=fit_steps, lr=1e-3, device=torch.device("cpu"))
    current_embeddings = {}
    forecast_embeddings = {}
    traces_by_regime = {}
    for idx, regime in enumerate(regimes):
        trace = np.stack([vectors[item] for item in cyclic_window(regimes, idx, window)], axis=0)
        traces_by_regime[regime] = trace
        current, forecast, _prediction = model.encode_with_forecast(trace, device=torch.device("cpu"))
        current_embeddings[regime] = np.asarray(current, dtype=np.float64)
        forecast_embeddings[regime] = np.asarray(forecast, dtype=np.float64)

    # The default values are the actual OW-CMORL defaults.  All H2 variants
    # retain the same fitted encoder and bank; only one target-key term is
    # disabled at a time.
    # Measure the query-key path separately from retrieval. This makes it
    # explicit that H1 removes compute but is evaluated for decision quality
    # and recovery rather than being falsely presented as a slower encoder.
    timing_repeats = 64
    start = time.perf_counter()
    for _ in range(timing_repeats):
        for regime in regimes:
            model.encode_with_forecast(traces_by_regime[regime], device=torch.device("cpu"))
    encoder_ms_per_query = 1000.0 * (time.perf_counter() - start) / (timing_repeats * len(regimes))
    start = time.perf_counter()
    for _ in range(timing_repeats):
        for regime in regimes:
            _normalize_vector(vectors[regime])
    current_token_ms_per_query = 1000.0 * (time.perf_counter() - start) / (timing_repeats * len(regimes))

    return ContextRepresentations(
        full=_build_target_embeddings(
            regimes,
            current_embeddings,
            forecast_embeddings,
            forecast_lambda=0.50,
            history_lambda=0.10,
        ),
        no_forecast=_build_target_embeddings(
            regimes,
            current_embeddings,
            forecast_embeddings,
            forecast_lambda=0.0,
            history_lambda=0.10,
        ),
        no_history=_build_target_embeddings(
            regimes,
            current_embeddings,
            forecast_embeddings,
            forecast_lambda=0.50,
            history_lambda=0.0,
        ),
        # H1 disables the Transformer and uses only the current observed
        # environment token as both the bank anchor and the query.
        current_token={regime: _normalize_vector(vector) for regime, vector in vectors.items()},
        encoder_ms_per_query=float(encoder_ms_per_query),
        current_token_ms_per_query=float(current_token_ms_per_query),
    )


def build_entries(snapshots: Iterable[PolicySnapshot], embeddings: dict[str, np.ndarray]) -> list[BankEntry]:
    snapshots = list(snapshots)
    all_returns = np.concatenate(
        [np.stack(list(snapshot.returns.values()), axis=0) for snapshot in snapshots], axis=0
    )
    lower = all_returns.min(axis=0)
    span = np.maximum(all_returns.max(axis=0) - lower, 1e-12)
    entries = [
        BankEntry(
            policy=snapshot,
            anchor_regime=regime,
            embedding=embeddings[regime],
            score=float(np.mean((snapshot.returns[regime] - lower) / span)) + 0.05 * snapshot.quality,
        )
        for snapshot in snapshots
        for regime in sorted(snapshot.returns)
        if regime in embeddings
    ]
    return sorted(entries, key=lambda item: (-item.score, item.policy.policy_id, item.anchor_regime))


def pareto_front(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    idx = NonDominatedSorting().do(-np.asarray(points, dtype=np.float64), only_non_dominated_front=True)
    return np.asarray(points[idx], dtype=np.float64)


def preferences(num_objectives: int) -> np.ndarray:
    rows = []
    for weights in product(range(5), repeat=num_objectives):
        total = sum(weights)
        if total == 4:
            rows.append(np.asarray(weights, dtype=np.float64) / 4.0)
    return np.stack(rows, axis=0)


def utility(point: np.ndarray, lower: np.ndarray, span: np.ndarray) -> float:
    return float(np.mean((np.asarray(point, dtype=np.float64) - lower) / span))


def route_variant(
    entries: list[BankEntry],
    embeddings: dict[str, np.ndarray],
    regimes: list[str],
    *,
    mode: str,
    retrieval_k: int,
    hnsw_ef_search: int,
    hnsw_m: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]]]:
    all_returns = np.concatenate(
        [np.stack([entry.policy.returns[regime] for entry in entries if regime in entry.policy.returns]) for regime in regimes], axis=0
    )
    lower = all_returns.min(axis=0)
    span = np.maximum(all_returns.max(axis=0) - lower, 1e-12)
    fronts: dict[str, np.ndarray] = {}
    selected_segments: dict[str, np.ndarray] = {}
    selected_ids: dict[str, list[str]] = {}
    for regime in regimes:
        # Fine-regime snapshots are saved only under conditions where they
        # were actually evaluated. Filter before routing so a selected policy
        # is never credited with a fabricated return for another condition.
        available_entries = [entry for entry in entries if regime in entry.policy.returns]
        if not available_entries:
            raise ValueError(f"no saved policy return for {regime!r}")
        vectors = np.stack([entry.embedding for entry in available_entries], axis=0)
        retriever = (
            ExactRetriever(vectors)
            if mode == "exact"
            else HNSWRetriever(vectors, ef_search=hnsw_ef_search, m=hnsw_m)
        )
        candidate_count = min(len(available_entries), retrieval_k)
        candidates = [
            available_entries[int(index)]
            for index in retriever.query(embeddings[regime], candidate_count)
        ]
        # Both backends use the same final candidate budget and the same
        # context-slot score. HNSW differs only in approximate neighbor
        # recall; exact search scores the complete bank before this step.
        candidates = sorted(candidates, key=lambda item: (-item.score, item.policy.policy_id))[:retrieval_k]
        unique: dict[str, BankEntry] = {}
        for candidate in candidates:
            unique.setdefault(candidate.policy.policy_id, candidate)
        candidates = list(unique.values())
        returns = np.stack([candidate.policy.returns[regime] for candidate in candidates], axis=0)
        fronts[regime] = pareto_front(returns)
        best = max(
            candidates,
            key=lambda item: (
                utility(item.policy.returns[regime], lower, span)
                + 0.20 * item.score
                + 0.30 * cosine_affinity(item.embedding, embeddings[regime])
            ),
        )
        selected_segments[regime] = best.policy.segments.get(regime, best.policy.returns[regime][None, :])
        selected_ids[regime] = [candidate.policy.policy_id for candidate in candidates]
    return fronts, selected_segments, selected_ids


def score_front(front: np.ndarray, ref_point: np.ndarray) -> tuple[float, float]:
    hv = float(Hypervolume(ref_point=-ref_point).do(-front))
    eu = float(np.mean(np.max(front @ preferences(front.shape[1]).T, axis=0)))
    return hv, eu


def score_variants(variant_fronts: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    all_points = np.concatenate(
        [front for fronts in variant_fronts.values() for front in fronts.values()], axis=0
    )
    reference = all_points.min(axis=0) - 0.1 * np.maximum(np.abs(all_points.min(axis=0)), 1.0)
    regime_metrics: dict[str, dict[str, tuple[float, float]]] = {}
    for variant, fronts in variant_fronts.items():
        regime_metrics[variant] = {regime: score_front(front, reference) for regime, front in fronts.items()}
    best = {
        regime: (
            max(regime_metrics[variant][regime][0] for variant in regime_metrics),
            max(regime_metrics[variant][regime][1] for variant in regime_metrics),
        )
        for regime in next(iter(variant_fronts.values()))
    }
    output = {}
    for variant, rows in regime_metrics.items():
        hvs = np.asarray([value[0] for value in rows.values()], dtype=np.float64)
        eus = np.asarray([value[1] for value in rows.values()], dtype=np.float64)
        adapt = []
        for regime, (hv, eu) in rows.items():
            best_hv, best_eu = best[regime]
            gap_hv = max(0.0, best_hv - hv) / max(abs(best_hv), 1e-12)
            gap_eu = max(0.0, best_eu - eu) / max(abs(best_eu), 1e-12)
            adapt.append(1.0 - 0.5 * (gap_hv + gap_eu))
        output[variant] = {
            "mean_regime_hv": float(hvs.mean()),
            "mean_regime_eu": float(eus.mean()),
            "adapt_score": float(np.mean(adapt)),
            "per_regime": {
                regime: {"hv": values[0], "eu": values[1]} for regime, values in rows.items()
            },
        }
    return output


def composite_trace(regimes: list[str], segments: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Repeat the observed regime order so every transition contributes a real stored rollout segment.
    schedule = regimes * 4
    rows = []
    ids = []
    regime_to_id = {regime: index for index, regime in enumerate(regimes)}
    for regime in schedule:
        segment = np.asarray(segments[regime], dtype=np.float64)
        rows.append(segment)
        ids.append(np.full(len(segment), regime_to_id[regime], dtype=np.int64))
    return {"obj": np.concatenate(rows, axis=0), "regime_id": np.concatenate(ids, axis=0)}


def benchmark_latency(
    entries: list[BankEntry],
    query_embeddings: dict[str, np.ndarray],
    *,
    replicas: int,
    repeats: int,
    retrieval_k: int,
    ef_search: int,
    m: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    base = np.stack([entry.embedding for entry in entries], axis=0)
    expanded = np.repeat(base, max(1, int(replicas)), axis=0)
    expanded += rng.normal(0.0, 1e-4, size=expanded.shape)
    expanded = _normalize_rows(expanded)
    queries = list(query_embeddings.values()) * max(1, int(repeats))
    results = {}
    for name, factory in {
        "hnsw": lambda: HNSWRetriever(expanded, ef_search=ef_search, m=m),
        "exact": lambda: ExactRetriever(expanded),
    }.items():
        start = time.perf_counter()
        retriever = factory()
        build_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for query_embedding in queries:
            retriever.query(query_embedding, retrieval_k)
        inference_seconds = time.perf_counter() - start
        results[name] = {
            "bank_entries": int(len(expanded)),
            "queries": int(len(queries)),
            "index_refresh_ms": 1000.0 * build_seconds,
            "inference_ms": 1000.0 * inference_seconds,
            "refresh_and_inference_ms": 1000.0 * (build_seconds + inference_seconds),
        }
    return results


def table1_owcmorl_baseline() -> dict[str, dict[str, float]]:
    # Copied from innovation_paper/iclr2026_conference.tex, Table 1.
    return {
        "building": {"hv": 1.4032e10, "eu": 8880.1323, "adapt": 0.7515, "regret": 3.1113, "latency": 5.7895},
        "evcharging": {"hv": 0.0720, "eu": 1.1956, "adapt": 0.9544, "regret": 0.4614, "latency": 0.8279},
        "cogen": {"hv": 5.1418e23, "eu": -2.4208e6, "adapt": 0.5710, "regret": 7999.5839, "latency": 5.5714},
        "chlor_alkali": {"hv": 1.0261e12, "eu": -8.1809e4, "adapt": 0.9823, "regret": 96.1114, "latency": 6.5718},
    }


def write_report(path: Path, results: dict) -> None:
    labels = {
        "full_hnsw": "Full (encoder + forecast + history + HNSW)",
        "h1_current_token_no_encoder": "H1: current token, no encoder",
        "h2_no_forecast": "H2a: forecast lambda = 0",
        "h2_no_history": "H2b: history lambda = 0",
        "full_exact": "HNSW ablation: exact retrieval",
    }

    def pct(value: float, baseline: float) -> str:
        if abs(float(baseline)) < 1e-12:
            return "n/a"
        return f"{100.0 * (float(value) - float(baseline)) / abs(float(baseline)):+.1f}%"

    def routing_ms(env_result: dict, variant: str) -> float:
        latency = env_result["latency"]
        reps = env_result["representation_latency_ms"]
        if variant == "h1_current_token_no_encoder":
            row = latency[variant]
            return float(reps["current_token"]) + 1000.0 * float(row["inference_ms"]) / max(int(row["queries"]), 1)
        backend = "full_exact" if variant == "full_exact" else "full_hnsw"
        row = latency[backend]
        return float(reps["full_encoder"]) + 1000.0 * float(row["inference_ms"]) / max(int(row["queries"]), 1)

    lines = [
        "# OW-CMORL H1/H2/HNSW lightweight ablation",
        "",
        "This CPU replay and re-inference diagnostic uses saved policy snapshots and per-regime rollouts. Candidate policies, regimes, reference points, Top-K, and HNSW settings stay fixed; only the specified context representation or retrieval backend changes. It does not replace end-to-end ablations retrained with independent seeds.",
        "",
        "## Controlled variants",
        "",
        "- **Full**: Transformer window, `context_forecast_lambda=0.50`, `context_history_lambda=0.10`, and HNSW.",
        "- **H1**: Disable the Transformer and use the current environment context token as both bank anchor and query.",
        "- **H2a/H2b**: Keep the encoder, bank, and HNSW fixed; set `context_forecast_lambda=0` or `context_history_lambda=0`, respectively.",
        "- **HNSW**: Keep the full context key and all candidates; replace HNSW with exact top-K on the same bank.",
        "",
        "## Solution-set and dynamic metrics",
        "",
        "`delta` is relative to Full HNSW in the same environment. Higher HV, EU, and adaptation score are better; lower shift regret and recovery latency (episodes) are better.",
        "",
        "| Environment | Variant | HV | delta | EU | delta | adapt. | regret | delta | recovery episodes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for env_key, env_result in results["environments"].items():
        for key in labels:
            row = env_result["variants"][key]
            trace = row["trace"]
            base = env_result["variants"]["full_hnsw"]
            base_trace = base["trace"]
            lines.append(
                f"| {env_key} | {labels[key]} | {row['mean_regime_hv']:.4g} | {pct(row['mean_regime_hv'], base['mean_regime_hv'])} | {row['mean_regime_eu']:.4g} | {pct(row['mean_regime_eu'], base['mean_regime_eu'])} | {row['adapt_score']:.3f} | {trace['trace_shift_regret']:.4g} | {pct(trace['trace_shift_regret'], base_trace['trace_shift_regret'])} | {trace['trace_recovery_latency']:.3g} |"
            )
    lines.extend([
        "",
        "## Context-routing latency",
        "",
        "Latency includes context-key construction and top-K retrieval per query. H2a/H2b and Full use the same encoder and HNSW path. Faster H1 token normalization does not imply faster recovery. Index refresh is reported separately because the index can be reused at deployment.",
        "",
        "| Environment | Full HNSW (ms/query) | H1 token (ms/query) | Exact (ms/query) | exact / HNSW | HNSW refresh (ms) | exact refresh (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for env_key, env_result in results["environments"].items():
        full = env_result["latency"]["full_hnsw"]
        exact = env_result["latency"]["full_exact"]
        full_ms = routing_ms(env_result, "full_hnsw")
        exact_ms = routing_ms(env_result, "full_exact")
        token_ms = routing_ms(env_result, "h1_current_token_no_encoder")
        lines.append(
            f"| {env_key} | {full_ms:.4f} | {token_ms:.4f} | {exact_ms:.4f} | {exact_ms / max(full_ms, 1e-12):.2f}x | {full['index_refresh_ms']:.2f} | {exact['index_refresh_ms']:.2f} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- This lightweight retrieval diagnostic holds snapshots fixed. It isolates the context key and retrieval backend but cannot establish the causal effect of representation learning after retraining.",
        "- H1 should cost less than a Transformer key. Judge its quality using HV/EU, shift regret, and recovery episodes rather than query time alone.",
        "- Exact and HNSW retrieval use the same bank, queries, and Top-K. Their routing latency excludes shared environment rollout and policy action costs.",
        "",
        "Raw data are stored in `ablation_metrics.json` and `latency_benchmark.json` in the output directory. Rerun with `scripts/run_owcmorl_ablation_study.py`.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {"config": vars(args), "table1_owcmorl": table1_owcmorl_baseline(), "environments": {}}
    results["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in results["config"].items()}
    for env_key in args.env_keys:
        snapshots = load_snapshots(args.snapshot_root, env_key)
        regimes = sorted({regime for snapshot in snapshots for regime in snapshot.returns})
        representations = build_context_representations(
            env_key,
            regimes,
            snapshot_root=args.snapshot_root,
            window=args.context_window,
            fit_steps=args.context_fit_steps,
            seed=args.seed,
        )
        full_entries = build_entries(snapshots, representations.full)
        current_token_entries = build_entries(snapshots, representations.current_token)
        single_algorithm = str(args.single_algorithm)
        if not [snapshot for snapshot in snapshots if snapshot.algorithm == single_algorithm]:
            raise ValueError(f"no {single_algorithm!r} snapshots available for {env_key}")
        front_rows = {}
        trace_segments = {}
        selections = {}
        variants = {
            "full_hnsw": (full_entries, representations.full, "hnsw"),
            "h1_current_token_no_encoder": (
                current_token_entries,
                representations.current_token,
                "hnsw",
            ),
            "h2_no_forecast": (full_entries, representations.no_forecast, "hnsw"),
            "h2_no_history": (full_entries, representations.no_history, "hnsw"),
            "full_exact": (full_entries, representations.full, "exact"),
        }
        for name, (entries, query_embeddings, mode) in variants.items():
            fronts, segments, ids = route_variant(
                entries,
                query_embeddings,
                regimes,
                mode=mode,
                retrieval_k=args.retrieval_k,
                hnsw_ef_search=args.hnsw_ef_search,
                hnsw_m=args.hnsw_m,
            )
            front_rows[name] = fronts
            trace_segments[name] = segments
            selections[name] = ids
        summaries = score_variants(front_rows)
        for name, summary in summaries.items():
            summary["trace"] = compute_trace_shift_metrics(
                composite_trace(regimes, trace_segments[name]), eval_delta_weight=0.5
            )
        full_latency = benchmark_latency(
            full_entries,
            representations.full,
            replicas=args.latency_replicas,
            repeats=args.latency_query_repeats,
            retrieval_k=args.retrieval_k,
            ef_search=args.hnsw_ef_search,
            m=args.hnsw_m,
            seed=args.seed,
        )
        current_token_latency = benchmark_latency(
            current_token_entries,
            representations.current_token,
            replicas=args.latency_replicas,
            repeats=args.latency_query_repeats,
            retrieval_k=args.retrieval_k,
            ef_search=args.hnsw_ef_search,
            m=args.hnsw_m,
            seed=args.seed,
        )["hnsw"]
        results["environments"][env_key] = {
            "snapshot_count": len(snapshots),
            "bank_entry_count": len(full_entries),
            "single_algorithm": single_algorithm,
            "current_token_bank_entry_count": len(current_token_entries),
            "variants": summaries,
            "latency": {
                "full_hnsw": full_latency["hnsw"],
                "full_exact": full_latency["exact"],
                "h1_current_token_no_encoder": current_token_latency,
            },
            "representation_latency_ms": {
                "full_encoder": representations.encoder_ms_per_query,
                "current_token": representations.current_token_ms_per_query,
            },
            "selected_policy_ids": selections,
        }
        print(json.dumps({env_key: results["environments"][env_key]}, indent=2))
    (args.out_dir / "ablation_metrics.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "latency_benchmark.json").write_text(
        json.dumps({key: value["latency"] for key, value in results["environments"].items()}, indent=2)
    )
    write_report(PROJECT_ROOT / "OW_CMORL_H1_H2_HNSW_ABLATION_ZH.md", results)


if __name__ == "__main__":
    main()
