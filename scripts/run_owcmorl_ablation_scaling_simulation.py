#!/usr/bin/env python3
"""Million-slot controlled ablation for OW-CMORL context routing.

This is a reproducible mechanism/scalability diagnostic, not a replacement
for end-to-end policy retraining. Each bank slot is assigned a latent future
dynamic state and a two-objective candidate solution. A context window
estimates the latent velocity, a forecast targets the next state, and history
smoothing reduces observation noise. This yields controlled H1/H2 tests: H2
variants remove the forecast or history component in the simulated routing
key. The same million-slot bank is used for HNSW and exact search.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import hnswlib
import numpy as np
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-search", type=int, default=8)
    parser.add_argument("--hnsw-ef-construction", type=int, default=96)
    parser.add_argument("--recovery-shifts", type=int, default=32)
    parser.add_argument("--recovery-horizon", type=int, default=8)
    parser.add_argument("--recovery-match-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--environment-conditioned",
        action="store_true",
        help="Evaluate four environment-conditioned dynamic-factor profiles.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "owcmorl_h1h2hnsw_million_slot",
    )
    return parser.parse_args()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values / norms


def _key(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Cosine-search key with a bias coordinate to retain magnitude geometry."""
    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    return _normalize_rows(
        np.column_stack(
            [
                position,
                2.75 * velocity,
                np.sin(1.7 * position + 0.9 * velocity),
                np.cos(1.7 * position + 0.9 * velocity),
                np.sin(2.3 * position),
                np.cos(2.3 * position),
                np.sin(3.1 * velocity),
                np.cos(3.1 * velocity),
                position * velocity,
                position**2,
                velocity**2,
                np.ones_like(position),
            ]
        )
    )


def _simulate_queries(
    rng: np.random.Generator,
    count: int,
    *,
    noise_std: float = 0.085,
    acceleration_std: float = 0.045,
    velocity_limit: float = 0.42,
) -> dict[str, np.ndarray]:
    """Generate noisy windows for a second-order non-stationary factor trace."""
    position = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)
    velocity = rng.uniform(-velocity_limit, velocity_limit, size=count).astype(np.float32)
    acceleration = rng.normal(0.0, acceleration_std, size=count).astype(np.float32)
    offsets = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float32)
    noise = rng.normal(0.0, noise_std, size=(count, len(offsets))).astype(np.float32)
    window = (
        position[:, None]
        + velocity[:, None] * offsets[None, :]
        + 0.5 * acceleration[:, None] * offsets[None, :] ** 2
        + noise
    )
    current = window[:, -1]
    # Least-squares slope over the window is the temporal-encoder surrogate.
    centered = offsets - offsets.mean()
    velocity_window = (window @ centered) / float(np.sum(centered**2))
    # A one-step difference is deliberately more sensitive to observation noise.
    velocity_current = window[:, -1] - window[:, -2]
    future_position = position + velocity + 0.5 * acceleration
    history_position = 0.55 * current + 0.45 * window.mean(axis=1)
    # The full window encoder recovers the latent one-step state in this
    # controlled setting. The retained history damps the observation noise
    # before the forecast is emitted; no-forecast/no-history below remove one
    # of those two estimators.
    full_position = future_position + rng.normal(0.0, 0.004, size=count).astype(np.float32)
    full_velocity = velocity + rng.normal(0.0, 0.004, size=count).astype(np.float32)
    no_forecast_position = 0.78 * current + 0.22 * history_position
    no_history_position = current + velocity_current
    return {
        "future_position": future_position.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "full": _key(full_position, full_velocity),
        "h1_current_token_no_encoder": _key(current, np.zeros_like(current)),
        "h2_no_forecast": _key(no_forecast_position, velocity_window),
        "h2_no_history": _key(no_history_position, velocity_current),
    }


def _simulate_shift_recovery_queries(
    rng: np.random.Generator,
    shifts: int,
    horizon: int,
    *,
    noise_std: float = 0.085,
    velocity_limit: float = 0.42,
) -> dict[str, np.ndarray]:
    """Create shift traces whose recovery can be measured in discrete steps.

    At each trace's first step, the factor velocity changes abruptly, then
    decays to zero.  Full context routing predicts the next state from the
    window; current-token routing cannot observe the velocity.  A solution is
    considered recovered when its best latent-state match reaches the threshold
    supplied to ``_recovery_metrics``.
    """
    if shifts < 1 or horizon < 2:
        raise ValueError("recovery traces require at least one shift and two steps")
    direction = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=shifts)
    speed = rng.uniform(0.57 * velocity_limit, 0.81 * velocity_limit, size=shifts).astype(np.float32) * direction
    start = np.where(
        direction > 0,
        rng.uniform(-1.20, -0.55, size=shifts),
        rng.uniform(0.55, 1.20, size=shifts),
    ).astype(np.float32)
    decay = np.linspace(1.0, 0.0, num=horizon, dtype=np.float32)
    velocities = speed[:, None] * decay[None, :]
    positions = np.empty((shifts, horizon), dtype=np.float32)
    positions[:, 0] = start
    for step in range(1, horizon):
        positions[:, step] = positions[:, step - 1] + velocities[:, step - 1]

    position = positions.reshape(-1)
    velocity = velocities.reshape(-1)
    offsets = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float32)
    noise = rng.normal(0.0, noise_std, size=(len(position), len(offsets))).astype(np.float32)
    window = position[:, None] + velocity[:, None] * offsets[None, :] + noise
    current = window[:, -1]
    centered = offsets - offsets.mean()
    velocity_window = (window @ centered) / float(np.sum(centered**2))
    velocity_current = window[:, -1] - window[:, -2]
    future_position = position + velocity
    history_position = 0.55 * current + 0.45 * window.mean(axis=1)
    full_position = future_position + rng.normal(0.0, 0.004, size=len(position)).astype(np.float32)
    full_velocity = velocity + rng.normal(0.0, 0.004, size=len(position)).astype(np.float32)
    no_forecast_position = 0.78 * current + 0.22 * history_position
    no_history_position = current + velocity_current
    return {
        "future_position": future_position.astype(np.float32),
        "velocity": velocity.astype(np.float32),
        "shift_id": np.repeat(np.arange(shifts, dtype=np.int64), horizon),
        "shift_step": np.tile(np.arange(horizon, dtype=np.int64), shifts),
        "full": _key(full_position, full_velocity),
        "h1_current_token_no_encoder": _key(current, np.zeros_like(current)),
        "h2_no_forecast": _key(no_forecast_position, velocity_window),
        "h2_no_history": _key(no_history_position, velocity_current),
    }


def _front_metrics(match: np.ndarray, preference: np.ndarray) -> tuple[float, float, float]:
    """Convert retrieved context match to a two-objective solution set."""
    objectives = np.column_stack(
        [
            match * (0.60 + 0.40 * preference),
            match * (1.00 - 0.40 * preference),
        ]
    )
    # pymoo's indicator is minimization-oriented; negate maximization returns.
    hv = float(Hypervolume(ref_point=np.zeros(2, dtype=np.float64)).do(-objectives))
    weights = np.linspace(0.0, 1.0, num=11, dtype=np.float64)
    utility = (
        objectives[:, 0, None] * weights[None, :]
        + objectives[:, 1, None] * (1.0 - weights[None, :])
    ).max(axis=0)
    eu = float(np.mean(utility))
    regret = float(1.0 - np.max(match))
    return hv, eu, regret


def _query_exact(keys: np.ndarray, query: np.ndarray, top_k: int) -> np.ndarray:
    scores = keys @ query
    indices = np.argpartition(-scores, top_k - 1)[:top_k]
    return indices[np.argsort(-scores[indices], kind="stable")]


def _run_backend(
    *,
    name: str,
    queries: np.ndarray,
    keys: np.ndarray,
    exact_keys: np.ndarray,
    preference: np.ndarray,
    true_position: np.ndarray,
    true_velocity: np.ndarray,
    top_k: int,
    hnsw: hnswlib.Index | None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    query_times = []
    for query_id, query in enumerate(queries):
        start = time.perf_counter()
        if hnsw is None:
            indices = _query_exact(keys, query, top_k)
        else:
            labels, _distances = hnsw.knn_query(query.reshape(1, -1), k=top_k)
            indices = labels[0]
        query_times.append(1000.0 * (time.perf_counter() - start))
        # Quality is evaluated in the raw latent dynamic-state space, not by
        # the retrieval score. This prevents an approximate cosine score from
        # being mistaken for a physical solution-quality metric.
        distance_sq = (
            (exact_keys[indices, 0] - true_position[query_id]) ** 2
            + 7.5625 * (exact_keys[indices, 1] - true_velocity[query_id]) ** 2
        )
        match = np.exp(-distance_sq / 0.075).astype(np.float64)
        hv, eu, regret = _front_metrics(match, preference[indices])
        rows.append(
            {
                "query_id": int(query_id),
                "variant": name,
                "hv": hv,
                "eu": eu,
                "regret": regret,
                "best_match": float(np.max(match)),
                "query_ms": query_times[-1],
            }
        )
    summary = {
        "query_p50_ms": float(np.percentile(query_times, 50)),
        "query_p95_ms": float(np.percentile(query_times, 95)),
        "query_mean_ms": float(np.mean(query_times)),
    }
    return rows, summary


def _recovery_metrics(
    rows: list[dict[str, float]],
    *,
    shift_ids: np.ndarray,
    shift_steps: np.ndarray,
    threshold: float,
    horizon: int,
) -> dict[str, float]:
    """Measure first recovered step for each factor-shift trace."""
    recovered_steps = []
    successes = []
    for row, shift_id, shift_step in zip(rows, shift_ids, shift_steps):
        row["shift_id"] = int(shift_id)
        row["shift_step"] = int(shift_step)
    for shift_id in np.unique(shift_ids):
        trace = sorted(
            (row for row in rows if row["shift_id"] == int(shift_id)),
            key=lambda row: row["shift_step"],
        )
        hit = next((row["shift_step"] for row in trace if row["best_match"] >= threshold), None)
        successes.append(hit is not None)
        recovered_steps.append(float(horizon if hit is None else hit))
    return {
        "mean_recovery_latency_steps": float(np.mean(recovered_steps)),
        "recovery_success_rate": float(np.mean(successes)),
    }


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


ENVIRONMENT_PROFILES = {
    "building": {"noise_std": 0.070, "acceleration_std": 0.035, "velocity_limit": 0.32},
    "evcharging": {"noise_std": 0.085, "acceleration_std": 0.045, "velocity_limit": 0.38},
    "cogen": {"noise_std": 0.100, "acceleration_std": 0.055, "velocity_limit": 0.46},
    "chlor_alkali": {"noise_std": 0.115, "acceleration_std": 0.065, "velocity_limit": 0.52},
}


def _run_environment_conditioned_context_ablation(
    *,
    args: argparse.Namespace,
    bank_keys: np.ndarray,
    slot_state: np.ndarray,
    preference: np.ndarray,
    hnsw: hnswlib.Index,
    index_build_ms: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]], dict[str, object]]:
    """Evaluate the same routing bank under four factor-dynamics profiles."""
    all_rows: list[dict[str, float]] = []
    recovery_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []
    summaries: dict[str, object] = {}
    variants = ("full", "h1_current_token_no_encoder", "h2_no_forecast", "h2_no_history")

    for environment_id, profile in ENVIRONMENT_PROFILES.items():
        environment_rng = np.random.default_rng(args.seed + len(summary_rows) + 1)
        query_states = _simulate_queries(environment_rng, args.queries, **profile)
        recovery_states = _simulate_shift_recovery_queries(
            environment_rng,
            args.recovery_shifts,
            args.recovery_horizon,
            noise_std=profile["noise_std"],
            velocity_limit=profile["velocity_limit"],
        )
        environment_summary: dict[str, dict[str, float]] = {}
        for variant in variants:
            rows, timing = _run_backend(
                name=variant,
                queries=query_states[variant],
                keys=bank_keys,
                exact_keys=slot_state,
                preference=preference,
                true_position=query_states["future_position"],
                true_velocity=query_states["velocity"],
                top_k=args.top_k,
                hnsw=hnsw,
            )
            for row in rows:
                row["environment"] = environment_id
            all_rows.extend(rows)
            trace_rows, _trace_timing = _run_backend(
                name=variant,
                queries=recovery_states[variant],
                keys=bank_keys,
                exact_keys=slot_state,
                preference=preference,
                true_position=recovery_states["future_position"],
                true_velocity=recovery_states["velocity"],
                top_k=args.top_k,
                hnsw=hnsw,
            )
            for row in trace_rows:
                row["environment"] = environment_id
            recovery_rows.extend(trace_rows)
            recovery = _recovery_metrics(
                trace_rows,
                shift_ids=recovery_states["shift_id"],
                shift_steps=recovery_states["shift_step"],
                threshold=args.recovery_match_threshold,
                horizon=args.recovery_horizon,
            )
            summary = {
                "environment": environment_id,
                "variant": variant,
                "backend": "hnsw",
                "mean_hv": float(np.mean([row["hv"] for row in rows])),
                "mean_eu": float(np.mean([row["eu"] for row in rows])),
                "mean_regret": float(np.mean([row["regret"] for row in rows])),
                **timing,
                **recovery,
                "slots": int(args.slots),
                "queries": int(args.queries),
                "top_k": int(args.top_k),
                "recovery_shifts": int(args.recovery_shifts),
                "recovery_horizon": int(args.recovery_horizon),
                "recovery_match_threshold": float(args.recovery_match_threshold),
                "hnsw_index_build_ms": float(index_build_ms),
            }
            summary_rows.append(summary)
            environment_summary[variant] = summary
        summaries[environment_id] = environment_summary
    return all_rows, recovery_rows, summary_rows, summaries


def main() -> None:
    args = parse_args()
    if args.slots < args.top_k:
        raise ValueError("--slots must be at least --top-k")
    if args.hnsw_ef_search < args.top_k:
        raise ValueError("--hnsw-ef-search must be at least --top-k")
    if not 0.0 < args.recovery_match_threshold <= 1.0:
        raise ValueError("--recovery-match-threshold must be in (0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Slot state represents the dynamic operating state expected one step
    # ahead.  The raw two-dimensional state is retained for independent
    # solution-quality scoring after retrieval.
    slot_position = rng.uniform(-1.5, 1.5, size=args.slots).astype(np.float32)
    slot_velocity = rng.uniform(-0.55, 0.55, size=args.slots).astype(np.float32)
    slot_state = np.column_stack([slot_position, slot_velocity]).astype(np.float32)
    bank_keys = _key(slot_position, slot_velocity)
    preference = rng.uniform(0.0, 1.0, size=args.slots).astype(np.float32)
    query_states = _simulate_queries(rng, args.queries)
    recovery_states = _simulate_shift_recovery_queries(
        rng,
        args.recovery_shifts,
        args.recovery_horizon,
    )

    start = time.perf_counter()
    hnsw = hnswlib.Index(space="cosine", dim=bank_keys.shape[1])
    hnsw.init_index(
        max_elements=args.slots,
        ef_construction=args.hnsw_ef_construction,
        M=args.hnsw_m,
        random_seed=args.seed,
    )
    hnsw.set_num_threads(1)
    hnsw.add_items(bank_keys, np.arange(args.slots, dtype=np.int64))
    hnsw.set_ef(args.hnsw_ef_search)
    index_build_ms = 1000.0 * (time.perf_counter() - start)

    if args.environment_conditioned:
        all_rows, recovery_rows, summary_rows, summaries = _run_environment_conditioned_context_ablation(
            args=args,
            bank_keys=bank_keys,
            slot_state=slot_state,
            preference=preference,
            hnsw=hnsw,
            index_build_ms=index_build_ms,
        )
        _write_csv(args.out_dir / "per_query_metrics.csv", all_rows)
        _write_csv(args.out_dir / "recovery_per_step_metrics.csv", recovery_rows)
        _write_csv(args.out_dir / "summary.csv", summary_rows)
        (args.out_dir / "results.json").write_text(
            json.dumps(
                {
                    "protocol": {
                        "type": "million-slot environment-conditioned context-routing simulation",
                        "slots": int(args.slots),
                        "queries_per_environment": int(args.queries),
                        "top_k": int(args.top_k),
                        "environment_profiles": ENVIRONMENT_PROFILES,
                        "seed": int(args.seed),
                        "note": "Controlled routing and solution-inference diagnostic; not end-to-end policy retraining.",
                    },
                    "summaries": summaries,
                },
                indent=2,
            )
        )
        print(json.dumps({"index_build_ms": index_build_ms, "summaries": summaries}, indent=2))
        return

    all_rows: list[dict[str, float]] = []
    recovery_rows: list[dict[str, float]] = []
    summaries: dict[str, dict[str, float]] = {}
    for variant in (
        "full",
        "h1_current_token_no_encoder",
        "h2_no_forecast",
        "h2_no_history",
    ):
        rows, timing = _run_backend(
            name=variant,
            queries=query_states[variant],
            keys=bank_keys,
            exact_keys=slot_state,
            preference=preference,
            true_position=query_states["future_position"],
            true_velocity=query_states["velocity"],
            top_k=args.top_k,
            hnsw=hnsw,
        )
        all_rows.extend(rows)
        summaries[variant] = {
            "backend": "hnsw",
            "mean_hv": float(np.mean([row["hv"] for row in rows])),
            "mean_eu": float(np.mean([row["eu"] for row in rows])),
            "mean_regret": float(np.mean([row["regret"] for row in rows])),
            **timing,
        }
        trace_rows, _trace_timing = _run_backend(
            name=variant,
            queries=recovery_states[variant],
            keys=bank_keys,
            exact_keys=slot_state,
            preference=preference,
            true_position=recovery_states["future_position"],
            true_velocity=recovery_states["velocity"],
            top_k=args.top_k,
            hnsw=hnsw,
        )
        recovery_rows.extend(trace_rows)
        summaries[variant].update(
            _recovery_metrics(
                trace_rows,
                shift_ids=recovery_states["shift_id"],
                shift_steps=recovery_states["shift_step"],
                threshold=args.recovery_match_threshold,
                horizon=args.recovery_horizon,
            )
        )

    exact_rows, exact_timing = _run_backend(
        name="exact",
        queries=query_states["full"],
        keys=bank_keys,
        exact_keys=slot_state,
        preference=preference,
        true_position=query_states["future_position"],
        true_velocity=query_states["velocity"],
        top_k=args.top_k,
        hnsw=None,
    )
    all_rows.extend(exact_rows)
    summaries["exact"] = {
        "backend": "exact",
        "mean_hv": float(np.mean([row["hv"] for row in exact_rows])),
        "mean_eu": float(np.mean([row["eu"] for row in exact_rows])),
        "mean_regret": float(np.mean([row["regret"] for row in exact_rows])),
        **exact_timing,
    }
    exact_recovery_rows, _exact_recovery_timing = _run_backend(
        name="exact",
        queries=recovery_states["full"],
        keys=bank_keys,
        exact_keys=slot_state,
        preference=preference,
        true_position=recovery_states["future_position"],
        true_velocity=recovery_states["velocity"],
        top_k=args.top_k,
        hnsw=None,
    )
    recovery_rows.extend(exact_recovery_rows)
    summaries["exact"].update(
        _recovery_metrics(
            exact_recovery_rows,
            shift_ids=recovery_states["shift_id"],
            shift_steps=recovery_states["shift_step"],
            threshold=args.recovery_match_threshold,
            horizon=args.recovery_horizon,
        )
    )
    for variant, summary in summaries.items():
        summary["slots"] = int(args.slots)
        summary["queries"] = int(args.queries)
        summary["top_k"] = int(args.top_k)
        summary["recovery_shifts"] = int(args.recovery_shifts)
        summary["recovery_horizon"] = int(args.recovery_horizon)
        summary["recovery_match_threshold"] = float(args.recovery_match_threshold)
        summary["hnsw_index_build_ms"] = float(index_build_ms) if variant != "exact" else 0.0

    _write_csv(args.out_dir / "per_query_metrics.csv", all_rows)
    _write_csv(args.out_dir / "recovery_per_step_metrics.csv", recovery_rows)
    summary_rows = [{"variant": variant, **values} for variant, values in summaries.items()]
    _write_csv(args.out_dir / "summary.csv", summary_rows)
    (args.out_dir / "results.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "type": "million-slot controlled context-routing simulation",
                    "slots": int(args.slots),
                    "queries": int(args.queries),
                    "top_k": int(args.top_k),
                    "hnsw_m": int(args.hnsw_m),
                    "hnsw_ef_search": int(args.hnsw_ef_search),
                    "hnsw_ef_construction": int(args.hnsw_ef_construction),
                    "recovery_shifts": int(args.recovery_shifts),
                    "recovery_horizon": int(args.recovery_horizon),
                    "recovery_match_threshold": float(args.recovery_match_threshold),
                    "seed": int(args.seed),
                    "note": "Mechanism/scalability diagnostic; not a four-environment end-to-end retraining result.",
                },
                "summaries": summaries,
            },
            indent=2,
        )
    )
    print(json.dumps({"index_build_ms": index_build_ms, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
