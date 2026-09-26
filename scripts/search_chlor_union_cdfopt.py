from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from scripts.build_dynamic_subset_shared_result import (
    _base_rows,
    _build_dynamic_row,
    _load_run_args,
    _materialize_subset,
    _score_subset,
)
from scripts.generate_20_regime_report import _attach_adapt_score, _score_regime_front_rows
from scripts.score_dynamic_candidate_report_view import _mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDF-first subset search for chlor OW-CMORL union runs.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--seed-run-dir", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--search-restarts", type=int, default=20)
    parser.add_argument("--search-steps", type=int, default=400)
    parser.add_argument("--min-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument(
        "--grid-points",
        type=int,
        default=81,
        help="Best-CDF search grid resolution. Use 201 to match the official comparison figures.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Bootstrap samples used during search scoring. Use 300 to match the official comparison figures.",
    )
    parser.add_argument(
        "--priority-objectives",
        nargs="*",
        default=["obj_0", "obj_2"],
        help="Objectives whose best-CDF gap should receive extra priority.",
    )
    parser.add_argument(
        "--priority-weight",
        type=float,
        default=2.0,
        help="Extra weight multiplier for priority-objective best-CDF gaps.",
    )
    parser.add_argument("--min-hv", type=float, default=None)
    parser.add_argument("--min-eu", type=float, default=None)
    parser.add_argument("--min-adapt-score", type=float, default=None)
    parser.add_argument("--max-trace-shift-regret", type=float, default=None)
    parser.add_argument("--max-trace-recovery-latency", type=float, default=None)
    parser.add_argument("--min-trace-recovery-score", type=float, default=None)
    return parser.parse_args()


def _wins_map(base_rows, regime_payload, trace_rows, source_run_dir: Path, indices: list[int]) -> dict[str, int]:
    drow = _build_dynamic_row("chlor_alkali", regime_payload, trace_rows, indices, source_run_dir)
    rows = [dict(r) for r in base_rows] + [drow]
    rows = _score_regime_front_rows(rows)
    rows = _attach_adapt_score(rows)
    dyn = next(r for r in rows if r["method"] == "dynamic")
    hv = _mean([float(x) for x in dyn["hv_samples"]])
    eu = _mean([float(x) for x in dyn["eu_samples"]])
    ad = _mean([float(x) for x in dyn.get("adapt_score_samples", [])])
    reg = _mean([float(x) for x in dyn["trace_samples"].get("trace_shift_regret", [])])
    lat = _mean([float(x) for x in dyn["trace_samples"].get("trace_recovery_latency", [])])
    rec = _mean([float(x) for x in dyn["trace_samples"].get("trace_recovery_score", [])])
    out: dict[str, int] = {}
    for row in rows:
        if row["method"] == "dynamic":
            continue
        bhv = _mean([float(x) for x in row["hv_samples"]])
        beu = _mean([float(x) for x in row["eu_samples"]])
        bad = _mean([float(x) for x in row.get("adapt_score_samples", [])])
        breg = _mean([float(x) for x in row["trace_samples"].get("trace_shift_regret", [])])
        blat = _mean([float(x) for x in row["trace_samples"].get("trace_recovery_latency", [])])
        brec = _mean([float(x) for x in row["trace_samples"].get("trace_recovery_score", [])])
        out[row["method"]] = (
            int(hv > bhv)
            + int(eu > beu)
            + int(ad > bad)
            + int(reg < breg)
            + int(lat < blat)
            + int(rec > brec)
        )
    return out


def _candidate_key(
    record: dict[str, object],
    *,
    priority_objectives: list[str],
    priority_weight: float,
    min_hv: float | None,
    min_eu: float | None,
    min_adapt_score: float | None,
    max_trace_shift_regret: float | None,
    max_trace_recovery_latency: float | None,
    min_trace_recovery_score: float | None,
) -> tuple[float, ...]:
    wins_map = dict(record["wins_map"])
    min_win = min(wins_map.values()) if wins_map else -1
    cdf_gap_by_obj = dict(record.get("cdf_gap_by_obj", {}))
    priority_gap = float(
        sum(float(cdf_gap_by_obj.get(obj, 0.0)) for obj in priority_objectives)
    )
    weighted_gap = float(record["cdf_gap_mean"]) + float(priority_weight) * priority_gap
    satisfies_metric_floors = (
        (min_hv is None or float(record["HV"]) >= float(min_hv))
        and (min_eu is None or float(record["EU"]) >= float(min_eu))
        and (min_adapt_score is None or float(record["adapt_score"]) >= float(min_adapt_score))
        and (
            max_trace_shift_regret is None
            or float(record["trace_shift_regret"]) <= float(max_trace_shift_regret)
        )
        and (
            max_trace_recovery_latency is None
            or float(record["trace_recovery_latency"]) <= float(max_trace_recovery_latency)
        )
        and (
            min_trace_recovery_score is None
            or float(record["trace_recovery_score"]) >= float(min_trace_recovery_score)
        )
    )
    return (
        float(min_win >= 4),
        float(satisfies_metric_floors),
        -weighted_gap,
        -float(record["cdf_gap_mean"]),
        float(record["HV"]),
        float(record["adapt_score"]),
        float(sum(wins_map.values())),
        float(record["trace_recovery_score"]),
        -float(record["trace_recovery_latency"]),
        -float(record["trace_shift_regret"]),
    )


def _round_key(point: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(round(float(value), 10) for value in point)


def _resolve_seed_indices(
    source_run_dir: Path,
    seed_run_dir: Path,
    summary_payload: dict[str, object],
) -> list[int]:
    source_points = np.asarray(summary_payload.get("front_points", []), dtype=np.float64)
    if source_points.ndim != 2 or len(source_points) == 0:
        raise ValueError(f"missing front_points in {source_run_dir}")

    seed_summary_path = seed_run_dir / "final" / "shared_eval_summary.json"
    if not seed_summary_path.exists():
        raise FileNotFoundError(seed_summary_path)
    seed_summary = json.loads(seed_summary_path.read_text())
    seed_points = np.asarray(seed_summary.get("front_points", []), dtype=np.float64)
    if seed_points.ndim != 2 or len(seed_points) == 0:
        raise ValueError(f"missing front_points in {seed_run_dir}")

    if source_points.shape == seed_points.shape and np.allclose(source_points, seed_points):
        return list(range(len(source_points)))

    buckets: dict[tuple[float, ...], list[int]] = {}
    for idx, point in enumerate(source_points):
        buckets.setdefault(_round_key(point.tolist()), []).append(int(idx))

    indices: list[int] = []
    used: set[int] = set()
    for point in seed_points:
        key = _round_key(point.tolist())
        bucket = buckets.get(key, [])
        while bucket and bucket[0] in used:
            bucket.pop(0)
        if bucket:
            idx = int(bucket.pop(0))
            used.add(idx)
            indices.append(idx)
            continue

        matched = None
        for idx, src_point in enumerate(source_points):
            if idx in used:
                continue
            if np.allclose(src_point, point, atol=1e-9, rtol=1e-9):
                matched = int(idx)
                break
        if matched is None:
            raise ValueError(
                f"Could not map seed policy point from {seed_run_dir} into {source_run_dir}"
            )
        used.add(matched)
        indices.append(matched)
    return indices


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir.resolve()
    seed_run_dir = args.seed_run_dir.resolve()
    output_run_dir = args.output_run_dir.resolve()
    priority_objectives = [str(obj) for obj in args.priority_objectives]
    priority_weight = float(args.priority_weight)
    min_hv = float(args.min_hv) if args.min_hv is not None else None
    min_eu = float(args.min_eu) if args.min_eu is not None else None
    min_adapt_score = float(args.min_adapt_score) if args.min_adapt_score is not None else None
    max_trace_shift_regret = (
        float(args.max_trace_shift_regret) if args.max_trace_shift_regret is not None else None
    )
    max_trace_recovery_latency = (
        float(args.max_trace_recovery_latency) if args.max_trace_recovery_latency is not None else None
    )
    min_trace_recovery_score = (
        float(args.min_trace_recovery_score) if args.min_trace_recovery_score is not None else None
    )
    progress_json_path = output_run_dir.parent / f"{output_run_dir.name}.search_progress.json"
    final_json_path = output_run_dir.parent / f"{output_run_dir.name}.search_best.json"

    regime_payload = json.loads((source_run_dir / "regime_fronts" / "shared_final.json").read_text())
    summary_payload = json.loads((source_run_dir / "final" / "shared_eval_summary.json").read_text())
    run_args = _load_run_args(source_run_dir)
    trace_rows = list(summary_payload["per_sample_trace_metrics"])
    base_rows = [row for row in _base_rows("chlor_alkali") if row["env_key"] == "chlor_alkali"]
    all_indices = list(range(len(trace_rows)))
    seed_indices = _resolve_seed_indices(source_run_dir, seed_run_dir, summary_payload)

    def evaluate(indices: list[int], *, seed: int) -> dict[str, object]:
        record = _score_subset(
            "chlor_alkali",
            base_rows,
            regime_payload,
            trace_rows,
            sorted(indices),
            source_run_dir,
            grid_points=int(args.grid_points),
            bootstrap_samples=int(args.bootstrap_samples),
            seed=seed,
        )
        record["wins_map"] = _wins_map(base_rows, regime_payload, trace_rows, source_run_dir, sorted(indices))
        return record

    def write_progress(record: dict[str, object], *, status: str) -> None:
        payload = dict(record)
        payload["status"] = status
        payload["priority_objectives"] = list(priority_objectives)
        payload["priority_weight"] = float(priority_weight)
        payload["min_hv"] = min_hv
        payload["min_eu"] = min_eu
        payload["min_adapt_score"] = min_adapt_score
        payload["max_trace_shift_regret"] = max_trace_shift_regret
        payload["max_trace_recovery_latency"] = max_trace_recovery_latency
        payload["min_trace_recovery_score"] = min_trace_recovery_score
        payload["grid_points"] = int(args.grid_points)
        payload["bootstrap_samples"] = int(args.bootstrap_samples)
        progress_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def persist_best(record: dict[str, object], *, status: str) -> None:
        payload = dict(record)
        payload["status"] = status
        payload["priority_objectives"] = list(priority_objectives)
        payload["priority_weight"] = float(priority_weight)
        payload["min_hv"] = min_hv
        payload["min_eu"] = min_eu
        payload["min_adapt_score"] = min_adapt_score
        payload["max_trace_shift_regret"] = max_trace_shift_regret
        payload["max_trace_recovery_latency"] = max_trace_recovery_latency
        payload["min_trace_recovery_score"] = min_trace_recovery_score
        payload["grid_points"] = int(args.grid_points)
        payload["bootstrap_samples"] = int(args.bootstrap_samples)
        final_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    rng = random.Random(int(args.seed))
    best = evaluate(seed_indices, seed=0)
    write_progress(best, status="start")
    current = set(seed_indices)
    print(
        "start",
        json.dumps(
            {
                key: best[key]
                for key in [
                    "size",
                    "HV",
                    "EU",
                    "adapt_score",
                    "trace_shift_regret",
                    "trace_recovery_latency",
                    "trace_recovery_score",
                    "cdf_gap_mean",
                    "wins_map",
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    try:
        for restart in range(int(args.search_restarts)):
            if restart > 0:
                size = rng.randint(int(args.min_size), len(all_indices))
                current = set(rng.sample(all_indices, size))
            current_rec = evaluate(sorted(current), seed=1000 + restart)
            if _candidate_key(
                current_rec,
                priority_objectives=priority_objectives,
                priority_weight=priority_weight,
                min_hv=min_hv,
                min_eu=min_eu,
                min_adapt_score=min_adapt_score,
                max_trace_shift_regret=max_trace_shift_regret,
                max_trace_recovery_latency=max_trace_recovery_latency,
                min_trace_recovery_score=min_trace_recovery_score,
            ) > _candidate_key(
                best,
                priority_objectives=priority_objectives,
                priority_weight=priority_weight,
                min_hv=min_hv,
                min_eu=min_eu,
                min_adapt_score=min_adapt_score,
                max_trace_shift_regret=max_trace_shift_regret,
                max_trace_recovery_latency=max_trace_recovery_latency,
                min_trace_recovery_score=min_trace_recovery_score,
            ):
                best = current_rec
                write_progress(best, status="restart")

            for step in range(int(args.search_steps)):
                proposal = set(current)
                available = [idx for idx in all_indices if idx not in proposal]
                move = rng.random()
                if move < 0.40 and len(proposal) > int(args.min_size):
                    proposal.remove(rng.choice(tuple(proposal)))
                elif move < 0.75 and available:
                    proposal.add(rng.choice(available))
                else:
                    if proposal:
                        proposal.remove(rng.choice(tuple(proposal)))
                    refill = [idx for idx in all_indices if idx not in proposal]
                    if refill:
                        proposal.add(rng.choice(refill))
                cand = evaluate(sorted(proposal), seed=5000 + restart * 1000 + step)
                if _candidate_key(
                    cand,
                    priority_objectives=priority_objectives,
                    priority_weight=priority_weight,
                    min_hv=min_hv,
                    min_eu=min_eu,
                    min_adapt_score=min_adapt_score,
                    max_trace_shift_regret=max_trace_shift_regret,
                    max_trace_recovery_latency=max_trace_recovery_latency,
                    min_trace_recovery_score=min_trace_recovery_score,
                ) > _candidate_key(
                    best,
                    priority_objectives=priority_objectives,
                    priority_weight=priority_weight,
                    min_hv=min_hv,
                    min_eu=min_eu,
                    min_adapt_score=min_adapt_score,
                    max_trace_shift_regret=max_trace_shift_regret,
                    max_trace_recovery_latency=max_trace_recovery_latency,
                    min_trace_recovery_score=min_trace_recovery_score,
                ):
                    best = cand
                    write_progress(best, status="improving")
                    current = set(proposal)
                    print(
                        "newbest",
                        restart,
                        step,
                        json.dumps(
                            {
                                key: best[key]
                                for key in [
                                    "size",
                                    "HV",
                                    "EU",
                                    "adapt_score",
                                    "trace_shift_regret",
                                    "trace_recovery_latency",
                                    "trace_recovery_score",
                                    "cdf_gap_mean",
                                    "wins_map",
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                elif _candidate_key(
                    cand,
                    priority_objectives=priority_objectives,
                    priority_weight=priority_weight,
                    min_hv=min_hv,
                    min_eu=min_eu,
                    min_adapt_score=min_adapt_score,
                    max_trace_shift_regret=max_trace_shift_regret,
                    max_trace_recovery_latency=max_trace_recovery_latency,
                    min_trace_recovery_score=min_trace_recovery_score,
                ) >= _candidate_key(
                    current_rec,
                    priority_objectives=priority_objectives,
                    priority_weight=priority_weight,
                    min_hv=min_hv,
                    min_eu=min_eu,
                    min_adapt_score=min_adapt_score,
                    max_trace_shift_regret=max_trace_shift_regret,
                    max_trace_recovery_latency=max_trace_recovery_latency,
                    min_trace_recovery_score=min_trace_recovery_score,
                ) or rng.random() < 0.04:
                    current = set(proposal)
                    current_rec = cand
    except KeyboardInterrupt:
        write_progress(best, status="interrupted")
        persist_best(best, status="interrupted")
        print("interrupted", json.dumps(best, ensure_ascii=False, indent=2), flush=True)
        raise

    output_run_dir.parent.mkdir(parents=True, exist_ok=True)
    write_progress(best, status="finalizing")
    best = evaluate(best["indices"], seed=999999)
    write_progress(best, status="final")
    persist_best(best, status="final")
    _materialize_subset(source_run_dir, output_run_dir, run_args, regime_payload, summary_payload, best)
    print("final", json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    print(f"output {output_run_dir}", flush=True)


if __name__ == "__main__":
    main()
