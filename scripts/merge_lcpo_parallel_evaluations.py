#!/usr/bin/env python3
"""Merge one-policy LCPO evaluation shards into a shared-condition result."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_eval import pareto_front, save_baseline_summary, save_front_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-key", required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--expected-preferences", type=int, required=True)
    return parser.parse_args()


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }


def _write_regime_artifacts(path: Path, regime_fronts: list[dict[str, Any]]) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    for row in regime_fronts:
        regime_id = int(row["regime_id"])
        name = str(row["regime"]).replace("/", "_").replace(" ", "_")
        stem = f"regime_{regime_id:03d}_{name}"
        points = np.asarray(row["points"], dtype=np.float64)
        front = pareto_front(points)
        payload = {
            "regime": row["regime"],
            "regime_id": regime_id,
            "regime_meta": row["regime_meta"],
            "num_solutions": int(len(points)),
            "front_points": front.tolist(),
            "solution_points": points.tolist(),
        }
        (path / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        np.savez_compressed(
            path / f"{stem}.npz",
            solutions=np.asarray(points, dtype=np.float32),
            pareto_front_points=np.asarray(front, dtype=np.float32),
            regime_id=np.asarray([regime_id], dtype=np.int32),
        )


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.shard_root.glob("policy_*/summary.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No LCPO evaluation shards found in {args.shard_root}")

    payloads = [json.loads(path.read_text()) for path in shard_paths]
    if any(payload.get("method") != "LCPO" or payload.get("env_key") != args.env_key for payload in payloads):
        raise ValueError("Every shard must be an LCPO result for the requested environment.")
    if any(len(payload.get("preference_ids", [])) != 1 for payload in payloads):
        raise ValueError("Each shard must contain exactly one original preference ID.")
    ids = [int(payload["preference_ids"][0]) for payload in payloads]
    if sorted(ids) != list(range(int(args.expected_preferences))):
        raise ValueError(f"Expected preference IDs 0..{args.expected_preferences - 1}, found {sorted(ids)}")

    first = payloads[0]
    plan = first["shared_regime_seed_plan"]
    plan_json = json.dumps(plan, sort_keys=True)
    if any(json.dumps(payload["shared_regime_seed_plan"], sort_keys=True) != plan_json for payload in payloads[1:]):
        raise ValueError("LCPO shards were not evaluated on the same condition plan.")

    regimes: dict[int, dict[str, Any]] = {}
    per_policy = []
    all_front_points = []
    trace_rows = []
    preferences: dict[int, list[float]] = {}
    for payload in payloads:
        preference_id = int(payload["preference_ids"][0])
        preferences[preference_id] = list(payload["preferences"][0])
        all_front_points.extend(list(payload.get("front_points", [])))
        trace_rows.append(dict(payload.get("trace_metrics", {})))
        for policy_row in payload.get("per_policy", []):
            per_policy.append({"policy_id": preference_id, **dict(policy_row)})
        for row in payload.get("regime_fronts", []):
            regime_id = int(row["regime_id"])
            bucket = regimes.setdefault(
                regime_id,
                {
                    "regime": str(row["regime"]),
                    "regime_id": regime_id,
                    "regime_meta": dict(row.get("regime_meta", {})),
                    "points": [],
                },
            )
            bucket["points"].extend(list(row.get("points", [])))
    if len(regimes) != len(plan):
        raise ValueError(f"Expected {len(plan)} regime fronts, found {len(regimes)}")

    ordered_regimes = [regimes[regime_id] for regime_id in sorted(regimes)]
    args.save_dir.mkdir(parents=True, exist_ok=True)
    _write_regime_artifacts(args.save_dir / "shared_regime_returns", ordered_regimes)
    summary = {
        "method": "LCPO",
        "env_key": args.env_key,
        "env_name": first["env_name"],
        "seed": int(first["seed"]),
        "obj_num": int(first["obj_num"]),
        "total_timesteps_per_preference": int(first["total_timesteps_per_preference"]),
        "total_training_timesteps": int(first["total_timesteps_per_preference"]) * len(payloads),
        "policy_source": "checkpoint_parallel_evaluation",
        "preferences": [preferences[preference_id] for preference_id in sorted(preferences)],
        "preference_ids": sorted(preferences),
        "env_kwargs": first["env_kwargs"],
        "lcpo_config": first["lcpo_config"],
        "front_points": all_front_points,
        "regime_fronts": ordered_regimes,
        "trace_metrics": _mean_metrics(trace_rows),
        "per_policy": sorted(per_policy, key=lambda row: int(row["policy_id"])),
        "shared_eval_episode_seeds": first["shared_eval_episode_seeds"],
        "shared_regime_seed_plan": plan,
        "parallel_evaluation_shards": [str(path) for path in shard_paths],
    }
    save_baseline_summary(args.save_dir / "summary.json", summary)
    save_front_csv(args.save_dir / "front_points.csv", all_front_points)
    print(json.dumps({"env_key": args.env_key, "preferences": len(payloads), "regimes": len(ordered_regimes)}))


if __name__ == "__main__":
    main()
