#!/usr/bin/env python3
"""Run a small, real shared-protocol study for condition-level paired tests.

This runner evaluates a fixed, evenly-spaced subset of ten conditions from the
registered 100-condition plans.  The ten aligned conditions are the sample
units in the companion scorer; they are not ten independent training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_quick_adaptation_validation as quick


ENV_KEYS = ("building", "evcharging", "cogen", "chlor_alkali")
METHODS = ("dynamic", "capql", "qpensieve", "pgmorl", "morlca")
PLAN_ROOT = PROJECT_ROOT / "analysis" / "lcpo_lightweight_100"
DEFAULT_CONDITION_IDS = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-keys", nargs="+", choices=ENV_KEYS, default=list(ENV_KEYS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--condition-ids", nargs="+", type=int, default=list(DEFAULT_CONDITION_IDS))
    parser.add_argument("--training-seed", type=int, default=20260814)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_10_condition_ttest_study",
    )
    parser.add_argument("--study-name", default="quick_real_v1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=96)
    # Plans retain IDs through 99, so the complete catalog must stay at 100.
    parser.add_argument("--fine-regime-clusters", type=int, default=100)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=128)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--trace-eval-episodes", type=int, default=1)
    parser.add_argument("--trace-recovery-window", type=int, default=3)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)

    parser.add_argument("--dynamic-num-time-steps", type=int, default=1024)
    parser.add_argument("--dynamic-num-init-steps", type=int, default=512)
    parser.add_argument("--dynamic-num-steps", type=int, default=8)
    parser.add_argument("--dynamic-num-processes", type=int, default=1)
    parser.add_argument("--dynamic-ppo-epoch", type=int, default=2)
    parser.add_argument("--dynamic-num-mini-batch", type=int, default=2)
    parser.add_argument("--dynamic-num-select", type=int, default=4)
    parser.add_argument("--dynamic-eval-num", type=int, default=1)
    parser.add_argument("--dynamic-rl-eval-interval", type=int, default=4)
    parser.add_argument("--dynamic-delta-weight", type=float, default=0.5)
    parser.add_argument("--dynamic-trace-eval-samples", type=int, default=1)
    parser.add_argument("--dynamic-regime-eval-samples", type=int, default=1)
    parser.add_argument("--dynamic-context-probe-samples", type=int, default=2)
    parser.add_argument("--dynamic-context-trace-steps", type=int, default=4)
    parser.add_argument("--dynamic-context-buffer-steps", type=int, default=32)
    parser.add_argument("--dynamic-context-history-size", type=int, default=8)
    parser.add_argument("--dynamic-context-nearest-k", type=int, default=2)
    parser.add_argument("--dynamic-bank-enable", type=int, default=1)
    parser.add_argument("--dynamic-bank-num-slots", type=int, default=4)
    parser.add_argument("--dynamic-bank-slot-size", type=int, default=2)
    parser.add_argument("--dynamic-bank-merge-threshold", type=float, default=0.82)
    parser.add_argument("--dynamic-bank-query-slots", type=int, default=2)
    parser.add_argument("--dynamic-bank-query-topk", type=int, default=1)
    parser.add_argument("--dynamic-knee-lambda", type=float, default=0.2)
    parser.add_argument("--dynamic-lambda", type=float, default=0.5)
    parser.add_argument("--dynamic-resilience-lambda", type=float, default=1.0)
    parser.add_argument("--dynamic-diversity-lambda", type=float, default=0.1)
    parser.add_argument("--dynamic-shift-gap-lambda", type=float, default=0.5)
    parser.add_argument("--dynamic-repeat-topk", type=int, default=2)
    parser.add_argument("--dynamic-online-shift-threshold", type=float, default=0.6)
    parser.add_argument("--dynamic-online-shift-min-gap", type=int, default=6)
    parser.add_argument("--dynamic-final-archive-mode", choices=("pareto", "union", "resilient-diverse"), default="resilient-diverse")
    parser.add_argument("--dynamic-final-archive-max-samples", type=int, default=7)
    parser.add_argument("--dynamic-finalization-timeout-seconds", type=int, default=1800)

    parser.add_argument("--capql-total-timesteps", type=int, default=1024)
    parser.add_argument("--capql-start-steps", type=int, default=128)
    parser.add_argument("--capql-batch-size", type=int, default=128)
    parser.add_argument("--capql-hidden-size", type=int, default=128)
    parser.add_argument("--qpensieve-total-timesteps", type=int, default=1024)
    parser.add_argument("--qpensieve-start-steps", type=int, default=128)
    parser.add_argument("--qpensieve-batch-size", type=int, default=128)
    parser.add_argument("--qpensieve-prefer-num", type=int, default=4)
    parser.add_argument("--qpensieve-hidden-size", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--pgmorl-total-timesteps", type=int, default=1024)
    parser.add_argument("--pgmorl-num-steps", type=int, default=64)
    parser.add_argument("--pgmorl-warmup-iter", type=int, default=4)
    parser.add_argument("--pgmorl-update-iter", type=int, default=2)
    parser.add_argument("--pgmorl-max-eval-policies", type=int, default=7)
    parser.add_argument("--morlca-total-timesteps", type=int, default=1024)
    parser.add_argument("--morlca-start-steps", type=int, default=128)
    parser.add_argument("--morlca-batch-size", type=int, default=128)
    parser.add_argument("--morlca-hidden-size", type=int, default=128)
    parser.add_argument("--morlca-aow-aux-coef", type=float, default=0.2)
    parser.add_argument("--morlca-eval-pref-bias-scale", type=float, default=0.75)

    parser.add_argument("--max-runs", type=int, default=0, help="run at most this many missing jobs")
    parser.add_argument("--force", action="store_true", help="replace completed or incomplete study runs")
    parser.add_argument("--rerun-incomplete", action="store_true", help="replace incomplete study runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = list(payload.get("shared_regime_seed_plan", []))
    ids = [int(row.get("regime_id", -1)) for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100 or set(ids) != set(range(100)):
        raise ValueError(f"{path}: expected exact regime ids 0..99")
    return payload


def _prepare_plans(study_root: Path, env_keys: list[str], condition_ids: list[int]) -> dict[str, Path]:
    if len(condition_ids) < 2 or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("At least two distinct conditions are required for a paired t-test.")
    if any(regime_id < 0 or regime_id >= 100 for regime_id in condition_ids):
        raise ValueError("Condition IDs must be within the registered range 0..99.")
    plans: dict[str, Path] = {}
    plan_dir = study_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for env_key in env_keys:
        source = PLAN_ROOT / f"{env_key}_shared_plan.json"
        payload = _load_source_plan(source)
        rows_by_id = {int(row["regime_id"]): row for row in payload["shared_regime_seed_plan"]}
        destination_payload = {
            "study": "quick_real_10_condition_paired_ttest",
            "source_plan": str(source),
            "source_plan_sha256": _sha256(source),
            "conditions": len(condition_ids),
            "selected_regime_ids": condition_ids,
            "shared_regime_seed_plan": [rows_by_id[regime_id] for regime_id in condition_ids],
        }
        destination = plan_dir / f"{env_key}_shared_plan.json"
        serialized = json.dumps(destination_payload, indent=2) + "\n"
        if destination.exists() and destination.read_text() != serialized:
            raise ValueError(f"Refusing to overwrite a different registered subset plan: {destination}")
        if not destination.exists():
            destination.write_text(serialized)
        plans[env_key] = destination
    return plans


def run_is_complete(run_dir: Path, method: str, expected_plan: list[dict[str, Any]]) -> bool:
    expected_ids = [int(row.get("regime_id", -1)) for row in expected_plan]
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        return False
    expected_solution_count: int | None = None
    if method == "dynamic":
        # OW-CMORL rewrites shared exports after each persisted archive policy.
        expected_solution_count = len(list((run_dir / "final").glob("EP_policy_*.pt")))
        if expected_solution_count <= 0:
            return False
    else:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        expected_solution_count = (
            int(summary.get("num_policies", 0))
            if method == "pgmorl"
            else len(list(summary.get("preferences", [])))
        )
        summary_rows = list(summary.get("regime_fronts", []))
        summary_ids = [int(row.get("regime_id", -1)) for row in summary_rows]
        if expected_solution_count <= 0 or len(summary_rows) != len(expected_ids) or set(summary_ids) != set(expected_ids):
            return False

    artifact_dir = run_dir / "shared_regime_returns"
    artifacts = list(artifact_dir.glob("*.json"))
    if len(artifacts) != len(expected_ids) or len(list(artifact_dir.glob("*.npz"))) != len(expected_ids):
        return False
    artifact_ids: list[int] = []
    for artifact in artifacts:
        try:
            payload = json.loads(artifact.read_text())
            regime_id = int(payload.get("regime_id", -1))
            front = np.asarray(payload.get("front_points", []), dtype=np.float64)
            solutions = np.asarray(payload.get("solution_points", []), dtype=np.float64)
        except (json.JSONDecodeError, OSError, ValueError):
            return False
        if (
            front.ndim != 2
            or len(front) == 0
            or solutions.ndim != 2
            or len(solutions) != expected_solution_count
        ):
            return False
        artifact_ids.append(regime_id)
    if len(set(artifact_ids)) != len(expected_ids) or set(artifact_ids) != set(expected_ids):
        return False
    if method != "dynamic":
        return True

    shared_final = run_dir / "regime_fronts" / "shared_final.json"
    if not shared_final.exists():
        return False
    try:
        payload = json.loads(shared_final.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    rows = list(payload.get("regimes", []))
    saved_plan = list(payload.get("shared_regime_seed_plan", []))
    return (
        len(rows) == len(expected_ids)
        and {int(row.get("regime_id", -1)) for row in rows} == set(expected_ids)
        and [int(row.get("regime_id", -1)) for row in saved_plan] == expected_ids
    )


def _quick_args(args: argparse.Namespace) -> SimpleNamespace:
    values = vars(args).copy()
    values["seed"] = int(args.training_seed)
    values["shared_regime_eval_episodes"] = len(args.condition_ids)
    values["shared_regime_seed_offset"] = 0
    return SimpleNamespace(**values)


def _command(method: str, env_key: str, run_args: SimpleNamespace, run_dir: Path, plan: Path) -> list[str]:
    if method == "dynamic":
        return quick._dynamic_command(env_key, run_args, run_dir, plan)
    return quick._method_command(method, env_key, run_args, run_dir, plan)


def _write_protocol(study_root: Path, args: argparse.Namespace, plans: dict[str, Path]) -> None:
    payload = {
        "study": "quick_real_10_condition_paired_ttest",
        "purpose": "Short-budget real simulator validation; not a multi-seed or full-protocol replacement.",
        "methods": list(args.methods),
        "environments": list(args.env_keys),
        "training_seed": int(args.training_seed),
        "conditions_per_environment": len(args.condition_ids),
        "selected_regime_ids": [int(value) for value in args.condition_ids],
        "shared_plans": {env: {"path": str(path), "sha256": _sha256(path)} for env, path in plans.items()},
        "budgets": {
            "dynamic_num_time_steps": int(args.dynamic_num_time_steps),
            "dynamic_final_archive_mode": str(args.dynamic_final_archive_mode),
            "dynamic_final_archive_max_samples": int(args.dynamic_final_archive_max_samples),
            "capql_total_timesteps": int(args.capql_total_timesteps),
            "qpensieve_total_timesteps": int(args.qpensieve_total_timesteps),
            "pgmorl_total_timesteps": int(args.pgmorl_total_timesteps),
            "pgmorl_max_eval_policies": int(args.pgmorl_max_eval_policies),
            "morlca_total_timesteps": int(args.morlca_total_timesteps),
        },
        "scoring": {
            "hv_eu": "condition-wise common reference point and preference grid",
            "adaptation_score": "condition-wise best HV/EU among compared methods",
            "inference_unit": "fixed aligned operating condition in a single shared trajectory",
            "test": "paired one-sided t-test; quick diagnostic only, not independent-seed inference",
        },
    }
    (study_root / "STUDY_PROTOCOL.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_export_manifest(
    run_dir: Path,
    *,
    environment: str,
    method: str,
    plan: Path,
    training_seed: int,
    process_status: str,
    conditions: int,
) -> None:
    payload = {
        "environment": environment,
        "method": method,
        "training_seed": int(training_seed),
        "shared_plan": str(plan),
        "shared_plan_sha256": _sha256(plan),
        "primary_artifact": "shared_regime_returns/*.json and *.npz",
        "conditions": int(conditions),
        "process_status": process_status,
    }
    (run_dir / "study_export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    study_root = args.results_root / args.study_name
    study_root.mkdir(parents=True, exist_ok=True)
    condition_ids = [int(value) for value in args.condition_ids]
    plans = _prepare_plans(study_root, list(args.env_keys), condition_ids)
    _write_protocol(study_root, args, plans)
    run_args = _quick_args(args)

    launched = 0
    for env_key in args.env_keys:
        expected_plan = list(json.loads(plans[env_key].read_text())["shared_regime_seed_plan"])
        for method in args.methods:
            run_dir = study_root / "runs" / env_key / method
            if run_is_complete(run_dir, method, expected_plan) and not args.force:
                if not (run_dir / "study_export_manifest.json").exists():
                    _write_export_manifest(
                        run_dir,
                        environment=env_key,
                        method=method,
                        plan=plans[env_key],
                        training_seed=args.training_seed,
                        process_status="existing_complete_artifacts",
                        conditions=len(expected_plan),
                    )
                print(f"skip complete: {run_dir}", flush=True)
                continue
            command = _command(method, env_key, run_args, run_dir, plans[env_key])
            if args.dry_run:
                print("would run:", " ".join(command), flush=True)
                continue
            if run_dir.exists():
                if not (args.force or args.rerun_incomplete):
                    raise RuntimeError(f"Incomplete run exists: {run_dir}; pass --rerun-incomplete to replace it.")
                shutil.rmtree(run_dir)
            log_path = study_root / "logs" / env_key / f"{method}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            record_path = run_dir / "run_record.json"
            record: dict[str, Any] = {
                "environment": env_key,
                "method": method,
                "training_seed": int(args.training_seed),
                "command": command,
                "shared_plan": str(plans[env_key]),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            print("running:", " ".join(command), flush=True)
            stopped_after_export = False
            timed_out = False
            with log_path.open("w") as log_handle:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log_handle, stderr=subprocess.STDOUT)
                start_time = time.monotonic()
                returncode: int | None = None
                while returncode is None:
                    returncode = process.poll()
                    if returncode is not None:
                        break
                    if method == "dynamic" and run_is_complete(run_dir, method, expected_plan):
                        process.terminate()
                        try:
                            returncode = int(process.wait(timeout=30))
                        except subprocess.TimeoutExpired:
                            process.kill()
                            returncode = int(process.wait())
                        stopped_after_export = True
                        break
                    if method == "dynamic" and int(args.dynamic_finalization_timeout_seconds) > 0 and time.monotonic() - start_time >= int(args.dynamic_finalization_timeout_seconds):
                        process.terminate()
                        try:
                            returncode = int(process.wait(timeout=30))
                        except subprocess.TimeoutExpired:
                            process.kill()
                            returncode = int(process.wait())
                        timed_out = True
                        break
                    time.sleep(2.0)
            record.update(
                {
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "stopped_after_complete_export": stopped_after_export,
                    "complete": run_is_complete(run_dir, method, expected_plan),
                }
            )
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            complete = bool(record["complete"])
            if (returncode not in {0, None} and not (method == "dynamic" and complete)) or not complete:
                raise RuntimeError(f"{method}/{env_key} failed or lacks complete exports; see {log_path}")
            status = "completed" if returncode == 0 else "terminated_after_complete_artifact_export"
            _write_export_manifest(
                run_dir,
                environment=env_key,
                method=method,
                plan=plans[env_key],
                training_seed=args.training_seed,
                process_status=status,
                conditions=len(expected_plan),
            )
            launched += 1
            if int(args.max_runs) and launched >= int(args.max_runs):
                print(f"stopped after --max-runs={args.max_runs}", flush=True)
                return
    print(json.dumps({"study_root": str(study_root), "new_runs": launched, "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
