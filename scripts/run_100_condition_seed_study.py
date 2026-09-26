#!/usr/bin/env python3
"""Run a reproducible, short-budget 100-condition multi-seed MORL study.

The script deliberately keeps training budgets small.  It is a real simulator
experiment, not a replacement for the long-run comparison in the paper.  All
methods in a replication share one pre-registered training seed and the same
fixed 100-condition evaluation plan for each environment.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-keys", nargs="+", choices=ENV_KEYS, default=list(ENV_KEYS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--replication-ids",
        nargs="+",
        type=int,
        default=None,
        help="optional registered replication subset, useful for independent environment shards",
    )
    parser.add_argument("--seed-manifest-seed", type=int, default=20260813)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_100_condition_seed_study",
    )
    parser.add_argument("--study-name", default="quick_real_v1")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=100)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=96)
    parser.add_argument("--fine-regime-clusters", type=int, default=100)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=128)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--trace-eval-episodes", type=int, default=1)
    parser.add_argument("--trace-recovery-window", type=int, default=3)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)

    # The following values match the existing quick validation configuration.
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
    parser.add_argument(
        "--dynamic-final-archive-mode",
        choices=("pareto", "union", "resilient-diverse"),
        default="resilient-diverse",
        help="final OW-CMORL archive selector used before the 100-condition sweep",
    )
    parser.add_argument(
        "--dynamic-final-archive-max-samples",
        type=int,
        default=7,
        help="fixed final OW-CMORL candidate count for the quick multi-seed study",
    )
    parser.add_argument("--dynamic-timeout-seconds", type=int, default=0)
    parser.add_argument(
        "--dynamic-finalization-timeout-seconds",
        type=int,
        default=1800,
        help="wall-clock cap for OW-CMORL; accept only after all 100 condition exports validate",
    )

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
    parser.add_argument(
        "--pgmorl-max-eval-policies",
        type=int,
        default=7,
        help="fixed PGMORL policy count, matched to the 7-candidate quick-study budget",
    )
    parser.add_argument("--morlca-total-timesteps", type=int, default=1024)
    parser.add_argument("--morlca-start-steps", type=int, default=128)
    parser.add_argument("--morlca-batch-size", type=int, default=128)
    parser.add_argument("--morlca-hidden-size", type=int, default=128)
    parser.add_argument("--morlca-aow-aux-coef", type=float, default=0.2)
    parser.add_argument("--morlca-eval-pref-bias-scale", type=float, default=0.75)

    parser.add_argument("--max-runs", type=int, default=0, help="run at most this many missing jobs")
    parser.add_argument("--force", action="store_true", help="delete and rerun completed or incomplete study jobs")
    parser.add_argument("--rerun-incomplete", action="store_true", help="delete and rerun incomplete study jobs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(source: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text())
    plan = list(payload.get("shared_regime_seed_plan", []))
    ids = [int(row.get("regime_id", -1)) for row in plan]
    if len(plan) != 100 or len(set(ids)) != 100 or min(ids) < 0:
        raise ValueError(f"{source}: expected 100 unique nonnegative regime ids")
    return payload


def _write_seed_manifest(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if path.exists():
        manifest = json.loads(path.read_text())
        seeds = [int(item["training_seed"]) for item in manifest.get("replications", [])]
        if len(seeds) != int(args.repetitions):
            raise ValueError(f"Existing manifest has {len(seeds)} repetitions, requested {args.repetitions}")
        return manifest
    if int(args.repetitions) < 2:
        raise ValueError("At least two independent training seeds are required for a t-test.")
    rng = np.random.default_rng(int(args.seed_manifest_seed))
    # Do not materialize the full 31-bit seed domain just to sample ten values.
    # Collision resolution remains deterministic because it uses this RNG only.
    seed_set: set[int] = set()
    while len(seed_set) < int(args.repetitions):
        seed_set.add(int(rng.integers(1, 2**31 - 1, endpoint=False)))
    seeds = sorted(seed_set)
    manifest = {
        "study": "quick_real_100_condition_multi_seed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_manifest_rng": "numpy.default_rng",
        "seed_manifest_seed": int(args.seed_manifest_seed),
        "replications": [
            {"replication": int(index), "training_seed": int(seed)}
            for index, seed in enumerate(seeds, start=1)
        ],
        "pairing": "Each method receives the same training_seed within a replication.",
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _prepare_plans(study_root: Path, env_keys: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    plan_dir = study_root / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    for env_key in env_keys:
        source = PLAN_ROOT / f"{env_key}_shared_plan.json"
        if not source.exists():
            raise FileNotFoundError(source)
        payload = _load_plan(source)
        destination = plan_dir / source.name
        source_hash = _sha256(source)
        if destination.exists() and _sha256(destination) != source_hash:
            raise ValueError(f"Refusing to overwrite different registered plan: {destination}")
        if not destination.exists():
            destination.write_text(json.dumps(payload, indent=2) + "\n")
        out[env_key] = destination
    return out


def run_is_complete(run_dir: Path, method: str, expected_plan: list[dict[str, Any]]) -> bool:
    expected_ids = [int(row.get("regime_id", -1)) for row in expected_plan]
    expected_solution_count: int | None = None
    if method == "dynamic":
        # Shared-final artifacts are written incrementally after every final
        # archive sample.  A 100-row artifact can therefore contain only the
        # first candidate unless we require every condition to include the
        # complete persisted archive before accepting an early termination.
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
        if method == "pgmorl":
            expected_solution_count = int(summary.get("num_policies", 0))
        else:
            expected_solution_count = len(list(summary.get("preferences", [])))
        if expected_solution_count <= 0:
            return False
        summary_rows = list(summary.get("regime_fronts", []))
        summary_ids = {int(row.get("regime_id", -1)) for row in summary_rows}
        if len(summary_rows) != 100 or summary_ids != set(expected_ids):
            return False
    artifact_dir = run_dir / "shared_regime_returns"
    json_ids: set[int] = set()
    for artifact in artifact_dir.glob("*.json"):
        try:
            artifact_payload = json.loads(artifact.read_text())
            regime_id = int(artifact_payload.get("regime_id", -1))
            front = np.asarray(artifact_payload.get("front_points", []), dtype=np.float64)
            solutions = np.asarray(artifact_payload.get("solution_points", []), dtype=np.float64)
            if (
                regime_id < 0
                or front.ndim != 2
                or len(front) == 0
                or solutions.ndim != 2
                or (expected_solution_count is not None and len(solutions) != expected_solution_count)
            ):
                return False
            json_ids.add(regime_id)
        except (json.JSONDecodeError, OSError, ValueError):
            return False
    npz_count = len(list(artifact_dir.glob("*.npz")))
    if json_ids != set(expected_ids) or npz_count < 100:
        return False
    # Baseline summary files are written only after a second trace sweep, so
    # their raw per-condition exports are the primary completion evidence.
    if method != "dynamic":
        return True
    path = run_dir / "regime_fronts" / "shared_final.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    rows = list(payload.get("regimes", []))
    row_ids = {int(row.get("regime_id", -1)) for row in rows}
    plan = list(payload.get("shared_regime_seed_plan", []))
    plan_ids = [int(row.get("regime_id", -1)) for row in plan]
    return len(rows) == 100 and row_ids == set(expected_ids) and len(plan) == 100 and plan_ids == expected_ids


def _quick_args(args: argparse.Namespace, training_seed: int) -> SimpleNamespace:
    values = vars(args).copy()
    values["seed"] = int(training_seed)
    values["shared_regime_seed_offset"] = 0
    values["lcpo_total_timesteps"] = 0
    values["lcpo_batch_size"] = 0
    values["lcpo_hidden_size"] = 0
    values["lcpo_action_bins"] = 0
    values["lcpo_max_preferences"] = 0
    return SimpleNamespace(**values)


def _command(method: str, env_key: str, run_args: SimpleNamespace, run_dir: Path, plan: Path) -> list[str]:
    if method == "dynamic":
        return quick._dynamic_command(env_key, run_args, run_dir, plan)
    return quick._method_command(method, env_key, run_args, run_dir, plan)


def _write_protocol(study_root: Path, args: argparse.Namespace, plans: dict[str, Path]) -> None:
    payload = {
        "study": "quick_real_100_condition_multi_seed",
        "purpose": "Short-budget real simulator verification; not a long-run replacement.",
        "methods": list(args.methods),
        "environments": list(args.env_keys),
        "conditions_per_environment": 100,
        "training_seed_pairing": "same seed across methods within each replication",
        "shared_plans": {
            env_key: {"path": str(path), "sha256": _sha256(path)} for env_key, path in plans.items()
        },
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
            "adaptation_score": "condition-wise best HV/EU across the compared methods in the same replication",
            "regret_latency": "reconstructed from aligned solution traces",
            "inference_unit": "independent training replication (not condition)",
        },
    }
    (study_root / "STUDY_PROTOCOL.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_export_manifest(
    run_dir: Path,
    *,
    replication: int,
    training_seed: int,
    environment: str,
    method: str,
    plan: Path,
    process_status: str,
) -> None:
    """Record that scoring uses raw exported fronts, including early exits."""
    payload = {
        "replication": int(replication),
        "training_seed": int(training_seed),
        "environment": environment,
        "method": method,
        "shared_plan": str(plan),
        "shared_plan_sha256": _sha256(plan),
        "primary_artifact": "shared_regime_returns/*.json and *.npz",
        "conditions": 100,
        "process_status": process_status,
    }
    (run_dir / "study_export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if int(args.shared_regime_eval_episodes) != 100:
        raise ValueError("This study is defined for exactly 100 shared operating conditions.")
    study_root = args.results_root / args.study_name
    study_root.mkdir(parents=True, exist_ok=True)
    plans = _prepare_plans(study_root, list(args.env_keys))
    manifest = _write_seed_manifest(study_root / "seed_manifest.json", args)
    _write_protocol(study_root, args, plans)
    replications = list(manifest["replications"])
    if args.replication_ids is not None:
        requested = {int(value) for value in args.replication_ids}
        replications = [row for row in replications if int(row.get("replication", -1)) in requested]
        if len(replications) != len(requested):
            raise ValueError(f"Requested replication ids missing from manifest: {sorted(requested)}")

    launched = 0
    for replication in replications:
        replication_id = int(replication["replication"])
        training_seed = int(replication["training_seed"])
        run_args = _quick_args(args, training_seed)
        for env_key in args.env_keys:
            expected_plan = list(json.loads(plans[env_key].read_text())["shared_regime_seed_plan"])
            for method in args.methods:
                run_dir = study_root / "runs" / f"rep_{replication_id:02d}_seed_{training_seed}" / env_key / method
                if run_is_complete(run_dir, method, expected_plan) and not args.force:
                    manifest_path = run_dir / "study_export_manifest.json"
                    if not manifest_path.exists():
                        _write_export_manifest(
                            run_dir,
                            replication=replication_id,
                            training_seed=training_seed,
                            environment=env_key,
                            method=method,
                            plan=plans[env_key],
                            process_status="existing_complete_artifacts",
                        )
                    print(f"skip complete: {run_dir}", flush=True)
                    continue
                command = _command(method, env_key, run_args, run_dir, plans[env_key])
                if args.dry_run:
                    print("would run:", " ".join(command), flush=True)
                    continue
                if run_dir.exists():
                    if not (args.force or args.rerun_incomplete):
                        raise RuntimeError(
                            f"Incomplete run exists: {run_dir}. Use --rerun-incomplete to replace only this dedicated study run."
                        )
                    shutil.rmtree(run_dir)
                log_path = study_root / "logs" / f"rep_{replication_id:02d}_seed_{training_seed}" / env_key / f"{method}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                record_path = run_dir / "run_record.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                record = {
                    "replication": replication_id,
                    "training_seed": training_seed,
                    "environment": env_key,
                    "method": method,
                    "command": command,
                    "shared_plan": str(plans[env_key]),
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                print("running:", " ".join(command), flush=True)
                timed_out = False
                stopped_after_export = False
                with log_path.open("w") as log_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    start_time = time.monotonic()
                    returncode: int | None = None
                    while returncode is None:
                        returncode = process.poll()
                        if returncode is not None:
                            break
                        if method == "dynamic" and run_is_complete(run_dir, method, expected_plan):
                            # All scoring inputs are persisted at this point.
                            # The remaining OW-CMORL evaluator pass replays the
                            # archive on the same conditions solely to write its
                            # private summary and is intentionally not used.
                            process.terminate()
                            try:
                                returncode = int(process.wait(timeout=30))
                            except subprocess.TimeoutExpired:
                                process.kill()
                                returncode = int(process.wait())
                            stopped_after_export = True
                            break
                        if (
                            method == "dynamic"
                            and int(args.dynamic_finalization_timeout_seconds) > 0
                            and time.monotonic() - start_time >= int(args.dynamic_finalization_timeout_seconds)
                        ):
                            process.terminate()
                            try:
                                returncode = int(process.wait(timeout=30))
                            except subprocess.TimeoutExpired:
                                process.kill()
                                returncode = int(process.wait())
                            timed_out = True
                            break
                        time.sleep(2.0)
                record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                record["returncode"] = returncode
                record["timed_out"] = timed_out
                record["stopped_after_complete_export"] = stopped_after_export
                record["complete"] = run_is_complete(run_dir, method, expected_plan)
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                # OW-CMORL first writes the complete shared-plan archive and
                # then runs a redundant archive-level trace sweep. Baselines
                # instead add one solution per preference/policy to the same
                # exports, so they must exit normally after all policies run.
                accepted_early_exit = method == "dynamic" and bool(record["complete"])
                if (returncode not in {0, None} and not accepted_early_exit) or (returncode is None and not accepted_early_exit) or not record["complete"]:
                    raise RuntimeError(
                        f"{method}/{env_key}/rep{replication_id} failed or lacked complete 100-condition exports; see {log_path}"
                    )
                process_status = (
                    "completed"
                    if returncode == 0
                    else "terminated_after_complete_artifact_export"
                    if stopped_after_export
                    else "interrupted_after_complete_artifact_export"
                )
                _write_export_manifest(
                    run_dir,
                    replication=replication_id,
                    training_seed=training_seed,
                    environment=env_key,
                    method=method,
                    plan=plans[env_key],
                    process_status=process_status,
                )
                if accepted_early_exit and returncode != 0:
                    print(f"accepted early exit after complete 100-condition exports: {run_dir}", flush=True)
                launched += 1
                if int(args.max_runs) and launched >= int(args.max_runs):
                    print(f"stopped after --max-runs={args.max_runs}", flush=True)
                    return
    print(json.dumps({"study_root": str(study_root), "new_runs": launched, "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
