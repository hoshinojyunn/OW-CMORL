from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_envs import get_env_spec
from src.dynamic_morl.metrics import (
    compute_trace_shift_metrics,
    shared_regime_seed_plan as build_shared_regime_seed_plan,
)


ENV_KEYS = ["building", "evcharging", "cogen", "chlor_alkali"]
METHODS = ["dynamic", "capql", "qpensieve", "pgmorl", "morlca", "lcpo"]
REGIME_TRACE_WINDOW = 3
PLAN_SOURCE_CANDIDATES = {
    "building": [
        PROJECT_ROOT / "analysis" / "building_shared_plan_from_summary.json",
        PROJECT_ROOT
       
        / "results_shared_protocol_repair"
        / "candidate_building_tune1_morl_online_plan"
        / "final"
        / "shared_eval_summary.json",
    ],
    "evcharging": [
        PROJECT_ROOT
       
        / "results_shared_protocol_repair"
        / "candidate_evcharging_v7_hvboost_subset_ci"
        / "final"
        / "shared_eval_summary.json",
    ],
    "cogen": [
        PROJECT_ROOT / "analysis" / "cogen_shared_plan_from_summary.json",
        PROJECT_ROOT
       
        / "results_shared_protocol_repair"
        / "candidate_cogen_formal_best_morl_online_plan"
        / "final"
        / "shared_eval_summary.json",
    ],
    "chlor_alkali": [
        PROJECT_ROOT
       
        / "results_shared_protocol"
        / "chlor_owcmorl_union_meta_v1_cdfopt_long_v1.20260608"
        / "final"
        / "shared_eval_summary.json",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a low-budget adaptation smoke benchmark across the four dynamic MORL environments.")
    parser.add_argument("--env-keys", nargs="+", default=ENV_KEYS, choices=ENV_KEYS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results_quick_adaptation_validation",
    )
    parser.add_argument("--run-name", type=str, default="quick_adapt_smoke")
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=20)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=96)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=128)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--trace-eval-episodes", type=int, default=1)
    parser.add_argument("--trace-recovery-window", type=int, default=12)
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
    parser.add_argument("--dynamic-timeout-seconds", type=int, default=300)

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
    parser.add_argument("--pgmorl-max-eval-policies", type=int, default=16)

    parser.add_argument("--morlca-total-timesteps", type=int, default=1024)
    parser.add_argument("--morlca-start-steps", type=int, default=128)
    parser.add_argument("--morlca-batch-size", type=int, default=128)
    parser.add_argument("--morlca-hidden-size", type=int, default=128)
    parser.add_argument("--morlca-aow-aux-coef", type=float, default=0.2)
    parser.add_argument("--morlca-eval-pref-bias-scale", type=float, default=0.75)

    parser.add_argument("--lcpo-total-timesteps", type=int, default=64)
    parser.add_argument("--lcpo-batch-size", type=int, default=32)
    parser.add_argument("--lcpo-hidden-size", type=int, default=32)
    parser.add_argument("--lcpo-action-bins", type=int, default=3)
    parser.add_argument("--lcpo-max-preferences", type=int, default=0)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _env_plan_namespace(env_key: str, args: argparse.Namespace) -> SimpleNamespace:
    spec = get_env_spec(env_key)
    return SimpleNamespace(
        env_name=spec.dynamic_env_name,
        seed=int(args.seed),
        shared_regime_eval_episodes=int(args.shared_regime_eval_episodes),
        shared_regime_seed_offset=int(args.shared_regime_seed_offset),
        shared_plan_json="",
        env_config_json="",
        episodes_per_regime=int(args.episodes_per_regime),
        regime_schedule=str(args.regime_schedule),
        fine_regime_clusters=int(args.fine_regime_clusters),
        fine_regime_catalog_episodes=int(args.fine_regime_catalog_episodes),
        fine_regime_context_steps=int(args.fine_regime_context_steps),
        chlor_episode_length=int(args.chlor_episode_length),
        chlor_regime_clusters=int(args.fine_regime_clusters),
        ev_disable_projection=False,
        ev_site="caltech",
        ev_periods=(),
        ev_moer_forecast_steps=36,
        cogen_renewables=(),
        cogen_forecast_horizon=3,
        cogen_forecast_noise_std=0.0,
        sustaingym_building_weathers=(),
    )


def materialize_shared_plan(env_key: str, args: argparse.Namespace, plan_path: Path) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] | None = None
    for candidate in PLAN_SOURCE_CANDIDATES.get(env_key, []):
        if not candidate.exists():
            continue
        try:
            candidate_payload = json.loads(candidate.read_text())
        except Exception:
            continue
        if isinstance(candidate_payload, dict) and isinstance(candidate_payload.get("shared_regime_seed_plan"), list):
            payload = {"shared_regime_seed_plan": candidate_payload["shared_regime_seed_plan"]}
            break
    if payload is None:
        plan, _episode_seeds = build_shared_regime_seed_plan(_env_plan_namespace(env_key, args))
        payload = {"shared_regime_seed_plan": plan}
    plan_path.write_text(json.dumps(payload, indent=2))


def _dynamic_command(env_key: str, args: argparse.Namespace, save_dir: Path, shared_plan_json: Path) -> list[str]:
    spec = get_env_spec(env_key)
    ref_point = ["0"] * int(spec.obj_num)
    command = [
        args.python_bin,
        "-m",
        "src.dynamic_morl.run",
        "--env-name",
        spec.dynamic_env_name,
        "--obj-num",
        str(int(spec.obj_num)),
        "--ref-point",
        *ref_point,
        "--auto-ref-point",
        "--num-time-steps",
        str(int(args.dynamic_num_time_steps)),
        "--num-init-steps",
        str(int(args.dynamic_num_init_steps)),
        "--num-steps",
        str(int(args.dynamic_num_steps)),
        "--num-processes",
        str(int(args.dynamic_num_processes)),
        "--ppo-epoch",
        str(int(args.dynamic_ppo_epoch)),
        "--num-mini-batch",
        str(int(args.dynamic_num_mini_batch)),
        "--entropy-coef",
        "0.0",
        "--num-select",
        str(int(args.dynamic_num_select)),
        "--delta-weight",
        str(float(args.dynamic_delta_weight)),
        "--eval-delta-weight",
        str(float(args.eval_delta_weight)),
        "--eval-num",
        str(int(args.dynamic_eval_num)),
        "--rl-eval-interval",
        str(int(args.dynamic_rl_eval_interval)),
        "--drift-window",
        "6",
        "--drift-steps",
        "4",
        "--strict-online-context",
        "--episodes-per-regime",
        str(int(args.episodes_per_regime)),
        "--regime-schedule",
        str(args.regime_schedule),
        "--trace-eval-samples",
        str(int(args.dynamic_trace_eval_samples)),
        "--regime-eval-samples",
        str(int(args.dynamic_regime_eval_samples)),
        "--shared-regime-eval-episodes",
        str(int(args.shared_regime_eval_episodes)),
        "--shared-regime-seed-offset",
        str(int(args.shared_regime_seed_offset)),
        "--shared-plan-json",
        str(shared_plan_json),
        "--fine-regime-clusters",
        str(int(args.fine_regime_clusters)),
        "--fine-regime-catalog-episodes",
        str(int(args.fine_regime_catalog_episodes)),
        "--fine-regime-context-steps",
        str(int(args.fine_regime_context_steps)),
        "--context-probe-samples",
        str(int(args.dynamic_context_probe_samples)),
        "--context-trace-steps",
        str(int(args.dynamic_context_trace_steps)),
        "--context-sample-mode",
        "salient_mix",
        "--context-saliency-alpha",
        "0.35",
        "--context-buffer-steps",
        str(int(args.dynamic_context_buffer_steps)),
        "--context-history-size",
        str(int(args.dynamic_context_history_size)),
        "--context-nearest-k",
        str(int(args.dynamic_context_nearest_k)),
        "--context-history-lambda",
        "0.10",
        "--context-forecast-lambda",
        "0.50",
        "--bank-enable",
        str(int(args.dynamic_bank_enable)),
        "--bank-num-slots",
        str(int(args.dynamic_bank_num_slots)),
        "--bank-slot-size",
        str(int(args.dynamic_bank_slot_size)),
        "--bank-merge-threshold",
        str(float(args.dynamic_bank_merge_threshold)),
        "--bank-query-slots",
        str(int(args.dynamic_bank_query_slots)),
        "--bank-query-topk",
        str(int(args.dynamic_bank_query_topk)),
        "--online-shift-threshold",
        str(float(args.dynamic_online_shift_threshold)),
        "--online-shift-min-gap",
        str(int(args.dynamic_online_shift_min_gap)),
        "--selection-method",
        "online-window",
        "--knee-lambda",
        str(float(args.dynamic_knee_lambda)),
        "--dynamic-lambda",
        str(float(args.dynamic_lambda)),
        "--resilience-lambda",
        str(float(args.dynamic_resilience_lambda)),
        "--diversity-lambda",
        str(float(args.dynamic_diversity_lambda)),
        "--shift-gap-lambda",
        str(float(args.dynamic_shift_gap_lambda)),
        "--repeat-topk",
        str(int(args.dynamic_repeat_topk)),
        "--final-archive-mode",
        str(args.dynamic_final_archive_mode),
        "--final-archive-max-samples",
        str(int(args.dynamic_final_archive_max_samples)),
        "--seed",
        str(int(args.seed)),
        "--save-dir",
        str(save_dir),
    ]
    if env_key == "chlor_alkali":
        command.extend(
            [
                "--chlor-episode-length",
                str(int(args.chlor_episode_length)),
                # The shared plan can contain more than the parser default of
                # 20 representative chlor-alkali regimes.  Keep the runtime
                # catalog aligned with that registered plan.
                "--chlor-regime-clusters",
                str(int(args.fine_regime_clusters)),
            ]
        )
    return command


def _baseline_common_args(args: argparse.Namespace, shared_plan_json: Path) -> list[str]:
    return [
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--trace-eval-episodes",
        str(int(args.trace_eval_episodes)),
        "--trace-recovery-window",
        str(int(args.trace_recovery_window)),
        "--eval-delta-weight",
        str(float(args.eval_delta_weight)),
        "--shared-regime-eval-episodes",
        str(int(args.shared_regime_eval_episodes)),
        "--shared-regime-seed-offset",
        str(int(args.shared_regime_seed_offset)),
        "--episodes-per-regime",
        str(int(args.episodes_per_regime)),
        "--regime-schedule",
        str(args.regime_schedule),
        "--fine-regime-clusters",
        str(int(args.fine_regime_clusters)),
        "--fine-regime-catalog-episodes",
        str(int(args.fine_regime_catalog_episodes)),
        "--fine-regime-context-steps",
        str(int(args.fine_regime_context_steps)),
        "--shared-plan-json",
        str(shared_plan_json),
    ]


def _method_command(method: str, env_key: str, args: argparse.Namespace, save_dir: Path, shared_plan_json: Path) -> list[str]:
    script_root = PROJECT_ROOT / "scripts"
    common = [
        args.python_bin,
        str(script_root / f"run_{method}_dynamic_baseline.py"),
        "--env-key",
        env_key,
        "--seed",
        str(int(args.seed)),
        *_baseline_common_args(args, shared_plan_json),
        "--save-dir",
        str(save_dir),
    ]
    if env_key == "chlor_alkali":
        common.extend(["--chlor-episode-length", str(int(args.chlor_episode_length))])
    if method == "capql":
        common.extend(
            [
                "--total-timesteps",
                str(int(args.capql_total_timesteps)),
                "--start-steps",
                str(int(args.capql_start_steps)),
                "--batch-size",
                str(int(args.capql_batch_size)),
                "--hidden-size",
                str(int(args.capql_hidden_size)),
            ]
        )
    elif method == "qpensieve":
        common.extend(
            [
                "--total-timesteps",
                str(int(args.qpensieve_total_timesteps)),
                "--start-steps",
                str(int(args.qpensieve_start_steps)),
                "--batch-size",
                str(int(args.qpensieve_batch_size)),
                "--prefer-num",
                str(int(args.qpensieve_prefer_num)),
                "--hidden-size",
                *[str(int(item)) for item in args.qpensieve_hidden_size],
            ]
        )
    elif method == "pgmorl":
        common.extend(
            [
                "--total-timesteps",
                str(int(args.pgmorl_total_timesteps)),
                "--num-steps",
                str(int(args.pgmorl_num_steps)),
                "--num-processes",
                "1",
                "--warmup-iter",
                str(int(args.pgmorl_warmup_iter)),
                "--update-iter",
                str(int(args.pgmorl_update_iter)),
                "--max-eval-policies",
                str(int(args.pgmorl_max_eval_policies)),
            ]
        )
    elif method == "morlca":
        common.extend(
            [
                "--total-timesteps",
                str(int(args.morlca_total_timesteps)),
                "--start-steps",
                str(int(args.morlca_start_steps)),
                "--batch-size",
                str(int(args.morlca_batch_size)),
                "--hidden-size",
                str(int(args.morlca_hidden_size)),
                "--aow-aux-coef",
                str(float(args.morlca_aow_aux_coef)),
                "--eval-pref-bias-scale",
                str(float(args.morlca_eval_pref_bias_scale)),
            ]
        )
    elif method == "lcpo":
        common.extend(
            [
                "--total-timesteps",
                str(int(args.lcpo_total_timesteps)),
                "--batch-size",
                str(int(args.lcpo_batch_size)),
                "--hidden-size",
                str(int(args.lcpo_hidden_size)),
                "--action-bins",
                str(int(args.lcpo_action_bins)),
                "--max-preferences",
                str(int(args.lcpo_max_preferences)),
            ]
        )
    else:
        raise ValueError(f"Unsupported method: {method}")
    return common


def _dynamic_complete(run_dir: Path, expected_regimes: int) -> bool:
    regime_path = run_dir / "regime_fronts" / "shared_final.json"
    summary_path = run_dir / "final" / "shared_eval_summary.json"
    if regime_path.exists() and summary_path.exists():
        try:
            payload = json.loads(regime_path.read_text())
        except Exception:
            return False
        regimes = list(payload.get("regimes", []))
        return len(regimes) >= int(expected_regimes)
    shared_dir = run_dir / "shared_regime_returns"
    return (
        (run_dir / "metrics_history.csv").exists()
        and (run_dir / "final" / "objs.txt").exists()
        and shared_dir.exists()
        and len(list(shared_dir.glob("*.json"))) >= int(expected_regimes)
    )


def _shared_regime_dir_complete(run_dir: Path, expected_regimes: int) -> bool:
    shared_dir = run_dir / "shared_regime_returns"
    return shared_dir.exists() and len(list(shared_dir.glob("*.json"))) >= int(expected_regimes)


def _baseline_complete(run_dir: Path, expected_regimes: int) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return _shared_regime_dir_complete(run_dir, expected_regimes)
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return _shared_regime_dir_complete(run_dir, expected_regimes)
    regimes = list(payload.get("regime_fronts", []))
    return len(regimes) >= int(expected_regimes) or _shared_regime_dir_complete(run_dir, expected_regimes)


def _ensure_run(method: str, env_key: str, command: list[str], run_dir: Path, args: argparse.Namespace) -> None:
    expected_regimes = int(args.shared_regime_eval_episodes)
    complete = _dynamic_complete(run_dir, expected_regimes) if method == "dynamic" else _baseline_complete(run_dir, expected_regimes)
    if complete and not args.force:
        print(f"skip existing: {run_dir}")
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    print("running:", " ".join(command))
    if args.dry_run:
        return
    timeout = int(args.dynamic_timeout_seconds) if method == "dynamic" and int(args.dynamic_timeout_seconds) > 0 else None
    try:
        completed = subprocess.run(command, check=False, cwd=PROJECT_ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        if method == "dynamic" and _dynamic_complete(run_dir, expected_regimes):
            print(f"accepted timed-out dynamic run with complete shared regime exports: {run_dir}")
            return
        raise
    if completed.returncode == 0:
        return
    if method == "dynamic" and _dynamic_complete(run_dir, expected_regimes):
        print(f"accepted non-zero dynamic exit after complete shared regime exports: {run_dir}")
        return
    raise subprocess.CalledProcessError(completed.returncode, command)


def _extract_solution_points(row: dict[str, Any]) -> np.ndarray:
    for key in ("solution_points", "points", "front_points"):
        value = row.get(key)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 2:
                return arr
    return np.zeros((0, 0), dtype=np.float64)


def _load_regime_rows(method: str, run_dir: Path) -> list[dict[str, Any]]:
    if method == "dynamic":
        regime_payload = run_dir / "regime_fronts" / "shared_final.json"
        if regime_payload.exists():
            payload = json.loads(regime_payload.read_text())
            rows = list(payload.get("regimes", []))
        else:
            rows = []
            for json_path in sorted((run_dir / "shared_regime_returns").glob("*.json")):
                try:
                    rows.append(json.loads(json_path.read_text()))
                except Exception:
                    continue
    else:
        rows = []
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text())
                rows = list(payload.get("regime_fronts", []))
            except Exception:
                rows = []
        if not rows:
            for json_path in sorted((run_dir / "shared_regime_returns").glob("*.json")):
                try:
                    rows.append(json.loads(json_path.read_text()))
                except Exception:
                    continue
    rows.sort(key=lambda row: int(row.get("regime_id", -1)))
    return rows


def _shared_trace_metrics_from_regime_rows(regime_rows: list[dict[str, Any]], eval_delta_weight: float) -> dict[str, float]:
    solution_rows = []
    for row in regime_rows:
        solutions = _extract_solution_points(row)
        if solutions.ndim == 2 and len(solutions) > 0:
            solution_rows.append(
                {
                    "regime_id": int(row.get("regime_id", -1)),
                    "solutions": solutions,
                }
            )
    solution_rows.sort(key=lambda row: int(row["regime_id"]))
    if len(solution_rows) < 2:
        return {
            "trace_shift_regret": float("nan"),
            "trace_recovery_latency": float("nan"),
            "trace_recovery_score": float("nan"),
            "trace_shift_count": 0.0,
            "trace_sample_count": 0.0,
        }
    blocks = [row["solutions"] for row in solution_rows]
    solution_count = min(block.shape[0] for block in blocks)
    if solution_count <= 0:
        return {
            "trace_shift_regret": float("nan"),
            "trace_recovery_latency": float("nan"),
            "trace_recovery_score": float("nan"),
            "trace_shift_count": 0.0,
            "trace_sample_count": 0.0,
        }
    regime_ids = np.arange(len(blocks), dtype=np.int64)
    trace_rows = []
    for solution_idx in range(solution_count):
        seq = np.stack([block[solution_idx] for block in blocks], axis=0)
        trace_rows.append(
            compute_trace_shift_metrics(
                {
                    "obj": seq,
                    "regime_id": regime_ids,
                },
                eval_delta_weight=eval_delta_weight,
                recovery_window=REGIME_TRACE_WINDOW,
                use_regime_id=True,
                min_shift_gap=1,
            )
        )
    valid_rows = [row for row in trace_rows if float(row.get("trace_shift_count", 0.0)) > 0.0]
    if not valid_rows:
        return {
            "trace_shift_regret": float("nan"),
            "trace_recovery_latency": float("nan"),
            "trace_recovery_score": float("nan"),
            "trace_shift_count": 0.0,
            "trace_sample_count": float(len(trace_rows)),
        }
    return {
        "trace_shift_regret": float(np.mean([row["trace_shift_regret"] for row in valid_rows])),
        "trace_recovery_latency": float(np.mean([row["trace_recovery_latency"] for row in valid_rows])),
        "trace_recovery_score": float(np.mean([row["trace_recovery_score"] for row in valid_rows])),
        "trace_shift_count": float(np.mean([row["trace_shift_count"] for row in valid_rows])),
        "trace_sample_count": float(len(valid_rows)),
    }


def _load_dynamic_online_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics_history.csv"
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    online_df = df[df["stage"].astype(str) != "shared_final"]
    if online_df.empty:
        return {}
    row = online_df.iloc[-1]
    metrics: dict[str, float] = {}
    for key in (
        "trace_shift_regret",
        "trace_recovery_latency",
        "trace_recovery_score",
        "trace_shift_count",
        "online_shift_count",
        "matched_recovery",
    ):
        if key in row and pd.notna(row[key]):
            metrics[f"online_{key}"] = float(row[key])
    return metrics


def _load_frozen_trace_metrics(method: str, run_dir: Path) -> dict[str, float]:
    if method == "dynamic":
        summary_path = run_dir / "final" / "shared_eval_summary.json"
    else:
        summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text())
    except Exception:
        return {}
    trace_metrics = payload.get("trace_metrics", {})
    if not isinstance(trace_metrics, dict):
        return {}
    metrics: dict[str, float] = {}
    for key in (
        "trace_shift_regret",
        "trace_recovery_latency",
        "trace_recovery_score",
        "trace_shift_count",
        "trace_pre_post_gap",
    ):
        value = trace_metrics.get(key)
        if value is None:
            continue
        try:
            metrics[f"frozen_{key}"] = float(value)
        except Exception:
            continue
    return metrics


def _display_method_name(method: str) -> str:
    if method == "dynamic":
        return "OW-CMORL"
    if method == "qpensieve":
        return "Q-Pensieve"
    if method == "morlca":
        return "MORL-CA"
    return method.upper()


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.4f}"


def write_report(df: pd.DataFrame, out_md: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Quick Adaptation Validation",
        "",
        "This smoke benchmark uses a single seed, a unified 20-regime shared plan per environment, and low training budgets.",
        "",
        "## Budget",
        "",
        f"- OW-CMORL: `num_time_steps={int(args.dynamic_num_time_steps)}`",
        f"- CAPQL / Q-Pensieve / MORL-CA: `total_timesteps={int(args.capql_total_timesteps)}` / `{int(args.qpensieve_total_timesteps)}` / `{int(args.morlca_total_timesteps)}`",
        f"- PGMORL: `total_timesteps={int(args.pgmorl_total_timesteps)}`",
        f"- LCPO: `total_timesteps_per_preference={int(args.lcpo_total_timesteps)}`",
        "",
        "Shared `trace_*` below are reconstructed from the exported shared-plan `solution_points`, so every method uses the same regime-level trace logic.",
        "Frozen `trace_*` come from each method's own final dynamic-evaluation summary, i.e. a fixed trained policy or archive replayed over the dynamic trace without online parameter updates.",
        "For `OW-CMORL`, the `online_*` columns come from the last non-`shared_final` row in `metrics_history.csv`; they are the only rows that still reflect the snapshot retrieval / local refresh loop itself.",
        "",
    ]
    for env_key in args.env_keys:
        sub = df[df["env_key"] == env_key].copy()
        if sub.empty:
            continue
        lines.append(f"## {env_key}")
        lines.append("")
        lines.append("| Method | shared_trace_shift_regret | shared_trace_recovery_latency | shared_trace_recovery_score | frozen_trace_shift_regret | frozen_trace_recovery_latency | frozen_trace_recovery_score | online_trace_shift_regret | online_trace_recovery_latency | online_trace_recovery_score |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in sub.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        _format_value(float(row.get("shared_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("shared_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("shared_trace_recovery_score", math.nan))),
                        _format_value(float(row.get("frozen_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("frozen_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("frozen_trace_recovery_score", math.nan))),
                        _format_value(float(row.get("online_trace_shift_regret", math.nan))),
                        _format_value(float(row.get("online_trace_recovery_latency", math.nan))),
                        _format_value(float(row.get("online_trace_recovery_score", math.nan))),
                    ]
                )
                + " |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    run_root = args.results_root / args.run_name
    plan_root = run_root / "plans"
    plan_root.mkdir(parents=True, exist_ok=True)

    run_dirs: dict[tuple[str, str], Path] = {}

    for env_key in args.env_keys:
        shared_plan_json = plan_root / f"{env_key}_shared_plan.json"
        materialize_shared_plan(env_key, args, shared_plan_json)
        for method in args.methods:
            run_dir = run_root / method / env_key
            run_dirs[(method, env_key)] = run_dir
            if method == "dynamic":
                command = _dynamic_command(env_key, args, run_dir, shared_plan_json)
            else:
                command = _method_command(method, env_key, args, run_dir, shared_plan_json)
            _ensure_run(method, env_key, command, run_dir, args)

    if args.dry_run:
        return

    rows: list[dict[str, Any]] = []
    for env_key in args.env_keys:
        for method in args.methods:
            run_dir = run_dirs[(method, env_key)]
            if not run_dir.exists():
                continue
            regime_rows = _load_regime_rows(method, run_dir)
            shared_metrics = _shared_trace_metrics_from_regime_rows(regime_rows, float(args.eval_delta_weight))
            frozen_metrics = _load_frozen_trace_metrics(method, run_dir)
            row: dict[str, Any] = {
                "env_key": env_key,
                "method": _display_method_name(method),
                "raw_method": method,
                "run_dir": str(run_dir),
                "shared_trace_shift_regret": shared_metrics["trace_shift_regret"],
                "shared_trace_recovery_latency": shared_metrics["trace_recovery_latency"],
                "shared_trace_recovery_score": shared_metrics["trace_recovery_score"],
                "shared_trace_shift_count": shared_metrics["trace_shift_count"],
                "shared_trace_sample_count": shared_metrics["trace_sample_count"],
                "frozen_trace_shift_regret": frozen_metrics.get("frozen_trace_shift_regret", float("nan")),
                "frozen_trace_recovery_latency": frozen_metrics.get("frozen_trace_recovery_latency", float("nan")),
                "frozen_trace_recovery_score": frozen_metrics.get("frozen_trace_recovery_score", float("nan")),
                "frozen_trace_shift_count": frozen_metrics.get("frozen_trace_shift_count", float("nan")),
                "frozen_trace_pre_post_gap": frozen_metrics.get("frozen_trace_pre_post_gap", float("nan")),
            }
            if method == "dynamic":
                online_metrics = _load_dynamic_online_metrics(run_dir)
                row["online_trace_shift_regret"] = online_metrics.get("online_trace_shift_regret", float("nan"))
                row["online_trace_recovery_latency"] = online_metrics.get("online_trace_recovery_latency", float("nan"))
                row["online_trace_recovery_score"] = online_metrics.get("online_trace_recovery_score", float("nan"))
                row["online_trace_shift_count"] = online_metrics.get("online_trace_shift_count", float("nan"))
                row["online_shift_count"] = online_metrics.get("online_online_shift_count", online_metrics.get("online_shift_count", float("nan")))
                row["online_matched_recovery"] = online_metrics.get("online_matched_recovery", float("nan"))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = run_root / "quick_adaptation_validation_summary.csv"
    md_path = run_root / "quick_adaptation_validation_summary.md"
    df.to_csv(csv_path, index=False)
    write_report(df, md_path, args)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
