from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPECTED_SHARED_PROTOCOL_REGIMES = 20


ENV_CONFIGS = {
    "building": {
        "env_name": "building_3d_dynamic",
        "obj_num": 3,
        "ref_point": ["0", "0", "0"],
        "extra_args": ["--auto-ref-point"],
    },
    "evcharging": {
        "env_name": "evcharging_dynamic",
        "obj_num": 3,
        "ref_point": ["0", "0", "0"],
        "extra_args": ["--auto-ref-point"],
    },
    "cogen": {
        "env_name": "cogen_dynamic",
        "obj_num": 4,
        "ref_point": ["0", "0", "0", "0"],
        "extra_args": ["--auto-ref-point"],
    },
    "chlor_alkali": {
        "env_name": "chlor_alkali_dynamic",
        "obj_num": 3,
        "ref_point": ["0", "0", "0"],
        "extra_args": ["--auto-ref-point"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=sorted(ENV_CONFIGS.keys()), required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["dynamic", "static", "random", "no_dyn_ablation"],
        choices=["dynamic", "static", "random", "no_dyn_ablation"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--prefix", type=str, default="suite")
    parser.add_argument("--num-time-steps", type=int, default=2048)
    parser.add_argument("--num-init-steps", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--ppo-epoch", type=int, default=2)
    parser.add_argument("--num-mini-batch", type=int, default=2)
    parser.add_argument("--num-select", type=int, default=4)
    parser.add_argument("--eval-num", type=int, default=4)
    parser.add_argument("--rl-eval-interval", type=int, default=10)
    parser.add_argument("--ob-rms", action="store_true")
    parser.add_argument("--obj-rms", action="store_true")
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--knee-lambda", type=float, default=0.3)
    parser.add_argument("--dynamic-lambda", type=float, default=0.75)
    parser.add_argument("--resilience-lambda", type=float, default=1.75)
    parser.add_argument("--diversity-lambda", type=float, default=0.10)
    parser.add_argument("--shift-gap-lambda", type=float, default=1.0)
    parser.add_argument("--repeat-topk", type=int, default=2)
    parser.add_argument("--drift-window", type=int, default=6)
    parser.add_argument("--drift-steps", type=int, default=4)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--trace-eval-samples", type=int, default=2)
    parser.add_argument("--regime-eval-samples", type=int, default=2)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--context-probe-samples", type=int, default=3)
    parser.add_argument("--context-trace-steps", type=int, default=6)
    parser.add_argument("--context-sample-mode", type=str, default="salient_mix", choices=["tail", "uniform", "salient_mix"])
    parser.add_argument("--context-saliency-alpha", type=float, default=0.35)
    parser.add_argument("--context-buffer-steps", type=int, default=48)
    parser.add_argument("--context-history-size", type=int, default=10)
    parser.add_argument("--context-nearest-k", type=int, default=3)
    parser.add_argument("--context-encoder-enable", type=int, default=1)
    parser.add_argument("--context-history-lambda", type=float, default=0.10)
    parser.add_argument("--context-forecast-lambda", type=float, default=0.50)
    parser.add_argument("--bank-enable", type=int, default=1)
    parser.add_argument("--bank-num-slots", type=int, default=6)
    parser.add_argument("--bank-slot-size", type=int, default=3)
    parser.add_argument("--bank-merge-threshold", type=float, default=0.82)
    parser.add_argument("--bank-query-slots", type=int, default=3)
    parser.add_argument("--bank-query-topk", type=int, default=2)
    parser.add_argument("--online-shift-threshold", type=float, default=0.6)
    parser.add_argument("--online-shift-min-gap", type=int, default=6)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--ev-disable-projection", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_base_command(env_key: str, args: argparse.Namespace) -> list[str]:
    cfg = ENV_CONFIGS[env_key]
    command = [
        sys.executable,
        "-m",
        "src.dynamic_morl.run",
        "--env-name",
        cfg["env_name"],
        "--obj-num",
        str(cfg["obj_num"]),
        "--ref-point",
        *cfg["ref_point"],
        *cfg["extra_args"],
        "--num-time-steps",
        str(args.num_time_steps),
        "--num-init-steps",
        str(args.num_init_steps),
        "--num-steps",
        str(args.num_steps),
        "--num-processes",
        str(args.num_processes),
        "--ppo-epoch",
        str(args.ppo_epoch),
        "--num-mini-batch",
        str(args.num_mini_batch),
        "--entropy-coef",
        str(args.entropy_coef),
        "--num-select",
        str(args.num_select),
        "--delta-weight",
        str(args.delta_weight),
        "--eval-delta-weight",
        str(args.eval_delta_weight),
        "--eval-num",
        str(args.eval_num),
        "--rl-eval-interval",
        str(args.rl_eval_interval),
        "--drift-window",
        str(args.drift_window),
        "--drift-steps",
        str(args.drift_steps),
        "--strict-online-context",
        "--episodes-per-regime",
        str(args.episodes_per_regime),
        "--regime-schedule",
        args.regime_schedule,
        "--trace-eval-samples",
        str(args.trace_eval_samples),
        "--regime-eval-samples",
        str(args.regime_eval_samples),
        "--shared-regime-eval-episodes",
        str(args.shared_regime_eval_episodes),
        "--shared-regime-seed-offset",
        str(args.shared_regime_seed_offset),
        "--fine-regime-clusters",
        str(args.fine_regime_clusters),
        "--fine-regime-catalog-episodes",
        str(args.fine_regime_catalog_episodes),
        "--fine-regime-context-steps",
        str(args.fine_regime_context_steps),
        "--context-probe-samples",
        str(args.context_probe_samples),
        "--context-trace-steps",
        str(args.context_trace_steps),
        "--context-sample-mode",
        args.context_sample_mode,
        "--context-saliency-alpha",
        str(args.context_saliency_alpha),
        "--context-buffer-steps",
        str(args.context_buffer_steps),
        "--context-history-size",
        str(args.context_history_size),
        "--context-nearest-k",
        str(args.context_nearest_k),
        "--context-encoder-enable",
        str(args.context_encoder_enable),
        "--context-history-lambda",
        str(args.context_history_lambda),
        "--context-forecast-lambda",
        str(args.context_forecast_lambda),
        "--bank-enable",
        str(args.bank_enable),
        "--bank-num-slots",
        str(args.bank_num_slots),
        "--bank-slot-size",
        str(args.bank_slot_size),
        "--bank-merge-threshold",
        str(args.bank_merge_threshold),
        "--bank-query-slots",
        str(args.bank_query_slots),
        "--bank-query-topk",
        str(args.bank_query_topk),
        "--online-shift-threshold",
        str(args.online_shift_threshold),
        "--online-shift-min-gap",
        str(args.online_shift_min_gap),
    ]
    if env_key == "evcharging" and args.ev_disable_projection:
        command.append("--ev-disable-projection")
    if env_key == "chlor_alkali":
        command.extend(["--chlor-episode-length", str(args.chlor_episode_length)])
    if args.ob_rms:
        command.append("--ob-rms")
    if args.obj_rms:
        command.append("--obj-rms")
    return command

def build_config_args(config_name: str, args: argparse.Namespace) -> list[str]:
    if config_name == "dynamic":
        return [
            "--selection-method",
            "online-window",
            "--knee-lambda",
            str(args.knee_lambda),
            "--dynamic-lambda",
            str(args.dynamic_lambda),
            "--resilience-lambda",
            str(args.resilience_lambda),
            "--diversity-lambda",
            str(args.diversity_lambda),
            "--shift-gap-lambda",
            str(args.shift_gap_lambda),
            "--repeat-topk",
            str(args.repeat_topk),
        ]
    if config_name == "static":
        return ["--selection-method", "crowding"]
    if config_name == "random":
        return ["--selection-method", "random"]
    if config_name == "no_dyn_ablation":
        return [
            "--selection-method",
            "online-window",
            "--knee-lambda",
            "0.0",
            "--dynamic-lambda",
            "0.0",
            "--resilience-lambda",
            "0.0",
            "--context-forecast-lambda",
            "0.0",
            "--context-history-lambda",
            "0.0",
            "--bank-enable",
            "0",
            "--bank-query-slots",
            "0",
            "--shift-gap-lambda",
            "0.0",
            "--repeat-topk",
            "0",
        ]
    raise ValueError(f"unknown config: {config_name}")


def run_complete(run_dir: Path, env_key: str) -> bool:
    if not run_dir.exists():
        return False
    regime_path = run_dir / "regime_fronts" / "shared_final.json"
    summary_path = run_dir / "final" / "shared_eval_summary.json"
    shared_dir = run_dir / "shared_regime_returns"
    if not regime_path.exists() or not summary_path.exists() or not shared_dir.exists():
        return False
    try:
        payload = json.loads(regime_path.read_text())
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    regimes = payload.get("regimes", [])
    unique_ids = {int(row.get("regime_id", -1)) for row in regimes}
    json_count = len(list(shared_dir.glob("*.json")))
    npz_count = len(list(shared_dir.glob("*.npz")))
    shared_episode_seeds = [int(seed) for seed in summary.get("shared_eval_episode_seeds", [])]
    shared_plan = summary.get("shared_regime_seed_plan", [])
    planned_episode_seeds = [
        int(seed)
        for plan_row in shared_plan
        for seed in plan_row.get("episode_seeds", [])
    ]
    return (
        len(regimes) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(unique_ids) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and json_count >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and npz_count >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(shared_plan) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == EXPECTED_SHARED_PROTOCOL_REGIMES
    )


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)

    base_command = build_base_command(args.env_key, args)
    for seed in args.seeds:
        for config_name in args.configs:
            run_dir = args.results_root / f"{args.prefix}_{args.env_key}_{config_name}_seed{seed}"
            complete = run_complete(run_dir, args.env_key)
            if complete and not args.force:
                print(f"skip existing: {run_dir}")
                continue
            if run_dir.exists() and not complete:
                shutil.rmtree(run_dir)
            if run_dir.exists() and args.force:
                shutil.rmtree(run_dir)

            command = [
                *base_command,
                *build_config_args(config_name, args),
                "--seed",
                str(seed),
                "--save-dir",
                str(run_dir),
            ]
            print("running:", " ".join(command))
            if args.dry_run:
                continue
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
