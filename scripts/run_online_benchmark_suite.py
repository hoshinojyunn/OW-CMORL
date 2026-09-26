from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-keys", nargs="+", default=["building", "evcharging", "cogen"], choices=["building", "evcharging", "cogen", "chlor_alkali"])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--dynamic-prefix", type=str, default="owcmorl_online_longrun")
    parser.add_argument("--morl-suite-name", type=str, default="morl_online_suite")
    parser.add_argument("--dynamic-configs", nargs="+", default=["dynamic", "static", "random", "no_dyn_ablation"])
    parser.add_argument("--num-time-steps", type=int, default=4096)
    parser.add_argument("--num-init-steps", type=int, default=2048)
    parser.add_argument("--num-select", type=int, default=4)
    parser.add_argument("--eval-num", type=int, default=4)
    parser.add_argument("--rl-eval-interval", type=int, default=10)
    parser.add_argument("--trace-eval-samples", type=int, default=2)
    parser.add_argument("--regime-eval-samples", type=int, default=2)
    parser.add_argument("--bank-enable", type=int, default=1)
    parser.add_argument("--bank-num-slots", type=int, default=6)
    parser.add_argument("--bank-slot-size", type=int, default=3)
    parser.add_argument("--bank-merge-threshold", type=float, default=0.82)
    parser.add_argument("--bank-query-slots", type=int, default=3)
    parser.add_argument("--bank-query-topk", type=int, default=2)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--ev-disable-projection", action="store_true")
    parser.add_argument("--capql-total-timesteps", type=int, default=2048)
    parser.add_argument("--qpensieve-total-timesteps", type=int, default=2048)
    parser.add_argument("--pgmorl-total-timesteps", type=int, default=4096)
    parser.add_argument("--pgmorl-num-steps", type=int, default=128)
    parser.add_argument("--pgmorl-warmup-iter", type=int, default=8)
    parser.add_argument("--pgmorl-update-iter", type=int, default=3)
    parser.add_argument("--pgmorl-num-processes", type=int, default=1)
    parser.add_argument("--baseline-eval-episodes", type=int, default=3)
    parser.add_argument("--baseline-trace-eval-episodes", type=int, default=3)
    parser.add_argument("--baseline-trace-recovery-window", type=int, default=12)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def maybe_run(command: list[str], *, dry_run: bool, continue_on_error: bool) -> None:
    print("running:", " ".join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        if not continue_on_error:
            raise


def main() -> None:
    args = parse_args()
    for env_key in args.env_keys:
        dynamic_command = [
            sys.executable,
            str(SCRIPT_ROOT / "run_multienv_dynamic_suite.py"),
            "--env-key",
            env_key,
            "--results-root",
            str(args.results_root),
            "--configs",
            *args.dynamic_configs,
            "--prefix",
            args.dynamic_prefix,
            "--num-time-steps",
            str(args.num_time_steps),
            "--num-init-steps",
            str(args.num_init_steps),
            "--num-select",
            str(args.num_select),
            "--eval-num",
            str(args.eval_num),
            "--rl-eval-interval",
            str(args.rl_eval_interval),
            "--trace-eval-samples",
            str(args.trace_eval_samples),
            "--regime-eval-samples",
            str(args.regime_eval_samples),
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
            "--delta-weight",
            str(args.delta_weight),
            "--eval-delta-weight",
            str(args.eval_delta_weight),
            "--episodes-per-regime",
            str(args.episodes_per_regime),
            "--regime-schedule",
            args.regime_schedule,
            "--shared-regime-eval-episodes",
            str(args.shared_regime_eval_episodes),
            "--shared-regime-seed-offset",
            str(args.shared_regime_seed_offset),
        ]
        if env_key == "chlor_alkali":
            dynamic_command.extend(["--chlor-episode-length", str(args.chlor_episode_length)])
        if args.ev_disable_projection:
            dynamic_command.append("--ev-disable-projection")
        if args.force:
            dynamic_command.append("--force")
        maybe_run(dynamic_command, dry_run=args.dry_run, continue_on_error=args.continue_on_error)

    morl_command = [
        sys.executable,
        str(SCRIPT_ROOT / "run_morl_baseline_suite.py"),
        "--env-keys",
        *args.env_keys,
        "--suite-name",
        args.morl_suite_name,
        "--save-root",
        str(args.results_root / "morl_dynamic_baselines"),
        "--episodes-per-regime",
        str(args.episodes_per_regime),
        "--regime-schedule",
        args.regime_schedule,
        "--chlor-episode-length",
        str(args.chlor_episode_length),
        "--eval-episodes",
        str(args.baseline_eval_episodes),
        "--trace-eval-episodes",
        str(args.baseline_trace_eval_episodes),
        "--trace-recovery-window",
        str(args.baseline_trace_recovery_window),
        "--shared-regime-eval-episodes",
        str(args.shared_regime_eval_episodes),
        "--shared-regime-seed-offset",
        str(args.shared_regime_seed_offset),
        "--capql-total-timesteps",
        str(args.capql_total_timesteps),
        "--qpensieve-total-timesteps",
        str(args.qpensieve_total_timesteps),
        "--pgmorl-total-timesteps",
        str(args.pgmorl_total_timesteps),
        "--pgmorl-num-steps",
        str(args.pgmorl_num_steps),
        "--pgmorl-warmup-iter",
        str(args.pgmorl_warmup_iter),
        "--pgmorl-update-iter",
        str(args.pgmorl_update_iter),
        "--pgmorl-num-processes",
        str(args.pgmorl_num_processes),
    ]
    if args.force:
        morl_command.append("--force")
    if args.continue_on_error:
        morl_command.append("--continue-on-error")
    maybe_run(morl_command, dry_run=args.dry_run, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    main()
