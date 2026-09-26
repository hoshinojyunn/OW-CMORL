from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = Path("/root/anaconda3/envs/morl-pareto/bin/python")
PYTHON_BIN = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable))
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "results" / "scalarized_baselines"
DEFAULT_GROUPS = [
    ["a2c", "acer"],
    ["acktr", "ddpg"],
    ["deepq"],
    ["ppo1", "ppo2", "trpo_mpi"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=["building", "evcharging", "cogen", "chlor_alkali"], required=True)
    parser.add_argument("--total-timesteps", type=int, default=256)
    parser.add_argument("--trace-eval-episodes", type=int, default=4)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--ev-moer-forecast-steps", type=int, default=36)
    parser.add_argument("--ev-project-action-in-env", type=int, default=1)
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=1,
        help="Maximum number of concurrently active algorithm groups. All groups are still processed.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def launch_group(args: argparse.Namespace, idx: int, algs: list[str]) -> tuple[str, subprocess.Popen[bytes], object]:
    log_dir = args.save_root.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.env_key}_baseline_parallel_group{idx}.log"
    command = [
        PYTHON_BIN,
        str(PROJECT_ROOT / "scripts" / "run_baseline_sweep.py"),
        "--env-key",
        args.env_key,
        "--algs",
        *algs,
        "--seeds",
        "0",
        "--total-timesteps",
        str(args.total_timesteps),
        "--trace-eval-episodes",
        str(args.trace_eval_episodes),
        "--shared-regime-eval-episodes",
        str(args.shared_regime_eval_episodes),
        "--shared-regime-seed-offset",
        str(args.shared_regime_seed_offset),
        "--ev-moer-forecast-steps",
        str(args.ev_moer_forecast_steps),
        "--ev-project-action-in-env",
        str(args.ev_project_action_in_env),
        "--fine-regime-clusters",
        str(args.fine_regime_clusters),
        "--fine-regime-catalog-episodes",
        str(args.fine_regime_catalog_episodes),
        "--fine-regime-context-steps",
        str(args.fine_regime_context_steps),
        "--save-root",
        str(args.save_root),
    ]
    if args.env_key == "chlor_alkali":
        command.extend(["--chlor-episode-length", str(args.chlor_episode_length)])
    if args.force:
        command.append("--force")
    if args.continue_on_error:
        command.append("--continue-on-error")
    print("launch:", " ".join(command))
    fp = open(log_path, "wb")
    proc = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT)
    return " ".join(algs), proc, fp


def main() -> None:
    args = parse_args()
    max_groups = max(1, int(args.max_groups))
    pending = list(enumerate(DEFAULT_GROUPS, start=1))
    active: list[tuple[str, subprocess.Popen[bytes], object]] = []
    failures = []
    while pending or active:
        while pending and len(active) < max_groups:
            idx, algs = pending.pop(0)
            active.append(launch_group(args, idx, algs))
        if not active:
            break
        retired = False
        for idx, (label, proc, fp) in enumerate(active):
            ret = proc.poll()
            if ret is None:
                continue
            fp.close()
            if ret != 0:
                failures.append({"group": label, "returncode": ret})
            active.pop(idx)
            retired = True
            break
        if not retired:
            time.sleep(1.0)

    if failures:
        raise SystemExit(f"parallel baseline groups failed: {failures}")


if __name__ == "__main__":
    main()
