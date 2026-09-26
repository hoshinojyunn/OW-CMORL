from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"


COMMON_ARGS = [
    "--env-name",
    "building_3d_dynamic",
    "--obj-num",
    "3",
    "--ref-point",
    "0",
    "0",
    "0",
    "--num-time-steps",
    "1024",
    "--num-init-steps",
    "512",
    "--num-steps",
    "8",
    "--num-processes",
    "1",
    "--ppo-epoch",
    "2",
    "--num-mini-batch",
    "2",
    "--num-select",
    "2",
    "--delta-weight",
    "0.5",
    "--eval-delta-weight",
    "0.5",
    "--eval-num",
    "1",
    "--drift-window",
    "4",
    "--drift-steps",
    "2",
    "--strict-online-context",
    "--episodes-per-regime",
    "1",
    "--regime-schedule",
    "cyclic",
    "--context-trace-steps",
    "4",
    "--context-buffer-steps",
    "32",
    "--context-history-lambda",
    "0.15",
    "--context-forecast-lambda",
    "0.35",
]


CONFIGS = {
    "dynamic": [
        "--selection-method",
        "online-window",
        "--knee-lambda",
        "0.3",
        "--dynamic-lambda",
        "0.5",
        "--resilience-lambda",
        "1.5",
        "--shift-gap-lambda",
        "1.0",
    ],
    "static": [
        "--selection-method",
        "crowding",
    ],
    "random": [
        "--selection-method",
        "random",
    ],
    "no_dyn_ablation": [
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
        "--shift-gap-lambda",
        "0.0",
        "--repeat-topk",
        "0",
    ],
}


def build_command(config_name: str, seed: int, save_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.dynamic_morl.run",
        *COMMON_ARGS,
        *CONFIGS[config_name],
        "--seed",
        str(seed),
        "--save-dir",
        str(save_dir),
    ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["dynamic", "static", "random", "no_dyn_ablation"],
        choices=sorted(CONFIGS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--prefix", type=str, default="smoke")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for config_name in args.configs:
            run_dir = RESULTS_ROOT / f"{args.prefix}_{config_name}_seed{seed}"
            if run_dir.exists() and not args.force:
                print(f"skip existing: {run_dir}")
                continue

            command = build_command(config_name, seed, run_dir)
            print("running:", " ".join(command))
            if args.dry_run:
                continue
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
