from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-keys", nargs="+", default=["building", "evcharging", "cogen"])
    parser.add_argument("--prefix", type=str, default="expertbank")
    parser.add_argument("--bank-size", type=int, default=12)
    parser.add_argument("--router-resilience-lambda", type=float, default=0.15)
    parser.add_argument("--router-regret-lambda", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for env_key in args.env_keys:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_dynamic_expert_bank.py"),
            "--env-key",
            env_key,
            "--prefix",
            args.prefix,
            "--bank-size",
            str(args.bank_size),
            "--router-resilience-lambda",
            str(args.router_resilience_lambda),
            "--router-regret-lambda",
            str(args.router_regret_lambda),
        ]
        print("running:", " ".join(command))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
