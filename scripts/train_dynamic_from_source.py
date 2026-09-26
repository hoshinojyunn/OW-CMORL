from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ONLY_SOURCE_FLAGS = {"--shared-plan-json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain OW-CMORL with source-run hyperparameters and env overrides.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--env-config-json", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional OW-CMORL CLI arguments appended after the source-run configuration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_source_args(source_run_dir: Path) -> list[str]:
    args_path = source_run_dir / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"missing args.txt: {args_path}")
    raw_args = ast.literal_eval(args_path.read_text())
    if not isinstance(raw_args, list):
        raise ValueError(f"unexpected args.txt payload in {args_path}")
    return [str(item) for item in raw_args]


def _replace_or_append(args: list[str], flag: str, values: list[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    replaced = False
    while idx < len(args):
        token = args[idx]
        if token == flag:
            replaced = True
            out.extend([flag, *values])
            idx += 1
            while idx < len(args) and not str(args[idx]).startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    if not replaced:
        out.extend([flag, *values])
    return out


def _strip_evaluation_only_args(args: list[str]) -> list[str]:
    """Drop flags written by older evaluation wrappers, not the trainer."""
    out: list[str] = []
    idx = 0
    while idx < len(args):
        if args[idx] in EVALUATION_ONLY_SOURCE_FLAGS:
            idx += 2
            continue
        out.append(args[idx])
        idx += 1
    return out


def main() -> None:
    args = parse_args()
    cmd_args = _load_source_args(args.source_run_dir)
    cmd_args = _strip_evaluation_only_args(cmd_args)
    cmd_args = _replace_or_append(cmd_args, "--save-dir", [str(args.target_run_dir)])
    if args.env_config_json is not None:
        cmd_args = _replace_or_append(cmd_args, "--env-config-json", [str(args.env_config_json)])
    cmd_args.extend(str(value) for value in args.extra_args)
    command = [
        args.python_bin,
        "-m",
        "src.dynamic_morl.run",
        *cmd_args,
    ]
    print("running:", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
