from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results_ood"
DEFAULT_REPORT_MD = PROJECT_ROOT / "OOD_EXPERIMENT_RESULTS_partial.md"
DEFAULT_REPORT_CSV = PROJECT_ROOT / "analysis" / "ood_results_summary_partial.csv"
DEFAULT_FINAL_REPORT_MD = PROJECT_ROOT / "OOD_EXPERIMENT_RESULTS_ZH.md"
DEFAULT_FINAL_REPORT_CSV = PROJECT_ROOT / "analysis" / "ood_results_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the remaining OOD workload serially to avoid overloading the local machine."
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--profile", type=str, default="ood_v1")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--final-report-md", type=Path, default=DEFAULT_FINAL_REPORT_MD)
    parser.add_argument("--final-report-csv", type=Path, default=DEFAULT_FINAL_REPORT_CSV)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("running:", " ".join(str(token) for token in command), flush=True)
    subprocess.run([str(token) for token in command], check=True)


def _is_process_running(needle: str) -> bool:
    proc = subprocess.run(
        ["ps", "-eo", "cmd"],
        check=True,
        capture_output=True,
        text=True,
    )
    return any(needle in line for line in proc.stdout.splitlines())


def _wait_for_completion(
    *,
    target_file: Path,
    running_needle: str,
    launch_command: list[str],
    poll_seconds: int,
) -> None:
    if target_file.exists():
        print(f"already complete: {target_file}", flush=True)
        return
    while True:
        if target_file.exists():
            print(f"completed: {target_file}", flush=True)
            return
        if _is_process_running(running_needle):
            print(f"waiting for active process: {running_needle}", flush=True)
            time.sleep(max(5, int(poll_seconds)))
            continue
        print(f"target missing and no active process found; launching: {target_file}", flush=True)
        _run(launch_command)
        if target_file.exists():
            print(f"completed after launch: {target_file}", flush=True)
            return
        print(f"target still missing after launch, retrying poll: {target_file}", flush=True)
        time.sleep(max(5, int(poll_seconds)))


def _refresh_report(args: argparse.Namespace) -> None:
    _run(
        [
            args.python_bin,
            str(SCRIPT_ROOT / "generate_ood_report.py"),
            "--profile",
            args.profile,
            "--results-root",
            str(args.results_root),
            "--out-md",
            str(args.report_md),
            "--out-csv",
            str(args.report_csv),
        ]
    )


def _refresh_final_report(args: argparse.Namespace) -> None:
    _run(
        [
            args.python_bin,
            str(SCRIPT_ROOT / "generate_ood_report.py"),
            "--profile",
            args.profile,
            "--results-root",
            str(args.results_root),
            "--out-md",
            str(args.final_report_md),
            "--out-csv",
            str(args.final_report_csv),
        ]
    )


def _run_benchmark_step(args: argparse.Namespace, *, env_key: str, method: str, force: bool) -> None:
    command = [
        args.python_bin,
        str(SCRIPT_ROOT / "run_ood_benchmark.py"),
        "--env-keys",
        env_key,
        "--methods",
        method,
        "--profile",
        args.profile,
        "--python-bin",
        args.python_bin,
    ]
    if force:
        command.append("--force")
    _run(command)
    _refresh_report(args)


def main() -> None:
    args = parse_args()

    ev_target = args.results_root / args.profile / "dynamic" / "evcharging" / "final" / "shared_eval_summary.json"
    ev_target_dir = args.results_root / args.profile / "dynamic" / "evcharging"
    ev_needle = f"--target-run-dir {ev_target_dir}"
    ev_launch = [
        args.python_bin,
        str(SCRIPT_ROOT / "run_ood_benchmark.py"),
        "--env-keys",
        "evcharging",
        "--methods",
        "dynamic",
        "--profile",
        args.profile,
        "--python-bin",
        args.python_bin,
        "--force",
    ]
    _wait_for_completion(
        target_file=ev_target,
        running_needle=ev_needle,
        launch_command=ev_launch,
        poll_seconds=int(args.poll_seconds),
    )
    _refresh_report(args)

    ev_steps = [
        ("capql", True),
        ("pgmorl", True),
        ("qpensieve", True),
        ("morlca", True),
    ]
    for method, force in ev_steps:
        _run_benchmark_step(
            args,
            env_key="evcharging",
            method=method,
            force=force,
        )

    chlor_steps = [
        ("pgmorl", True),
        ("qpensieve", True),
        ("capql", True),
    ]
    for method, force in chlor_steps:
        _run_benchmark_step(
            args,
            env_key="chlor_alkali",
            method=method,
            force=force,
        )
    _refresh_final_report(args)


if __name__ == "__main__":
    main()
