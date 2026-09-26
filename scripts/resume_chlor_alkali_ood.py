from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_ood_benchmark import OOD_BUNDLE_ROOT, OOD_RESULTS_ROOT, SCRIPT_ROOT, _baseline_retrain_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for the in-flight chlor_alkali Q-Pensieve retrain to finish, then resume the remaining OOD steps with the patched orchestrator."
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--profile", type=str, default="ood_v1")
    parser.add_argument("--bundle-root", type=Path, default=OOD_BUNDLE_ROOT)
    parser.add_argument("--results-root", type=Path, default=OOD_RESULTS_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=10)
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("running:", " ".join(str(token) for token in command), flush=True)
    subprocess.run([str(token) for token in command], check=True)


def _ps_rows() -> list[tuple[int, str]]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,cmd="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, cmd = stripped.partition(" ")
        if not pid_text.isdigit():
            continue
        rows.append((int(pid_text), cmd))
    return rows


def _is_process_running(needle: str) -> bool:
    return any(needle in cmd for _pid, cmd in _ps_rows())


def _is_qpensieve_train_running(train_dir: Path) -> bool:
    train_dir_text = str(train_dir.resolve())
    for _pid, cmd in _ps_rows():
        if "run_qpensieve_dynamic_baseline.py --env-key chlor_alkali" not in cmd:
            continue
        if train_dir_text not in cmd:
            continue
        if "--train-only" not in cmd:
            continue
        return True
    return False


def _matching_pids(patterns: list[str]) -> list[int]:
    current_pid = os.getpid()
    matches = []
    for pid, cmd in _ps_rows():
        if pid == current_pid:
            continue
        if any(pattern in cmd for pattern in patterns):
            matches.append(pid)
    return sorted(set(matches))


def _terminate_processes(patterns: list[str], *, grace_seconds: int = 10) -> None:
    pids = _matching_pids(patterns)
    if not pids:
        return
    print(f"terminating stale queue processes: {pids}", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + max(1, int(grace_seconds))
    while time.time() < deadline:
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            return
        time.sleep(1)
    alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _refresh_partial_report(args: argparse.Namespace) -> None:
    _run(
        [
            args.python_bin,
            str(SCRIPT_ROOT / "generate_ood_report.py"),
            "--profile",
            args.profile,
            "--results-root",
            str(args.results_root),
            "--out-md",
            str(PROJECT_ROOT / "OOD_EXPERIMENT_RESULTS_partial.md"),
            "--out-csv",
            str(PROJECT_ROOT / "analysis" / "ood_results_summary_partial.csv"),
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
            str(PROJECT_ROOT / "OOD_EXPERIMENT_RESULTS_ZH.md"),
            "--out-csv",
            str(PROJECT_ROOT / "analysis" / "ood_results_summary.csv"),
        ]
    )


def _run_method(args: argparse.Namespace, method: str, *, force_retrain: bool = False) -> None:
    command = [
        args.python_bin,
        str(SCRIPT_ROOT / "run_ood_benchmark.py"),
        "--env-keys",
        "chlor_alkali",
        "--methods",
        method,
        "--profile",
        args.profile,
        "--python-bin",
        args.python_bin,
        "--force",
    ]
    if force_retrain:
        command.append("--force-retrain")
    _run(command)
    _refresh_partial_report(args)


def _retrain_ready(args: argparse.Namespace, method: str) -> bool:
    train_dir = args.results_root / args.profile / f"{method}_train" / "chlor_alkali"
    train_env_config_json = args.bundle_root / args.profile / "chlor_alkali" / "train_env_config.json"
    return _baseline_retrain_ready(method, train_dir, train_env_config_json)


def main() -> None:
    args = parse_args()
    qpensieve_train_dir = args.results_root / args.profile / "qpensieve_train" / "chlor_alkali"
    train_env_config_json = args.bundle_root / args.profile / "chlor_alkali" / "train_env_config.json"

    print("waiting for in-flight qpensieve chlor_alkali retrain to finish", flush=True)
    while True:
        ready = _baseline_retrain_ready("qpensieve", qpensieve_train_dir, train_env_config_json)
        active_train = _is_qpensieve_train_running(qpensieve_train_dir)
        print(f"status: qpensieve_train_ready={ready} active_train={active_train}", flush=True)
        if ready or not active_train:
            break
        time.sleep(max(2, int(args.poll_seconds)))

    force_qpensieve_retrain = not _baseline_retrain_ready("qpensieve", qpensieve_train_dir, train_env_config_json)
    _terminate_processes(
        [
            "scripts/run_ood_serial_queue.py",
            "scripts/run_ood_benchmark.py --env-keys chlor_alkali",
            "run_qpensieve_dynamic_baseline.py --env-key chlor_alkali",
            "run_pgmorl_dynamic_baseline.py --env-key chlor_alkali",
            "run_capql_dynamic_baseline.py --env-key chlor_alkali",
        ]
    )

    _run_method(args, "qpensieve", force_retrain=force_qpensieve_retrain)
    _run_method(args, "pgmorl", force_retrain=not _retrain_ready(args, "pgmorl"))
    _run_method(args, "capql", force_retrain=not _retrain_ready(args, "capql"))
    _refresh_final_report(args)
    _run(
        [
            args.python_bin,
            str(SCRIPT_ROOT / "audit_ood_results.py"),
            "--profile",
            args.profile,
            "--results-root",
            str(args.results_root),
        ]
    )


if __name__ == "__main__":
    main()
