from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol"
SCALAR_ROOT = RESULTS_ROOT / "scalarized_baselines" / "chlor_alkali"
DYNAMIC_ROOT = RESULTS_ROOT
MORL_ROOT = RESULTS_ROOT / "morl_dynamic_baselines" / "chlor_bench_t1024"
REPORT_MD = PROJECT_ROOT / "EXPERIMENT_RESULTS_ZH_20_REGIME.md"
REPORT_CSV = PROJECT_ROOT / "analysis" / "experiment_results_20_regime_summary.csv"
LOG_PATH = RESULTS_ROOT / "logs_fix" / "watch_chlor_repair_completion.log"
DONE_MARKER = RESULTS_ROOT / "logs_fix" / "watch_chlor_repair_completion.done"
CHLOR_LOG_ROOT = RESULTS_ROOT / "logs_fix"
CHLOR_SCALAR_ALGS = ["a2c", "acer", "acktr", "ddpg", "deepq", "ppo1", "ppo2", "trpo_mpi"]
CHLOR_SCALAR_WEIGHT_COUNT = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/root/anaconda3/envs/morl-pareto/bin/python"),
    )
    return parser.parse_args()


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as fp:
        fp.write(f"[{timestamp}] {message}\n")


def scalar_status() -> tuple[int, int]:
    good = bad = 0
    if not SCALAR_ROOT.exists():
        return 0, 56
    for alg in CHLOR_SCALAR_ALGS:
        for weight_idx in range(CHLOR_SCALAR_WEIGHT_COUNT):
            ok = scalar_run_complete(SCALAR_ROOT / alg / f"seed0_w{weight_idx}")
            good += int(ok)
            bad += int(not ok)
    return good, bad


def scalar_run_complete(run_dir: Path) -> bool:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text())
    except Exception:
        return False
    regime_rows = payload.get("regime_returns", [])
    unique_ids = {int(row.get("regime_id", -1)) for row in regime_rows}
    shared_dir = run_dir / "shared_regime_returns"
    json_count = len(list(shared_dir.glob("*.json"))) if shared_dir.exists() else 0
    npz_count = len(list(shared_dir.glob("*.npz"))) if shared_dir.exists() else 0
    return len(regime_rows) == 20 and len(unique_ids) == 20 and json_count == 20 and npz_count == 20


def scalar_incomplete_algs() -> list[str]:
    incomplete: list[str] = []
    for alg in CHLOR_SCALAR_ALGS:
        if any(
            not scalar_run_complete(SCALAR_ROOT / alg / f"seed0_w{weight_idx}")
            for weight_idx in range(CHLOR_SCALAR_WEIGHT_COUNT)
        ):
            incomplete.append(alg)
    return incomplete


def dynamic_status() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for config in ["dynamic", "static", "random", "no_dyn_ablation"]:
        path = DYNAMIC_ROOT / f"sharedprot_main_chlor_alkali_{config}_seed0" / "regime_fronts" / "shared_final.json"
        if not path.exists():
            out[config] = (0, 0)
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            out[config] = (-1, -1)
            continue
        regimes = payload.get("regimes", [])
        unique_ids = {int(row.get("regime_id", -1)) for row in regimes}
        out[config] = (len(regimes), len(unique_ids))
    return out


def morl_status() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for method in ["capql", "qpensieve", "pgmorl"]:
        path = MORL_ROOT / method / "chlor_alkali" / "seed0" / "summary.json"
        if not path.exists():
            out[method] = (0, 0)
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            out[method] = (-1, -1)
            continue
        regimes = payload.get("regime_fronts", [])
        unique_ids = {int(row.get("regime_id", -1)) for row in regimes}
        out[method] = (len(regimes), len(unique_ids))
    return out


def all_complete() -> bool:
    scalar_good, _scalar_bad = scalar_status()
    if scalar_good < 56:
        return False
    if any(pair != (20, 20) for pair in dynamic_status().values()):
        return False
    if any(pair != (20, 20) for pair in morl_status().values()):
        return False
    return True


def _ps_lines() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return proc.stdout.splitlines()


def _process_active(pattern: str) -> bool:
    for line in _ps_lines():
        if pattern in line and "watch_chlor_repair_completion.py" not in line:
            return True
    return False


def scalar_process_active() -> bool:
    patterns = [
        "scripts/run_parallel_baseline_groups.py --env-key chlor_alkali",
        "scripts/run_baseline_sweep.py --env-key chlor_alkali",
        "scripts/run_scalarized_baseline.py --alg",
    ]
    return any(_process_active(pattern) for pattern in patterns)


def scalar_alg_process_active(alg: str) -> bool:
    alg = str(alg)
    patterns = [
        f"scripts/run_baseline_sweep.py --env-key chlor_alkali --algs {alg}",
        f"scripts/run_scalarized_baseline.py --alg {alg}",
    ]
    return any(_process_active(pattern) for pattern in patterns)


def launch_background(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    append_log(f"launching repair: {' '.join(command)}")
    with open(log_path, "a", encoding="utf-8") as fp:
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=fp,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            },
            start_new_session=True,
        )


def maybe_resume_repairs(args: argparse.Namespace) -> None:
    incomplete_algs = scalar_incomplete_algs()
    for alg in incomplete_algs:
        if scalar_alg_process_active(alg):
            continue
        launch_background(
            [
                str(args.python_bin),
                "scripts/run_baseline_sweep.py",
                "--env-key",
                "chlor_alkali",
                "--algs",
                alg,
                "--seeds",
                "0",
                "--total-timesteps",
                "4096",
                "--trace-eval-episodes",
                "20",
                "--shared-regime-eval-episodes",
                "20",
                "--shared-regime-seed-offset",
                "0",
                "--chlor-episode-length",
                "96",
                "--save-root",
                str(RESULTS_ROOT / "scalarized_baselines"),
                "--continue-on-error",
            ],
            CHLOR_LOG_ROOT / f"chlor_scalar_resume_{alg}.log",
        )

    dynamic_pairs = dynamic_status()
    if any(pair != (20, 20) for pair in dynamic_pairs.values()) and not _process_active(
        "scripts/run_multienv_dynamic_suite.py --env-key chlor_alkali"
    ):
        launch_background(
            [
                str(args.python_bin),
                "scripts/run_multienv_dynamic_suite.py",
                "--env-key",
                "chlor_alkali",
                "--results-root",
                str(RESULTS_ROOT),
                "--prefix",
                "sharedprot_main",
                "--configs",
                "dynamic",
                "static",
                "random",
                "no_dyn_ablation",
                "--num-time-steps",
                "2048",
                "--num-init-steps",
                "1024",
                "--num-select",
                "4",
                "--eval-num",
                "2",
                "--ppo-epoch",
                "6",
                "--num-mini-batch",
                "4",
                "--delta-weight",
                "0.25",
                "--eval-delta-weight",
                "0.5",
                "--trace-eval-samples",
                "1",
                "--regime-eval-samples",
                "1",
                "--context-probe-samples",
                "2",
                "--context-trace-steps",
                "4",
                "--context-buffer-steps",
                "20",
                "--context-history-size",
                "6",
                "--context-nearest-k",
                "1",
                "--context-history-lambda",
                "0.1",
                "--context-forecast-lambda",
                "0.5",
                "--knee-lambda",
                "0.05",
                "--dynamic-lambda",
                "0.02",
                "--resilience-lambda",
                "0.05",
                "--diversity-lambda",
                "0.25",
                "--shift-gap-lambda",
                "0.0",
                "--repeat-topk",
                "1",
                "--chlor-episode-length",
                "96",
                "--ob-rms",
                "--obj-rms",
                "--shared-regime-eval-episodes",
                "20",
                "--force",
            ],
            CHLOR_LOG_ROOT / "chlor_dynamic_resume.log",
        )

    morl_pairs = morl_status()
    if any(pair != (20, 20) for pair in morl_pairs.values()) and not _process_active(
        "scripts/run_morl_baseline_suite.py --methods capql qpensieve pgmorl --env-keys chlor_alkali"
    ):
        launch_background(
            [
                str(args.python_bin),
                "scripts/run_morl_baseline_suite.py",
                "--methods",
                "capql",
                "qpensieve",
                "pgmorl",
                "--env-keys",
                "chlor_alkali",
                "--suite-name",
                "chlor_bench_t1024",
                "--save-root",
                str(RESULTS_ROOT / "morl_dynamic_baselines"),
                "--shared-regime-eval-episodes",
                "20",
                "--chlor-episode-length",
                "96",
            ],
            CHLOR_LOG_ROOT / "chlor_morl_resume.log",
        )


def run_post(args: argparse.Namespace) -> int:
    command = [
        str(args.python_bin),
        "scripts/run_shared_protocol_pool.py",
        "--phases",
        "post",
        "--max-workers",
        "1",
        "--force",
    ]
    append_log(f"launching post: {' '.join(command)}")
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )
    append_log(proc.stdout[-8000:] if proc.stdout else "(no stdout)")
    append_log(f"post finished returncode={proc.returncode}")
    return int(proc.returncode)


def run_report(args: argparse.Namespace) -> int:
    command = [
        str(args.python_bin),
        "scripts/generate_20_regime_report.py",
        "--out-md",
        str(REPORT_MD),
        "--out-csv",
        str(REPORT_CSV),
    ]
    append_log(f"launching report: {' '.join(command)}")
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        },
    )
    append_log(proc.stdout[-8000:] if proc.stdout else "(no stdout)")
    append_log(f"report finished returncode={proc.returncode}")
    return int(proc.returncode)


def main() -> None:
    args = parse_args()
    append_log("watcher started")
    last_snapshot = None
    while True:
        snapshot = {
            "scalar": scalar_status(),
            "dynamic": dynamic_status(),
            "morl": morl_status(),
        }
        if snapshot != last_snapshot:
            append_log(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
            last_snapshot = snapshot
        if all_complete():
            post_rc = run_post(args)
            report_rc = run_report(args) if post_rc == 0 else -1
            DONE_MARKER.write_text(
                json.dumps(
                    {
                        "post_returncode": post_rc,
                        "report_returncode": report_rc,
                        "report_md": str(REPORT_MD),
                        "report_csv": str(REPORT_CSV),
                        "completed_at": time.time(),
                    },
                    indent=2,
                )
            )
            if post_rc != 0:
                sys.exit(post_rc)
            if report_rc != 0:
                sys.exit(report_rc)
            append_log("watcher complete")
            return
        maybe_resume_repairs(args)
        time.sleep(max(10, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
