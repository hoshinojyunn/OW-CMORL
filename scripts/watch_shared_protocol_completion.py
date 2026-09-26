from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol"
SCALAR_ROOT = RESULTS_ROOT / "scalarized_baselines"
LOG_ROOT = RESULTS_ROOT / "logs_pool"
RAW_ARCHIVE_ROOT = PROJECT_ROOT / "archives_npz_shared_protocol"
RAW_MANIFEST = PROJECT_ROOT / "archives_npz_shared_protocol_manifest.csv"
CANON_ARCHIVE_ROOT = PROJECT_ROOT / "archives_npz_canonical_shared_protocol"
CANON_MANIFEST = PROJECT_ROOT / "archives_npz_canonical_shared_protocol_manifest.csv"
NPZ_FIG_DIR = PROJECT_ROOT / "figures" / "benchmark_npz_analysis_shared_protocol"
BENCH_FIG_DIR = PROJECT_ROOT / "figures" / "all_benchmarks_shared_protocol"
STATUS_JSON = LOG_ROOT / "shared_protocol_goal_status.json"
STATUS_MD = LOG_ROOT / "shared_protocol_goal_status.md"
DONE_MARKER = LOG_ROOT / "shared_protocol_watcher.done"

EXPECTED_SCALAR_COUNTS = {
    "building": 56,
    "evcharging": 56,
    "cogen": 88,
    "chlor_alkali": 56,
}
EXPECTED_CHLOR_REGIMES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path("/root/anaconda3/envs/morl-pareto/bin/python"))
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--resume-max-attempts", type=int, default=4)
    parser.add_argument("--stale-seconds", type=int, default=1800)
    parser.add_argument("--log-path", type=Path, default=LOG_ROOT / "watch_shared_protocol_completion.log")
    return parser.parse_args()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(f"[{timestamp}] {message}\n")


def tracked_progress_snapshot() -> dict[str, object]:
    counts = count_scalar_results()
    mtimes: dict[str, int] = {}
    patterns = [
        "cogen_baseline_parallel_group*.log",
        "chlor_alkali_baseline_parallel_group*.log",
        "chlor_baseline_parallel_group*.log",
    ]
    log_root = RESULTS_ROOT / "logs"
    if log_root.exists():
        for pattern in patterns:
            for path in sorted(log_root.glob(pattern)):
                try:
                    mtimes[str(path)] = int(path.stat().st_mtime)
                except FileNotFoundError:
                    continue
    return {
        "counts": counts,
        "mtimes": mtimes,
    }


def snapshot_changed(previous: dict[str, object] | None, current: dict[str, object]) -> bool:
    if previous is None:
        return True
    return previous != current


def count_scalar_results() -> dict[str, int]:
    counts: dict[str, int] = {}
    for env_key in EXPECTED_SCALAR_COUNTS:
        env_root = SCALAR_ROOT / env_key
        if not env_root.exists():
            counts[env_key] = 0
            continue
        if env_key != "chlor_alkali":
            counts[env_key] = sum(1 for _ in env_root.rglob("result.json"))
            continue
        complete = 0
        for result_path in env_root.rglob("result.json"):
            try:
                payload = json.loads(result_path.read_text())
            except Exception:
                continue
            regime_rows = payload.get("regime_returns", [])
            unique_ids = {
                int(row.get("regime_id", -1))
                for row in regime_rows
                if "regime_id" in row
            }
            shared_dir = result_path.parent / "shared_regime_returns"
            shared_json = list(shared_dir.glob("*.json")) if shared_dir.exists() else []
            shared_npz = list(shared_dir.glob("*.npz")) if shared_dir.exists() else []
            if (
                len(regime_rows) == EXPECTED_CHLOR_REGIMES
                and len(unique_ids) == EXPECTED_CHLOR_REGIMES
                and len(shared_json) == EXPECTED_CHLOR_REGIMES
                and len(shared_npz) == EXPECTED_CHLOR_REGIMES
            ):
                complete += 1
        counts[env_key] = complete
    return counts


def all_scalar_complete(counts: dict[str, int]) -> bool:
    return all(counts.get(env_key, 0) >= expected for env_key, expected in EXPECTED_SCALAR_COUNTS.items())


def missing_scalar(counts: dict[str, int]) -> dict[str, int]:
    return {
        env_key: max(0, expected - counts.get(env_key, 0))
        for env_key, expected in EXPECTED_SCALAR_COUNTS.items()
        if counts.get(env_key, 0) < expected
    }


def outputs_complete() -> bool:
    required = required_output_paths()
    if any(not path.exists() for path in required):
        return False
    latest_source = latest_result_mtime()
    oldest_output = min(path.stat().st_mtime for path in required)
    return oldest_output >= latest_source


def missing_outputs() -> list[str]:
    required = required_output_paths()
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return missing
    latest_source = latest_result_mtime()
    stale = [f"stale:{path}" for path in required if path.stat().st_mtime < latest_source]
    return stale


def required_output_paths() -> list[Path]:
    return [
        RAW_MANIFEST,
        CANON_MANIFEST,
        NPZ_FIG_DIR / "cdf_curves.csv",
        BENCH_FIG_DIR / "comparison_raw.csv",
        BENCH_FIG_DIR / "pairwise_summary.csv",
        BENCH_FIG_DIR / "wins_vs_baselines.csv",
    ]


def latest_result_mtime() -> float:
    latest = 0.0
    patterns = ("result.json", "shared_final.json")
    for pattern in patterns:
        for path in RESULTS_ROOT.rglob(pattern):
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                continue
    return latest


def run_pipeline(
    args: argparse.Namespace,
    phases: list[str],
    log_path: Path,
    *,
    force: bool = False,
) -> int:
    command = [
        str(args.python_bin),
        "scripts/run_shared_protocol_pool.py",
        "--phases",
        *phases,
        "--max-workers",
        "1",
    ]
    if force:
        command.append("--force")
    append_log(log_path, f"launching pipeline: {' '.join(command)}")
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    with open(log_path, "a", encoding="utf-8") as fp:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
    append_log(log_path, f"pipeline finished with returncode={proc.returncode}")
    return int(proc.returncode)


def load_pairwise_summary(path: Path) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    if not path.exists():
        return summary
    env_rows: dict[str, list[dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            env_rows.setdefault(row["env_key"], []).append(row)
    for env_key, rows in env_rows.items():
        wins = sum(1 for row in rows if float(row.get("win_rate", 0.0)) > 0.5)
        summary[env_key] = {
            "num_baselines": len(rows),
            "pairwise_beats": wins,
            "pairwise_majority": bool(len(rows) > 0 and wins > len(rows) / 2.0),
        }
    return summary


def load_metric_majorities(path: Path) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    if not path.exists():
        return summary
    env_rows: dict[str, list[dict[str, str]]] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            env_rows.setdefault(row["env_key"], []).append(row)
    for env_key, rows in env_rows.items():
        wins = sum(
            1
            for row in rows
            if float(row.get("wins_vs_baselines", 0.0)) > float(row.get("num_baselines", 0.0)) / 2.0
        )
        summary[env_key] = {
            "num_metrics": len(rows),
            "metric_majority_wins": wins,
            "metric_majority": bool(len(rows) > 0 and wins > len(rows) / 2.0),
        }
    return summary


def write_status(
    counts: dict[str, int],
    log_path: Path,
    *,
    resume_attempts: int,
) -> None:
    pairwise = load_pairwise_summary(BENCH_FIG_DIR / "pairwise_summary.csv")
    metric_majorities = load_metric_majorities(BENCH_FIG_DIR / "wins_vs_baselines.csv")
    scalar_complete = all_scalar_complete(counts)
    outputs_done = outputs_complete()
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scalar_counts": counts,
        "expected_scalar_counts": EXPECTED_SCALAR_COUNTS,
        "scalar_complete": scalar_complete,
        "outputs_complete": outputs_done,
        "resume_attempts": int(resume_attempts),
        "missing_scalar": missing_scalar(counts),
        "missing_outputs": missing_outputs(),
        "pairwise_summary": pairwise,
        "metric_majorities": metric_majorities,
    }
    STATUS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Shared Protocol Goal Status",
        "",
        f"- generated_at: {payload['generated_at']}",
        f"- scalar_complete: {payload['scalar_complete']}",
        f"- outputs_complete: {payload['outputs_complete']}",
        f"- resume_attempts: {payload['resume_attempts']}",
        "",
        "## Scalar Counts",
    ]
    for env_key, expected in EXPECTED_SCALAR_COUNTS.items():
        lines.append(f"- {env_key}: {counts.get(env_key, 0)}/{expected}")
    if payload["missing_scalar"]:
        lines.append("")
        lines.append("## Missing Scalar")
        for env_key, missing in payload["missing_scalar"].items():
            lines.append(f"- {env_key}: {missing}")
    if payload["missing_outputs"]:
        lines.append("")
        lines.append("## Missing Outputs")
        for item in payload["missing_outputs"]:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("## Dynamic vs Baselines")
    for env_key in sorted(set(pairwise) | set(metric_majorities)):
        pair = pairwise.get(env_key, {})
        metric = metric_majorities.get(env_key, {})
        lines.append(
            "- "
            f"{env_key}: pairwise {pair.get('pairwise_beats', 0)}/{pair.get('num_baselines', 0)} "
            f"(majority={pair.get('pairwise_majority', False)}), "
            f"metric-majority {metric.get('metric_majority_wins', 0)}/{metric.get('num_metrics', 0)} "
            f"(majority={metric.get('metric_majority', False)})"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    DONE_MARKER.write_text(f"done {payload['generated_at']}\n", encoding="utf-8")
    append_log(log_path, f"wrote status files: {STATUS_JSON} and {STATUS_MD}")


def main() -> None:
    args = parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    append_log(args.log_path, f"watcher started for pid={args.wait_pid}")
    last_snapshot: dict[str, object] | None = None
    last_progress_at = time.time()
    while pid_alive(args.wait_pid):
        snapshot = tracked_progress_snapshot()
        if snapshot_changed(last_snapshot, snapshot):
            last_progress_at = time.time()
            append_log(
                args.log_path,
                f"target process still running; progress updated counts={snapshot['counts']}",
            )
            last_snapshot = snapshot
        else:
            idle_for = int(time.time() - last_progress_at)
            append_log(
                args.log_path,
                f"target process still running; no progress change for {idle_for}s",
            )
            if idle_for >= int(args.stale_seconds):
                append_log(
                    args.log_path,
                    f"detected stale main process after {idle_for}s; sending SIGTERM to pid={args.wait_pid}",
                )
                try:
                    os.kill(args.wait_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                break
        time.sleep(max(30, int(args.poll_seconds)))

    append_log(args.log_path, f"target pid={args.wait_pid} exited")
    counts = count_scalar_results()
    append_log(args.log_path, f"scalar counts after exit: {counts}")
    resume_attempts = 0
    while resume_attempts < max(1, int(args.resume_max_attempts)):
        scalar_done = all_scalar_complete(counts)
        outputs_done = outputs_complete()
        if scalar_done and outputs_done:
            break
        resume_attempts += 1
        append_log(
            args.log_path,
            f"resume attempt {resume_attempts}: scalar_complete={scalar_done} outputs_complete={outputs_done}",
        )
        rc = 0
        if not scalar_done:
            rc = run_pipeline(args, ["scalar"], args.log_path, force=False)
            counts = count_scalar_results()
            scalar_done = all_scalar_complete(counts)
            append_log(
                args.log_path,
                f"after scalar resume {resume_attempts}: rc={rc} counts={counts}",
            )
        if scalar_done and not outputs_complete():
            rc_post = run_pipeline(args, ["post"], args.log_path, force=True)
            rc = rc or rc_post
            append_log(
                args.log_path,
                f"after post refresh {resume_attempts}: rc={rc_post} missing_outputs={missing_outputs()}",
            )
        counts = count_scalar_results()
        append_log(
            args.log_path,
            f"after resume attempt {resume_attempts}: rc={rc} counts={counts} missing_outputs={missing_outputs()}",
        )
        if rc != 0:
            time.sleep(10)

    write_status(counts, args.log_path, resume_attempts=resume_attempts)


if __name__ == "__main__":
    main()
