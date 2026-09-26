from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results_shared_protocol"
LOG_ROOT = RESULTS_ROOT / "logs_pool"
RAW_ARCHIVE_ROOT = PROJECT_ROOT / "archives_npz_shared_protocol"
RAW_MANIFEST = PROJECT_ROOT / "archives_npz_shared_protocol_manifest.csv"
CANON_ARCHIVE_ROOT = PROJECT_ROOT / "archives_npz_canonical_shared_protocol"
CANON_MANIFEST = PROJECT_ROOT / "archives_npz_canonical_shared_protocol_manifest.csv"
NPZ_FIG_DIR = PROJECT_ROOT / "figures" / "benchmark_npz_analysis_shared_protocol"
BENCH_FIG_DIR = PROJECT_ROOT / "figures" / "all_benchmarks_shared_protocol"
CHLOR_SCALAR_ALGS = ["a2c", "acer", "acktr", "ddpg", "deepq", "ppo1", "ppo2", "trpo_mpi"]
DISPLAY_METHODS = ["dynamic", "capql", "pgmorl", "q_pensieve"]
TARGET_FINE_REGIMES = 20
TARGET_FINE_CATALOG_EPISODES = 256
TARGET_FINE_CONTEXT_STEPS = 4


@dataclass
class Job:
    name: str
    command: list[str]
    check_done: Callable[[], bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--scalar-max-groups",
        type=int,
        default=4,
        help="Maximum number of scalarized baseline groups to launch per non-chlor environment.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-post", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["dynamic", "morl", "scalar", "post"],
        default=["dynamic", "morl", "scalar", "post"],
    )
    return parser.parse_args()


def dynamic_done(env_key: str) -> bool:
    configs = ["dynamic", "static", "random", "no_dyn_ablation"]
    for config in configs:
        path = RESULTS_ROOT / f"sharedprot_main_{env_key}_{config}_seed0" / "final" / "shared_eval_summary.json"
        regime_path = RESULTS_ROOT / f"sharedprot_main_{env_key}_{config}_seed0" / "regime_fronts" / "shared_final.json"
        if not path.exists() or not regime_path.exists():
            return False
        try:
            payload = json.loads(regime_path.read_text())
        except Exception:
            return False
        regimes = payload.get("regimes", [])
        summary = json.loads(path.read_text())
        shared_episode_seeds = [int(seed) for seed in summary.get("shared_eval_episode_seeds", [])]
        shared_plan = list(summary.get("shared_regime_seed_plan", []))
        planned_episode_seeds = [
            int(seed)
            for plan_row in shared_plan
            for seed in plan_row.get("episode_seeds", [])
        ]
        unique_ids = {int(row.get("regime_id", -1)) for row in regimes if int(row.get("regime_id", -1)) >= 0}
        if (
            len(regimes) < TARGET_FINE_REGIMES
            or len(unique_ids) < TARGET_FINE_REGIMES
            or len(shared_plan) < TARGET_FINE_REGIMES
            or shared_episode_seeds != planned_episode_seeds
            or len(shared_episode_seeds) != TARGET_FINE_REGIMES
        ):
            return False
    return True


def morl_done(suite: str, method: str, env_key: str) -> bool:
    summary_path = RESULTS_ROOT / "morl_dynamic_baselines" / suite / method / env_key / "seed0" / "summary.json"
    shared_dir = RESULTS_ROOT / "morl_dynamic_baselines" / suite / method / env_key / "seed0" / "shared_regime_returns"
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return False
    if not (len(data.get("front_points", [])) > 0 and len(data.get("regime_fronts", [])) > 0):
        return False
    regime_fronts = data.get("regime_fronts", [])
    shared_episode_seeds = [int(seed) for seed in data.get("shared_eval_episode_seeds", [])]
    shared_plan = list(data.get("shared_regime_seed_plan", []))
    planned_episode_seeds = [
        int(seed)
        for plan_row in shared_plan
        for seed in plan_row.get("episode_seeds", [])
    ]
    unique_ids = {int(row.get("regime_id", -1)) for row in regime_fronts if int(row.get("regime_id", -1)) >= 0}
    json_count = len(list(shared_dir.glob("*.json"))) if shared_dir.exists() else 0
    npz_count = len(list(shared_dir.glob("*.npz"))) if shared_dir.exists() else 0
    return (
        len(regime_fronts) >= TARGET_FINE_REGIMES
        and len(unique_ids) >= TARGET_FINE_REGIMES
        and len(data.get("shared_regime_seed_plan", [])) >= TARGET_FINE_REGIMES
        and json_count >= TARGET_FINE_REGIMES
        and npz_count >= TARGET_FINE_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == TARGET_FINE_REGIMES
    )


def scalar_done(env_key: str, expected_count: int) -> bool:
    root = RESULTS_ROOT / "scalarized_baselines" / env_key
    if not root.exists():
        return False
    complete = 0
    for result_path in root.rglob("result.json"):
        try:
            payload = json.loads(result_path.read_text())
        except Exception:
            continue
        regime_rows = payload.get("regime_returns", [])
        shared_episode_seeds = [int(seed) for seed in payload.get("shared_eval_episode_seeds", [])]
        shared_plan = list(payload.get("shared_regime_seed_plan", []))
        planned_episode_seeds = [
            int(seed)
            for plan_row in shared_plan
            for seed in plan_row.get("episode_seeds", [])
        ]
        unique_ids = {int(row.get("regime_id", -1)) for row in regime_rows if int(row.get("regime_id", -1)) >= 0}
        shared_dir = result_path.parent / "shared_regime_returns"
        json_count = len(list(shared_dir.glob("*.json"))) if shared_dir.exists() else 0
        npz_count = len(list(shared_dir.glob("*.npz"))) if shared_dir.exists() else 0
        if (
            len(regime_rows) >= TARGET_FINE_REGIMES
            and len(unique_ids) >= TARGET_FINE_REGIMES
            and json_count >= TARGET_FINE_REGIMES
            and npz_count >= TARGET_FINE_REGIMES
            and shared_episode_seeds == planned_episode_seeds
            and len(shared_episode_seeds) == TARGET_FINE_REGIMES
        ):
            complete += 1
    return complete >= expected_count


def run_job(job: Job) -> tuple[str, int]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{job.name}.log"
    with open(log_path, "w") as fp:
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        fp.write(f"STARTED: {started_at}\n")
        fp.write("COMMAND: " + " ".join(job.command) + "\n")
        fp.flush()
        proc = subprocess.run(
            job.command,
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
        )
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
        verified = False
        if proc.returncode == 0:
            try:
                verified = bool(job.check_done())
            except Exception as exc:  # pragma: no cover
                fp.write(f"VERIFY_ERROR: {exc!r}\n")
                verified = False
        if proc.returncode == 0 and not verified:
            fp.write("VERIFY_DONE: False\n")
            proc_returncode = 2
        else:
            fp.write(f"VERIFY_DONE: {verified}\n")
            proc_returncode = proc.returncode
        fp.write(f"FINISHED: {finished_at}\n")
        fp.write(f"RETURNCODE: {proc_returncode}\n")
        fp.flush()
    return job.name, int(proc_returncode)


def phase_jobs(force: bool, enabled_phases: set[str], scalar_max_groups: int) -> list[list[Job]]:
    py = sys.executable
    jobs_dynamic = [
        Job(
            name="dynamic_building",
            command=[
                py,
                "scripts/run_multienv_dynamic_suite.py",
                "--env-key",
                "building",
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
                "6",
                "--eval-num",
                "3",
                "--delta-weight",
                "0.25",
                "--eval-delta-weight",
                "0.25",
                "--trace-eval-samples",
                "2",
                "--regime-eval-samples",
                "2",
                "--context-probe-samples",
                "5",
                "--context-trace-steps",
                "24",
                "--context-saliency-alpha",
                "0.6",
                "--context-buffer-steps",
                "48",
                "--context-history-size",
                "12",
                "--context-nearest-k",
                "4",
                "--context-history-lambda",
                "0.2",
                "--context-forecast-lambda",
                "0.5",
                "--knee-lambda",
                "0.3",
                "--dynamic-lambda",
                "0.7",
                "--resilience-lambda",
                "1.8",
                "--diversity-lambda",
                "0.6",
                "--shift-gap-lambda",
                "1.2",
                "--repeat-topk",
                "3",
                "--entropy-coef",
                "0.02",
                "--ob-rms",
                "--obj-rms",
                "--shared-regime-eval-episodes",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--force",
            ],
            check_done=lambda: dynamic_done("building"),
        ),
        Job(
            name="dynamic_evcharging",
            command=[
                py,
                "scripts/run_multienv_dynamic_suite.py",
                "--env-key",
                "evcharging",
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
                "6",
                "--eval-num",
                "3",
                "--delta-weight",
                "0.25",
                "--eval-delta-weight",
                "0.25",
                "--trace-eval-samples",
                "2",
                "--regime-eval-samples",
                "2",
                "--context-probe-samples",
                "5",
                "--context-trace-steps",
                "24",
                "--context-saliency-alpha",
                "0.6",
                "--context-buffer-steps",
                "48",
                "--context-history-size",
                "12",
                "--context-nearest-k",
                "4",
                "--context-history-lambda",
                "0.2",
                "--context-forecast-lambda",
                "0.5",
                "--knee-lambda",
                "0.3",
                "--dynamic-lambda",
                "0.7",
                "--resilience-lambda",
                "1.8",
                "--diversity-lambda",
                "0.6",
                "--shift-gap-lambda",
                "1.2",
                "--repeat-topk",
                "3",
                "--entropy-coef",
                "0.02",
                "--ob-rms",
                "--obj-rms",
                "--shared-regime-eval-episodes",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--force",
            ],
            check_done=lambda: dynamic_done("evcharging"),
        ),
        Job(
            name="dynamic_cogen",
            command=[
                py,
                "scripts/run_multienv_dynamic_suite.py",
                "--env-key",
                "cogen",
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
                "3072",
                "--num-init-steps",
                "1536",
                "--num-select",
                "4",
                "--eval-num",
                "3",
                "--delta-weight",
                "0.25",
                "--eval-delta-weight",
                "0.25",
                "--trace-eval-samples",
                "2",
                "--regime-eval-samples",
                "2",
                "--context-probe-samples",
                "3",
                "--context-trace-steps",
                "12",
                "--context-buffer-steps",
                "32",
                "--context-history-size",
                "10",
                "--context-nearest-k",
                "3",
                "--context-history-lambda",
                "0.1",
                "--context-forecast-lambda",
                "0.35",
                "--knee-lambda",
                "0.2",
                "--dynamic-lambda",
                "0.5",
                "--resilience-lambda",
                "1.0",
                "--diversity-lambda",
                "0.2",
                "--shift-gap-lambda",
                "0.5",
                "--repeat-topk",
                "2",
                "--ob-rms",
                "--obj-rms",
                "--shared-regime-eval-episodes",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--force",
            ],
            check_done=lambda: dynamic_done("cogen"),
        ),
        Job(
            name="dynamic_chlor",
            command=[
                py,
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
                "24",
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
                str(TARGET_FINE_REGIMES),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--force",
            ],
            check_done=lambda: dynamic_done("chlor_alkali"),
        ),
    ]

    def capql_job(suite: str, env_key: str, shared_eps: int, extra: list[str] | None = None) -> Job:
        extra = extra or []
        save_dir = RESULTS_ROOT / "morl_dynamic_baselines" / suite / "capql" / env_key / "seed0"
        cmd = [
            py,
            "scripts/run_capql_dynamic_baseline.py",
            "--env-key",
            env_key,
            "--seed",
            "0",
            "--shared-regime-eval-episodes",
            str(shared_eps),
            "--fine-regime-clusters",
            str(TARGET_FINE_REGIMES),
            "--fine-regime-catalog-episodes",
            str(TARGET_FINE_CATALOG_EPISODES),
            "--fine-regime-context-steps",
            str(TARGET_FINE_CONTEXT_STEPS),
            "--save-dir",
            str(save_dir),
            "--skip-train",
        ]
        cmd.extend(extra)
        return Job(
            name=f"capql_{suite}_{env_key}",
            command=cmd,
            check_done=lambda suite=suite, env_key=env_key: morl_done(suite, "capql", env_key),
        )

    def qpensieve_job(suite: str, env_key: str, shared_eps: int, extra: list[str] | None = None) -> Job:
        extra = extra or []
        save_dir = RESULTS_ROOT / "morl_dynamic_baselines" / suite / "qpensieve" / env_key / "seed0"
        cmd = [
            py,
            "scripts/run_qpensieve_dynamic_baseline.py",
            "--env-key",
            env_key,
            "--seed",
            "0",
            "--shared-regime-eval-episodes",
            str(shared_eps),
            "--fine-regime-clusters",
            str(TARGET_FINE_REGIMES),
            "--fine-regime-catalog-episodes",
            str(TARGET_FINE_CATALOG_EPISODES),
            "--fine-regime-context-steps",
            str(TARGET_FINE_CONTEXT_STEPS),
            "--save-dir",
            str(save_dir),
            "--skip-train",
        ]
        cmd.extend(extra)
        return Job(
            name=f"qpensieve_{suite}_{env_key}",
            command=cmd,
            check_done=lambda suite=suite, env_key=env_key: morl_done(suite, "qpensieve", env_key),
        )

    def pgmorl_job(suite: str, env_key: str, shared_eps: int, extra: list[str] | None = None) -> Job:
        extra = extra or []
        save_dir = RESULTS_ROOT / "morl_dynamic_baselines" / suite / "pgmorl" / env_key / "seed0"
        cmd = [
            py,
            "scripts/run_pgmorl_dynamic_baseline.py",
            "--env-key",
            env_key,
            "--seed",
            "0",
            "--shared-regime-eval-episodes",
            str(shared_eps),
            "--fine-regime-clusters",
            str(TARGET_FINE_REGIMES),
            "--fine-regime-catalog-episodes",
            str(TARGET_FINE_CATALOG_EPISODES),
            "--fine-regime-context-steps",
            str(TARGET_FINE_CONTEXT_STEPS),
            "--save-dir",
            str(save_dir),
            "--skip-train",
        ]
        cmd.extend(extra)
        return Job(
            name=f"pgmorl_{suite}_{env_key}",
            command=cmd,
            check_done=lambda suite=suite, env_key=env_key: morl_done(suite, "pgmorl", env_key),
        )

    jobs_morl = [
        capql_job("morl_online_longrun_v3", "building", TARGET_FINE_REGIMES),
        capql_job("morl_online_longrun_v3", "evcharging", TARGET_FINE_REGIMES),
        capql_job("morl_online_longrun_v3", "cogen", TARGET_FINE_REGIMES),
        capql_job("chlor_bench_t1024", "chlor_alkali", TARGET_FINE_REGIMES, ["--chlor-episode-length", "96"]),
        qpensieve_job("morl_online_longrun_v3", "building", TARGET_FINE_REGIMES),
        qpensieve_job("morl_online_longrun_v3", "evcharging", TARGET_FINE_REGIMES),
        qpensieve_job("morl_online_longrun_v3", "cogen", TARGET_FINE_REGIMES),
        qpensieve_job("chlor_bench_t1024", "chlor_alkali", TARGET_FINE_REGIMES, ["--chlor-episode-length", "96"]),
        pgmorl_job("morl_online_longrun_v3", "building", TARGET_FINE_REGIMES),
        pgmorl_job("morl_online_longrun_v5c", "evcharging", TARGET_FINE_REGIMES),
        pgmorl_job("morl_online_longrun_v5c", "cogen", TARGET_FINE_REGIMES),
        pgmorl_job("chlor_bench_t1024", "chlor_alkali", TARGET_FINE_REGIMES, ["--chlor-episode-length", "96"]),
    ]

    def scalar_job(
        env_key: str,
        *,
        total_timesteps: int,
        trace_eval_episodes: int,
        shared_eval_episodes: int,
        expected_count: int,
        max_groups: int,
        extra: list[str] | None = None,
    ) -> Job:
        extra = list(extra or [])
        if env_key == "chlor_alkali":
            command = [
                py,
                "scripts/run_baseline_sweep.py",
                "--env-key",
                env_key,
                "--algs",
                *CHLOR_SCALAR_ALGS,
                "--seeds",
                "0",
                "--total-timesteps",
                str(total_timesteps),
                "--trace-eval-episodes",
                str(trace_eval_episodes),
                "--shared-regime-eval-episodes",
                str(shared_eval_episodes),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--shared-regime-seed-offset",
                "0",
                "--save-root",
                str(RESULTS_ROOT / "scalarized_baselines"),
                "--continue-on-error",
                *extra,
            ]
        else:
            command = [
                py,
                "scripts/run_parallel_baseline_groups.py",
                "--env-key",
                env_key,
                "--total-timesteps",
                str(total_timesteps),
                "--trace-eval-episodes",
                str(trace_eval_episodes),
                "--shared-regime-eval-episodes",
                str(shared_eval_episodes),
                "--fine-regime-clusters",
                str(TARGET_FINE_REGIMES),
                "--fine-regime-catalog-episodes",
                str(TARGET_FINE_CATALOG_EPISODES),
                "--fine-regime-context-steps",
                str(TARGET_FINE_CONTEXT_STEPS),
                "--max-groups",
                str(max(1, int(max_groups))),
                "--save-root",
                str(RESULTS_ROOT / "scalarized_baselines"),
                "--continue-on-error",
                *extra,
            ]
        if force:
            command.append("--force")
        return Job(
            name=f"scalar_{'chlor' if env_key == 'chlor_alkali' else env_key}",
            command=command,
            check_done=lambda env_key=env_key, expected_count=expected_count: scalar_done(env_key, expected_count),
        )

    jobs_scalar = [
        scalar_job(
            "building",
            total_timesteps=4096,
            trace_eval_episodes=20,
            shared_eval_episodes=20,
            expected_count=56,
            max_groups=scalar_max_groups,
        ),
        scalar_job(
            "evcharging",
            total_timesteps=4096,
            trace_eval_episodes=20,
            shared_eval_episodes=20,
            expected_count=56,
            max_groups=scalar_max_groups,
        ),
        scalar_job(
            "cogen",
            total_timesteps=4096,
            trace_eval_episodes=20,
            shared_eval_episodes=20,
            expected_count=88,
            max_groups=scalar_max_groups,
        ),
        scalar_job(
            "chlor_alkali",
            total_timesteps=4096,
            trace_eval_episodes=20,
            shared_eval_episodes=20,
            expected_count=56,
            max_groups=1,
            extra=["--chlor-episode-length", "96"],
        ),
    ]

    jobs_post = [
        Job(
            name="export_npz",
            command=[
                py,
                "scripts/export_benchmark_npz_archives.py",
                "--results-root",
                str(RESULTS_ROOT),
                "--raw-out-root",
                str(RAW_ARCHIVE_ROOT),
                "--raw-manifest-path",
                str(RAW_MANIFEST),
                "--canonical-out-root",
                str(CANON_ARCHIVE_ROOT),
                "--canonical-manifest-path",
                str(CANON_MANIFEST),
                "--overwrite",
            ],
            check_done=lambda: CANON_MANIFEST.exists(),
        ),
        Job(
            name="analyze_npz",
            command=[
                py,
                "scripts/analyze_benchmark_npz_archives.py",
                "--manifest-path",
                str(CANON_MANIFEST),
                "--archive-root",
                str(CANON_ARCHIVE_ROOT),
                "--out-dir",
                str(NPZ_FIG_DIR),
                "--highlight-method",
                "dynamic",
                "--include-methods",
                *DISPLAY_METHODS,
                "--require-highlight-method",
            ],
            check_done=lambda: (NPZ_FIG_DIR / "cdf_curves.csv").exists(),
        ),
        Job(
            name="aggregate_benchmarks",
            command=[
                py,
                "scripts/aggregate_all_benchmarks.py",
                "--scalarized-root",
                str(RESULTS_ROOT / "scalarized_baselines"),
                "--dynamic-root",
                str(RESULTS_ROOT),
                "--dynamic-prefixes",
                "sharedprot_main",
                "--morl-baseline-root",
                str(RESULTS_ROOT / "morl_dynamic_baselines"),
                "--morl-suite-names",
                "morl_online_longrun_v5c",
                "morl_online_longrun_v3",
                "chlor_bench_t1024",
                "--out-dir",
                str(BENCH_FIG_DIR),
            ],
            check_done=lambda: (BENCH_FIG_DIR / "comparison_raw.csv").exists(),
        ),
    ]

    phase_specs: list[tuple[str, list[Job]]] = [
        ("dynamic", jobs_dynamic),
        ("morl", jobs_morl),
        ("scalar", jobs_scalar),
    ]
    if not force:
        phase_specs = [
            (name, [job for job in phase if not job.check_done()])
            for name, phase in phase_specs
        ]
        jobs_post = [job for job in jobs_post if not job.check_done()]
    phases = [phase for name, phase in phase_specs if name in enabled_phases and phase]
    if "post" in enabled_phases and jobs_post:
        phases.append(jobs_post)
    return phases


def run_phase(jobs: list[Job], max_workers: int, failures: list[dict[str, object]]) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                name, returncode = future.result()
            except Exception as exc:  # pragma: no cover
                failures.append({"job": job.name, "returncode": -999, "error": repr(exc)})
                print(f"[fail] {job.name}: {exc}", flush=True)
                continue
            if returncode != 0:
                failures.append({"job": name, "returncode": returncode})
                print(f"[fail] {name}: returncode={returncode}", flush=True)
            else:
                print(f"[ok] {name}", flush=True)


def run_phase_continue(jobs: list[Job], max_workers: int, failures: list[dict[str, object]]) -> None:
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                name, returncode = future.result()
            except Exception as exc:  # pragma: no cover
                failures.append({"job": job.name, "returncode": -999, "error": repr(exc)})
                print(f"[fail] {job.name}: {exc}", flush=True)
                continue
            if returncode != 0:
                failures.append({"job": name, "returncode": returncode})
                print(f"[fail-continue] {name}: returncode={returncode}", flush=True)
            else:
                print(f"[ok] {name}", flush=True)


def main() -> None:
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    phases = phase_jobs(
        force=args.force,
        enabled_phases=set(args.phases),
        scalar_max_groups=args.scalar_max_groups,
    )
    failures: list[dict[str, object]] = []
    for idx, jobs in enumerate(phases, start=1):
        print(f"[phase {idx}] jobs={len(jobs)} workers={args.max_workers}", flush=True)
        is_scalar_phase = all(job.name.startswith("scalar_") for job in jobs)
        if is_scalar_phase:
            run_phase_continue(jobs, args.max_workers, failures)
        else:
            run_phase(jobs, args.max_workers, failures)
            if failures:
                break
    if failures:
        failure_path = LOG_ROOT / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"[stop] failures written to {failure_path}", flush=True)
        raise SystemExit(1)
    print("[done] all queued jobs finished", flush=True)


if __name__ == "__main__":
    main()
