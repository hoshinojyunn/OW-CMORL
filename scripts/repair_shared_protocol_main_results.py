from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROOT = PROJECT_ROOT / "results_shared_protocol"
REPAIR_ROOT = PROJECT_ROOT / "results_shared_protocol_repair"
LOG_ROOT = REPAIR_ROOT / "logs_main_protocol_fix"
LOCK_PATH = LOG_ROOT / "repair.lock"
TARGET_REGIMES = 20
PREFERRED_PYTHON = Path("/root/anaconda3/envs/morl-pareto/bin/python")
PYTHON_BIN = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable))

METHOD_TO_SCRIPT = {
    "capql": PROJECT_ROOT / "scripts" / "run_capql_dynamic_baseline.py",
    "pgmorl": PROJECT_ROOT / "scripts" / "run_pgmorl_dynamic_baseline.py",
    "qpensieve": PROJECT_ROOT / "scripts" / "run_qpensieve_dynamic_baseline.py",
}


@dataclass
class Job:
    name: str
    command: list[str]
    check_done: callable
    prepare: callable | None = None


_LOCK_FP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair main shared-protocol results without retraining models."
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["dynamic", "morl"],
        default=["dynamic", "morl"],
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _planned_episode_seeds(plan: list[dict[str, object]]) -> list[int]:
    return [
        int(seed)
        for plan_row in plan
        for seed in plan_row.get("episode_seeds", [])
    ]


def dynamic_result_valid(run_dir: Path) -> bool:
    summary_path = run_dir / "final" / "shared_eval_summary.json"
    regime_path = run_dir / "regime_fronts" / "shared_final.json"
    shared_dir = run_dir / "shared_regime_returns"
    if not summary_path.exists() or not regime_path.exists() or not shared_dir.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        payload = json.loads(regime_path.read_text())
    except Exception:
        return False
    regimes = list(payload.get("regimes", []))
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in regimes
        if int(row.get("regime_id", -1)) >= 0
    }
    shared_episode_seeds = [int(seed) for seed in summary.get("shared_eval_episode_seeds", [])]
    shared_plan = list(summary.get("shared_regime_seed_plan", []))
    planned_episode_seeds = _planned_episode_seeds(shared_plan)
    per_sample_trace = list(summary.get("per_sample_trace_metrics", []))
    json_count = len(list(shared_dir.glob("*.json")))
    npz_count = len(list(shared_dir.glob("*.npz")))
    return (
        len(regimes) >= TARGET_REGIMES
        and len(unique_ids) >= TARGET_REGIMES
        and len(shared_plan) >= TARGET_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == TARGET_REGIMES
        and json_count >= TARGET_REGIMES
        and npz_count >= TARGET_REGIMES
        and len(per_sample_trace) > 0
    )


def _has_final_policy(run_dir: Path) -> bool:
    final_dir = run_dir / "final"
    return final_dir.exists() and any(final_dir.glob("EP_policy_*.pt"))


def _find_dynamic_source(env_key: str) -> Path | None:
    preferred = PRIMARY_ROOT / f"sharedprot_main_{env_key}_dynamic_seed0"
    if _has_final_policy(preferred):
        return preferred
    candidates = sorted(PRIMARY_ROOT.glob(f"*{env_key}*dynamic*"))
    ranked: list[tuple[int, int, Path]] = []
    for candidate in candidates:
        if not candidate.is_dir() or not _has_final_policy(candidate):
            continue
        score = 0
        if candidate.name.startswith("sharedprot_main_"):
            score += 100
        if "pre_hybrid" in candidate.name:
            score += 20
        if "protocolfix" in candidate.name:
            score += 10
        policy_count = len(list((candidate / "final").glob("EP_policy_*.pt")))
        ranked.append((score, policy_count, candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return ranked[0][2]


def morl_result_valid(summary_path: Path) -> bool:
    shared_dir = summary_path.parent / "shared_regime_returns"
    if not summary_path.exists() or not shared_dir.exists():
        return False
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return False
    regime_fronts = list(data.get("regime_fronts", []))
    unique_ids = {
        int(row.get("regime_id", -1))
        for row in regime_fronts
        if int(row.get("regime_id", -1)) >= 0
    }
    shared_episode_seeds = [int(seed) for seed in data.get("shared_eval_episode_seeds", [])]
    shared_plan = list(data.get("shared_regime_seed_plan", []))
    planned_episode_seeds = _planned_episode_seeds(shared_plan)
    json_count = len(list(shared_dir.glob("*.json")))
    npz_count = len(list(shared_dir.glob("*.npz")))
    seed_match = (
        shared_episode_seeds == planned_episode_seeds
        or (
            len(shared_episode_seeds) >= TARGET_REGIMES
            and len(planned_episode_seeds) == TARGET_REGIMES
            and shared_episode_seeds[:TARGET_REGIMES] == planned_episode_seeds
        )
    )
    return (
        len(regime_fronts) >= TARGET_REGIMES
        and len(unique_ids) >= TARGET_REGIMES
        and len(shared_plan) >= TARGET_REGIMES
        and seed_match
        and json_count >= TARGET_REGIMES
        and npz_count >= TARGET_REGIMES
    )


def _run_with_log(job: Job) -> tuple[str, int]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if job.prepare is not None:
        job.prepare()
    log_path = LOG_ROOT / f"{job.name}.log"
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write(f"STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
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
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        fp.write(f"FINISHED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fp.write(f"RETURNCODE: {proc.returncode}\n")
    return job.name, int(proc.returncode)


def _acquire_lock() -> None:
    global _LOCK_FP
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    _LOCK_FP = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(_LOCK_FP.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"repair lock is already held: {LOCK_PATH}") from exc
    _LOCK_FP.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}, indent=2))
    _LOCK_FP.flush()


def _prepare_target_copy(source_dir: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if not force:
            return
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)


def _load_config(source_dir: Path) -> dict[str, object]:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return {}


def _append_opt(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def build_dynamic_jobs(force: bool) -> list[Job]:
    jobs: list[Job] = []
    for env_key in ["building", "evcharging", "cogen", "chlor_alkali"]:
        source_dir = _find_dynamic_source(env_key)
        if source_dir is None:
            continue
        target_dir = REPAIR_ROOT / f"sharedprot_main_{env_key}_dynamic_seed0"
        should_force = force or env_key in {"building", "cogen"}
        if dynamic_result_valid(target_dir) and not should_force:
            continue
        jobs.append(
            Job(
                name=f"repair_dynamic_{env_key}",
                command=[
                    PYTHON_BIN,
                    str(PROJECT_ROOT / "scripts" / "reevaluate_dynamic_run_shared_protocol.py"),
                    "--source-run-dir",
                    str(source_dir),
                    "--target-run-dir",
                    str(target_dir),
                    "--shared-regime-eval-episodes",
                    str(TARGET_REGIMES),
                    "--shared-regime-seed-offset",
                    "0",
                    "--force",
                ],
                check_done=lambda target_dir=target_dir: dynamic_result_valid(target_dir),
            )
        )
    return jobs


def _find_morl_source(method: str, env_key: str) -> Path | None:
    method_dir = method
    root = PRIMARY_ROOT / "morl_dynamic_baselines"
    candidates = sorted(root.glob(f"*/{method_dir}/{env_key}/seed0"))

    def _checkpoint_score(candidate: Path) -> tuple[int, int, str]:
        suite_name = candidate.parts[-4]
        summary_exists = int((candidate / "summary.json").exists())
        if method == "pgmorl":
            policy_count = len(list((candidate / "final").glob("EP_policy_*.pt")))
            env_param_count = len(list((candidate / "final").glob("EP_env_params_*.pkl")))
            return (summary_exists, int(policy_count > 0), policy_count + env_param_count, suite_name)
        if method == "capql":
            model_dir = candidate / "model"
            has_policy = int((model_dir / "policy.pt").exists())
            has_critic = int((model_dir / "critic.pt").exists())
            return (summary_exists, has_policy and has_critic, has_policy + has_critic, suite_name)
        if method == "qpensieve":
            model_dir = candidate / "model"
            required = [
                model_dir / "policy_final.pth",
                model_dir / "critic_final.pth",
                model_dir / "critic_target.pth",
            ]
            present = sum(int(path.exists()) for path in required)
            return (summary_exists, int(present == len(required)), present, suite_name)
        return (summary_exists, 0, 0, suite_name)

    ranked: list[tuple[tuple[int, int, str], Path]] = []
    for candidate in candidates:
        suite_name = candidate.parts[-4]
        if suite_name.startswith("_") or "smoke" in suite_name:
            continue
        ranked.append((_checkpoint_score(candidate), candidate))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][1]
    if not (best / "summary.json").exists():
        return None
    return best


def build_morl_command(method: str, env_key: str, target_dir: Path, source_dir: Path) -> list[str]:
    config = _load_config(source_dir)
    command = [
        PYTHON_BIN,
        str(METHOD_TO_SCRIPT[method]),
        "--env-key",
        env_key,
        "--seed",
        "0",
        "--eval-episodes",
        str(config.get("eval_episodes", 3)),
        "--trace-eval-episodes",
        str(config.get("trace_eval_episodes", 3)),
        "--trace-recovery-window",
        str(config.get("trace_recovery_window", 12)),
        "--eval-delta-weight",
        str(config.get("eval_delta_weight", 0.5)),
        "--shared-regime-eval-episodes",
        str(config.get("shared_regime_eval_episodes", TARGET_REGIMES) or TARGET_REGIMES),
        "--shared-regime-seed-offset",
        str(config.get("shared_regime_seed_offset", 0)),
        "--episodes-per-regime",
        str(config.get("episodes_per_regime", 1)),
        "--regime-schedule",
        str(config.get("regime_schedule", "cyclic")),
        "--chlor-episode-length",
        str(config.get("chlor_episode_length", 288)),
        "--fine-regime-clusters",
        str(config.get("fine_regime_clusters", TARGET_REGIMES)),
        "--fine-regime-catalog-episodes",
        str(config.get("fine_regime_catalog_episodes", 256)),
        "--fine-regime-context-steps",
        str(config.get("fine_regime_context_steps", 4)),
        "--skip-train",
        "--save-dir",
        str(target_dir),
    ]
    if method == "capql":
        _append_opt(command, "--total-timesteps", config.get("total_timesteps"))
        _append_opt(command, "--batch-size", config.get("batch_size"))
        _append_opt(command, "--hidden-size", config.get("hidden_size"))
        _append_opt(command, "--start-steps", config.get("start_steps"))
        _append_opt(command, "--updates-per-step", config.get("updates_per_step"))
        _append_opt(command, "--angle", config.get("angle"))
    elif method == "qpensieve":
        _append_opt(command, "--total-timesteps", config.get("total_timesteps"))
        _append_opt(command, "--batch-size", config.get("batch_size"))
        _append_opt(command, "--hidden-size", config.get("hidden_size"))
        _append_opt(command, "--start-steps", config.get("start_steps"))
        _append_opt(command, "--updates-per-step", config.get("updates_per_step"))
        _append_opt(command, "--prefer-num", config.get("prefer_num"))
        _append_opt(command, "--q-frequency", config.get("q_frequency"))
    elif method == "pgmorl":
        _append_opt(command, "--total-timesteps", config.get("total_timesteps"))
        _append_opt(command, "--num-steps", config.get("num_steps"))
        _append_opt(command, "--warmup-iter", config.get("warmup_iter"))
        _append_opt(command, "--update-iter", config.get("update_iter"))
        _append_opt(command, "--num-processes", config.get("num_processes"))
    return command


def build_morl_jobs(force: bool) -> list[Job]:
    jobs: list[Job] = []
    for env_key in ["building", "evcharging", "cogen", "chlor_alkali"]:
        for method in ["capql", "pgmorl", "qpensieve"]:
            source_dir = _find_morl_source(method, env_key)
            if source_dir is None:
                continue
            rel = source_dir.relative_to(PRIMARY_ROOT)
            target_dir = REPAIR_ROOT / rel
            target_summary = target_dir / "summary.json"
            if morl_result_valid(target_summary) and not force:
                continue
            jobs.append(
                Job(
                    name=f"repair_{method}_{env_key}",
                    command=build_morl_command(method, env_key, target_dir, source_dir),
                    check_done=lambda target_summary=target_summary: morl_result_valid(target_summary),
                    prepare=lambda source_dir=source_dir, target_dir=target_dir, force=force: _prepare_target_copy(
                        source_dir,
                        target_dir,
                        force=force,
                    ),
                )
            )
    return jobs


def main() -> None:
    args = parse_args()
    _acquire_lock()
    jobs: list[Job] = []
    if "dynamic" in args.phases:
        jobs.extend(build_dynamic_jobs(force=args.force))
    if "morl" in args.phases:
        jobs.extend(build_morl_jobs(force=args.force))

    if not jobs:
        print("no repair jobs needed")
        return

    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        future_map = {executor.submit(_run_with_log, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                _name, returncode = future.result()
            except Exception as exc:
                failures.append({"job": job.name, "error": str(exc)})
                continue
            if returncode != 0 or not job.check_done():
                failures.append({"job": job.name, "returncode": returncode})

    status = {
        "jobs": [job.name for job in jobs],
        "failures": failures,
        "log_dir": str(LOG_ROOT),
    }
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "repair_status.json").write_text(json.dumps(status, indent=2))
    if failures:
        raise SystemExit(f"repair failures: {json.dumps(failures, ensure_ascii=False)}")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
