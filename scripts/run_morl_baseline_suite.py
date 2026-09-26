from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


METHOD_TO_SCRIPT = {
    "capql": SCRIPT_ROOT / "run_capql_dynamic_baseline.py",
    "qpensieve": SCRIPT_ROOT / "run_qpensieve_dynamic_baseline.py",
    "pgmorl": SCRIPT_ROOT / "run_pgmorl_dynamic_baseline.py",
    "morlca": SCRIPT_ROOT / "run_morlca_dynamic_baseline.py",
}
EXPECTED_SHARED_PROTOCOL_REGIMES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["capql", "qpensieve", "pgmorl", "morlca"], choices=sorted(METHOD_TO_SCRIPT.keys()))
    parser.add_argument("--env-keys", nargs="+", default=["building", "evcharging", "cogen"], choices=["building", "evcharging", "cogen", "chlor_alkali"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--suite-name", type=str, default="morl_online_suite")
    parser.add_argument(
        "--save-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "morl_dynamic_baselines",
    )
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--trace-eval-episodes", type=int, default=3)
    parser.add_argument("--trace-recovery-window", type=int, default=12)
    parser.add_argument("--eval-delta-weight", type=float, default=0.5)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--episodes-per-regime", type=int, default=1)
    parser.add_argument("--regime-schedule", type=str, default="cyclic")
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--capql-total-timesteps", type=int, default=2048)
    parser.add_argument("--capql-start-steps", type=int, default=256)
    parser.add_argument("--capql-batch-size", type=int, default=256)
    parser.add_argument("--capql-hidden-size", type=int, default=256)
    parser.add_argument("--qpensieve-total-timesteps", type=int, default=2048)
    parser.add_argument("--qpensieve-start-steps", type=int, default=256)
    parser.add_argument("--qpensieve-batch-size", type=int, default=256)
    parser.add_argument("--qpensieve-prefer-num", type=int, default=4)
    parser.add_argument("--pgmorl-total-timesteps", type=int, default=4096)
    parser.add_argument("--pgmorl-num-steps", type=int, default=128)
    parser.add_argument("--pgmorl-warmup-iter", type=int, default=8)
    parser.add_argument("--pgmorl-update-iter", type=int, default=3)
    parser.add_argument("--pgmorl-num-processes", type=int, default=1)
    parser.add_argument("--morlca-total-timesteps", type=int, default=4096)
    parser.add_argument("--morlca-start-steps", type=int, default=256)
    parser.add_argument("--morlca-batch-size", type=int, default=256)
    parser.add_argument("--morlca-hidden-size", type=int, default=256)
    parser.add_argument("--morlca-aow-aux-coef", type=float, default=0.2)
    parser.add_argument("--morlca-eval-pref-bias-scale", type=float, default=0.75)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summary_complete(summary_path: Path, env_key: str) -> bool:
    if not summary_path.exists():
        return False
    shared_dir = summary_path.parent / "shared_regime_returns"
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
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
        len(regime_fronts) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(unique_ids) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(data.get("shared_regime_seed_plan", [])) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and json_count >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and npz_count >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == EXPECTED_SHARED_PROTOCOL_REGIMES
    )


def common_eval_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--eval-episodes",
        str(args.eval_episodes),
        "--trace-eval-episodes",
        str(args.trace_eval_episodes),
        "--trace-recovery-window",
        str(args.trace_recovery_window),
        "--eval-delta-weight",
        str(args.eval_delta_weight),
        "--shared-regime-eval-episodes",
        str(args.shared_regime_eval_episodes),
        "--shared-regime-seed-offset",
        str(args.shared_regime_seed_offset),
        "--fine-regime-clusters",
        str(args.fine_regime_clusters),
        "--fine-regime-catalog-episodes",
        str(args.fine_regime_catalog_episodes),
        "--fine-regime-context-steps",
        str(args.fine_regime_context_steps),
        "--episodes-per-regime",
        str(args.episodes_per_regime),
        "--regime-schedule",
        args.regime_schedule,
    ]
    if int(args.chlor_episode_length) > 0:
        common.extend(["--chlor-episode-length", str(args.chlor_episode_length)])
    return common


def build_method_args(method: str, args: argparse.Namespace) -> list[str]:
    if method == "capql":
        return [
            "--total-timesteps",
            str(args.capql_total_timesteps),
            "--start-steps",
            str(args.capql_start_steps),
            "--batch-size",
            str(args.capql_batch_size),
            "--hidden-size",
            str(args.capql_hidden_size),
        ]
    if method == "qpensieve":
        return [
            "--total-timesteps",
            str(args.qpensieve_total_timesteps),
            "--start-steps",
            str(args.qpensieve_start_steps),
            "--batch-size",
            str(args.qpensieve_batch_size),
            "--prefer-num",
            str(args.qpensieve_prefer_num),
        ]
    if method == "pgmorl":
        return [
            "--total-timesteps",
            str(args.pgmorl_total_timesteps),
            "--num-steps",
            str(args.pgmorl_num_steps),
            "--warmup-iter",
            str(args.pgmorl_warmup_iter),
            "--update-iter",
            str(args.pgmorl_update_iter),
            "--num-processes",
            str(args.pgmorl_num_processes),
        ]
    if method == "morlca":
        return [
            "--total-timesteps",
            str(args.morlca_total_timesteps),
            "--start-steps",
            str(args.morlca_start_steps),
            "--batch-size",
            str(args.morlca_batch_size),
            "--hidden-size",
            str(args.morlca_hidden_size),
            "--aow-aux-coef",
            str(args.morlca_aow_aux_coef),
            "--eval-pref-bias-scale",
            str(args.morlca_eval_pref_bias_scale),
        ]
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    args = parse_args()
    failures: list[dict[str, object]] = []
    suite_root = args.save_root / args.suite_name
    suite_root.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        script_path = METHOD_TO_SCRIPT[method]
        for env_key in args.env_keys:
            for seed in args.seeds:
                save_dir = suite_root / method / env_key / f"seed{seed}"
                summary_path = save_dir / "summary.json"
                if summary_complete(summary_path, env_key) and not args.force:
                    print(f"skip existing: {summary_path}")
                    continue
                if save_dir.exists() and not summary_complete(summary_path, env_key):
                    shutil.rmtree(save_dir)

                command = [
                    sys.executable,
                    str(script_path),
                    "--env-key",
                    env_key,
                    "--seed",
                    str(seed),
                    *build_method_args(method, args),
                    *common_eval_args(args),
                    "--save-dir",
                    str(save_dir),
                ]
                if args.skip_train:
                    command.append("--skip-train")
                print("running:", " ".join(command))
                if args.dry_run:
                    continue
                try:
                    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
                except subprocess.CalledProcessError as exc:
                    failure = {
                        "method": method,
                        "env_key": env_key,
                        "seed": seed,
                        "returncode": exc.returncode,
                    }
                    failures.append(failure)
                    print(f"failed: {json.dumps(failure)}")
                    if not args.continue_on_error:
                        raise

    if failures:
        failure_path = suite_root / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"wrote failures to {failure_path}")


if __name__ == "__main__":
    main()
