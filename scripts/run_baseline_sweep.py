from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_PYTHON = Path("/root/anaconda3/envs/morl-pareto/bin/python")
PYTHON_BIN = str(PREFERRED_PYTHON if PREFERRED_PYTHON.exists() else Path(sys.executable))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_CONFIG = {
    "building": {
        "env_name": "building_3d_dynamic",
        "obj_num": 3,
        "weights": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]],
    },
    "evcharging": {
        "env_name": "evcharging_dynamic",
        "obj_num": 3,
        "weights": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]],
    },
    "cogen": {
        "env_name": "cogen_dynamic",
        "obj_num": 4,
        "weights": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.5, 0.5, 0, 0],
            [0.5, 0, 0.5, 0],
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0.25, 0.25, 0.25, 0.25],
        ],
    },
    "chlor_alkali": {
        "env_name": "chlor_alkali_dynamic",
        "obj_num": 3,
        "weights": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]],
    },
}

DEFAULT_ALGS = ["a2c", "acer", "acktr", "ddpg", "deepq", "ppo1", "ppo2", "trpo_mpi"]
EXPECTED_SHARED_PROTOCOL_REGIMES = 20


def run_complete(run_dir: Path, env_key: str) -> bool:
    result_path = run_dir / "result.json"
    if not result_path.exists():
        return False
    try:
        payload = json.loads(result_path.read_text())
    except Exception:
        return False
    regime_rows = payload.get("regime_returns", [])
    shared_episode_seeds = [int(seed) for seed in payload.get("shared_eval_episode_seeds", [])]
    shared_plan = list(payload.get("shared_regime_seed_plan", []))
    planned_episode_seeds = [
        int(seed)
        for plan_row in shared_plan
        for seed in plan_row.get("episode_seeds", [])
    ]
    unique_ids = {int(row.get("regime_id", -1)) for row in regime_rows if int(row.get("regime_id", -1)) >= 0}
    shared_dir = run_dir / "shared_regime_returns"
    shared_json = list(shared_dir.glob("*.json")) if shared_dir.exists() else []
    shared_npz = list(shared_dir.glob("*.npz")) if shared_dir.exists() else []
    return (
        len(unique_ids) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(regime_rows) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(shared_json) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and len(shared_npz) >= EXPECTED_SHARED_PROTOCOL_REGIMES
        and shared_episode_seeds == planned_episode_seeds
        and len(shared_episode_seeds) == EXPECTED_SHARED_PROTOCOL_REGIMES
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-key", choices=sorted(ENV_CONFIG.keys()), required=True)
    parser.add_argument("--algs", nargs="+", default=DEFAULT_ALGS)
    parser.add_argument(
        "--weight-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of weight indices to run for the selected environment.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--total-timesteps", type=int, default=4096)
    parser.add_argument("--trace-eval-episodes", type=int, default=4)
    parser.add_argument("--shared-regime-eval-episodes", type=int, default=0)
    parser.add_argument("--shared-regime-seed-offset", type=int, default=0)
    parser.add_argument("--ev-moer-forecast-steps", type=int, default=36)
    parser.add_argument("--ev-project-action-in-env", type=int, default=1)
    parser.add_argument("--chlor-episode-length", type=int, default=288)
    parser.add_argument("--fine-regime-clusters", type=int, default=20)
    parser.add_argument("--fine-regime-catalog-episodes", type=int, default=256)
    parser.add_argument("--fine-regime-context-steps", type=int, default=4)
    parser.add_argument("--save-root", type=Path, default=PROJECT_ROOT / "results" / "scalarized_baselines")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ENV_CONFIG[args.env_key]
    allowed_weight_indices = None if args.weight_indices is None else set(args.weight_indices)
    failures = []
    for alg in args.algs:
        for seed in args.seeds:
            for idx, weights in enumerate(cfg["weights"]):
                if allowed_weight_indices is not None and idx not in allowed_weight_indices:
                    continue
                run_dir = args.save_root / args.env_key / alg / f"seed{seed}_w{idx}"
                if run_complete(run_dir, args.env_key) and not args.force:
                    print(f"skip existing complete run: {run_dir}")
                    continue
                command = [
                    PYTHON_BIN,
                    str(PROJECT_ROOT / "scripts" / "run_scalarized_baseline.py"),
                    "--alg",
                    alg,
                    "--env-name",
                    cfg["env_name"],
                    "--obj-num",
                    str(cfg["obj_num"]),
                    "--weights",
                    *[str(x) for x in weights],
                    "--seed",
                    str(seed),
                    "--total-timesteps",
                    str(args.total_timesteps),
                    "--trace-eval-episodes",
                    str(args.trace_eval_episodes),
                    "--shared-regime-eval-episodes",
                    str(args.shared_regime_eval_episodes),
                    "--shared-regime-seed-offset",
                    str(args.shared_regime_seed_offset),
                    "--ev-moer-forecast-steps",
                    str(args.ev_moer_forecast_steps),
                    "--ev-project-action-in-env",
                    str(args.ev_project_action_in_env),
                    "--fine-regime-clusters",
                    str(args.fine_regime_clusters),
                    "--fine-regime-catalog-episodes",
                    str(args.fine_regime_catalog_episodes),
                    "--fine-regime-context-steps",
                    str(args.fine_regime_context_steps),
                    "--save-dir",
                    str(run_dir),
                ]
                if args.env_key == "chlor_alkali":
                    command.extend(
                        [
                            "--chlor-episode-length",
                            str(args.chlor_episode_length),
                            "--chlor-regime-clusters",
                            str(args.fine_regime_clusters),
                        ]
                    )
                print("running:", " ".join(command))
                try:
                    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
                except subprocess.CalledProcessError as exc:
                    failure = {
                        "env_key": args.env_key,
                        "alg": alg,
                        "seed": seed,
                        "weight_idx": idx,
                        "weights": weights,
                        "returncode": exc.returncode,
                    }
                    failures.append(failure)
                    print(f"failed: {json.dumps(failure)}")
                    if not args.continue_on_error:
                        raise
    if failures:
        failure_path = args.save_root / args.env_key / "failures.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failures, indent=2))
        print(f"wrote failures to {failure_path}")


if __name__ == "__main__":
    main()
