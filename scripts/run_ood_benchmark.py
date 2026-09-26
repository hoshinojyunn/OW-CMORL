from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.generate_20_regime_report as id_report
from src.ood_protocol import DEFAULT_OOD_PROFILE


OOD_RESULTS_ROOT = PROJECT_ROOT / "results_ood"
OOD_BUNDLE_ROOT = PROJECT_ROOT / "analysis" / "ood_protocols"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"


METHOD_TO_SCRIPT = {
    "capql": SCRIPT_ROOT / "run_capql_dynamic_baseline.py",
    "qpensieve": SCRIPT_ROOT / "run_qpensieve_dynamic_baseline.py",
    "pgmorl": SCRIPT_ROOT / "run_pgmorl_dynamic_baseline.py",
    "morlca": SCRIPT_ROOT / "run_morlca_dynamic_baseline.py",
    "lcpo": SCRIPT_ROOT / "run_lcpo_dynamic_baseline.py",
}

BASELINE_OVERRIDDEN_KEYS = {
    "env_key",
    "save_dir",
    "skip_train",
    "train_only",
    "env_config_json",
    "train_env_config_json",
    "shared_plan_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD reevaluation for OW-CMORL and MORL baselines.")
    parser.add_argument(
        "--env-keys",
        nargs="+",
        default=["building", "evcharging", "cogen", "chlor_alkali"],
        choices=["building", "evcharging", "cogen", "chlor_alkali"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["dynamic", "capql", "qpensieve", "pgmorl", "morlca", "lcpo"],
        choices=["dynamic", "capql", "qpensieve", "pgmorl", "morlca", "lcpo"],
    )
    parser.add_argument("--profile", type=str, default=DEFAULT_OOD_PROFILE)
    parser.add_argument("--bundle-root", type=Path, default=OOD_BUNDLE_ROOT)
    parser.add_argument("--results-root", type=Path, default=OOD_RESULTS_ROOT)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--retrain-env-keys",
        nargs="+",
        default=["chlor_alkali"],
        choices=["building", "evcharging", "cogen", "chlor_alkali"],
        help="Environments that must be retrained on the ID split before OOD evaluation.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-id-trained-profile",
        type=str,
        default=None,
        help="Reuse matching ID-trained artifacts from results_root/<profile> for OOD reevaluation.",
    )
    parser.add_argument(
        "--reuse-unchanged-ood-results-from",
        type=str,
        default=None,
        help="Reuse completed evaluation directories only when OOD config and shared-plan payloads are identical.",
    )
    parser.add_argument("--lcpo-total-timesteps", type=int, default=2048)
    parser.add_argument("--lcpo-batch-size", type=int, default=128)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force rebuilding retrain artifacts even if a matching retrain already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ood_paths(bundle_root: Path, profile: str, env_key: str) -> tuple[Path, Path, Path, int]:
    env_dir = bundle_root / profile / env_key
    env_payload = json.loads((env_dir / "eval_env_config.json").read_text())
    return (
        env_dir / "train_env_config.json",
        env_dir / "eval_env_config.json",
        env_dir / "shared_plan.json",
        int(env_payload["shared_regime_eval_episodes"]),
    )


def _discover_dynamic_sources() -> dict[str, Path]:
    rows = id_report._dynamic_rows()
    return {str(row["env_key"]): Path(str(row["source"])) for row in rows}


def _discover_morl_sources() -> dict[tuple[str, str], Path]:
    rows = id_report._morl_rows()
    sources: dict[tuple[str, str], Path] = {}
    for row in rows:
        method = str(row["method"]).replace("q_pensieve", "qpensieve")
        sources[(method, str(row["env_key"]))] = Path(str(row["source"])).parent
    return sources


def _load_env_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("env_kwargs", payload) if isinstance(payload, dict) else {}


def _dynamic_complete(target_dir: Path, expected_env_config_json: Path | None = None) -> bool:
    if not (target_dir / "final" / "shared_eval_summary.json").exists():
        return False
    if expected_env_config_json is None:
        return True
    args_path = target_dir / "args.txt"
    if not args_path.exists():
        return False
    try:
        raw_args = ast.literal_eval(args_path.read_text())
    except Exception:
        return False
    if not isinstance(raw_args, list):
        return False
    expected = str(expected_env_config_json.resolve())
    for idx, token in enumerate(raw_args[:-1]):
        if str(token) == "--env-config-json":
            return str(Path(str(raw_args[idx + 1])).resolve()) == expected
    return False


def _baseline_complete(target_dir: Path, expected_env_config_json: Path | None = None) -> bool:
    summary_path = target_dir / "summary.json"
    if not summary_path.exists():
        return False
    if expected_env_config_json is None:
        return True
    payload = json.loads(summary_path.read_text())
    env_kwargs = payload.get("env_kwargs", {}) if isinstance(payload, dict) else {}
    if not isinstance(env_kwargs, dict):
        return False
    expected_env_kwargs = _load_env_config(expected_env_config_json)
    return all(env_kwargs.get(key) == value for key, value in expected_env_kwargs.items())


def _baseline_retrain_ready(method: str, target_dir: Path, expected_env_config_json: Path | None = None) -> bool:
    config_path = target_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config_payload = json.loads(config_path.read_text())
    except Exception:
        return False
    if expected_env_config_json is not None:
        expected = str(expected_env_config_json.resolve())
        actual = config_payload.get("env_config_json")
        if actual is None or str(Path(str(actual)).resolve()) != expected:
            return False
    required: dict[str, tuple[str, ...]] = {
        "capql": ("model/policy.pt", "model/critic.pt"),
        "qpensieve": ("model/policy_final.pth", "model/critic_final.pth", "model/critic_target.pth"),
        "morlca": ("model/morlca.pt",),
        "pgmorl": ("final/EP_policy_0.pt",),
    }
    return all((target_dir / rel_path).exists() for rel_path in required[method])


def _reuse_train_dir(
    *,
    results_root: Path,
    profile: str | None,
    method: str,
    env_key: str,
) -> Path | None:
    if profile is None:
        return None
    candidate = results_root / profile / f"{method}_train" / env_key
    return candidate if candidate.exists() else None


def _reuse_unchanged_ood_result(
    *,
    results_root: Path,
    bundle_root: Path,
    source_profile: str | None,
    target_profile: str,
    method: str,
    env_key: str,
    target_dir: Path,
) -> bool:
    if source_profile is None or target_dir.exists():
        return False
    source_dir = results_root / source_profile / method / env_key
    source_bundle = bundle_root / source_profile / env_key
    target_bundle = bundle_root / target_profile / env_key
    source_config = source_bundle / "eval_env_config.json"
    target_config = target_bundle / "eval_env_config.json"
    source_plan = source_bundle / "shared_plan.json"
    target_plan = target_bundle / "shared_plan.json"
    if not all(path.exists() for path in (source_dir, source_config, target_config, source_plan, target_plan)):
        return False
    source_env_kwargs = json.loads(source_config.read_text()).get("env_kwargs", {})
    target_env_kwargs = json.loads(target_config.read_text()).get("env_kwargs", {})
    if source_env_kwargs != target_env_kwargs or json.loads(source_plan.read_text()) != json.loads(target_plan.read_text()):
        return False
    required = (source_dir / "final" / "shared_eval_summary.json",) if method == "dynamic" else (source_dir / "summary.json",)
    if not all(path.exists() for path in required):
        return False
    shutil.copytree(source_dir, target_dir)
    provenance = {
        "source_profile": source_profile,
        "source_dir": str(source_dir),
        "reason": "Resolved OOD evaluation config and shared-plan payload are byte-identical.",
    }
    (target_dir / "ood_reuse_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"reused unchanged OOD result: {source_dir} -> {target_dir}")
    return True


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("running:", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _append_cli_value(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def _baseline_source_config(source_dir: Path) -> dict[str, Any]:
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return {}
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        return {}
    return payload


def _baseline_command(
    *,
    python_bin: str,
    method: str,
    source_dir: Path,
    env_key: str,
    save_dir: Path,
    env_config_json: Path,
    shared_regime_eval_episodes: int | None = None,
    shared_plan_json: Path | None = None,
    skip_train: bool,
) -> list[str]:
    command = [
        python_bin,
        str(METHOD_TO_SCRIPT[method]),
        "--env-key",
        env_key,
    ]
    source_config = _baseline_source_config(source_dir)
    for key, value in source_config.items():
        if key in BASELINE_OVERRIDDEN_KEYS:
            continue
        _append_cli_value(command, f"--{key.replace('_', '-')}", value)
    if skip_train:
        command.append("--skip-train")
    if shared_regime_eval_episodes is not None:
        _append_cli_value(
            command,
            "--shared-regime-eval-episodes",
            int(shared_regime_eval_episodes),
        )
    if shared_plan_json is not None:
        _append_cli_value(command, "--shared-plan-json", str(shared_plan_json))
    _append_cli_value(command, "--env-config-json", str(env_config_json))
    _append_cli_value(command, "--save-dir", str(save_dir))
    return command


def main() -> None:
    args = parse_args()
    dynamic_sources = _discover_dynamic_sources()
    morl_sources = _discover_morl_sources()
    retrain_env_keys = {str(env_key) for env_key in args.retrain_env_keys}

    for env_key in args.env_keys:
        train_env_config_json, eval_env_config_json, shared_plan_json, shared_eval_episodes = _ood_paths(
            args.bundle_root,
            args.profile,
            env_key,
        )
        require_retrain = env_key in retrain_env_keys

        if "lcpo" in args.methods:
            target_dir = args.results_root / args.profile / "lcpo" / env_key
            if _baseline_complete(target_dir, eval_env_config_json) and not args.force:
                print(f"skip existing: {target_dir}")
            else:
                if target_dir.exists():
                    if not args.force:
                        raise FileExistsError(f"Target exists: {target_dir}")
                    shutil.rmtree(target_dir)
                command = [
                    args.python_bin,
                    str(METHOD_TO_SCRIPT["lcpo"]),
                    "--env-key",
                    env_key,
                    "--total-timesteps",
                    str(args.lcpo_total_timesteps),
                    "--batch-size",
                    str(args.lcpo_batch_size),
                    "--shared-regime-eval-episodes",
                    str(shared_eval_episodes),
                    "--shared-plan-json",
                    str(shared_plan_json),
                    "--train-env-config-json",
                    str(train_env_config_json),
                    "--env-config-json",
                    str(eval_env_config_json),
                    "--save-dir",
                    str(target_dir),
                ]
                _run_command(command, dry_run=args.dry_run)

        if "dynamic" in args.methods:
            source_dir = dynamic_sources.get(env_key)
            if source_dir is None:
                raise FileNotFoundError(f"Missing dynamic source for env={env_key}")
            train_dir = args.results_root / args.profile / "dynamic_train" / env_key
            target_dir = args.results_root / args.profile / "dynamic" / env_key
            if _reuse_unchanged_ood_result(
                results_root=args.results_root,
                bundle_root=args.bundle_root,
                source_profile=args.reuse_unchanged_ood_results_from,
                target_profile=args.profile,
                method="dynamic",
                env_key=env_key,
                target_dir=target_dir,
            ):
                print(f"skip reevaluation after guarded result reuse: {target_dir}")
            if require_retrain:
                reused_train_dir = _reuse_train_dir(
                    results_root=args.results_root,
                    profile=args.reuse_id_trained_profile,
                    method="dynamic",
                    env_key=env_key,
                )
                if reused_train_dir is not None and not args.force_retrain:
                    source_for_eval = reused_train_dir
                elif not _dynamic_complete(train_dir, train_env_config_json) or args.force_retrain:
                    command = [
                        args.python_bin,
                        str(SCRIPT_ROOT / "train_dynamic_from_source.py"),
                        "--source-run-dir",
                        str(source_dir),
                        "--target-run-dir",
                        str(train_dir),
                        "--env-config-json",
                        str(train_env_config_json),
                    ]
                    if args.dry_run:
                        command.append("--dry-run")
                    _run_command(command, dry_run=args.dry_run)
                    source_for_eval = train_dir
                else:
                    source_for_eval = train_dir
            else:
                source_for_eval = source_dir
            if _dynamic_complete(target_dir) and not args.force:
                print(f"skip existing: {target_dir}")
            else:
                command = [
                    args.python_bin,
                    str(SCRIPT_ROOT / "reevaluate_dynamic_run_shared_protocol.py"),
                    "--source-run-dir",
                    str(source_for_eval),
                    "--target-run-dir",
                    str(target_dir),
                    "--shared-regime-eval-episodes",
                    str(shared_eval_episodes),
                    "--shared-plan-json",
                    str(shared_plan_json),
                    "--env-config-json",
                    str(eval_env_config_json),
                ]
                if args.force:
                    command.append("--force")
                _run_command(command, dry_run=args.dry_run)

        for method in [item for item in args.methods if item not in {"dynamic", "lcpo"}]:
            source_dir = morl_sources.get((method, env_key))
            if source_dir is None:
                raise FileNotFoundError(f"Missing baseline source for method={method}, env={env_key}")
            train_dir = args.results_root / args.profile / f"{method}_train" / env_key
            target_dir = args.results_root / args.profile / method / env_key
            if _reuse_unchanged_ood_result(
                results_root=args.results_root,
                bundle_root=args.bundle_root,
                source_profile=args.reuse_unchanged_ood_results_from,
                target_profile=args.profile,
                method=method,
                env_key=env_key,
                target_dir=target_dir,
            ):
                continue
            if require_retrain:
                reused_train_dir = _reuse_train_dir(
                    results_root=args.results_root,
                    profile=args.reuse_id_trained_profile,
                    method=method,
                    env_key=env_key,
                )
                if reused_train_dir is not None and not args.force_retrain:
                    source_dir = reused_train_dir
                elif not _baseline_retrain_ready(method, train_dir, train_env_config_json) or args.force_retrain:
                    if train_dir.exists():
                        if not args.force_retrain:
                            raise FileExistsError(f"Target exists: {train_dir}")
                        shutil.rmtree(train_dir)
                    if not args.dry_run:
                        shutil.copytree(source_dir, train_dir)
                    train_command = _baseline_command(
                        python_bin=args.python_bin,
                        method=method,
                        source_dir=source_dir,
                        env_key=env_key,
                        save_dir=train_dir,
                        env_config_json=train_env_config_json,
                        skip_train=False,
                    )
                    train_command.append("--train-only")
                    _run_command(train_command, dry_run=args.dry_run)
                    source_dir = train_dir
                else:
                    source_dir = train_dir
            if _baseline_complete(target_dir) and not args.force:
                print(f"skip existing: {target_dir}")
                continue
            if target_dir.exists():
                if not args.force:
                    raise FileExistsError(f"Target exists: {target_dir}")
                shutil.rmtree(target_dir)
            if not args.dry_run:
                shutil.copytree(source_dir, target_dir)
            command = _baseline_command(
                python_bin=args.python_bin,
                method=method,
                source_dir=source_dir,
                env_key=env_key,
                save_dir=target_dir,
                env_config_json=eval_env_config_json,
                shared_regime_eval_episodes=shared_eval_episodes,
                shared_plan_json=shared_plan_json,
                skip_train=True,
            )
            _run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
