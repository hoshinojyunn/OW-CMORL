#!/usr/bin/env python
"""Causal phase runner for short-budget OOD adaptation experiments.

The command-line experiment wiring is added only after method adapters have
passed the checkpoint-continuation capability audit.  The reusable function in
this module is intentionally dependency-injected so its causal ordering and
zero-shot guard can be tested without a simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.short_ood_adapters import AdaptationAdapter, LCPOAdapter, MORLCAAdapter, assert_unchanged
from src.short_ood_protocol import BudgetLedger, ShortOODManifest


EvaluateFn = Callable[[AdaptationAdapter, Any, str, int], Mapping[str, Any]]


def _close_if_possible(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _event_row(
    *,
    adapter: AdaptationAdapter,
    trajectory_seed: int,
    phase: str,
    budget: int,
    metrics: Mapping[str, Any],
    transition_count: int,
    update_count: float,
    elapsed_seconds: float,
    env_key: str | None = None,
) -> dict[str, Any]:
    return {
        "env_key": env_key,
        "method": adapter.method,
        "trajectory_seed": int(trajectory_seed),
        "phase": phase,
        "budget": int(budget),
        "state_hash": adapter.state_hash(),
        "archive_size": adapter.archive_size(),
        "transition_count": int(transition_count),
        "update_count": float(update_count),
        "wall_time_seconds": float(elapsed_seconds),
        **dict(metrics),
    }


def run_single_trajectory(
    *,
    adapter: AdaptationAdapter,
    manifest: ShortOODManifest,
    trajectory_seed: int,
    make_id_env: Callable[[], Any],
    make_ood_env: Callable[[], Any],
    evaluate: EvaluateFn,
    env_key: str | None = None,
) -> list[dict[str, Any]]:
    """Execute ID-pre, frozen OOD, causal adaptation, and frozen ID-return."""

    if int(trajectory_seed) not in manifest.trajectory_seeds:
        raise ValueError("trajectory seed is not declared by the manifest")
    ledger = BudgetLedger(manifest.budgets)
    rows: list[dict[str, Any]] = []
    transition_count = 0
    update_count = 0.0

    start_hash = adapter.state_hash()
    id_env = make_id_env()
    try:
        start = perf_counter()
        id_metrics = evaluate(adapter, id_env, "id_pre", 0)
        assert_unchanged(start_hash, adapter.state_hash())
        rows.append(
            _event_row(
                adapter=adapter,
                trajectory_seed=trajectory_seed,
                phase="id_pre",
                budget=0,
                metrics=id_metrics,
                transition_count=transition_count,
                update_count=update_count,
                elapsed_seconds=perf_counter() - start,
                env_key=env_key,
            )
        )
    finally:
        _close_if_possible(id_env)

    ood_env = make_ood_env()
    try:
        ledger.record(0)
        start = perf_counter()
        ood_zero_metrics = evaluate(adapter, ood_env, "ood", 0)
        assert_unchanged(start_hash, adapter.state_hash())
        rows.append(
            _event_row(
                adapter=adapter,
                trajectory_seed=trajectory_seed,
                phase="ood",
                budget=0,
                metrics=ood_zero_metrics,
                transition_count=transition_count,
                update_count=update_count,
                elapsed_seconds=perf_counter() - start,
                env_key=env_key,
            )
        )

        for target_budget in manifest.budgets[1:]:
            increment = ledger.remaining_to(target_budget)
            start = perf_counter()
            update_stats = dict(adapter.adapt(ood_env, increment))
            # ``budget`` is a cumulative refinement-update budget.  Each
            # native adapter reports the actual environment transitions used
            # to realize those updates, so PPO rollout length never becomes an
            # implicit or mislabeled part of the x-axis.
            transition_count += int(round(float(update_stats.get("transitions", increment))))
            update_count += float(update_stats.get("updates", increment))
            ledger.record(target_budget)
            ood_metrics = evaluate(adapter, ood_env, "ood", target_budget)
            rows.append(
                _event_row(
                    adapter=adapter,
                    trajectory_seed=trajectory_seed,
                    phase="ood",
                    budget=target_budget,
                    metrics=ood_metrics,
                    transition_count=transition_count,
                    update_count=update_count,
                    elapsed_seconds=perf_counter() - start,
                    env_key=env_key,
                )
            )
    finally:
        _close_if_possible(ood_env)

    id_return_env = make_id_env()
    try:
        return_hash = adapter.state_hash()
        start = perf_counter()
        return_metrics = evaluate(adapter, id_return_env, "id_return", manifest.budgets[-1])
        assert_unchanged(return_hash, adapter.state_hash())
        rows.append(
            _event_row(
                adapter=adapter,
                trajectory_seed=trajectory_seed,
                phase="id_return",
                budget=manifest.budgets[-1],
                metrics=return_metrics,
                transition_count=transition_count,
                update_count=update_count,
                elapsed_seconds=perf_counter() - start,
                env_key=env_key,
            )
        )
    finally:
        _close_if_possible(id_return_env)
    return rows


class PhaseEnvironment:
    """Owns one adaptation stream and produces isolated frozen-evaluation clones."""

    def __init__(self, make_env: Callable[[], Any]):
        self._make_env = make_env
        self._env = make_env()

    def clone(self) -> Any:
        return self._make_env()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def close(self) -> None:
        self._env.close()


class RuntimePhase:
    """Phase metadata for OW-CMORL, whose native updater creates vector envs."""

    def __init__(self, env_kwargs: dict[str, Any]):
        self.env_kwargs = dict(env_kwargs)

    def close(self) -> None:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _checkpoint_reference_point(profile: str, env_key: str) -> np.ndarray:
    """Derive a fixed HV reference from pre-existing held-out checkpoint fronts."""

    root = PROJECT_ROOT / "results_ood" / profile
    fronts: list[np.ndarray] = []
    for method_dir in root.iterdir() if root.exists() else []:
        source = method_dir / env_key
        csv_path = source / "front_points.csv"
        if csv_path.exists():
            try:
                values = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                values = np.atleast_2d(np.asarray(values, dtype=np.float64))
                if values.size:
                    fronts.append(values)
            except ValueError:
                pass
        obj_path = source / "final" / "objs.txt"
        if obj_path.exists():
            try:
                values = np.loadtxt(obj_path, delimiter=",")
                values = np.atleast_2d(np.asarray(values, dtype=np.float64))
                if values.size:
                    fronts.append(values)
            except ValueError:
                pass
    if not fronts:
        raise FileNotFoundError(f"No predeclared checkpoint fronts for profile={profile}, env={env_key}.")
    lower = np.concatenate(fronts, axis=0).min(axis=0)
    return lower - 0.1 * np.maximum(np.abs(lower), 1.0)


def _make_morlca_adapter(checkpoint_dir: Path, env_key: str, id_env_kwargs: dict[str, Any]) -> MORLCAAdapter:
    from src.baseline_envs import get_env_spec, make_vector_env
    from src.morl_ca_baseline import MORLCABaseline

    config = _load_json(checkpoint_dir / "config.json")
    env = make_vector_env(env_key, seed=int(config.get("seed", 0)), env_kwargs=id_env_kwargs)
    try:
        model = MORLCABaseline(
            obs_dim=int(env.observation_space.shape[0]),
            action_low=np.asarray(env.action_space.low, dtype=np.float32),
            action_high=np.asarray(env.action_space.high, dtype=np.float32),
            obj_num=int(get_env_spec(env_key).obj_num),
            hidden_dim=int(config["hidden_size"]),
            gamma=float(config["gamma"]),
            tau=float(config["tau"]),
            actor_lr=float(config["actor_lr"]),
            critic_lr=float(config["critic_lr"]),
            alpha_lr=float(config["alpha_lr"]),
            alpha_init=float(config["alpha_init"]),
            batch_size=int(config["batch_size"]),
            replay_size=max(50000, int(config["total_timesteps"]) * 2),
            start_steps=int(config["start_steps"]),
            updates_per_step=int(config["updates_per_step"]),
            aow_aux_coef=float(config["aow_aux_coef"]),
            eval_pref_bias_scale=float(config["eval_pref_bias_scale"]),
            device="cpu",
        )
    finally:
        env.close()
    model.load(checkpoint_dir / "model" / "morlca.pt")
    return MORLCAAdapter(model)


def _make_lcpo_adapter(checkpoint_dir: Path, env_key: str, id_env_kwargs: dict[str, Any]) -> LCPOAdapter:
    from src.baseline_envs import make_scalar_env
    from src.lcpo_baseline import LCPOTrainer

    config = _load_json(checkpoint_dir / "config.json")
    summary = _load_json(checkpoint_dir / "summary.json")
    preferences = [tuple(float(value) for value in preference) for preference in summary["preferences"]]
    env = make_scalar_env(env_key, seed=int(config.get("seed", 0)), env_kwargs=id_env_kwargs)
    try:
        obs = np.asarray(env.reset(), dtype=np.float32)
        action_low = np.asarray(env.action_space.low, dtype=np.float32)
        action_high = np.asarray(env.action_space.high, dtype=np.float32)
    finally:
        env.close()
    trainers = []
    for preference_id, _preference in enumerate(preferences):
        trainer = LCPOTrainer(
            obs_dim=len(obs),
            action_low=action_low,
            action_high=action_high,
            context_dim=len(obs),
            action_bins=int(config["action_bins"]),
            hidden_sizes=(int(config["hidden_size"]), int(config["hidden_size"])),
            learning_rate=float(config["learning_rate"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            entropy_coef=float(config["entropy_coef"]),
            trpo_kl_in=float(config["trpo_kl_in"]),
            trpo_kl_out=float(config["trpo_kl_out"]),
            trpo_damping=float(config["trpo_damping"]),
            trpo_dual=bool(config["trpo_dual"]),
            cg_iterations=int(config["cg_iterations"]),
            line_search_backtracks=int(config["line_search_backtracks"]),
            ood_capacity=int(config["ood_capacity"]),
            ood_recent_window=int(config["ood_recent_window"]),
            ood_threshold=float(config["ood_threshold"]),
            seed=int(config["seed"]) + preference_id,
        )
        trainer.load_state_dict(torch.load(checkpoint_dir / f"policy_{preference_id:02d}.pt", map_location="cpu"))
        trainers.append(trainer)
    return LCPOAdapter(trainers, preferences=preferences)


def _evaluate_front(
    adapter: AdaptationAdapter,
    phase_env: PhaseEnvironment,
    *,
    env_key: str,
    episode_seed: int,
    reference_point: np.ndarray,
) -> Mapping[str, Any]:
    from src.baseline_envs import get_env_spec
    from src.baseline_eval import evaluate_old_gym_policy
    from src.dynamic_morl.metrics import compute_front_metrics, pareto_front

    # The wrappers contain legacy simulators that may draw from process-global
    # RNGs.  Evaluation must therefore be deterministic without perturbing the
    # causal adaptation stream that follows it.
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(episode_seed))
        np.random.seed(int(episode_seed))
        torch.manual_seed(int(episode_seed))
        spec = get_env_spec(env_key)
        returns = []
        for preference in spec.preference_grid:
            objective, _trace = evaluate_old_gym_policy(
                phase_env.clone,
                lambda observation, preference=preference: adapter.action(observation, preference),
                episodes=1,
                episode_seeds=[int(episode_seed)],
                record_trace=False,
            )
            returns.append(np.asarray(objective, dtype=np.float64))
        front = pareto_front(np.asarray(returns, dtype=np.float64))
        hv, eu, _sp = compute_front_metrics(front, reference_point, eval_delta_weight=0.25)
        return {
            "HV": float(hv),
            "EU": float(eu),
            "front_size": int(len(front)),
            "front_points": front.tolist(),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ood_v3_site_tariff_trace")
    parser.add_argument("--env-keys", nargs="+", default=["building", "evcharging", "cogen", "chlor_alkali"])
    parser.add_argument("--methods", nargs="+", default=["dynamic"], choices=["dynamic"])
    parser.add_argument("--trajectory-seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=[0, 8, 16, 32],
        help="Cumulative OW-CMORL local-refinement update budgets.",
    )
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "results_ood_short")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood_short_id",
        help="Root containing newly trained OW-CMORL ID runs with final/short_ood_runtime.pt.",
    )
    parser.add_argument(
        "--local-lr-scale",
        type=float,
        default=1.0,
        help="Multiplier applied only to the PPO optimizer during local OOD refinement.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from src.morl_ca_baseline import set_global_seed
    from src.ood_protocol import load_env_config_json
    from src.owcmorl_short_ood import OWCMORLAdapter

    args = parse_args()
    manifest = ShortOODManifest(
        budgets=tuple(args.budgets),
        trajectory_seeds=tuple(args.trajectory_seeds),
        profile=str(args.profile),
    )
    torch.set_num_threads(1)
    bundle_root = PROJECT_ROOT / "analysis" / "ood_protocols" / manifest.profile
    checkpoint_root = PROJECT_ROOT / "results_ood" / manifest.profile

    for env_key in args.env_keys:
        if env_key not in {"building", "evcharging", "cogen", "chlor_alkali"}:
            raise ValueError(f"Unsupported environment: {env_key}")
        bundle_dir = bundle_root / env_key
        id_env_kwargs = load_env_config_json(bundle_dir / "train_env_config.json")
        ood_env_kwargs = load_env_config_json(bundle_dir / "eval_env_config.json")
        for method in args.methods:
            for trajectory_seed in manifest.trajectory_seeds:
                out_dir = args.out_root / manifest.profile / method / env_key / f"seed_{trajectory_seed}"
                events_path = out_dir / "events.csv"
                if events_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Existing result: {events_path}; use --overwrite to replace it.")
                set_global_seed(int(trajectory_seed))
                checkpoint_dir = args.runtime_root / manifest.profile / method / env_key
                adapter = OWCMORLAdapter(checkpoint_dir, local_lr_scale=float(args.local_lr_scale))
                adapter.set_trajectory_seed(int(trajectory_seed))
                make_id_env = lambda: RuntimePhase(id_env_kwargs)
                make_ood_env = lambda: RuntimePhase(ood_env_kwargs)
                rows = run_single_trajectory(
                    adapter=adapter,
                    manifest=manifest,
                    trajectory_seed=int(trajectory_seed),
                    make_id_env=make_id_env,
                    make_ood_env=make_ood_env,
                    evaluate=lambda current_adapter, phase_env, _phase, _budget: current_adapter.evaluate(
                        phase_env.env_kwargs,
                        episode_seed=int(trajectory_seed),
                    ),
                    env_key=env_key,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                _write_rows(events_path, rows)
                (out_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            **manifest.as_dict(),
                            "env_key": env_key,
                            "method": method,
                            "trajectory_seed": int(trajectory_seed),
                            "id_env_kwargs": id_env_kwargs,
                            "ood_env_kwargs": ood_env_kwargs,
                            "source_checkpoint": str(checkpoint_dir),
                            "budget_unit": "cumulative local-refinement updates",
                            "transition_accounting": "adapter-reported OOD rollout transitions; excluded from the response-curve x-axis",
                            "evaluation": "one fixed-seed evaluation episode per persisted archive snapshot; evaluation does not update runtime state",
                            "local_lr_scale": float(args.local_lr_scale),
                        },
                        indent=2,
                    )
                )
                (out_dir / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "initial_state_hash": rows[0]["state_hash"],
                            "zero_shot_state_hash": rows[1]["state_hash"],
                            "final_state_hash": rows[-1]["state_hash"],
                            "transition_count": rows[-1]["transition_count"],
                            "update_count": rows[-1]["update_count"],
                            "hnsw_backend": getattr(adapter, "_last_backend", "unqueried"),
                            "retrieved_snapshots": int(getattr(adapter, "_last_retrieved", 0)),
                        },
                        indent=2,
                    )
                )
                print(f"[short-ood] completed {method}/{env_key}/seed_{trajectory_seed}", flush=True)


if __name__ == "__main__":
    main()
