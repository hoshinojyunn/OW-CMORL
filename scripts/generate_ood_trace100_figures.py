#!/usr/bin/env python
"""Create illustrative 100-regime ID/OOD context trajectories and support checks.

The figures visualize the pre-registered OOD families; benchmark results still
use the smaller deterministic shared plans emitted by build_ood_shared_plans.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_envs import build_dynamic_env, get_env_spec
from src.ood_protocol import DEFAULT_OOD_PROFILE, resolve_ood_eval_env_config, resolve_ood_train_env_config


ENV_ORDER = ["building", "evcharging", "cogen", "chlor_alkali"]
PLOT_FIELDS = {
    "building": ["occupancy_scale", "carbon_scale", "price_scale", "target_temp"],
    "evcharging": ["active_ratio", "demand_ratio", "deadline_pressure", "forecast_mean"],
    "cogen": ["renewables_magnitude", "ambient_temp", "target_power", "energy_price"],
    "chlor_alkali": ["current_price", "aging_index", "current_limit_ratio", "stress_index"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=str, default="ood_v2_price_trace")
    parser.add_argument("--regimes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "ood_trace100",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_trace100",
    )
    return parser.parse_args()


def _collect_trace(env_key: str, env_kwargs: dict[str, Any], *, count: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    env = build_dynamic_env(get_env_spec(env_key).dynamic_env_name, seed=seed, env_kwargs=env_kwargs)
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    feature_names: list[str] = []
    try:
        for regime_index in range(count):
            _obs, info = env.reset(seed=seed + regime_index)
            context = dict(info.get("context_dict", {}))
            vector = np.asarray(info.get("context_vector", []), dtype=np.float64).reshape(-1)
            names = [str(name) for name in info.get("context_feature_names", [])]
            if not feature_names:
                feature_names = names
            if names != feature_names or len(vector) != len(feature_names):
                raise ValueError(f"Inconsistent context shape for {env_key} at regime {regime_index}.")
            record: dict[str, Any] = {
                "regime_index": int(regime_index),
                "context_label": str(info.get("context_label", info.get("regime_name", ""))),
            }
            for field in PLOT_FIELDS[env_key]:
                record[field] = float(context.get(field, np.nan))
            record.update({f"context_{name}": float(value) for name, value in zip(feature_names, vector)})
            records.append(record)
            vectors.append(vector)
    finally:
        env.close()
    return pd.DataFrame(records), np.stack(vectors), feature_names


def _support_summary(train: np.ndarray, ood: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    combined = np.vstack([train, ood])
    scale = np.maximum(combined.max(axis=0) - combined.min(axis=0), 1e-8)
    normalized_train = (train - combined.min(axis=0)) / scale
    normalized_ood = (ood - combined.min(axis=0)) / scale
    distances = np.linalg.norm(
        normalized_train[:, None, :] - normalized_ood[None, :, :], axis=2
    )
    train_rows = {tuple(np.round(row, 8)) for row in train}
    ood_rows = {tuple(np.round(row, 8)) for row in ood}
    return {
        "context_features": feature_names,
        "train_context_min": {name: float(value) for name, value in zip(feature_names, train.min(axis=0))},
        "train_context_max": {name: float(value) for name, value in zip(feature_names, train.max(axis=0))},
        "ood_context_min": {name: float(value) for name, value in zip(feature_names, ood.min(axis=0))},
        "ood_context_max": {name: float(value) for name, value in zip(feature_names, ood.max(axis=0))},
        "exact_context_vector_overlap_count": int(len(train_rows & ood_rows)),
        "minimum_normalized_train_to_ood_l2": float(distances.min()),
    }


def _factorwise_supports(train: pd.DataFrame, ood: pd.DataFrame, env_key: str) -> dict[str, Any]:
    factors: dict[str, dict[str, Any]] = {}
    separating_factors: list[str] = []
    for field in PLOT_FIELDS[env_key]:
        train_min = float(train[field].min())
        train_max = float(train[field].max())
        ood_min = float(ood[field].min())
        ood_max = float(ood[field].max())
        disjoint = bool(ood_min > train_max or train_min > ood_max)
        factors[field] = {
            "train_min": train_min,
            "train_max": train_max,
            "ood_min": ood_min,
            "ood_max": ood_max,
            "support_disjoint": disjoint,
        }
        if disjoint:
            separating_factors.append(field)
    return {"plot_factor_supports": factors, "separating_numeric_factors": separating_factors}


def _plot(env_key: str, train: pd.DataFrame, ood: pd.DataFrame, out_path: Path) -> None:
    fields = PLOT_FIELDS[env_key]
    figure, axes = plt.subplots(len(fields), 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, field in zip(axes, fields):
        axis.plot(train["regime_index"], train[field], color="#0072B2", linewidth=1.5, label="ID training trace")
        axis.plot(ood["regime_index"], ood[field], color="#D55E00", linewidth=1.5, label="OOD evaluation trace")
        axis.set_ylabel(field.replace("_", " "))
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Illustrative regime index")
    figure.suptitle(f"{env_key}: illustrative 100-regime ID/OOD context trajectories")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "profile": args.profile,
        "regimes_per_trace": int(args.regimes),
        "purpose": "illustrative only; benchmark uses the smaller shared OOD plan",
        "environments": {},
    }
    for env_offset, env_key in enumerate(ENV_ORDER):
        train_cfg = resolve_ood_train_env_config(env_key, args.profile)
        ood_cfg = resolve_ood_eval_env_config(env_key, args.profile)
        train, train_vectors, feature_names = _collect_trace(
            env_key, train_cfg, count=args.regimes, seed=args.seed + 1000 * env_offset
        )
        ood, ood_vectors, ood_feature_names = _collect_trace(
            env_key, ood_cfg, count=args.regimes, seed=args.seed + 1000 * env_offset
        )
        if feature_names != ood_feature_names:
            raise ValueError(f"ID/OOD context features differ for {env_key}.")
        train.to_csv(args.analysis_dir / f"{env_key}_id_trace.csv", index=False)
        ood.to_csv(args.analysis_dir / f"{env_key}_ood_trace.csv", index=False)
        _plot(env_key, train, ood, args.out_dir / f"ood_trace100_{env_key}.png")
        support = _support_summary(train_vectors, ood_vectors, feature_names)
        support["train_context_labels"] = sorted(train["context_label"].unique().tolist())
        support["ood_context_labels"] = sorted(ood["context_label"].unique().tolist())
        support["context_label_overlap"] = sorted(
            set(support["train_context_labels"]) & set(support["ood_context_labels"])
        )
        support.update(_factorwise_supports(train, ood, env_key))
        if support["separating_numeric_factors"]:
            support["factorwise_disjoint_evidence"] = {
                "kind": "numeric_factor",
                "factors": support["separating_numeric_factors"],
            }
        elif not support["context_label_overlap"]:
            support["factorwise_disjoint_evidence"] = {
                "kind": "disjoint_context_labels",
                "train_labels": support["train_context_labels"],
                "ood_labels": support["ood_context_labels"],
            }
        else:
            support["factorwise_disjoint_evidence"] = None
        manifest["environments"][env_key] = support
        print(
            json.dumps(
                {
                    "env_key": env_key,
                    "exact_overlap": support["exact_context_vector_overlap_count"],
                    "min_normalized_l2": support["minimum_normalized_train_to_ood_l2"],
                    "figure": str(args.out_dir / f"ood_trace100_{env_key}.png"),
                }
            )
        )
    (args.analysis_dir / "support_disjointness.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
