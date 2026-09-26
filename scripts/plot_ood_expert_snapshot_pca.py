#!/usr/bin/env python
"""Plot archived expert-snapshot and OOD dynamic factors in PCA coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamic_morl.fine_regimes import (
    CATALOG_CONTEXT_SALIENCY_ALPHA,
    CATALOG_CONTEXT_SAMPLE_MODE,
    _focus_evcharging_context_trace,
)
from src.dynamic_morl.online_context import select_trace_indices
from src.dynamic_morl.runtime_checkpoint import load_owcmorl_runtime_checkpoint


ENV_ORDER = ("building", "evcharging", "cogen", "chlor_alkali")
CONTEXT_STEPS = 4


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


def _summarize_snapshot_context(sample) -> tuple[list[str], np.ndarray]:
    """Use the same four-step salient context summary as the OOD shared plan."""

    metadata = sample.metadata
    feature_names = list(metadata.get("context_feature_names", []))
    context = np.asarray(metadata.get("trace", {}).get("context", []), dtype=np.float64)
    if context.ndim != 2 or len(context) == 0:
        raise ValueError("An archived expert snapshot has no dynamic-factor trace.")
    if len(feature_names) != context.shape[1]:
        raise ValueError("Archived expert snapshot has incompatible context feature names.")
    context = _focus_evcharging_context_trace(context, feature_names)
    sample_indices = select_trace_indices(
        context,
        sample_size=CONTEXT_STEPS,
        sample_mode=CATALOG_CONTEXT_SAMPLE_MODE,
        saliency_alpha=CATALOG_CONTEXT_SALIENCY_ALPHA,
    )
    return feature_names, np.mean(context[sample_indices], axis=0)


def _load_expert_points(runtime_path: Path) -> tuple[list[str], np.ndarray]:
    payload = load_owcmorl_runtime_checkpoint(runtime_path)
    expert_bank = payload.get("expert_bank")
    if expert_bank is None:
        raise ValueError(f"No expert bank is stored in {runtime_path}.")
    samples = expert_bank.export_samples()
    if not samples:
        raise ValueError(f"No expert snapshots are stored in {runtime_path}.")

    expected_names: list[str] | None = None
    summaries = []
    for sample in samples:
        feature_names, summary = _summarize_snapshot_context(sample)
        if expected_names is None:
            expected_names = feature_names
        elif feature_names != expected_names:
            raise ValueError(f"Expert snapshots in {runtime_path} use different context schemas.")
        summaries.append(summary)
    assert expected_names is not None
    return expected_names, np.asarray(summaries, dtype=np.float64)


def _load_ood_points(plan_path: Path) -> np.ndarray:
    payload = json.loads(plan_path.read_text())
    plan = payload.get("shared_regime_seed_plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError(f"Missing OOD condition plan in {plan_path}.")
    vectors = []
    for row in plan:
        values = row.get("context_vectors", [])
        if not values:
            raise ValueError(f"Missing OOD context vector in {plan_path}.")
        vectors.append(np.asarray(values[0], dtype=np.float64).reshape(-1))
    return np.stack(vectors, axis=0)


def _pca(experts: np.ndarray, ood: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joined = np.vstack((experts, ood))
    mean = np.mean(joined, axis=0, keepdims=True)
    scale = np.maximum(np.std(joined, axis=0, keepdims=True), 1e-12)
    standardized = (joined - mean) / scale
    _left, singular_values, vectors_t = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ vectors_t[:2].T
    explained = singular_values[:2] ** 2 / max(float(np.sum(singular_values ** 2)), 1e-12) * 100.0
    return coordinates[: len(experts)], coordinates[len(experts) :], explained, standardized


def _cross_set_distances(experts_z: np.ndarray, ood_z: np.ndarray) -> np.ndarray:
    return np.linalg.norm(experts_z[:, None, :] - ood_z[None, :, :], axis=2)


def _write_points(path: Path, feature_names: list[str], experts: np.ndarray, ood: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "point_id", *feature_names])
        writer.writeheader()
        for source, points in (("expert_snapshot", experts), ("ood_condition", ood)):
            for point_id, point in enumerate(points):
                writer.writerow(
                    {
                        "source": source,
                        "point_id": point_id,
                        **{name: float(value) for name, value in zip(feature_names, point)},
                    }
                )


def _plot_environment(
    *,
    env_key: str,
    expert_xy: np.ndarray,
    ood_xy: np.ndarray,
    explained: np.ndarray,
    out_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(2.45, 2.18))
    axis.scatter(
        expert_xy[:, 0],
        expert_xy[:, 1],
        s=16,
        color="#7F7F7F",
        alpha=0.78,
        linewidths=0,
        zorder=1,
    )
    axis.scatter(
        ood_xy[:, 0],
        ood_xy[:, 1],
        s=24,
        marker="x",
        color="#D55E00",
        linewidths=0.85,
        zorder=2,
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
    axis.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
    axis.tick_params(length=2.0, width=0.5)
    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.22, top=0.98)
    output = out_dir / f"ood_expert_snapshot_pca_{env_key}.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="ood_v3_site_tariff_trace")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT / "results_ood_short_id",
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_protocols",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "innovation_paper" / "fig",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_expert_snapshot_pca",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "profile": str(args.profile),
        "expert_snapshot_summary": {
            "context_steps": CONTEXT_STEPS,
            "sample_mode": CATALOG_CONTEXT_SAMPLE_MODE,
            "saliency_alpha": CATALOG_CONTEXT_SALIENCY_ALPHA,
        },
        "environments": {},
    }
    for env_key in ENV_ORDER:
        runtime_path = args.runtime_root / args.profile / "dynamic" / env_key / "final" / "short_ood_runtime.pt"
        plan_path = args.plan_root / args.profile / env_key / "shared_plan.json"
        feature_names, expert_points = _load_expert_points(runtime_path)
        ood_points = _load_ood_points(plan_path)
        if ood_points.shape[1] != len(feature_names):
            raise ValueError(f"OOD condition schema differs from expert snapshots for {env_key}.")

        expert_xy, ood_xy, explained, standardized = _pca(expert_points, ood_points)
        distances = _cross_set_distances(standardized[: len(expert_points)], standardized[len(expert_points) :])
        exact_overlap_count = int(np.sum(distances <= 1e-12))
        if exact_overlap_count:
            raise ValueError(f"OOD conditions overlap archived expert snapshots in {env_key}.")

        _plot_environment(
            env_key=env_key,
            expert_xy=expert_xy,
            ood_xy=ood_xy,
            explained=explained,
            out_dir=args.out_dir,
        )
        _write_points(args.analysis_dir / f"{env_key}_points.csv", feature_names, expert_points, ood_points)
        manifest["environments"][env_key] = {
            "feature_names": feature_names,
            "expert_snapshot_count": int(len(expert_points)),
            "ood_condition_count": int(len(ood_points)),
            "exact_cross_set_overlap_count": exact_overlap_count,
            "minimum_standardized_distance": float(np.min(distances)),
            "pca_explained_variance_percent": [float(value) for value in explained],
            "runtime_checkpoint": str(runtime_path.relative_to(PROJECT_ROOT)),
            "ood_shared_plan": str(plan_path.relative_to(PROJECT_ROOT)),
        }
    (args.analysis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ood-expert-pca] wrote four figures to {args.out_dir}")


if __name__ == "__main__":
    main()
