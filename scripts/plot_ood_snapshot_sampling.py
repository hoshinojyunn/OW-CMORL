#!/usr/bin/env python
"""Plot ID factors stored with expert snapshots and synthetic OOD samples."""

from __future__ import annotations

import argparse
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

from src.dynamic_morl.runtime_checkpoint import load_owcmorl_runtime_checkpoint


ENV_ORDER = ("building", "evcharging", "cogen", "chlor_alkali")
ENV_SEEDS = {"building": 101, "evcharging": 202, "cogen": 303, "chlor_alkali": 404}
DISPLAYED_ID_POINTS = 1_000
SYNTHETIC_OOD_POINTS = 4


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_snapshot_contexts(runtime_path: Path) -> tuple[list[str], np.ndarray]:
    payload = load_owcmorl_runtime_checkpoint(runtime_path)
    expert_bank = payload.get("expert_bank")
    if expert_bank is None:
        raise ValueError(f"No expert bank in {runtime_path}.")

    feature_names: list[str] | None = None
    contexts = []
    for sample in expert_bank.export_samples():
        metadata = sample.metadata
        names = list(metadata.get("context_feature_names", []))
        trace_context = np.asarray(metadata.get("trace", {}).get("context", []), dtype=np.float64)
        if trace_context.ndim != 2 or len(trace_context) == 0:
            raise ValueError(f"An expert snapshot in {runtime_path} has no context trace.")
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Expert snapshots in {runtime_path} use inconsistent context schemas.")
        contexts.append(trace_context)
    if feature_names is None or not contexts:
        raise ValueError(f"No expert snapshot contexts in {runtime_path}.")
    return feature_names, np.concatenate(contexts, axis=0)


def _pca2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, keepdims=True)
    scale = np.maximum(np.std(values, axis=0, keepdims=True), 1e-12)
    standardized = (values - mean) / scale
    _left, singular_values, vectors_t = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ vectors_t[:2].T
    explained = singular_values[:2] ** 2 / max(float(np.sum(singular_values ** 2)), 1e-12) * 100.0
    return coordinates, mean, scale, vectors_t[:2], explained


def _synthetic_ood_points(id_xy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample separated OOD locations outside the displayed ID point cloud."""

    center = np.median(id_xy, axis=0)
    radius = np.sqrt(np.sum((id_xy - center) ** 2, axis=1))
    support_radius = max(float(np.quantile(radius, 0.999)), 1e-12)
    points = []
    for _ in range(10_000):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        candidate = center + rng.uniform(1.20, 1.45) * support_radius * np.array((np.cos(angle), np.sin(angle)))
        nearest = float(np.min(np.linalg.norm(id_xy - candidate, axis=1)))
        separated = all(np.linalg.norm(candidate - point) >= 0.30 * support_radius for point in points)
        if nearest >= 0.18 * support_radius and separated:
            points.append(candidate)
            if len(points) == SYNTHETIC_OOD_POINTS:
                return np.asarray(points, dtype=np.float64)
    raise RuntimeError("Could not place separated synthetic OOD points outside the ID cloud.")


def _plot_environment(
    *,
    env_key: str,
    id_xy: np.ndarray,
    ood_xy: np.ndarray,
    explained: np.ndarray,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    displayed_count = min(DISPLAYED_ID_POINTS, len(id_xy))
    sampled_idx = rng.choice(len(id_xy), size=displayed_count, replace=False)
    figure, axis = plt.subplots(figsize=(2.45, 2.28))
    axis.scatter(
        id_xy[sampled_idx, 0],
        id_xy[sampled_idx, 1],
        s=5.5,
        color="#9E9E9E",
        alpha=0.42,
        linewidths=0,
        zorder=1,
    )
    axis.scatter(
        ood_xy[:, 0],
        ood_xy[:, 1],
        s=29,
        marker="x",
        color="#D55E00",
        linewidths=0.95,
        zorder=2,
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
    axis.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
    axis.tick_params(length=2.0, width=0.5)
    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.22, top=0.98)
    output = out_dir / f"ood_snapshot_sampling_{env_key}.pdf"
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
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "innovation_paper" / "fig")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_snapshot_sampling",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": str(args.profile),
        "displayed_id_contexts_per_environment": DISPLAYED_ID_POINTS,
        "synthetic_ood_points_per_environment": SYNTHETIC_OOD_POINTS,
        "environments": {},
    }
    for env_key in ENV_ORDER:
        rng = np.random.default_rng(ENV_SEEDS[env_key])
        runtime_path = args.runtime_root / args.profile / "dynamic" / env_key / "final" / "short_ood_runtime.pt"
        feature_names, snapshot_contexts = _load_snapshot_contexts(runtime_path)
        id_xy, _mean, _scale, _components, explained = _pca2(snapshot_contexts)
        ood_xy = _synthetic_ood_points(id_xy, rng)
        _plot_environment(
            env_key=env_key,
            id_xy=id_xy,
            ood_xy=ood_xy,
            explained=explained,
            rng=rng,
            out_dir=args.out_dir,
        )
        min_distance = float(np.min(np.linalg.norm(id_xy[:, None, :] - ood_xy[None, :, :], axis=2)))
        manifest["environments"][env_key] = {
            "context_feature_names": feature_names,
            "expert_snapshot_context_count": int(len(snapshot_contexts)),
            "expert_snapshot_count": int(len(load_owcmorl_runtime_checkpoint(runtime_path)["expert_bank"].export_samples())),
            "pca_explained_variance_percent": [float(value) for value in explained],
            "minimum_pca_distance_to_synthetic_ood": min_distance,
            "runtime_checkpoint": str(runtime_path.relative_to(PROJECT_ROOT)),
        }
    (args.analysis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ood-snapshot-sampling] wrote four figures to {args.out_dir}")


if __name__ == "__main__":
    main()
