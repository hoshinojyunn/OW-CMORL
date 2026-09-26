#!/usr/bin/env python
"""Create reproducible PCA schematics for the OOD protocol illustration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_ORDER = ("building", "evcharging", "cogen", "chlor_alkali")
FEATURE_DIMS = {"building": 11, "evcharging": 8, "cogen": 10, "chlor_alkali": 6}
ENV_SEEDS = {"building": 11, "evcharging": 23, "cogen": 37, "chlor_alkali": 53}
NUM_ID_VECTORS = 100_000
NUM_DISPLAYED_ID_POINTS = 1_000
NUM_OOD_POINTS = 4


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


def _simulate_id_vectors(rng: np.random.Generator, feature_dim: int) -> np.ndarray:
    """Sample one continuous in-distribution factor cloud without discrete modes."""

    latent = rng.normal(size=(NUM_ID_VECTORS, 3))
    latent[:, 1] += 0.28 * latent[:, 0]
    latent[:, 2] -= 0.18 * latent[:, 0]
    loadings = rng.normal(size=(3, feature_dim))
    loadings /= np.maximum(np.linalg.norm(loadings, axis=0, keepdims=True), 1e-12)
    return latent @ loadings + rng.normal(scale=0.10, size=(NUM_ID_VECTORS, feature_dim))


def _pca2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, keepdims=True)
    scale = np.maximum(np.std(values, axis=0, keepdims=True), 1e-12)
    standardized = (values - mean) / scale
    _left, singular_values, vectors_t = np.linalg.svd(standardized, full_matrices=False)
    coordinates = standardized @ vectors_t[:2].T
    explained = singular_values[:2] ** 2 / np.sum(singular_values ** 2) * 100.0
    return coordinates, mean, scale, vectors_t[:2], explained


def _make_uncovered_ood_points(
    id_xy: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Place synthetic OOD points in separated, uncovered PCA directions."""

    rng = np.random.default_rng(30_000 + int(round(float(np.sum(np.abs(components))) * 1_000)))
    center = np.median(id_xy, axis=0)
    distances = np.linalg.norm(id_xy - center, axis=1)
    support_radius = max(float(np.quantile(distances, 0.999)), 1e-12)
    ood_locations = []
    for _ in range(10_000):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        candidate = center + rng.uniform(1.20, 1.45) * support_radius * np.array((np.cos(angle), np.sin(angle)))
        nearest = float(np.min(np.linalg.norm(id_xy - candidate, axis=1)))
        separated = all(np.linalg.norm(candidate - point) >= 0.30 * support_radius for point in ood_locations)
        if nearest >= 0.18 * support_radius and separated:
            ood_locations.append(candidate)
            if len(ood_locations) == NUM_OOD_POINTS:
                break
    if len(ood_locations) != NUM_OOD_POINTS:
        raise RuntimeError("Could not place separated synthetic OOD points outside the ID cloud.")
    ood_xy = np.asarray(ood_locations, dtype=np.float64)
    ood_standardized = ood_xy @ components
    ood_vectors = mean + ood_standardized * scale
    return ood_vectors, ood_xy


def _plot_environment(env_key: str, id_xy: np.ndarray, ood_xy: np.ndarray, explained: np.ndarray, out_dir: Path) -> None:
    rng = np.random.default_rng(10_000 + ENV_SEEDS[env_key])
    subset_idx = rng.choice(len(id_xy), size=NUM_DISPLAYED_ID_POINTS, replace=False)
    figure, axis = plt.subplots(figsize=(2.45, 2.28))
    axis.scatter(
        id_xy[subset_idx, 0],
        id_xy[subset_idx, 1],
        s=5.5,
        color="#9E9E9E",
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )
    axis.scatter(
        ood_xy[:, 0],
        ood_xy[:, 1],
        s=27,
        marker="x",
        color="#D55E00",
        linewidths=0.95,
        zorder=2,
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.0f}%)")
    axis.set_ylabel(f"PC2 ({explained[1]:.0f}%)")
    axis.tick_params(length=2.0, width=0.5)
    figure.subplots_adjust(left=0.23, right=0.98, bottom=0.22, top=0.98)
    output = out_dir / f"ood_pca_schematic_{env_key}.pdf"
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "innovation_paper" / "fig")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "ood_pca_schematic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "kind": "illustrative_synthetic_pca_schematic",
        "id_vectors_simulated_per_environment": NUM_ID_VECTORS,
        "id_points_displayed_per_environment": NUM_DISPLAYED_ID_POINTS,
        "ood_points_per_environment": NUM_OOD_POINTS,
        "environments": {},
    }
    for env_key in ENV_ORDER:
        rng = np.random.default_rng(ENV_SEEDS[env_key])
        id_vectors = _simulate_id_vectors(rng, FEATURE_DIMS[env_key])
        if len(id_vectors) != NUM_ID_VECTORS:
            raise RuntimeError(f"Unexpected simulated vector count for {env_key}.")
        id_xy, mean, scale, components, explained = _pca2(id_vectors)
        ood_vectors, ood_xy = _make_uncovered_ood_points(id_xy, mean, scale, components)
        if float(np.min(np.linalg.norm(id_xy[:, None, :] - ood_xy[None, :, :], axis=2))) <= 1e-12:
            raise RuntimeError(f"Synthetic OOD points overlap the ID PCA cloud for {env_key}.")
        _plot_environment(env_key, id_xy, ood_xy, explained, args.out_dir)
        manifest["environments"][env_key] = {
            "feature_dimension": FEATURE_DIMS[env_key],
            "seed": ENV_SEEDS[env_key],
            "pca_explained_variance_percent": [float(value) for value in explained],
            "ood_pc1_min": float(np.min(ood_xy[:, 0])),
            "id_pc1_max": float(np.max(id_xy[:, 0])),
            "ood_vectors_shape": list(ood_vectors.shape),
        }
    (args.analysis_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ood-pca-schematic] wrote four figures to {args.out_dir}")


if __name__ == "__main__":
    main()
