from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.dynamic_morl.utils import compute_eu, compute_sparsity, generate_w_batch_test


def pareto_front(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    nd_idx = NonDominatedSorting().do(-points, only_non_dominated_front=True)
    return np.asarray(points[nd_idx], dtype=np.float64)


def as_2d(points: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def derive_reference_point(fronts: list[np.ndarray]) -> np.ndarray:
    valid_fronts = [front for front in fronts if front.ndim == 2 and len(front) > 0]
    if not valid_fronts:
        raise ValueError("Cannot derive reference point from empty fronts.")
    stacked = np.concatenate(valid_fronts, axis=0)
    lower = stacked.min(axis=0)
    margin = 0.1 * np.maximum(np.abs(lower), 1.0)
    return lower - margin


def front_metrics(front: np.ndarray, ref_point: np.ndarray) -> dict[str, float]:
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or len(front) == 0:
        return {"HV": 0.0, "EU": 0.0, "SP": 0.0}
    hv = Hypervolume(ref_point=-ref_point).do(-front)
    prefs = generate_w_batch_test(front.shape[1], 0.5)
    eu = compute_eu(front, prefs)
    sp = compute_sparsity(front)
    return {"HV": float(hv), "EU": float(eu), "SP": float(sp)}


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        value = float(values[0])
        return value, value
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def empirical_cdf(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    if len(values) == 0:
        return np.zeros_like(x_grid, dtype=np.float64)
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / float(len(values))


def normalize_to_unit(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    span = max(float(vmax) - float(vmin), 1e-12)
    return np.clip((values - float(vmin)) / span, 0.0, 1.0)


def save_npz_archive(path: Path, *, solutions: np.ndarray, pareto_front_points: np.ndarray, meta: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        solutions=np.asarray(solutions, dtype=np.float32),
        pareto_front=np.asarray(pareto_front_points, dtype=np.float32),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False), dtype=np.unicode_),
    )


def load_npz_archive(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    solutions = np.asarray(data["solutions"], dtype=np.float64)
    pareto_front_points = np.asarray(data["pareto_front"], dtype=np.float64)
    meta = {}
    if "meta_json" in data.files:
        meta_raw = data["meta_json"]
        raw_text = meta_raw.item() if hasattr(meta_raw, "item") else meta_raw
        meta = json.loads(str(raw_text))
    return solutions, pareto_front_points, meta
