from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

try:
    import gym
except ModuleNotFoundError:  # pragma: no cover - fallback for gymnasium-only envs
    import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

from .chlor_alkali_price import (
    ANHUI_CHLOR_ALKALI_35KV_PROXY,
    TOUPriceProxy,
    chlor_alkali_dynamic_price,
    chlor_alkali_price_period,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHLOR_ALKALI_ALL = BASE_DIR / "chlor-alkali" / "CA_all.csv"
DEFAULT_CHLOR_ALKALI_TRAIN = BASE_DIR / "chlor-alkali" / "CA_train.csv"
DEFAULT_CHLOR_ALKALI_TEST = BASE_DIR / "chlor-alkali" / "CA_test.csv"
CACHE_DIR = BASE_DIR / ".cache" / "chlor_alkali"
CACHE_VERSION = 3
DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS = 20

LOCAL_PREFIXES = (
    "current_",
    "temp_",
    "hcl_",
    "p_flow_",
    "n_flow_",
    "nacl_den_",
    "naoh_con_",
)
OUTPUT_COLUMNS = ("daily_prod", "curr_eff", "djdh")
MIN_TOTAL_CURRENT_KA = 86.0
MAX_TOTAL_CURRENT_KA = 119.0
CURRENT_LIMIT_SPAN_KA = MAX_TOTAL_CURRENT_KA - MIN_TOTAL_CURRENT_KA
AGING_LIMIT_PENALTY_KA = 10.0
THERMAL_LIMIT_PENALTY_KA = 5.0
OUTAGE_LIMIT_PENALTY_KA = 8.0
FLOW_LIMIT_PENALTY_KA = 3.0
CONTEXT_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "price_norm",
    "peak_flag",
    "valley_flag",
    "glo_temp_norm",
    "temp_naoh_norm",
    "active_cell_ratio",
    "aging_index",
    "current_limit_ratio",
    "flow_imbalance_norm",
    "stress_index",
    "eff_gap_ema",
)
OBSERVATION_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "price_norm",
    "peak_flag",
    "valley_flag",
    "total_current_norm",
    "mean_temp_norm",
    "mean_hcl_norm",
    "mean_p_flow_norm",
    "mean_n_flow_norm",
    "mean_nacl_den_norm",
    "glo_temp_norm",
    "temp_naoh_norm",
    "active_cell_ratio",
    "aging_index",
    "current_limit_ratio",
    "flow_imbalance_norm",
    "stress_index",
    "eff_gap_ema",
    "prev_prod_norm",
    "prev_eff_norm",
    "prev_djdh_norm",
)
SURROGATE_FEATURE_COLUMNS = (
    [f"{prefix}{idx}" for idx in range(1, 9) for prefix in LOCAL_PREFIXES]
    + [
        "glo_temp",
        "temp_naoh",
        "current_price",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "peak_flag",
        "flat_flag",
        "valley_flag",
        "summer_flag",
        "total_current",
        "mean_temp",
        "mean_hcl",
        "mean_p_flow",
        "mean_n_flow",
        "mean_nacl_den",
        "mean_naoh_con",
        "flow_imbalance",
        "active_cells",
        "aging_proxy",
        "price_limit_kA_proxy",
        "current_limit_kA_proxy",
        "current_limit_ratio_proxy",
        "stress_proxy",
    ]
)
REGIME_FEATURE_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "current_price",
    "glo_temp",
    "temp_naoh",
    "active_cells",
    "aging_proxy",
    "price_limit_kA_proxy",
    "current_limit_kA_proxy",
    "current_limit_ratio_proxy",
    "flow_imbalance",
    "stress_proxy",
)
REGIME_LABEL_COLUMNS = (
    "current_price",
    "current_limit_ratio_proxy",
    "aging_proxy",
    "stress_proxy",
    "glo_temp",
)


def _safe_value_key(value: float, digits: int = 3) -> str:
    text = f"{float(value):.{digits}f}"
    return text.replace("-", "m").replace(".", "p")


_FRAME_CACHE: dict[tuple[str, bool, str], pd.DataFrame] = {}


def _atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _frame_cache_path(
    csv_path: str | Path,
    *,
    dynamic_price: bool,
    price_proxy: TOUPriceProxy,
) -> Path:
    source = Path(csv_path).resolve()
    fingerprint = "|".join(
        [
            str(CACHE_VERSION),
            str(source),
            str(source.stat().st_mtime_ns),
            str(dynamic_price),
            price_proxy.name,
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return CACHE_DIR / "frames" / f"{digest}.pkl"


def _surrogate_cache_path(
    train_path: str | Path,
    *,
    dynamic_price: bool,
    seed: int,
) -> Path:
    source = Path(train_path).resolve()
    fingerprint = "|".join(
        [
            str(CACHE_VERSION),
            str(source),
            str(source.stat().st_mtime_ns),
            str(dynamic_price),
            str(seed),
            "rf160_depth18_leaf4_jobs1",
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return CACHE_DIR / "surrogates" / f"{digest}.pkl"


def _regime_encoder_cache_path(
    train_path: str | Path,
    *,
    dynamic_price: bool,
    price_proxy: TOUPriceProxy,
    seed: int,
    n_clusters: int,
) -> Path:
    source = Path(train_path).resolve()
    fingerprint = "|".join(
        [
            str(CACHE_VERSION),
            str(source),
            str(source.stat().st_mtime_ns),
            str(dynamic_price),
            price_proxy.name,
            str(seed),
            str(n_clusters),
            "kmeans_chlor_context_v1",
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return CACHE_DIR / "regime_encoders" / f"{digest}.pkl"


class ChlorAlkaliRegimeEncoder:
    def __init__(self, n_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS, random_state: int = 0):
        self.n_clusters = int(max(1, n_clusters))
        self.random_state = int(random_state)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.model: KMeans | None = None
        self.raw_to_stable: dict[int, int] = {}
        self.stable_to_raw: dict[int, int] = {}
        self.labels: dict[int, str] = {}
        self.centers: np.ndarray | None = None
        self.cluster_counts: dict[int, int] = {}
        self._fitted = False
        self._regime_feature_indices: list[int] | None = None

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, list(REGIME_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)

    def fit(self, frame: pd.DataFrame) -> None:
        x = self._matrix(frame)
        if len(x) == 0:
            raise ValueError("Cannot fit ChlorAlkaliRegimeEncoder on an empty frame.")
        self.feature_mean = x.mean(axis=0)
        self.feature_std = np.maximum(x.std(axis=0), 1e-6)
        x_norm = (x - self.feature_mean) / self.feature_std
        n_clusters = min(self.n_clusters, len(x_norm))
        self.model = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.random_state)
        raw_ids = self.model.fit_predict(x_norm)
        self.centers = self.model.cluster_centers_ * self.feature_std + self.feature_mean

        idx_price = REGIME_FEATURE_COLUMNS.index("current_price")
        idx_limit = REGIME_FEATURE_COLUMNS.index("current_limit_ratio_proxy")
        idx_aging = REGIME_FEATURE_COLUMNS.index("aging_proxy")
        idx_stress = REGIME_FEATURE_COLUMNS.index("stress_proxy")
        stable_order = np.lexsort(
            (
                self.centers[:, idx_stress],
                self.centers[:, idx_aging],
                self.centers[:, idx_limit],
                self.centers[:, idx_price],
            )
        )
        self.raw_to_stable = {int(raw): int(stable) for stable, raw in enumerate(stable_order)}
        self.stable_to_raw = {int(stable): int(raw) for stable, raw in enumerate(stable_order)}

        stable_ids = np.asarray([self.raw_to_stable[int(raw)] for raw in raw_ids], dtype=np.int64)
        counts = np.bincount(stable_ids, minlength=n_clusters)
        self.cluster_counts = {int(idx): int(count) for idx, count in enumerate(counts)}

        label_cols = {name: REGIME_FEATURE_COLUMNS.index(name) for name in REGIME_LABEL_COLUMNS}
        labels: dict[int, str] = {}
        for stable_idx in range(n_clusters):
            raw_idx = self.stable_to_raw[stable_idx]
            center = self.centers[raw_idx]
            labels[stable_idx] = (
                f"ctx_{stable_idx:02d}"
                f"_p{_safe_value_key(center[label_cols['current_price']], 3)}"
                f"_lim{_safe_value_key(center[label_cols['current_limit_ratio_proxy']], 2)}"
                f"_age{_safe_value_key(center[label_cols['aging_proxy']], 2)}"
                f"_st{_safe_value_key(center[label_cols['stress_proxy']], 2)}"
                f"_t{_safe_value_key(center[label_cols['glo_temp']], 1)}"
            )
        self.labels = labels
        self.n_clusters = int(n_clusters)
        self._fitted = True
        self._regime_feature_indices = None

    def _transform_matrix(self, x: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("ChlorAlkaliRegimeEncoder must be fitted before transform.")
        x_norm = (x - self.feature_mean) / self.feature_std
        raw_ids = self.model.predict(x_norm)
        return np.asarray([self.raw_to_stable[int(raw)] for raw in raw_ids], dtype=np.int64)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        stable_ids = self._transform_matrix(self._matrix(frame))
        labels = np.asarray([self.labels[int(idx)] for idx in stable_ids], dtype=object)
        return stable_ids, labels

    def assign_row(self, row: pd.Series) -> tuple[int, str]:
        frame = pd.DataFrame([{column: float(row[column]) for column in REGIME_FEATURE_COLUMNS}])
        stable_id = int(self._transform_matrix(frame.to_numpy(dtype=np.float64))[0])
        return stable_id, self.labels[stable_id]

    def assign_values(self, values: np.ndarray) -> tuple[int, str]:
        stable_id = int(self._transform_matrix(np.asarray(values, dtype=np.float64).reshape(1, -1))[0])
        return stable_id, self.labels[stable_id]

    def catalog(self) -> list[dict[str, Any]]:
        if not self._fitted or self.centers is None:
            raise RuntimeError("ChlorAlkaliRegimeEncoder must be fitted before catalog.")
        rows: list[dict[str, Any]] = []
        for stable_idx in range(self.n_clusters):
            raw_idx = self.stable_to_raw[stable_idx]
            center = self.centers[raw_idx]
            center_dict = {
                column: float(center[REGIME_FEATURE_COLUMNS.index(column)])
                for column in REGIME_FEATURE_COLUMNS
            }
            rows.append(
                {
                    "regime_id": int(stable_idx),
                    "regime_name": self.labels[stable_idx],
                    "count": int(self.cluster_counts.get(stable_idx, 0)),
                    **center_dict,
                }
            )
        return rows


_REGIME_ENCODER_CACHE: dict[tuple[str, bool, str, int, int], ChlorAlkaliRegimeEncoder] = {}


def fit_chlor_alkali_regime_encoder(
    train_path: str | Path,
    *,
    dynamic_price: bool,
    price_proxy: TOUPriceProxy = ANHUI_CHLOR_ALKALI_35KV_PROXY,
    seed: int = 0,
    n_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
) -> ChlorAlkaliRegimeEncoder:
    cache_key = (str(Path(train_path).resolve()), bool(dynamic_price), price_proxy.name, int(seed), int(n_clusters))
    cached = _REGIME_ENCODER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache_path = _regime_encoder_cache_path(
        train_path,
        dynamic_price=dynamic_price,
        price_proxy=price_proxy,
        seed=int(seed),
        n_clusters=int(n_clusters),
    )
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fp:
                encoder = pickle.load(fp)
            if isinstance(encoder, ChlorAlkaliRegimeEncoder) and encoder._fitted:
                _REGIME_ENCODER_CACHE[cache_key] = encoder
                return encoder
        except Exception:
            pass
    train_frame = load_chlor_alkali_frame(
        train_path,
        dynamic_price=dynamic_price,
        price_proxy=price_proxy,
    )
    encoder = ChlorAlkaliRegimeEncoder(n_clusters=n_clusters, random_state=int(seed))
    encoder.fit(train_frame)
    _REGIME_ENCODER_CACHE[cache_key] = encoder
    try:
        _atomic_pickle_dump(encoder, cache_path)
    except Exception:
        pass
    return encoder


def annotate_chlor_alkali_regimes(frame: pd.DataFrame, encoder: ChlorAlkaliRegimeEncoder) -> pd.DataFrame:
    annotated = frame.copy()
    regime_ids, regime_names = encoder.transform(annotated)
    annotated["regime_id"] = regime_ids.astype(np.int64)
    annotated["regime_name"] = regime_names
    return annotated


def chlor_alkali_regime_catalog(
    *,
    train_path: str | Path = DEFAULT_CHLOR_ALKALI_TRAIN,
    dynamic_price: bool = True,
    price_proxy: TOUPriceProxy = ANHUI_CHLOR_ALKALI_35KV_PROXY,
    seed: int = 0,
    n_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    encoder = fit_chlor_alkali_regime_encoder(
        train_path,
        dynamic_price=dynamic_price,
        price_proxy=price_proxy,
        seed=seed,
        n_clusters=n_clusters,
    )
    rows = [row for row in encoder.catalog() if int(row["count"]) >= int(min_count)]
    rows.sort(key=lambda row: int(row["regime_id"]))
    return rows


def _cell_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in df.columns if column.startswith(prefix)]


def _normalized(value: float, low: float, high: float, *, clip: bool = True) -> float:
    span = max(float(high - low), 1e-6)
    scaled = (float(value) - float(low)) / span
    normalized = 2.0 * scaled - 1.0
    return float(np.clip(normalized, -1.0, 1.0)) if clip else float(normalized)


def _rolling_aging_features(df: pd.DataFrame) -> pd.DataFrame:
    current_ref = max(float(df["total_current"].quantile(0.95)), 1e-6)
    temp_ref = float(df["mean_temp"].median())
    imbalance_ref = max(float(df["flow_imbalance"].abs().quantile(0.95)), 1e-6)
    hcl_ref = float(df["mean_hcl"].median())
    brine_ref = float(df["mean_nacl_den"].median())

    aging = 0.08
    eff_gap_ema = 0.0
    current_limit = 1.0
    current_limit_ka = MAX_TOTAL_CURRENT_KA
    price_limit_trace: list[float] = []
    limit_ka_trace: list[float] = []
    aging_trace: list[float] = []
    limit_trace: list[float] = []
    stress_trace: list[float] = []
    eff_trace: list[float] = []

    for row in df.itertuples(index=False):
        active_ratio = float(getattr(row, "active_cells")) / 8.0
        current_ratio = float(getattr(row, "total_current")) / current_ref
        temp_stress = max(float(getattr(row, "mean_temp")) - temp_ref, 0.0) / 4.0
        imbalance_stress = abs(float(getattr(row, "flow_imbalance"))) / imbalance_ref
        outage_stress = max(0.0, 1.0 - active_ratio)
        brine_recovery = max(float(getattr(row, "mean_nacl_den")) - brine_ref, 0.0) / max(brine_ref, 1.0)
        acid_recovery = max(float(getattr(row, "mean_hcl")) - hcl_ref, 0.0) / max(hcl_ref, 1.0)
        price_ratio = np.clip(
            (float(getattr(row, "current_price")) - float(df["current_price"].min()))
            / max(float(df["current_price"].max()) - float(df["current_price"].min()), 1e-6),
            0.0,
            1.0,
        )
        price_limit_ka = float(MAX_TOTAL_CURRENT_KA - CURRENT_LIMIT_SPAN_KA * price_ratio)

        stress = (
            0.55 * current_ratio
            + 0.20 * temp_stress
            + 0.15 * imbalance_stress
            + 0.10 * outage_stress
        )
        recovery = 0.07 * acid_recovery + 0.03 * brine_recovery
        aging = float(np.clip(0.996 * aging + 0.012 * stress - 0.008 * recovery, 0.0, 1.0))
        current_limit_ka = float(
            np.clip(
                price_limit_ka
                - AGING_LIMIT_PENALTY_KA * aging
                - THERMAL_LIMIT_PENALTY_KA * temp_stress
                - OUTAGE_LIMIT_PENALTY_KA * outage_stress
                - FLOW_LIMIT_PENALTY_KA * min(imbalance_stress, 1.0),
                MIN_TOTAL_CURRENT_KA,
                price_limit_ka,
            )
        )
        current_limit = float(np.clip(current_limit_ka / MAX_TOTAL_CURRENT_KA, MIN_TOTAL_CURRENT_KA / MAX_TOTAL_CURRENT_KA, 1.0))
        eff_gap_ema = float(0.97 * eff_gap_ema + 0.03 * ((float(getattr(row, "curr_eff")) - 93.5) / 5.0))

        price_limit_trace.append(price_limit_ka)
        limit_ka_trace.append(current_limit_ka)
        aging_trace.append(aging)
        limit_trace.append(current_limit)
        stress_trace.append(float(np.clip(stress / 1.5, 0.0, 1.0)))
        eff_trace.append(float(np.clip(eff_gap_ema, -1.0, 1.0)))

    df = df.copy()
    df["aging_proxy"] = np.asarray(aging_trace, dtype=np.float32)
    df["price_limit_kA_proxy"] = np.asarray(price_limit_trace, dtype=np.float32)
    df["current_limit_kA_proxy"] = np.asarray(limit_ka_trace, dtype=np.float32)
    df["current_limit_ratio_proxy"] = np.asarray(limit_trace, dtype=np.float32)
    df["stress_proxy"] = np.asarray(stress_trace, dtype=np.float32)
    df["eff_gap_ema_proxy"] = np.asarray(eff_trace, dtype=np.float32)
    return df


def load_chlor_alkali_frame(
    csv_path: str | Path,
    *,
    dynamic_price: bool = True,
    price_proxy: TOUPriceProxy = ANHUI_CHLOR_ALKALI_35KV_PROXY,
) -> pd.DataFrame:
    cache_key = (str(Path(csv_path).resolve()), bool(dynamic_price), price_proxy.name)
    cached = _FRAME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cache_path = _frame_cache_path(
        csv_path,
        dynamic_price=dynamic_price,
        price_proxy=price_proxy,
    )
    if cache_path.exists():
        try:
            frame = pd.read_pickle(cache_path)
            _FRAME_CACHE[cache_key] = frame
            return frame
        except Exception:
            pass

    frame = pd.read_csv(csv_path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], format="%y/%m/%d %H:%M:%S")

    if dynamic_price:
        prices = [
            chlor_alkali_dynamic_price(timestamp, proxy=price_proxy)
            for timestamp in frame["datetime"]
        ]
        frame["current_price"] = np.asarray(prices, dtype=np.float32)

    current_cols = _cell_columns(frame, "current_")
    temp_cols = [column for column in _cell_columns(frame, "temp_") if column != "temp_naoh"]
    hcl_cols = _cell_columns(frame, "hcl_")
    p_flow_cols = _cell_columns(frame, "p_flow_")
    n_flow_cols = _cell_columns(frame, "n_flow_")
    nacl_cols = _cell_columns(frame, "nacl_den_")
    naoh_cols = _cell_columns(frame, "naoh_con_")

    minutes = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    month_angle = 2.0 * np.pi * (frame["datetime"].dt.month.to_numpy(dtype=np.float32) - 1.0) / 12.0
    day_angle = 2.0 * np.pi * minutes.to_numpy(dtype=np.float32) / (24.0 * 60.0)
    periods = frame["datetime"].map(chlor_alkali_price_period)

    frame["hour_sin"] = np.sin(day_angle).astype(np.float32)
    frame["hour_cos"] = np.cos(day_angle).astype(np.float32)
    frame["month_sin"] = np.sin(month_angle).astype(np.float32)
    frame["month_cos"] = np.cos(month_angle).astype(np.float32)
    frame["peak_flag"] = (periods == "peak").astype(np.float32)
    frame["flat_flag"] = (periods == "flat").astype(np.float32)
    frame["valley_flag"] = (periods == "valley").astype(np.float32)
    frame["summer_flag"] = frame["datetime"].dt.month.isin({7, 8, 9}).astype(np.float32)
    frame["total_current"] = frame[current_cols].sum(axis=1)
    frame["mean_temp"] = frame[temp_cols].mean(axis=1)
    frame["mean_hcl"] = frame[hcl_cols].mean(axis=1)
    frame["mean_p_flow"] = frame[p_flow_cols].mean(axis=1)
    frame["mean_n_flow"] = frame[n_flow_cols].mean(axis=1)
    frame["mean_nacl_den"] = frame[nacl_cols].mean(axis=1)
    frame["mean_naoh_con"] = frame[naoh_cols].mean(axis=1)
    frame["flow_imbalance"] = frame["mean_p_flow"] - frame["mean_n_flow"]
    frame["active_cells"] = (frame[current_cols] > 1.0).sum(axis=1).astype(np.float32)
    frame = _rolling_aging_features(frame)
    _FRAME_CACHE[cache_key] = frame
    try:
        frame.to_pickle(cache_path)
    except Exception:
        pass
    return frame


def _apply_eval_price_shift(
    frame: pd.DataFrame,
    *,
    price_multiplier: float,
    price_offset: float,
) -> pd.DataFrame:
    """Apply a deterministic evaluation-only tariff shift to a prepared frame."""
    multiplier = float(price_multiplier)
    offset = float(price_offset)
    if multiplier <= 0.0:
        raise ValueError("price_multiplier must be positive.")
    if abs(multiplier - 1.0) <= 1e-12 and abs(offset) <= 1e-12:
        return frame
    shifted = frame.copy()
    shifted["current_price"] = (
        shifted["current_price"].to_numpy(dtype=np.float32) * multiplier + offset
    ).astype(np.float32)
    # Price-linked limits and rolling state proxies must match the shifted trace.
    return _rolling_aging_features(shifted)


class ChlorAlkaliSurrogate:
    def __init__(self, random_state: int = 0):
        self.random_state = int(random_state)
        self.model = RandomForestRegressor(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=4,
            n_jobs=1,
            random_state=self.random_state,
        )
        self._fitted = False
        self.target_min: np.ndarray | None = None
        self.target_max: np.ndarray | None = None
        self.feature_min: dict[str, float] = {}
        self.feature_max: dict[str, float] = {}
        self._tree_predictors: list[Any] = []
        self._tree_count: int = 0

    def fit(self, frame: pd.DataFrame) -> None:
        x = frame.loc[:, SURROGATE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y = frame.loc[:, list(OUTPUT_COLUMNS)].to_numpy(dtype=np.float32)
        self.model.fit(x, y)
        self.target_min = y.min(axis=0)
        self.target_max = y.max(axis=0)
        self.feature_min = {column: float(frame[column].min()) for column in SURROGATE_FEATURE_COLUMNS}
        self.feature_max = {column: float(frame[column].max()) for column in SURROGATE_FEATURE_COLUMNS}
        self._tree_predictors = [est.tree_ for est in self.model.estimators_]
        self._tree_count = len(self._tree_predictors)
        self._fitted = True

    def _predict_fast(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_tree_predictors"):
            self._tree_predictors = [est.tree_ for est in self.model.estimators_]
        if not hasattr(self, "_tree_count"):
            self._tree_count = len(self._tree_predictors)
        if self._tree_count <= 0:
            return self.model.predict(x).reshape(-1)
        accum = None
        for tree in self._tree_predictors:
            pred = np.asarray(tree.predict(x), dtype=np.float64).reshape(-1)
            if accum is None:
                accum = pred
            else:
                accum += pred
        assert accum is not None
        return (accum / float(self._tree_count)).astype(np.float32, copy=False)

    def predict(self, row: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ChlorAlkaliSurrogate must be fitted before prediction.")
        x = row.loc[list(SURROGATE_FEATURE_COLUMNS)].to_numpy(dtype=np.float32).reshape(1, -1)
        prediction = self._predict_fast(x)
        assert self.target_min is not None and self.target_max is not None
        return np.clip(prediction, self.target_min, self.target_max).astype(np.float32)

    def predict_from_values(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ChlorAlkaliSurrogate must be fitted before prediction.")
        x = np.asarray(values, dtype=np.float32).reshape(1, -1)
        prediction = self._predict_fast(x)
        assert self.target_min is not None and self.target_max is not None
        return np.clip(prediction, self.target_min, self.target_max).astype(np.float32)


_SURROGATE_CACHE: dict[tuple[str, bool, int], ChlorAlkaliSurrogate] = {}


class ChlorAlkaliEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dataset_path: str | Path | None = DEFAULT_CHLOR_ALKALI_ALL,
        train_path: str | Path | None = DEFAULT_CHLOR_ALKALI_TRAIN,
        dynamic_price: bool = True,
        aging_dynamics: bool = True,
        episode_length: int = 288,
        schedule: str = "random",
        allowed_regimes: tuple[str, ...] | None = None,
        allowed_regime_ids: tuple[int, ...] | None = None,
        num_regime_clusters: int = DEFAULT_CHLOR_ALKALI_REGIME_CLUSTERS,
        seed: int | None = None,
        price_proxy: TOUPriceProxy = ANHUI_CHLOR_ALKALI_35KV_PROXY,
        price_multiplier: float = 1.0,
        price_offset: float = 0.0,
        surrogate_seed: int = 0,
    ):
        assert schedule in {"random", "sequential", "cyclic"}
        self.dataset_path = Path(dataset_path or DEFAULT_CHLOR_ALKALI_ALL)
        self.train_path = Path(train_path or DEFAULT_CHLOR_ALKALI_TRAIN)
        self.dynamic_price = bool(dynamic_price)
        self.aging_dynamics = bool(aging_dynamics)
        self.episode_length = int(episode_length)
        self.schedule = schedule
        self.allowed_regimes = tuple(allowed_regimes or ())
        self.allowed_regime_ids = tuple(int(regime_id) for regime_id in (allowed_regime_ids or ()))
        self.num_regime_clusters = int(max(1, num_regime_clusters))
        self.price_proxy = price_proxy
        self.price_multiplier = float(price_multiplier)
        self.price_offset = float(price_offset)
        # The surrogate represents the fixed process model trained from the
        # ID partition.  Rollout seeds control only episode sampling; letting
        # them alter the random-forest seed would silently change the process
        # between OOD budget points.
        self.surrogate_seed = int(surrogate_seed)
        if self.price_multiplier <= 0.0:
            raise ValueError("price_multiplier must be positive.")
        self.rng = np.random.default_rng(seed)

        self.train_frame = load_chlor_alkali_frame(
            self.train_path,
            dynamic_price=self.dynamic_price,
            price_proxy=self.price_proxy,
        ).reset_index(drop=True)
        self.regime_encoder = fit_chlor_alkali_regime_encoder(
            self.train_path,
            dynamic_price=self.dynamic_price,
            price_proxy=self.price_proxy,
            seed=0,
            n_clusters=self.num_regime_clusters,
        )
        self.train_frame = annotate_chlor_alkali_regimes(self.train_frame, self.regime_encoder).reset_index(drop=True)
        eval_frame = load_chlor_alkali_frame(
                self.dataset_path,
                dynamic_price=self.dynamic_price,
                price_proxy=self.price_proxy,
            ).reset_index(drop=True)
        self.frame = annotate_chlor_alkali_regimes(
            _apply_eval_price_shift(
                eval_frame,
                price_multiplier=self.price_multiplier,
                price_offset=self.price_offset,
            ),
            self.regime_encoder,
        ).reset_index(drop=True)
        self.surrogate = self._load_or_fit_surrogate(self.surrogate_seed)
        self._surrogate_feature_indices = [
            int(self.frame.columns.get_loc(name))
            for name in SURROGATE_FEATURE_COLUMNS
        ]
        self._regime_feature_indices = [
            int(self.frame.columns.get_loc(name))
            for name in REGIME_FEATURE_COLUMNS
        ]

        self.current_cols = _cell_columns(self.frame, "current_")
        self.hcl_cols = _cell_columns(self.frame, "hcl_")
        self.p_flow_cols = _cell_columns(self.frame, "p_flow_")
        self.n_flow_cols = _cell_columns(self.frame, "n_flow_")
        self.nacl_cols = _cell_columns(self.frame, "nacl_den_")
        self.temp_cols = [column for column in _cell_columns(self.frame, "temp_") if column != "temp_naoh"]
        self._current_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.current_cols]
        self._hcl_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.hcl_cols]
        self._p_flow_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.p_flow_cols]
        self._n_flow_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.n_flow_cols]
        self._nacl_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.nacl_cols]
        self._temp_col_idx = [int(self.frame.columns.get_loc(column)) for column in self.temp_cols]
        self._naoh_con_col_idx = [
            int(self.frame.columns.get_loc(f"naoh_con_{idx}"))
            for idx in range(1, 9)
        ]
        self.cell_current_caps = {
            column: float(self.train_frame[column].quantile(0.995))
            for column in self.current_cols
        }
        self.summary_ranges = {
            name: (
                float(self.train_frame[name].min()),
                float(self.train_frame[name].max()),
            )
            for name in (
                "total_current",
                "mean_temp",
                "mean_hcl",
                "mean_p_flow",
                "mean_n_flow",
                "mean_nacl_den",
                "glo_temp",
                "temp_naoh",
                "current_price",
                "daily_prod",
                "curr_eff",
                "djdh",
                "flow_imbalance",
            )
        }
        self._range_total_current = self.summary_ranges["total_current"]
        self._range_mean_temp = self.summary_ranges["mean_temp"]
        self._range_mean_hcl = self.summary_ranges["mean_hcl"]
        self._range_mean_p_flow = self.summary_ranges["mean_p_flow"]
        self._range_mean_n_flow = self.summary_ranges["mean_n_flow"]
        self._range_mean_nacl_den = self.summary_ranges["mean_nacl_den"]
        self._range_glo_temp = self.summary_ranges["glo_temp"]
        self._range_temp_naoh = self.summary_ranges["temp_naoh"]
        self._range_current_price = self.summary_ranges["current_price"]
        self._range_daily_prod = self.summary_ranges["daily_prod"]
        self._range_curr_eff = self.summary_ranges["curr_eff"]
        self._range_djdh = self.summary_ranges["djdh"]
        self._range_flow_imbalance = self.summary_ranges["flow_imbalance"]
        self.balance_pattern = np.linspace(-1.0, 1.0, len(self.current_cols), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=-np.ones(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
            high=np.ones(len(OBSERVATION_FEATURE_NAMES), dtype=np.float32),
            dtype=np.float32,
        )

        self._elapsed_steps = 0
        self._max_episode_steps = self.episode_length
        self._start_index = 0
        self._cursor = 0
        self._next_sequential_start = 0
        self._aging_index = 0.0
        self._current_limit_ratio = 1.0
        self._current_limit_ka = MAX_TOTAL_CURRENT_KA
        self._price_limit_ka = MAX_TOTAL_CURRENT_KA
        self._eff_gap_ema = 0.0
        self._previous_outputs = np.zeros(len(OUTPUT_COLUMNS), dtype=np.float32)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._sequence_indices: np.ndarray = np.arange(len(self.frame), dtype=np.int64)
        self._sequence_pos = 0

    def _load_or_fit_surrogate(self, seed: int | None) -> ChlorAlkaliSurrogate:
        cache_key = (str(self.train_path.resolve()), self.dynamic_price, int(seed or 0))
        cached = _SURROGATE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        cache_path = _surrogate_cache_path(
            self.train_path,
            dynamic_price=self.dynamic_price,
            seed=int(seed or 0),
        )
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fp:
                    surrogate = pickle.load(fp)
                if isinstance(surrogate, ChlorAlkaliSurrogate) and surrogate._fitted:
                    _SURROGATE_CACHE[cache_key] = surrogate
                    return surrogate
            except Exception:
                pass
        surrogate = ChlorAlkaliSurrogate(random_state=int(seed or 0))
        surrogate.fit(self.train_frame)
        _SURROGATE_CACHE[cache_key] = surrogate
        try:
            _atomic_pickle_dump(surrogate, cache_path)
        except Exception:
            pass
        return surrogate

    def _valid_start_indices(self) -> np.ndarray:
        if self.allowed_regime_ids:
            regime_ids = self.frame["regime_id"].to_numpy(dtype=np.int64, copy=False)
            allowed = np.asarray(self.allowed_regime_ids, dtype=np.int64)
            indices = np.flatnonzero(np.isin(regime_ids, allowed))
            if len(indices) == 0:
                raise ValueError(
                    f"No chlor-alkali rows matched allowed_regime_ids={self.allowed_regime_ids} "
                    f"under n_clusters={self.num_regime_clusters}."
                )
            return indices.astype(np.int64)
        if self.allowed_regimes:
            regime_names = self.frame["regime_name"].astype(str)
            indices = np.flatnonzero(regime_names.isin(self.allowed_regimes).to_numpy(dtype=bool))
            if len(indices) == 0:
                raise ValueError(
                    f"No chlor-alkali rows matched allowed_regimes={self.allowed_regimes} "
                    f"under n_clusters={self.num_regime_clusters}."
                )
            return indices.astype(np.int64)
        upper = max(len(self.frame) - self.episode_length - 1, 1)
        return np.arange(0, upper, dtype=np.int64)

    def _select_start_index(self) -> int:
        candidates = self._valid_start_indices()
        if self.schedule in {"sequential", "cyclic"}:
            if len(candidates) == 0:
                return 0
            start = int(candidates[self._next_sequential_start % len(candidates)])
            self._next_sequential_start += 1
            return start
        return int(self.rng.choice(candidates)) if len(candidates) > 0 else 0

    def _price_ratio(self, price: float) -> float:
        price_low, price_high = self.summary_ranges["current_price"]
        return float(np.clip((float(price) - price_low) / max(price_high - price_low, 1e-6), 0.0, 1.0))

    def _price_linked_limit_ka(self, price: float) -> float:
        return float(MAX_TOTAL_CURRENT_KA - CURRENT_LIMIT_SPAN_KA * self._price_ratio(price))

    def _compute_total_current_limit_ka(self, row: pd.Series) -> tuple[float, float]:
        price_limit_ka = self._price_linked_limit_ka(float(row["current_price"]))
        temp_stress = max(float(row["mean_temp"]) - 83.0, 0.0) / 4.0
        outage_stress = max(0.0, 1.0 - float(row["active_cells"]) / 8.0)
        imbalance_low, imbalance_high = self.summary_ranges["flow_imbalance"]
        imbalance_stress = abs(float(row["flow_imbalance"])) / max(abs(imbalance_low), abs(imbalance_high), 1e-6)
        limit_ka = float(
            np.clip(
                price_limit_ka
                - AGING_LIMIT_PENALTY_KA * self._aging_index
                - THERMAL_LIMIT_PENALTY_KA * temp_stress
                - OUTAGE_LIMIT_PENALTY_KA * outage_stress
                - FLOW_LIMIT_PENALTY_KA * min(imbalance_stress, 1.0),
                MIN_TOTAL_CURRENT_KA,
                price_limit_ka,
            )
        )
        return price_limit_ka, limit_ka

    def _refresh_price_and_features(self, row: pd.Series) -> pd.Series:
        refreshed = row.copy()
        timestamp = pd.Timestamp(refreshed["datetime"])
        if self.dynamic_price:
            base_price = chlor_alkali_dynamic_price(timestamp, proxy=self.price_proxy)
            refreshed["current_price"] = float(base_price) * self.price_multiplier + self.price_offset
        period = chlor_alkali_price_period(timestamp)
        refreshed["peak_flag"] = float(period == "peak")
        refreshed["flat_flag"] = float(period == "flat")
        refreshed["valley_flag"] = float(period == "valley")
        refreshed["summer_flag"] = float(timestamp.month in {7, 8, 9})
        minutes = timestamp.hour * 60 + timestamp.minute
        day_angle = 2.0 * np.pi * minutes / (24.0 * 60.0)
        month_angle = 2.0 * np.pi * (timestamp.month - 1.0) / 12.0
        refreshed["hour_sin"] = float(np.sin(day_angle))
        refreshed["hour_cos"] = float(np.cos(day_angle))
        refreshed["month_sin"] = float(np.sin(month_angle))
        refreshed["month_cos"] = float(np.cos(month_angle))
        current_values = refreshed.iloc[self._current_col_idx].to_numpy(dtype=np.float32, copy=False)
        temp_values = refreshed.iloc[self._temp_col_idx].to_numpy(dtype=np.float32, copy=False)
        hcl_values = refreshed.iloc[self._hcl_col_idx].to_numpy(dtype=np.float32, copy=False)
        p_flow_values = refreshed.iloc[self._p_flow_col_idx].to_numpy(dtype=np.float32, copy=False)
        n_flow_values = refreshed.iloc[self._n_flow_col_idx].to_numpy(dtype=np.float32, copy=False)
        nacl_values = refreshed.iloc[self._nacl_col_idx].to_numpy(dtype=np.float32, copy=False)
        naoh_values = refreshed.iloc[self._naoh_con_col_idx].to_numpy(dtype=np.float32, copy=False)
        refreshed["total_current"] = float(current_values.sum())
        refreshed["mean_temp"] = float(temp_values.mean())
        refreshed["mean_hcl"] = float(hcl_values.mean())
        refreshed["mean_p_flow"] = float(p_flow_values.mean())
        refreshed["mean_n_flow"] = float(n_flow_values.mean())
        refreshed["mean_nacl_den"] = float(nacl_values.mean())
        refreshed["mean_naoh_con"] = float(naoh_values.mean())
        refreshed["flow_imbalance"] = float(refreshed["mean_p_flow"] - refreshed["mean_n_flow"])
        refreshed["active_cells"] = float((current_values > 1.0).sum())
        refreshed["aging_proxy"] = float(self._aging_index)
        price_limit_ka, current_limit_ka = self._compute_total_current_limit_ka(refreshed)
        refreshed["price_limit_kA_proxy"] = float(price_limit_ka)
        refreshed["current_limit_kA_proxy"] = float(current_limit_ka)
        refreshed["current_limit_ratio_proxy"] = float(current_limit_ka / MAX_TOTAL_CURRENT_KA)
        refreshed["stress_proxy"] = float(self._stress_index(refreshed))
        refreshed["eff_gap_ema_proxy"] = float(self._eff_gap_ema)
        regime_values = refreshed.iloc[self._regime_feature_indices].to_numpy(dtype=np.float64, copy=False)
        regime_id, regime_name = self.regime_encoder.assign_values(regime_values)
        refreshed["regime_id"] = int(regime_id)
        refreshed["regime_name"] = regime_name
        return refreshed

    def _stress_index(self, row: pd.Series) -> float:
        total_low, total_high = self._range_total_current
        temp_low, temp_high = self._range_mean_temp
        imbalance_low, imbalance_high = self._range_flow_imbalance
        current_ratio = max(float(row["total_current"]) - total_low, 0.0) / max(total_high - total_low, 1e-6)
        temp_ratio = max(float(row["mean_temp"]) - 83.0, 0.0) / max(temp_high - temp_low, 1e-6)
        imbalance_ratio = abs(float(row["flow_imbalance"])) / max(abs(imbalance_low), abs(imbalance_high), 1e-6)
        outage_ratio = max(0.0, 1.0 - float(row["active_cells"]) / 8.0)
        return float(np.clip(0.55 * current_ratio + 0.20 * temp_ratio + 0.15 * imbalance_ratio + 0.10 * outage_ratio, 0.0, 1.0))

    def _apply_action(self, baseline_row: pd.Series, action: np.ndarray) -> tuple[pd.Series, dict[str, float]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        row = baseline_row.copy()
        current_scale = 1.0 + 0.08 * float(action[0])
        current_balance = 0.03 * float(action[1])
        acid_scale = 1.0 + 0.10 * float(action[2])
        anolyte_scale = 1.0 + 0.06 * float(action[3])
        catholyte_scale = 1.0 + 0.06 * float(action[4])
        brine_scale = 1.0 + 0.04 * float(action[5])

        raw_currents: list[float] = []
        current_violation = 0.0
        cell_violation = 0.0
        for offset, column in enumerate(self.current_cols):
            scaled = float(row[column]) * current_scale * (1.0 + current_balance * float(self.balance_pattern[offset]))
            raw_currents.append(max(0.0, scaled))

        total_preclip = float(np.sum(raw_currents))
        total_limit_ka = float(self._current_limit_ka)
        current_violation += max(0.0, total_preclip - total_limit_ka)
        total_scale = min(1.0, total_limit_ka / max(total_preclip, 1e-6))

        clipped_currents: list[float] = []
        for column, scaled in zip(self.current_cols, raw_currents):
            scaled *= total_scale
            cell_cap = float(self.cell_current_caps[column])
            cell_violation += max(0.0, scaled - cell_cap)
            clipped_currents.append(float(np.clip(scaled, 0.0, cell_cap)))

        clipped_total = float(np.sum(clipped_currents))
        if clipped_total > total_limit_ka + 1e-6:
            final_scale = total_limit_ka / clipped_total
            clipped_currents = [float(value * final_scale) for value in clipped_currents]
        for column, value in zip(self.current_cols, clipped_currents):
            row[column] = value
        current_violation += cell_violation

        for column in self.hcl_cols:
            row[column] = max(0.0, float(row[column]) * acid_scale)
        for column in self.p_flow_cols:
            row[column] = max(0.0, float(row[column]) * anolyte_scale)
        for column in self.n_flow_cols:
            row[column] = max(0.0, float(row[column]) * catholyte_scale)
        for column in self.nacl_cols:
            row[column] = max(0.0, float(row[column]) * brine_scale)

        row = self._refresh_price_and_features(row)
        thermal_violation = max(0.0, float(row["mean_temp"]) - 85.0)
        return row, {
            "current_scale": current_scale,
            "current_balance": current_balance,
            "acid_scale": acid_scale,
            "anolyte_scale": anolyte_scale,
            "catholyte_scale": catholyte_scale,
            "brine_scale": brine_scale,
            "current_violation": float(current_violation),
            "cell_violation": float(cell_violation),
            "total_current_preclip": total_preclip,
            "total_current_postclip": float(np.sum(clipped_currents)),
            "total_current_limit_kA": total_limit_ka,
            "thermal_violation": float(thermal_violation),
        }

    def _update_latent_state(self, controlled_row: pd.Series, control_meta: dict[str, float]) -> None:
        if not self.aging_dynamics:
            self._aging_index = 0.0
            self._current_limit_ratio = 1.0
            self._price_limit_ka = self._price_linked_limit_ka(float(controlled_row["current_price"]))
            self._current_limit_ka = self._price_limit_ka
            self._eff_gap_ema = 0.0
            return
        stress = self._stress_index(controlled_row)
        recovery = 0.07 * max(control_meta["acid_scale"] - 1.0, 0.0) + 0.03 * max(control_meta["brine_scale"] - 1.0, 0.0)
        self._aging_index = float(np.clip(0.996 * self._aging_index + 0.015 * stress - 0.010 * recovery, 0.0, 1.0))
        self._price_limit_ka, self._current_limit_ka = self._compute_total_current_limit_ka(controlled_row)
        self._current_limit_ratio = float(
            np.clip(self._current_limit_ka / MAX_TOTAL_CURRENT_KA, MIN_TOTAL_CURRENT_KA / MAX_TOTAL_CURRENT_KA, 1.0)
        )

    def _build_observation(self, exogenous_row: pd.Series) -> np.ndarray:
        row = self._refresh_price_and_features(exogenous_row)
        prod_low, prod_high = self._range_daily_prod
        eff_low, eff_high = self._range_curr_eff
        djdh_low, djdh_high = self._range_djdh
        total_low, total_high = self._range_total_current
        temp_low, temp_high = self._range_mean_temp
        hcl_low, hcl_high = self._range_mean_hcl
        pf_low, pf_high = self._range_mean_p_flow
        nf_low, nf_high = self._range_mean_n_flow
        nacl_low, nacl_high = self._range_mean_nacl_den
        glo_low, glo_high = self._range_glo_temp
        naoh_low, naoh_high = self._range_temp_naoh
        price_low, price_high = self._range_current_price
        imbalance_low, imbalance_high = self._range_flow_imbalance

        obs = np.asarray(
            [
                float(row["hour_sin"]),
                float(row["hour_cos"]),
                float(row["month_sin"]),
                float(row["month_cos"]),
                _normalized(float(row["current_price"]), price_low, price_high, clip=False),
                1.0 if float(row["peak_flag"]) > 0.5 else -1.0,
                1.0 if float(row["valley_flag"]) > 0.5 else -1.0,
                _normalized(float(row["total_current"]), total_low, total_high),
                _normalized(float(row["mean_temp"]), temp_low, temp_high),
                _normalized(float(row["mean_hcl"]), hcl_low, hcl_high),
                _normalized(float(row["mean_p_flow"]), pf_low, pf_high),
                _normalized(float(row["mean_n_flow"]), nf_low, nf_high),
                _normalized(float(row["mean_nacl_den"]), nacl_low, nacl_high),
                _normalized(float(row["glo_temp"]), glo_low, glo_high),
                _normalized(float(row["temp_naoh"]), naoh_low, naoh_high),
                float(np.clip(2.0 * float(row["active_cells"]) / 8.0 - 1.0, -1.0, 1.0)),
                float(np.clip(2.0 * self._aging_index - 1.0, -1.0, 1.0)),
                float(np.clip(2.0 * (self._current_limit_ratio - 0.55) / 0.45 - 1.0, -1.0, 1.0)),
                _normalized(float(row["flow_imbalance"]), imbalance_low, imbalance_high),
                float(np.clip(2.0 * float(row["stress_proxy"]) - 1.0, -1.0, 1.0)),
                float(np.clip(self._eff_gap_ema, -1.0, 1.0)),
                _normalized(float(self._previous_outputs[0]), prod_low, prod_high),
                _normalized(float(self._previous_outputs[1]), eff_low, eff_high),
                _normalized(float(self._previous_outputs[2]), djdh_low, djdh_high),
            ],
            dtype=np.float32,
        )
        return obs

    def _context_info(self, row: pd.Series) -> dict[str, Any]:
        price_low, price_high = self._range_current_price
        glo_low, glo_high = self._range_glo_temp
        naoh_low, naoh_high = self._range_temp_naoh
        imbalance_low, imbalance_high = self._range_flow_imbalance
        vector = np.asarray(
            [
                float(row["hour_sin"]),
                float(row["hour_cos"]),
                float(row["month_sin"]),
                float(row["month_cos"]),
                _normalized(float(row["current_price"]), price_low, price_high, clip=False),
                1.0 if float(row["peak_flag"]) > 0.5 else 0.0,
                1.0 if float(row["valley_flag"]) > 0.5 else 0.0,
                _normalized(float(row["glo_temp"]), glo_low, glo_high),
                _normalized(float(row["temp_naoh"]), naoh_low, naoh_high),
                float(np.clip(float(row["active_cells"]) / 8.0, 0.0, 1.0)),
                float(np.clip(self._aging_index, 0.0, 1.0)),
                float(np.clip(self._current_limit_ratio, 0.0, 1.0)),
                _normalized(float(row["flow_imbalance"]), imbalance_low, imbalance_high),
                float(np.clip(self._stress_index(row), 0.0, 1.0)),
                float(np.clip(self._eff_gap_ema, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return {
            "context_source": "chlor_alkali_environment_window",
            "context_label": str(row["regime_name"]),
            "context_feature_names": list(CONTEXT_FEATURE_NAMES),
            "context_vector": vector,
            "context_dict": {
                "datetime": str(pd.Timestamp(row["datetime"])),
                "current_price": float(row["current_price"]),
                "regime_name": str(row["regime_name"]),
                "regime_id": int(row["regime_id"]),
                "price_period": chlor_alkali_price_period(pd.Timestamp(row["datetime"])),
                "glo_temp": float(row["glo_temp"]),
                "temp_naoh": float(row["temp_naoh"]),
                "active_cells": float(row["active_cells"]),
                "aging_index": float(self._aging_index),
                "price_limit_kA": float(self._price_limit_ka),
                "current_limit_kA": float(self._current_limit_ka),
                "current_limit_ratio": float(self._current_limit_ratio),
                "flow_imbalance": float(row["flow_imbalance"]),
                "stress_index": float(row["stress_proxy"]),
                "eff_gap_ema": float(self._eff_gap_ema),
            },
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = dict(options or {})
        self._start_index = int(options.get("start_index", self._select_start_index()))
        candidates = self._valid_start_indices()
        if self.allowed_regimes and len(candidates) > 0:
            start_pos = int(np.where(candidates == self._start_index)[0][0]) if np.any(candidates == self._start_index) else 0
            self._sequence_indices = candidates
            self._sequence_pos = start_pos
            self._cursor = int(self._sequence_indices[self._sequence_pos])
        else:
            self._sequence_indices = np.arange(len(self.frame), dtype=np.int64)
            self._sequence_pos = int(self._start_index)
            self._cursor = self._start_index
        self._elapsed_steps = 0

        initial_row = self.frame.iloc[self._cursor].copy()
        if self.aging_dynamics:
            self._aging_index = float(initial_row["aging_proxy"])
            self._price_limit_ka = float(initial_row.get("price_limit_kA_proxy", MAX_TOTAL_CURRENT_KA))
            self._current_limit_ka = float(initial_row.get("current_limit_kA_proxy", MAX_TOTAL_CURRENT_KA))
            self._current_limit_ratio = float(initial_row["current_limit_ratio_proxy"])
            self._eff_gap_ema = float(initial_row["eff_gap_ema_proxy"])
        else:
            self._aging_index = 0.0
            self._price_limit_ka = self._price_linked_limit_ka(float(initial_row["current_price"]))
            self._current_limit_ka = self._price_limit_ka
            self._current_limit_ratio = float(self._current_limit_ka / MAX_TOTAL_CURRENT_KA)
            self._eff_gap_ema = 0.0
        self._previous_outputs = initial_row.loc[list(OUTPUT_COLUMNS)].to_numpy(dtype=np.float32)
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        obs = self._build_observation(initial_row)
        regime_name = str(initial_row["regime_name"])
        info = {
            "price_proxy": self.price_proxy.name,
            "price_multiplier": self.price_multiplier,
            "price_offset": self.price_offset,
            "dynamic_price": self.dynamic_price,
            "aging_dynamics": self.aging_dynamics,
            "regime_name": regime_name,
            "regime_id": int(initial_row["regime_id"]),
        }
        info.update(self._context_info(initial_row))
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        baseline_row = self.frame.iloc[self._cursor].copy()
        controlled_row, control_meta = self._apply_action(baseline_row, action)
        surrogate_values = controlled_row.iloc[self._surrogate_feature_indices].to_numpy(dtype=np.float32, copy=False)
        predicted_outputs = self.surrogate.predict_from_values(surrogate_values)

        current_cost = float(predicted_outputs[2] * controlled_row["current_price"])
        constraint_penalty = 25.0 * float(control_meta["current_violation"]) + 15.0 * float(control_meta["thermal_violation"])
        reward_vector = np.asarray(
            [
                float(predicted_outputs[0]),
                float(predicted_outputs[1]),
                -current_cost - constraint_penalty,
            ],
            dtype=np.float32,
        )

        self._eff_gap_ema = float(0.95 * self._eff_gap_ema + 0.05 * ((float(predicted_outputs[1]) - float(baseline_row["curr_eff"])) / 5.0))
        self._previous_outputs = predicted_outputs.astype(np.float32)
        self._last_action = action.copy()
        self._update_latent_state(controlled_row, control_meta)

        self._elapsed_steps += 1
        self._sequence_pos += 1
        if self.allowed_regimes:
            has_next = self._sequence_pos < len(self._sequence_indices)
            if has_next:
                self._cursor = int(self._sequence_indices[self._sequence_pos])
        else:
            self._cursor += 1
            has_next = self._cursor < len(self.frame)
        truncated = self._elapsed_steps >= self.episode_length or not has_next or self._cursor >= len(self.frame) - 1
        terminated = False

        next_row = self.frame.iloc[self._cursor].copy()
        observation = self._build_observation(next_row)
        regime_name = str(next_row["regime_name"])
        info: dict[str, Any] = {
            "reward_vector": reward_vector.copy(),
            "obj": reward_vector.copy(),
            "obj_raw": reward_vector.copy(),
            "predicted_outputs": {
                "daily_prod": float(predicted_outputs[0]),
                "curr_eff": float(predicted_outputs[1]),
                "djdh": float(predicted_outputs[2]),
            },
            "constraint_meta": control_meta,
            "price_proxy": self.price_proxy.name,
            "price_multiplier": self.price_multiplier,
            "price_offset": self.price_offset,
            "dynamic_price": self.dynamic_price,
            "aging_dynamics": self.aging_dynamics,
            "action_applied": {
                "current_scale": control_meta["current_scale"],
                "current_balance": control_meta["current_balance"],
                "acid_scale": control_meta["acid_scale"],
                "anolyte_scale": control_meta["anolyte_scale"],
                "catholyte_scale": control_meta["catholyte_scale"],
                "brine_scale": control_meta["brine_scale"],
            },
            "regime_name": regime_name,
            "regime_id": int(next_row["regime_id"]),
        }
        info.update(self._context_info(next_row))
        return observation, reward_vector, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
