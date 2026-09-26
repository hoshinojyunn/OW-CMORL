from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans

from .online_context import select_trace_indices


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / ".cache" / "fine_regimes"
CACHE_VERSION = 4
DEFAULT_FINE_REGIME_CLUSTERS = 20
DEFAULT_FINE_REGIME_EPISODES = 256
DEFAULT_FINE_REGIME_CONTEXT_STEPS = 4
CATALOG_CONTEXT_SAMPLE_MODE = "salient_mix"
CATALOG_CONTEXT_SALIENCY_ALPHA = 0.35
EVCHARGING_CONTEXT_FOCUS_FEATURES = {"active_ratio", "demand_ratio", "deadline_pressure"}


def coerce_regime_seed_plan(plan_source: Any) -> list[dict[str, Any]]:
    if isinstance(plan_source, (str, os.PathLike, Path)):
        payload = json.loads(Path(plan_source).read_text())
    else:
        payload = plan_source
    if isinstance(payload, dict):
        if isinstance(payload.get("shared_regime_seed_plan"), list):
            payload = payload["shared_regime_seed_plan"]
        elif isinstance(payload.get("plan"), list):
            payload = payload["plan"]
    if not isinstance(payload, list):
        raise ValueError("Expected shared regime seed plan as a list or JSON payload containing one.")
    rows = [dict(row) for row in payload if isinstance(row, dict)]
    if not rows:
        raise ValueError("Shared regime seed plan is empty.")
    return rows


def _safe_value_key(value: float, digits: int = 2) -> str:
    text = f"{float(value):.{digits}f}"
    return text.replace("-", "m").replace(".", "p")


def _atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _catalog_cache_path(
    *,
    env_key: str,
    feature_names: list[str],
    n_clusters: int,
    episodes: int,
    context_steps: int,
    seed: int,
    cache_signature: str = "",
) -> Path:
    fingerprint = "|".join(
        [
            str(CACHE_VERSION),
            env_key,
            ",".join(feature_names),
            str(int(n_clusters)),
            str(int(episodes)),
            str(int(context_steps)),
            str(int(seed)),
            str(cache_signature),
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{env_key}_{digest}.pkl"


def _plan_cache_path(
    *,
    env_key: str,
    search_start_seed: int,
    seeds_per_regime: int,
    context_steps: int,
    max_search_episodes: int,
    target_regime_ids: list[int],
    cache_signature: str = "",
) -> Path:
    fingerprint = "|".join(
        [
            str(CACHE_VERSION),
            "plan",
            env_key,
            str(int(search_start_seed)),
            str(int(seeds_per_regime)),
            str(int(context_steps)),
            str(int(max_search_episodes)),
            ",".join(str(int(regime_id)) for regime_id in target_regime_ids),
            str(cache_signature),
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{env_key}_plan_{digest}.pkl"


def _jsonable_fingerprint(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable_fingerprint(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_fingerprint(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable_fingerprint(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def catalog_cache_signature(value: Any) -> str:
    return json.dumps(
        _jsonable_fingerprint(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_cluster_order(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or len(centers) == 0:
        return np.zeros(0, dtype=np.int64)
    keys = [centers[:, idx] for idx in reversed(range(centers.shape[1]))]
    return np.lexsort(keys)


def _top_feature_indices(centers: np.ndarray, top_k: int = 3) -> list[int]:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] == 0:
        return []
    spread = np.std(centers, axis=0)
    order = np.argsort(-spread)
    return [int(idx) for idx in order[: min(top_k, len(order))]]


def _focus_evcharging_context_trace(
    episode_contexts: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    episode_contexts = np.asarray(episode_contexts, dtype=np.float64)
    if episode_contexts.ndim != 2 or len(episode_contexts) == 0:
        return episode_contexts
    if not EVCHARGING_CONTEXT_FOCUS_FEATURES.issubset(set(feature_names)):
        return episode_contexts
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    mask = np.zeros(len(episode_contexts), dtype=bool)
    for key in EVCHARGING_CONTEXT_FOCUS_FEATURES:
        idx = name_to_idx.get(key)
        if idx is None or idx >= episode_contexts.shape[1]:
            continue
        mask |= episode_contexts[:, idx] > 1e-6
    focused = episode_contexts[mask]
    if len(focused) == 0:
        return episode_contexts
    return focused


@dataclass
class FineRegimeCatalog:
    env_key: str
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    centers: np.ndarray
    labels: dict[int, str]
    counts: dict[int, int]
    raw_to_stable: dict[int, int]
    stable_to_raw: dict[int, int]
    sample_episode_seeds: np.ndarray | None = None
    sample_vectors: np.ndarray | None = None
    sample_regime_ids: np.ndarray | None = None

    def assign(self, vector: np.ndarray) -> tuple[int, str]:
        vector = np.asarray(vector, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != self.centers.shape[1]:
            raise ValueError(f"Expected vector dim {self.centers.shape[1]}, got {vector.shape[1]}")
        z = (vector - self.feature_mean) / self.feature_std
        center_z = (self.centers - self.feature_mean) / self.feature_std
        dist = np.sum((center_z - z) ** 2, axis=1)
        raw_idx = int(np.argmin(dist))
        stable_idx = int(self.raw_to_stable[raw_idx])
        return stable_idx, self.labels[stable_idx]

    def regime_meta(self, stable_idx: int) -> dict[str, float]:
        raw_idx = int(self.stable_to_raw[int(stable_idx)])
        center = self.centers[raw_idx]
        return {
            name: float(center[idx])
            for idx, name in enumerate(self.feature_names)
        }

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for stable_idx in sorted(self.labels):
            row = {
                "regime_id": int(stable_idx),
                "regime_name": self.labels[stable_idx],
                "count": int(self.counts.get(stable_idx, 0)),
            }
            row.update(self.regime_meta(stable_idx))
            out.append(row)
        return out

    def representative_records(
        self,
        target_regime_ids: list[int],
        *,
        seeds_per_regime: int = 1,
    ) -> dict[int, list[dict[str, Any]]]:
        if (
            self.sample_episode_seeds is None
            or self.sample_vectors is None
            or self.sample_regime_ids is None
        ):
            return {}

        sample_episode_seeds = np.asarray(self.sample_episode_seeds, dtype=np.int64)
        sample_vectors = np.asarray(self.sample_vectors, dtype=np.float64)
        sample_regime_ids = np.asarray(self.sample_regime_ids, dtype=np.int64)
        if (
            sample_vectors.ndim != 2
            or len(sample_vectors) == 0
            or len(sample_episode_seeds) != len(sample_vectors)
            or len(sample_regime_ids) != len(sample_vectors)
        ):
            return {}

        sample_z = (sample_vectors - self.feature_mean) / self.feature_std
        center_z = (self.centers - self.feature_mean) / self.feature_std
        out: dict[int, list[dict[str, Any]]] = {}
        for stable_idx in [int(idx) for idx in target_regime_ids]:
            raw_idx = int(self.stable_to_raw[stable_idx])
            candidate_idx = np.where(sample_regime_ids == stable_idx)[0]
            if len(candidate_idx) == 0:
                continue
            distances = np.sum((sample_z[candidate_idx] - center_z[raw_idx]) ** 2, axis=1)
            ranked_idx = candidate_idx[np.argsort(distances)]
            rows: list[dict[str, Any]] = []
            for sample_idx in ranked_idx[: max(1, int(seeds_per_regime))]:
                rows.append(
                    {
                        "episode_seed": int(sample_episode_seeds[sample_idx]),
                        "context_vector": sample_vectors[sample_idx].astype(np.float64).tolist(),
                        "regime": str(self.labels[stable_idx]),
                        "regime_id": int(stable_idx),
                        "regime_meta": dict(self.regime_meta(stable_idx)),
                    }
                )
            out[stable_idx] = rows
        return out


def build_catalog_from_episode_vectors(
    *,
    env_key: str,
    feature_names: list[str],
    vectors: np.ndarray,
    episode_seeds: np.ndarray | list[int] | None = None,
    n_clusters: int = DEFAULT_FINE_REGIME_CLUSTERS,
    seed: int = 0,
) -> FineRegimeCatalog:
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError(f"Cannot build regime catalog for {env_key} from empty vectors.")
    feature_mean = x.mean(axis=0)
    feature_std = np.maximum(x.std(axis=0), 1e-6)
    x_norm = (x - feature_mean) / feature_std
    num_clusters = min(int(max(1, n_clusters)), len(x_norm))
    model = KMeans(n_clusters=num_clusters, n_init=20, random_state=int(seed))
    raw_ids = model.fit_predict(x_norm)
    centers = model.cluster_centers_ * feature_std + feature_mean

    stable_order = _stable_cluster_order(centers)
    raw_to_stable = {int(raw): int(stable) for stable, raw in enumerate(stable_order)}
    stable_to_raw = {int(stable): int(raw) for stable, raw in enumerate(stable_order)}

    stable_ids = np.asarray([raw_to_stable[int(raw)] for raw in raw_ids], dtype=np.int64)
    counts_arr = np.bincount(stable_ids, minlength=num_clusters)
    counts = {int(idx): int(count) for idx, count in enumerate(counts_arr)}

    feature_idx = _top_feature_indices(centers, top_k=min(3, len(feature_names)))
    labels: dict[int, str] = {}
    for stable_idx in range(num_clusters):
        raw_idx = stable_to_raw[stable_idx]
        center = centers[raw_idx]
        parts = [f"ctx_{stable_idx:02d}"]
        for idx in feature_idx:
            parts.append(f"{feature_names[idx]}_{_safe_value_key(center[idx], 2)}")
        labels[stable_idx] = "_".join(parts)

    if episode_seeds is None:
        sample_episode_seeds_arr = np.arange(len(x), dtype=np.int64)
    else:
        sample_episode_seeds_arr = np.asarray(episode_seeds, dtype=np.int64)
        if len(sample_episode_seeds_arr) != len(x):
            raise ValueError(
                f"Expected {len(x)} episode seeds for {env_key}, got {len(sample_episode_seeds_arr)}."
            )

    return FineRegimeCatalog(
        env_key=env_key,
        feature_names=list(feature_names),
        feature_mean=feature_mean.astype(np.float64),
        feature_std=feature_std.astype(np.float64),
        centers=centers.astype(np.float64),
        labels=labels,
        counts=counts,
        raw_to_stable=raw_to_stable,
        stable_to_raw=stable_to_raw,
        sample_episode_seeds=sample_episode_seeds_arr.astype(np.int64),
        sample_vectors=x.astype(np.float64),
        sample_regime_ids=stable_ids.astype(np.int64),
    )


def sample_episode_context_vectors(
    env_factory: Callable[[int], Any],
    *,
    episodes: int = DEFAULT_FINE_REGIME_EPISODES,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    def _zero_action_for(env) -> Any:
        if getattr(env.action_space, "shape", None) is not None:
            return np.zeros(env.action_space.shape, dtype=np.float32)
        return 0

    def _collect_episode_context_trace(
        env,
        *,
        episode_seed: int,
        zero_action: Any,
    ) -> tuple[np.ndarray, list[str]]:
        reset_out = env.reset(seed=int(episode_seed))
        if isinstance(reset_out, tuple):
            _obs, info = reset_out
        else:
            info = {}

        feature_names: list[str] = []
        episode_contexts: list[np.ndarray] = []
        if info.get("context_vector") is not None and len(np.asarray(info["context_vector"]).reshape(-1)) > 0:
            episode_contexts.append(np.asarray(info["context_vector"], dtype=np.float64).reshape(-1))
            if info.get("context_feature_names"):
                feature_names = list(info.get("context_feature_names", []))

        done = False
        while not done:
            step_out = env.step(zero_action)
            if len(step_out) == 5:
                _obs, _reward, terminated, truncated, step_info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                _obs, _reward, done, step_info = step_out
            context_vec = np.asarray(step_info.get("context_vector", []), dtype=np.float64).reshape(-1)
            if len(context_vec) > 0:
                episode_contexts.append(context_vec)
                if not feature_names and step_info.get("context_feature_names"):
                    feature_names = list(step_info.get("context_feature_names", []))

        if not episode_contexts:
            return np.zeros((0, 0), dtype=np.float64), feature_names
        return np.stack(episode_contexts, axis=0), feature_names

    def _summarize_episode_context_trace(
        episode_contexts: np.ndarray,
        *,
        feature_names: list[str],
        sample_steps: int,
    ) -> np.ndarray | None:
        episode_contexts = np.asarray(episode_contexts, dtype=np.float64)
        if episode_contexts.ndim != 2 or len(episode_contexts) == 0:
            return None
        episode_contexts = _focus_evcharging_context_trace(episode_contexts, feature_names)
        sample_idx = select_trace_indices(
            episode_contexts,
            sample_size=max(1, int(sample_steps)),
            sample_mode=CATALOG_CONTEXT_SAMPLE_MODE,
            saliency_alpha=CATALOG_CONTEXT_SALIENCY_ALPHA,
        )
        return np.mean(episode_contexts[sample_idx], axis=0)

    env = env_factory(0)
    feature_names: list[str] = []
    vectors = []
    episode_seeds = []
    try:
        zero_action = _zero_action_for(env)
        for episode_idx in range(max(1, int(episodes))):
            episode_contexts, sampled_feature_names = _collect_episode_context_trace(
                env,
                episode_seed=episode_idx,
                zero_action=zero_action,
            )
            vector = _summarize_episode_context_trace(
                episode_contexts,
                feature_names=sampled_feature_names,
                sample_steps=context_steps,
            )
            if vector is not None:
                vectors.append(vector)
                episode_seeds.append(int(episode_idx))
                if not feature_names and sampled_feature_names:
                    feature_names = list(sampled_feature_names)
    finally:
        env.close()
    if not vectors:
        raise ValueError("Failed to sample any context vectors from environment.")
    return (
        np.asarray(vectors, dtype=np.float64),
        feature_names,
        np.asarray(episode_seeds, dtype=np.int64),
    )


def sample_episode_context_vector(
    env,
    *,
    episode_seed: int,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
) -> np.ndarray | None:
    if getattr(env.action_space, "shape", None) is not None:
        zero_action: Any = np.zeros(env.action_space.shape, dtype=np.float32)
    else:
        zero_action = 0

    episode_contexts: list[np.ndarray] = []
    reset_out = env.reset(seed=int(episode_seed))
    if isinstance(reset_out, tuple):
        _obs, info = reset_out
    else:
        info = {}
    feature_names: list[str] = []
    if info.get("context_vector") is not None and len(np.asarray(info["context_vector"]).reshape(-1)) > 0:
        episode_contexts.append(np.asarray(info["context_vector"], dtype=np.float64).reshape(-1))
        if info.get("context_feature_names"):
            feature_names = list(info.get("context_feature_names", []))
    done = False
    while not done:
        step_out = env.step(zero_action)
        if len(step_out) == 5:
            _obs, _reward, terminated, truncated, step_info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            _obs, _reward, done, step_info = step_out
        context_vec = np.asarray(step_info.get("context_vector", []), dtype=np.float64).reshape(-1)
        if len(context_vec) > 0:
            episode_contexts.append(context_vec)
            if not feature_names and step_info.get("context_feature_names"):
                feature_names = list(step_info.get("context_feature_names", []))
    if not episode_contexts:
        return None
    episode_contexts_arr = np.stack(episode_contexts, axis=0)
    episode_contexts_arr = _focus_evcharging_context_trace(episode_contexts_arr, feature_names)
    sample_idx = select_trace_indices(
        episode_contexts_arr,
        sample_size=max(1, int(context_steps)),
        sample_mode=CATALOG_CONTEXT_SAMPLE_MODE,
        saliency_alpha=CATALOG_CONTEXT_SALIENCY_ALPHA,
    )
    return np.mean(episode_contexts_arr[sample_idx], axis=0)


def episode_context_vectors_from_trace(
    trace: dict[str, Any],
    *,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
) -> dict[int, np.ndarray]:
    context = np.asarray(trace.get("context", []), dtype=np.float64)
    episodes = np.asarray(trace.get("episode", []), dtype=np.int64)
    if context.ndim != 2 or len(context) == 0 or len(episodes) != len(context):
        return {}
    out: dict[int, np.ndarray] = {}
    feature_names = list(trace.get("context_feature_names", []))
    for episode_id in np.unique(episodes):
        mask = episodes == episode_id
        episode_context = context[mask]
        if len(episode_context) == 0:
            continue
        episode_context = _focus_evcharging_context_trace(episode_context, feature_names)
        sample_idx = select_trace_indices(
            episode_context,
            sample_size=max(1, int(context_steps)),
            sample_mode=CATALOG_CONTEXT_SAMPLE_MODE,
            saliency_alpha=CATALOG_CONTEXT_SALIENCY_ALPHA,
        )
        out[int(episode_id)] = episode_context[sample_idx].mean(axis=0)
    return out


def assign_episode_returns_to_catalog(
    trace: dict[str, Any],
    *,
    catalog: FineRegimeCatalog,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
) -> list[dict[str, Any]]:
    episode_vectors = episode_context_vectors_from_trace(trace, context_steps=context_steps)
    regime_records = trace.get("regime_records", [])
    assigned = []
    for record in regime_records:
        episode_id = int(record.get("eval_id", record.get("episode", -1)))
        vector = episode_vectors.get(episode_id)
        if vector is None:
            continue
        regime_id, regime_name = catalog.assign(vector)
        assigned.append(
            {
                "episode": int(episode_id),
                "regime_id": int(regime_id),
                "regime": regime_name,
                "regime_meta": catalog.regime_meta(regime_id),
                "context_vector": vector.astype(np.float64),
                "return": np.asarray(record["return"], dtype=np.float64),
            }
        )
    return assigned


def assign_episode_returns_to_plan(
    trace: dict[str, Any],
    *,
    plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    regime_records = list(trace.get("regime_records", []))
    assigned: list[dict[str, Any]] = []
    for plan_idx, plan_row in enumerate(plan):
        if plan_idx >= len(regime_records):
            break
        record = regime_records[plan_idx]
        assigned.append(
            {
                "episode": int(record.get("eval_id", record.get("episode", plan_idx))),
                "regime_id": int(plan_row["regime_id"]),
                "regime": str(plan_row["regime"]),
                "regime_meta": dict(plan_row.get("regime_meta", {})),
                "context_vector": np.asarray([], dtype=np.float64),
                "return": np.asarray(record["return"], dtype=np.float64),
            }
        )
    return assigned


def load_or_build_catalog(
    *,
    env_key: str,
    env_factory: Callable[[int], Any],
    n_clusters: int = DEFAULT_FINE_REGIME_CLUSTERS,
    episodes: int = DEFAULT_FINE_REGIME_EPISODES,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    seed: int = 0,
    cache_signature: str = "",
) -> FineRegimeCatalog:
    temp_env = env_factory(0)
    try:
        feature_names = list(getattr(temp_env, "context_feature_names", []))
    finally:
        temp_env.close()
    cache_path = _catalog_cache_path(
        env_key=env_key,
        feature_names=feature_names,
        n_clusters=n_clusters,
        episodes=episodes,
        context_steps=context_steps,
        seed=seed,
        cache_signature=cache_signature,
    )
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fp:
                catalog = pickle.load(fp)
            if isinstance(catalog, FineRegimeCatalog):
                return catalog
        except Exception:
            pass
    vectors, sampled_feature_names, sampled_episode_seeds = sample_episode_context_vectors(
        env_factory,
        episodes=episodes,
        context_steps=context_steps,
    )
    catalog = build_catalog_from_episode_vectors(
        env_key=env_key,
        feature_names=sampled_feature_names,
        vectors=vectors,
        episode_seeds=sampled_episode_seeds,
        n_clusters=n_clusters,
        seed=seed,
    )
    try:
        _atomic_pickle_dump(catalog, cache_path)
    except Exception:
        pass
    return catalog


def build_regime_seed_plan(
    *,
    env_factory: Callable[[int], Any],
    catalog: FineRegimeCatalog,
    search_start_seed: int,
    seeds_per_regime: int = 1,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    max_search_episodes: int = 2048,
    target_regime_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    target_ids = [int(idx) for idx in (target_regime_ids or sorted(catalog.labels))]
    if not target_ids:
        return []

    buckets: dict[int, list[dict[str, Any]]] = {int(idx): [] for idx in target_ids}
    accepted_records: list[dict[str, Any]] = []
    used_episode_seeds: set[int] = set()

    representative_records = catalog.representative_records(
        target_ids,
        seeds_per_regime=int(seeds_per_regime),
    )
    for regime_id in target_ids:
        for row in representative_records.get(int(regime_id), []):
            episode_seed = int(row["episode_seed"])
            if episode_seed in used_episode_seeds or len(buckets[int(regime_id)]) >= int(seeds_per_regime):
                continue
            record = {
                **row,
                "sequence_index": int(len(accepted_records)),
                "probe_index": -1,
            }
            buckets[int(regime_id)].append(record)
            accepted_records.append(record)
            used_episode_seeds.add(episode_seed)

    if all(len(rows) >= int(seeds_per_regime) for rows in buckets.values()):
        plan = []
        for regime_id in sorted(buckets):
            rows = buckets[regime_id]
            rows = sorted(rows, key=lambda row: int(row.get("sequence_index", 0)))
            plan.append(
                {
                    "regime_id": int(regime_id),
                    "regime": str(catalog.labels[regime_id]),
                    "regime_meta": dict(catalog.regime_meta(regime_id)),
                    "episode_seeds": [int(row["episode_seed"]) for row in rows],
                    "context_vectors": [list(row["context_vector"]) for row in rows],
                    "episode_seed_records": [
                        {
                            "episode_seed": int(row["episode_seed"]),
                            "context_vector": list(row["context_vector"]),
                            "sequence_index": int(row["sequence_index"]),
                            "probe_index": int(row["probe_index"]),
                        }
                        for row in rows
                    ],
                }
            )
        return plan

    env = env_factory(0)
    try:
        for offset in range(max(1, int(max_search_episodes))):
            if all(len(rows) >= int(seeds_per_regime) for rows in buckets.values()):
                break
            episode_seed = int(search_start_seed) + int(offset)
            if episode_seed in used_episode_seeds:
                continue
            vector = sample_episode_context_vector(
                env,
                episode_seed=episode_seed,
                context_steps=context_steps,
            )
            if vector is None:
                continue
            regime_id, regime_name = catalog.assign(vector)
            regime_id = int(regime_id)
            if regime_id not in buckets or len(buckets[regime_id]) >= int(seeds_per_regime):
                continue
            record = {
                "episode_seed": int(episode_seed),
                "context_vector": np.asarray(vector, dtype=np.float64).tolist(),
                "regime": str(regime_name),
                "regime_id": int(regime_id),
                "regime_meta": dict(catalog.regime_meta(regime_id)),
                "sequence_index": int(len(accepted_records)),
                "probe_index": int(offset),
            }
            buckets[regime_id].append(record)
            accepted_records.append(record)
            used_episode_seeds.add(episode_seed)
            if all(len(rows) >= int(seeds_per_regime) for rows in buckets.values()):
                break
    finally:
        env.close()

    missing = [regime_id for regime_id, rows in buckets.items() if len(rows) < int(seeds_per_regime)]
    if missing:
        raise RuntimeError(
            f"Failed to cover fine regimes {missing} with {seeds_per_regime} seed(s) each "
            f"from seed {int(search_start_seed)} over {int(max_search_episodes)} probes."
        )

    plan = []
    for regime_id in sorted(buckets):
        rows = buckets[regime_id]
        rows = sorted(rows, key=lambda row: int(row.get("sequence_index", 0)))
        plan.append(
            {
                "regime_id": int(regime_id),
                "regime": str(catalog.labels[regime_id]),
                "regime_meta": dict(catalog.regime_meta(regime_id)),
                "episode_seeds": [int(row["episode_seed"]) for row in rows],
                "context_vectors": [list(row["context_vector"]) for row in rows],
                "episode_seed_records": [
                    {
                        "episode_seed": int(row["episode_seed"]),
                        "context_vector": list(row["context_vector"]),
                        "sequence_index": int(row["sequence_index"]),
                        "probe_index": int(row["probe_index"]),
                    }
                    for row in rows
                ],
            }
    )
    return plan


def load_or_build_regime_seed_plan(
    *,
    env_key: str,
    env_factory: Callable[[int], Any],
    catalog: FineRegimeCatalog,
    search_start_seed: int,
    seeds_per_regime: int = 1,
    context_steps: int = DEFAULT_FINE_REGIME_CONTEXT_STEPS,
    max_search_episodes: int = 2048,
    target_regime_ids: list[int] | None = None,
    cache_signature: str = "",
) -> list[dict[str, Any]]:
    target_ids = [int(idx) for idx in (target_regime_ids or sorted(catalog.labels))]
    if not target_ids:
        return []
    cache_path = _plan_cache_path(
        env_key=env_key,
        search_start_seed=int(search_start_seed),
        seeds_per_regime=int(seeds_per_regime),
        context_steps=int(context_steps),
        max_search_episodes=int(max_search_episodes),
        target_regime_ids=target_ids,
        cache_signature=cache_signature,
    )
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fp:
                plan = pickle.load(fp)
            if isinstance(plan, list):
                return plan
        except Exception:
            pass
    plan = build_regime_seed_plan(
        env_factory=env_factory,
        catalog=catalog,
        search_start_seed=int(search_start_seed),
        seeds_per_regime=int(seeds_per_regime),
        context_steps=int(context_steps),
        max_search_episodes=int(max_search_episodes),
        target_regime_ids=target_ids,
    )
    try:
        _atomic_pickle_dump(plan, cache_path)
    except Exception:
        pass
    return plan


def selected_episode_seeds_from_plan(plan: list[dict[str, Any]]) -> list[int]:
    ordered_records: list[int] = []
    for row in plan:
        episode_seeds = [int(seed) for seed in row.get("episode_seeds", [])]
        if episode_seeds:
            ordered_records.extend(episode_seeds)
            continue
        records = row.get("episode_seed_records", [])
        ordered_records.extend(int(record["episode_seed"]) for record in records)
    return ordered_records


def flatten_regime_seed_plan(plan: list[dict[str, Any]]) -> list[int]:
    ordered_records: list[tuple[int, int]] = []
    probe_records: list[tuple[int, int]] = []
    fallback_records: list[int] = []
    for row in plan:
        records = row.get("episode_seed_records", [])
        if records:
            for record in records:
                ordered_records.append((int(record.get("sequence_index", 0)), int(record["episode_seed"])))
                probe_index = int(record.get("probe_index", -1))
                if probe_index >= 0:
                    probe_records.append((probe_index, int(record["episode_seed"])))
        else:
            for seed in row.get("episode_seeds", []):
                fallback_records.append(int(seed))
    if probe_records:
        start_seed = min(seed - probe_idx for probe_idx, seed in probe_records)
        stop_probe_idx = max(probe_idx for probe_idx, _seed in probe_records)
        return [int(start_seed + probe_idx) for probe_idx in range(stop_probe_idx + 1)]
    if ordered_records:
        ordered_records.sort(key=lambda item: item[0])
        return [seed for _idx, seed in ordered_records]
    return fallback_records
