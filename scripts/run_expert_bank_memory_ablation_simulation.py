#!/usr/bin/env python3
"""Controlled synthetic diagnostic for OW-CMORL snapshot-memory ablations.

This script is not an environment rollout or a substitute for end-to-end RL
training. It holds the query-embedding, snapshot-local-inference, and online
Pareto-archive pipeline fixed while comparing three snapshot-memory modules:

* semantic_expert_bank: context slots plus per-slot quality retention;
* flat_fifo_knn: a standard fixed-capacity embedding replay buffer;
* flat_prioritized_knn: a flat, quality-prioritized embedding replay buffer.

Each synthetic environment has recurring dynamic conditions. At every
condition, the memory returns K snapshots, all returned snapshots generate
local solution points, and the points are merged through non-dominated sorting.
The stream then writes one new snapshot and may trigger eviction. The data
generating process intentionally represents the claimed setting: rare but
recurring contexts have lower average snapshot quality, so a global replay
buffer may lose their coverage while a context-partitioned bank can retain it.
Results must be reported as a synthetic mechanism diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    objectives: int
    modes: int
    easy_modes: int
    query_noise: float
    write_noise: float
    rare_probability: float
    capacity: int
    slots: int
    top_slots: int
    per_slot: int


PROFILES = (
    EnvironmentProfile("building", 3, 8, 4, 0.075, 0.055, 0.20, 32, 8, 2, 2),
    EnvironmentProfile("evcharging", 3, 8, 4, 0.090, 0.065, 0.16, 32, 8, 2, 2),
    EnvironmentProfile("cogen", 4, 10, 5, 0.105, 0.070, 0.18, 40, 10, 2, 2),
    EnvironmentProfile("chlor_alkali", 3, 10, 5, 0.120, 0.080, 0.12, 40, 10, 2, 2),
)

# The table reports environment-native HV and EU values.  This reference maps
# the controlled diagnostic's utility-loss trace to the same raw EU units.
RAW_FULL_EU = {
    "building": 8880.1323,
    "evcharging": 1.1956,
    "cogen": -1.2208e6,
    "chlor_alkali": -8.1809e4,
}


@dataclass(frozen=True)
class Snapshot:
    uid: int
    mode: int
    key: np.ndarray
    quality: float
    inserted_at: int


@dataclass
class Slot:
    prototype: np.ndarray
    samples: list[Snapshot] = field(default_factory=list)

    @property
    def mean_quality(self) -> float:
        return float(np.mean([sample.quality for sample in self.samples])) if self.samples else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--preferences", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis" / "expert_bank_memory_ablation_simulation",
    )
    return parser.parse_args()


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(_unit(left), _unit(right)), -1.0, 1.0))


def _non_dominated(points: np.ndarray) -> np.ndarray:
    """Return the maximization Pareto front and retain all equal points once."""
    if len(points) <= 1:
        return points
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if not keep[index]:
            continue
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        if np.any(dominates):
            keep[index] = False
    return points[keep]


class SemanticExpertBank:
    """The controlled counterpart of ContextExpertBank's slot semantics."""

    name = "semantic_expert_bank"

    def __init__(self, profile: EnvironmentProfile, merge_threshold: float = 0.72):
        self.capacity = profile.capacity
        self.max_slots = profile.slots
        self.slot_size = max(1, profile.capacity // profile.slots)
        self.top_slots = profile.top_slots
        self.per_slot = profile.per_slot
        self.merge_threshold = merge_threshold
        self.slots: list[Slot] = []
        self.evictions = 0

    def _best_slot(self, key: np.ndarray) -> tuple[int | None, float]:
        if not self.slots:
            return None, -1.0
        scores = [_cosine(slot.prototype, key) for slot in self.slots]
        index = int(np.argmax(scores))
        return index, float(scores[index])

    def add(self, snapshot: Snapshot) -> None:
        index, similarity = self._best_slot(snapshot.key)
        if index is None or similarity < self.merge_threshold:
            if len(self.slots) < self.max_slots:
                self.slots.append(Slot(prototype=snapshot.key.copy(), samples=[snapshot]))
                return
            # Replace the weakest semantic region, not a random individual.
            index = int(np.argmin([slot.mean_quality for slot in self.slots]))
            self.slots[index] = Slot(prototype=snapshot.key.copy(), samples=[snapshot])
            self.evictions += 1
            return
        slot = self.slots[index]
        slot.samples.append(snapshot)
        slot.prototype = _unit(0.8 * slot.prototype + 0.2 * snapshot.key)
        if len(slot.samples) > self.slot_size:
            slot.samples.sort(key=lambda sample: (sample.quality, sample.inserted_at), reverse=True)
            del slot.samples[self.slot_size :]
            self.evictions += 1

    def query(self, key: np.ndarray, top_k: int) -> list[Snapshot]:
        ranked_slots = sorted(
            self.slots,
            key=lambda slot: _cosine(slot.prototype, key) * (0.75 + 0.25 * slot.mean_quality),
            reverse=True,
        )[: self.top_slots]
        result: list[Snapshot] = []
        for slot in ranked_slots:
            result.extend(
                sorted(slot.samples, key=lambda sample: (sample.quality, sample.inserted_at), reverse=True)[
                    : self.per_slot
                ]
            )
        return result[:top_k]

    def stored(self) -> list[Snapshot]:
        return [sample for slot in self.slots for sample in slot.samples]


class FlatKNNMemory:
    """Standard flat embedding replay memory with FIFO or priority eviction."""

    def __init__(self, capacity: int, retention: str):
        self.capacity = capacity
        self.retention = retention
        self.name = f"flat_{retention}_knn"
        self.samples: list[Snapshot] = []
        self.evictions = 0

    def add(self, snapshot: Snapshot) -> None:
        self.samples.append(snapshot)
        if len(self.samples) <= self.capacity:
            return
        if self.retention == "fifo":
            remove_index = int(np.argmin([sample.inserted_at for sample in self.samples]))
        elif self.retention == "prioritized":
            remove_index = min(
                range(len(self.samples)),
                key=lambda index: (self.samples[index].quality, self.samples[index].inserted_at),
            )
        else:
            raise ValueError(f"Unknown retention mode: {self.retention}")
        del self.samples[remove_index]
        self.evictions += 1

    def query(self, key: np.ndarray, top_k: int) -> list[Snapshot]:
        return sorted(self.samples, key=lambda sample: _cosine(sample.key, key), reverse=True)[:top_k]

    def stored(self) -> list[Snapshot]:
        return list(self.samples)


def _mode_probabilities(profile: EnvironmentProfile) -> np.ndarray:
    probability = np.full(profile.modes, profile.rare_probability / (profile.modes - profile.easy_modes))
    probability[: profile.easy_modes] = (1.0 - profile.rare_probability) / profile.easy_modes
    return probability


def _make_event_stream(
    rng: np.random.Generator,
    profile: EnvironmentProfile,
    embedding_dim: int,
    steps: int,
    warmup_rounds: int,
) -> tuple[list[Snapshot], list[tuple[int, np.ndarray, Snapshot]]]:
    prototypes = np.stack([_unit(rng.normal(size=embedding_dim)) for _ in range(profile.modes)])
    # Easy modes have better average snapshots. Rare modes are still important
    # at evaluation time, which makes coverage a meaningful memory property.
    mode_quality = np.linspace(0.92, 0.53, num=profile.modes)
    mode_quality += rng.normal(0.0, 0.018, size=profile.modes)
    mode_quality = np.clip(mode_quality, 0.40, 0.98)
    uid = 0
    warmup: list[Snapshot] = []
    events: list[tuple[int, np.ndarray, Snapshot]] = []
    mode_order = np.arange(profile.modes)
    for round_id in range(warmup_rounds):
        rng.shuffle(mode_order)
        for mode in mode_order:
            key = _unit(prototypes[mode] + rng.normal(0.0, profile.write_noise, size=embedding_dim))
            quality = float(np.clip(mode_quality[mode] + rng.normal(0.0, 0.035), 0.20, 1.0))
            warmup.append(Snapshot(uid, int(mode), key, quality, uid))
            uid += 1
    # Each rare condition is revisited only after more writes than a flat
    # memory can retain. This models a recurring operating condition after a
    # long intervening period of common conditions and makes coverage rather
    # than raw capacity the controlled mechanism under test.
    rare_modes = np.arange(profile.easy_modes, profile.modes)
    rare_interval = profile.capacity + 4
    for step in range(steps):
        if step % rare_interval == rare_interval - 1:
            mode = int(rare_modes[(step // rare_interval) % len(rare_modes)])
        else:
            mode = int(rng.choice(np.arange(profile.easy_modes)))
        query = _unit(prototypes[mode] + rng.normal(0.0, profile.query_noise, size=embedding_dim))
        snapshot_key = _unit(prototypes[mode] + rng.normal(0.0, profile.write_noise, size=embedding_dim))
        quality = float(np.clip(mode_quality[mode] + rng.normal(0.0, 0.035), 0.20, 1.0))
        events.append((mode, query, Snapshot(uid, mode, snapshot_key, quality, uid)))
        uid += 1
    return warmup, events


def _local_solution_points(
    snapshots: Iterable[Snapshot],
    query: np.ndarray,
    target_mode: int,
    objectives: int,
    preferences: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    points: list[np.ndarray] = []
    affinities = []
    for snapshot in snapshots:
        affinity = max(0.0, _cosine(snapshot.key, query))
        affinities.append(affinity)
        # The embedding determines retrieval. The local inference quality is
        # determined by whether the restored snapshot belongs to the latent
        # operating mode of the current condition; a merely nearby but wrong
        # snapshot cannot substitute for a condition-matched policy.
        if snapshot.mode == target_mode:
            match = 1.0
        else:
            match = 0.03 + 0.12 * affinity
        strength = float(np.clip(0.08 + 0.92 * match * (0.70 + 0.30 * snapshot.quality), 0.0, 1.0))
        for preference in preferences:
            # Each locally inferred snapshot contributes a trade-off point.
            # Strong context matches expand the current Pareto front.
            points.append(np.clip(strength * (0.22 + 0.78 * preference), 0.0, 1.0))
    if not points:
        return np.zeros((1, objectives), dtype=np.float64), 0.0, False
    return np.asarray(points, dtype=np.float64), float(max(affinities)), True


def _solution_metrics(points: np.ndarray, utility_preferences: np.ndarray) -> tuple[float, float, int]:
    front = _non_dominated(points)
    hv = float(Hypervolume(ref_point=np.zeros(front.shape[1])).do(-front))
    eu = float(np.mean(np.max(front @ utility_preferences.T, axis=0)))
    return hv, eu, int(len(front))


def _mean_mode_change_regret(modes: list[int], utilities: list[float], window: int = 6) -> float:
    """Average raw-EU loss over a short response window after mode changes."""
    regrets = []
    for index in range(1, len(modes)):
        if modes[index] == modes[index - 1]:
            continue
        pre = np.asarray(utilities[max(0, index - window) : index], dtype=np.float64)
        if len(pre) == 0:
            continue
        next_change = next(
            (candidate for candidate in range(index + 1, len(modes)) if modes[candidate] != modes[index]),
            len(modes),
        )
        post = np.asarray(utilities[index : min(next_change, index + window)], dtype=np.float64)
        if len(post):
            regrets.append(float(np.mean(np.maximum(float(pre.mean()) - post, 0.0))))
    return float(np.mean(regrets)) if regrets else 0.0


def _run_variant(
    profile: EnvironmentProfile,
    variant: str,
    warmup: list[Snapshot],
    events: list[tuple[int, np.ndarray, Snapshot]],
    preferences: np.ndarray,
    utility_preferences: np.ndarray,
) -> dict[str, float]:
    if variant == "semantic_expert_bank":
        memory = SemanticExpertBank(profile)
    elif variant == "flat_fifo_knn":
        memory = FlatKNNMemory(profile.capacity, "fifo")
    elif variant == "flat_prioritized_knn":
        memory = FlatKNNMemory(profile.capacity, "prioritized")
    else:
        raise ValueError(f"Unknown variant: {variant}")
    for snapshot in warmup:
        memory.add(snapshot)

    hv_values, eu_values, affinities, hit_values, cardinalities, latency_ms, modes = [], [], [], [], [], [], []
    rare_hv, rare_eu, rare_hits = [], [], []
    for mode, query, new_snapshot in events:
        start = time.perf_counter()
        retrieved = memory.query(query, profile.top_slots * profile.per_slot)
        local_points, affinity, has_candidates = _local_solution_points(
            retrieved, query, mode, profile.objectives, preferences
        )
        hv, eu, cardinality = _solution_metrics(local_points, utility_preferences)
        # The write occurs after evaluating the current condition, so this
        # condition cannot be solved by a snapshot generated from itself.
        memory.add(new_snapshot)
        latency_ms.append(1000.0 * (time.perf_counter() - start))
        hit = float(any(snapshot.mode == mode for snapshot in retrieved)) if has_candidates else 0.0
        hv_values.append(hv)
        eu_values.append(eu)
        modes.append(mode)
        affinities.append(affinity)
        hit_values.append(hit)
        cardinalities.append(cardinality)
        if mode >= profile.easy_modes:
            rare_hv.append(hv)
            rare_eu.append(eu)
            rare_hits.append(hit)

    return {
        "mean_hv": float(np.mean(hv_values)),
        "mean_eu": float(np.mean(eu_values)),
        "mean_regret": _mean_mode_change_regret(modes, eu_values),
        "mean_affinity": float(np.mean(affinities)),
        "snapshot_hit_rate": float(np.mean(hit_values)),
        "mean_archive_cardinality": float(np.mean(cardinalities)),
        "rare_hv": float(np.mean(rare_hv)),
        "rare_eu": float(np.mean(rare_eu)),
        "rare_snapshot_hit_rate": float(np.mean(rare_hits)),
        "latency_p50_ms": float(np.percentile(latency_ms, 50)),
        "latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "stored_snapshots": float(len(memory.stored())),
        "evictions": float(memory.evictions),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 4000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))


def _aggregate(rows: list[dict[str, object]], seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics = (
        "mean_hv",
        "mean_eu",
        "mean_regret",
        "mean_affinity",
        "snapshot_hit_rate",
        "rare_hv",
        "rare_eu",
        "rare_snapshot_hit_rate",
        "mean_archive_cardinality",
        "latency_p95_ms",
        "evictions",
    )
    rng = np.random.default_rng(seed)
    summary: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    environments = sorted({str(row["environment"]) for row in rows})
    variants = sorted({str(row["variant"]) for row in rows})
    for environment in environments:
        env_rows = [row for row in rows if row["environment"] == environment]
        for variant in variants:
            variant_rows = sorted((row for row in env_rows if row["variant"] == variant), key=lambda row: int(row["replicate"]))
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in variant_rows], dtype=np.float64)
                ci_low, ci_high = _bootstrap_ci(values, rng)
                summary.append(
                    {
                        "environment": environment,
                        "variant": variant,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "replicates": int(len(values)),
                    }
                )
        full_rows = {int(row["replicate"]): row for row in env_rows if row["variant"] == "semantic_expert_bank"}
        for variant in ("flat_fifo_knn", "flat_prioritized_knn"):
            ablated_rows = {int(row["replicate"]): row for row in env_rows if row["variant"] == variant}
            common = sorted(set(full_rows) & set(ablated_rows))
            for metric in metrics:
                differences = np.asarray(
                    [float(full_rows[index][metric]) - float(ablated_rows[index][metric]) for index in common],
                    dtype=np.float64,
                )
                ci_low, ci_high = _bootstrap_ci(differences, rng)
                comparisons.append(
                    {
                        "environment": environment,
                        "comparison": f"semantic_expert_bank - {variant}",
                        "metric": metric,
                        "mean_paired_difference": float(differences.mean()),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "full_wins": int(
                            np.sum(differences < 0) if metric == "mean_regret" else np.sum(differences > 0)
                        ),
                        "ties": int(np.sum(np.isclose(differences, 0.0))),
                        "replicates": int(len(differences)),
                    }
                )
    return summary, comparisons


def _raw_regret_scales(summary: list[dict[str, object]]) -> dict[str, float]:
    scales = {}
    for profile in PROFILES:
        simulated_eu = next(
            float(row["mean"])
            for row in summary
            if row["environment"] == profile.name
            and row["variant"] == "semantic_expert_bank"
            and row["metric"] == "mean_eu"
        )
        scales[profile.name] = abs(float(RAW_FULL_EU[profile.name])) / max(simulated_eu, 1e-12)
    return scales


def _plot(summary: list[dict[str, object]], path: Path) -> None:
    environments = [profile.name for profile in PROFILES]
    variants = ["semantic_expert_bank", "flat_fifo_knn", "flat_prioritized_knn"]
    labels = {"semantic_expert_bank": "Semantic bank", "flat_fifo_knn": "Flat FIFO", "flat_prioritized_knn": "Flat priority"}
    colors = {"semantic_expert_bank": "#1b5e20", "flat_fifo_knn": "#d95f02", "flat_prioritized_knn": "#386cb0"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), constrained_layout=True)
    for axis, metric, title in zip(axes, ("mean_hv", "mean_eu"), ("Hypervolume", "Expected utility")):
        positions = np.arange(len(environments), dtype=float)
        width = 0.24
        for offset, variant in enumerate(variants):
            values, errors = [], []
            for environment in environments:
                row = next(item for item in summary if item["environment"] == environment and item["variant"] == variant and item["metric"] == metric)
                values.append(float(row["mean"]))
                errors.append(float(row["std"]) / math.sqrt(float(row["replicates"])))
            axis.bar(positions + (offset - 1) * width, values, width=width, color=colors[variant], label=labels[variant], yerr=errors, capsize=2.5)
        axis.set_title(title)
        axis.set_xticks(positions, ["Building", "EV charging", "Cogen", "Chlor-alkali"], rotation=0)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Synthetic diagnostic score")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, args: argparse.Namespace, summary: list[dict[str, object]], comparisons: list[dict[str, object]]) -> None:
    lines = [
        "# Expert-Bank Memory Ablation: Synthetic Mechanism Diagnostic",
        "",
        "This report is generated from a controlled synthetic simulation. It does not measure real environment rollouts or establish an end-to-end OW-CMORL performance claim.",
        "",
        f"Replicates per environment: {args.replicates}. Conditions per replicate after warm-up: {args.steps}.",
        "Regret is converted to each environment's raw EU unit using the same semantic-bank calibration as Table 3.",
        "",
        "## Paired Quality Differences",
        "",
        "Positive values mean that the semantic expert bank exceeds the flat-memory comparator for quality metrics; negative Regret differences mean that it incurs less post-change utility loss.",
        "",
        "| Environment | Comparison | Metric | Paired difference | 95% bootstrap CI | Full wins |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        if row["metric"] not in {"mean_hv", "mean_eu", "mean_regret", "rare_hv", "rare_eu", "snapshot_hit_rate", "rare_snapshot_hit_rate"}:
            continue
        lines.append(
            f"| {row['environment']} | {row['comparison']} | {row['metric']} | {float(row['mean_paired_difference']):.5f} | [{float(row['ci95_low']):.5f}, {float(row['ci95_high']):.5f}] | {row['full_wins']}/{row['replicates']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "The semantic bank is supported by this diagnostic only when its paired HV/EU difference is positive with a positive bootstrap interval and the same direction is visible for rare recurring conditions. Any result that does not satisfy this rule must be reported as no supported advantage in this synthetic setting.",
            "",
            "Raw per-replicate data are in `per_replicate_metrics.csv`; aggregate values are in `summary_metrics.csv` and `paired_comparisons.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.replicates < 2 or args.steps < 16 or args.warmup_rounds < 1:
        raise ValueError("Use at least two replicates, 16 conditions, and one warm-up round.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    variants = ("semantic_expert_bank", "flat_fifo_knn", "flat_prioritized_knn")
    for profile_index, profile in enumerate(PROFILES):
        for replicate in range(args.replicates):
            rng = np.random.default_rng(args.seed + 1009 * profile_index + replicate)
            warmup, events = _make_event_stream(
                rng, profile, args.embedding_dim, args.steps, args.warmup_rounds
            )
            preferences = rng.dirichlet(np.ones(profile.objectives), size=args.preferences)
            utility_preferences = rng.dirichlet(np.ones(profile.objectives), size=args.preferences)
            for variant in variants:
                metrics = _run_variant(profile, variant, warmup, events, preferences, utility_preferences)
                rows.append({"environment": profile.name, "replicate": replicate, "variant": variant, **metrics})
    initial_summary, _ = _aggregate(rows, args.seed + 77)
    regret_scales = _raw_regret_scales(initial_summary)
    for row in rows:
        row["mean_regret"] = float(row["mean_regret"]) * regret_scales[str(row["environment"])]
    summary, comparisons = _aggregate(rows, args.seed + 77)
    _write_csv(args.out_dir / "per_replicate_metrics.csv", rows)
    _write_csv(args.out_dir / "summary_metrics.csv", summary)
    _write_csv(args.out_dir / "paired_comparisons.csv", comparisons)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_mechanism_diagnostic",
                "warning": "Not an environment rollout or end-to-end OW-CMORL result.",
                "replicates": args.replicates,
                "steps": args.steps,
                "warmup_rounds": args.warmup_rounds,
                "embedding_dim": args.embedding_dim,
                "profiles": [profile.__dict__ for profile in PROFILES],
                "variants": list(variants),
                "raw_regret_scales": regret_scales,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _plot(summary, args.out_dir / "quality_comparison.pdf")
    _write_report(args.out_dir / "REPORT.md", args, summary, comparisons)
    print(f"Wrote synthetic diagnostic to {args.out_dir}")


if __name__ == "__main__":
    main()
