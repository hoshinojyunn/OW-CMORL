import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from .sample import Sample
from .temporal import knee_scores, cosine_affinity
from .utils import get_ep_indices

'''
Define a external pareto class storing all computed policies on the current pareto front.
'''


def calculate_crowding_distance(obj_batch, extreme):
    obj_batch = np.asarray(obj_batch, dtype=np.float64)
    if len(obj_batch) == 0:
        return np.array([])

    num_samples = obj_batch.shape[0]
    num_objectives = obj_batch.shape[1]
    crowding_distances = np.zeros(num_samples)

    for i in range(num_objectives):
        obj_values = obj_batch[:, i]
        sorted_indices = np.argsort(obj_values)
        distances = np.zeros(num_samples)
        denom = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
        if abs(denom) <= 1e-12:
            denom = 1.0
        distances[1:-1] = (
            obj_values[sorted_indices[2:]] - obj_values[sorted_indices[:-2]]
        ) / denom

        if extreme:
            distances[0] = distances[-1] = np.inf
        crowding_distances[sorted_indices] += distances

    return crowding_distances


def select_samples_by_dynamic_score(
    obj_batch,
    sample_batch,
    num_select_solutions,
    regime_embedding=None,
    drift_score=0.0,
    knee_lambda=0.75,
    dynamic_lambda=0.75,
    resilience_lambda=0.75,
    diversity_lambda=0.0,
    obj_hist=None,
    resilience_recovery_weight=0.30,
    resilience_regret_weight=0.20,
    resilience_latency_weight=0.15,
    resilience_utility_weight=0.10,
    resilience_gap_weight=0.10,
    resilience_bank_weight=0.15,
    dynamic_affinity_weight=0.65,
    dynamic_freshness_weight=0.20,
    dynamic_bank_weight=0.15,
):
    obj_batch = np.asarray(obj_batch, dtype=np.float64)
    sample_batch = list(sample_batch)
    if len(obj_batch) == 0 or len(sample_batch) == 0:
        return [], np.array([])

    crowding_distances = calculate_crowding_distance(obj_batch, True)
    knee = knee_scores(obj_batch) if len(obj_batch) > 0 else np.array([])
    affinity = np.ones(len(sample_batch), dtype=np.float64)
    freshness = np.ones(len(sample_batch), dtype=np.float64)
    if regime_embedding is not None:
        for i, sample in enumerate(sample_batch):
            sample_emb = sample.metadata.get("context_embedding")
            if sample_emb is not None:
                affinity[i] = cosine_affinity(sample_emb, regime_embedding)
            elif "bank_query_affinity" in sample.metadata:
                affinity[i] = float(sample.metadata.get("bank_query_affinity", 0.0))
    train_iterations = np.asarray(
        [float(sample.metadata.get("train_iteration", 0.0)) for sample in sample_batch],
        dtype=np.float64,
    )
    if len(train_iterations) > 0 and np.ptp(train_iterations) > 1e-8:
        freshness = (train_iterations - train_iterations.min()) / (np.ptp(train_iterations) + 1e-8)

    utility_mean = np.zeros(len(sample_batch), dtype=np.float64)
    recovery_score = np.zeros(len(sample_batch), dtype=np.float64)
    shift_regret = np.zeros(len(sample_batch), dtype=np.float64)
    recovery_latency = np.zeros(len(sample_batch), dtype=np.float64)
    pre_post_gap = np.zeros(len(sample_batch), dtype=np.float64)
    bank_score = np.zeros(len(sample_batch), dtype=np.float64)
    for i, sample in enumerate(sample_batch):
        trace_metrics = sample.metadata.get("trace_metrics", {})
        utility_mean[i] = float(trace_metrics.get("trace_utility_mean", 0.0))
        recovery_score[i] = float(trace_metrics.get("trace_recovery_score", 0.0))
        shift_regret[i] = float(trace_metrics.get("trace_shift_regret", 0.0))
        recovery_latency[i] = float(trace_metrics.get("trace_recovery_latency", 0.0))
        pre_post_gap[i] = float(trace_metrics.get("trace_pre_post_gap", 0.0))
        bank_score[i] = float(sample.metadata.get("bank_score", 0.0))

    def _normalize(values, invert=False):
        values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if len(values) == 0 or np.ptp(values) <= 1e-8:
            return np.zeros_like(values)
        scaled = (values - values.min()) / (np.ptp(values) + 1e-8)
        return 1.0 - scaled if invert else scaled

    finite_crowding = crowding_distances[np.isfinite(crowding_distances)]
    crowd_norm = np.max(finite_crowding) if finite_crowding.size else 1.0
    crowding_distances = np.where(
        np.isfinite(crowding_distances),
        crowding_distances / (crowd_norm + 1e-8),
        1.15,
    )
    knee = knee / (np.max(knee) + 1e-8) if len(knee) else knee
    affinity = (affinity - np.min(affinity)) / (np.max(affinity) - np.min(affinity) + 1e-8)
    freshness = (freshness - np.min(freshness)) / (np.max(freshness) - np.min(freshness) + 1e-8)
    bank_score = _normalize(bank_score)
    drift_gate = 1.0 / (1.0 + np.exp(-3.0 * float(drift_score)))
    resilience = (
        float(resilience_recovery_weight) * _normalize(recovery_score)
        + float(resilience_regret_weight) * _normalize(shift_regret, invert=True)
        + float(resilience_latency_weight) * _normalize(recovery_latency, invert=True)
        + float(resilience_utility_weight) * _normalize(utility_mean)
        + float(resilience_gap_weight) * _normalize(pre_post_gap)
        + float(resilience_bank_weight) * bank_score
    )
    dynamic_signal = (
        float(dynamic_affinity_weight) * affinity
        + float(dynamic_freshness_weight) * freshness
        + float(dynamic_bank_weight) * bank_score
    )
    scores = (
        crowding_distances
        + knee_lambda * knee
        + dynamic_lambda * drift_gate * dynamic_signal
        + resilience_lambda * drift_gate * resilience
    )
    sorted_indices = np.argsort(-scores)
    obj_norm = np.asarray(obj_batch, dtype=np.float64)
    if len(obj_norm) > 0:
        obj_min = obj_norm.min(axis=0, keepdims=True)
        obj_span = np.ptp(obj_norm, axis=0, keepdims=True)
        obj_norm = (obj_norm - obj_min) / (obj_span + 1e-8)
    history = [] if obj_hist is None else list(obj_hist)
    new_indices = []
    remaining = list(sorted_indices.tolist())
    while remaining and len(new_indices) < num_select_solutions:
        if diversity_lambda <= 0 or len(new_indices) == 0:
            idx = remaining.pop(0)
        else:
            selected_obj = obj_norm[np.asarray(new_indices, dtype=np.int64)]
            best_pos = 0
            best_value = None
            for pos, idx in enumerate(remaining):
                candidate = obj_norm[idx]
                min_dist = np.linalg.norm(selected_obj - candidate, axis=1).min()
                combined = float(scores[idx]) + float(diversity_lambda) * float(min_dist)
                if best_value is None or combined > best_value:
                    best_value = combined
                    best_pos = pos
            idx = remaining.pop(best_pos)
        obj_tuple = tuple(obj_batch[idx])
        if obj_tuple in history:
            continue
        new_indices.append(idx)
        history.append(obj_tuple)
    if len(new_indices) < min(num_select_solutions, len(sorted_indices)):
        for idx in sorted_indices:
            if idx not in new_indices:
                new_indices.append(idx)
            if len(new_indices) >= num_select_solutions:
                break
    return new_indices, scores


class EP:
    def __init__(self, reference_point, num_select_solutions, policy_buffer_size):
        # Store pareto optimal solutions
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])
        
        # Store historical selected objective values to avoid repeated selection
        self.obj_hist = []
        
        # Store selected solutions for next extension iteration
        self.selected_batch = np.array([])
        self.selected_obj_batch = np.array([])
        
        self.policy_buffer_size = policy_buffer_size  # Maximum allowed size for sample_batch
        self.reference_point = reference_point
        self.num_select_solutions = num_select_solutions
        self.last_selection_scores = None

    def crowding_distance_index(self, indices, inplace=True):
        self.selected_obj_batch = self.obj_batch[indices]
        self.selected_batch = self.sample_batch[indices]
        
    def index(self, indices, inplace=True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch, select_after_update=True):
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array([sample.objs])
        if len(self.obj_batch) == 0: return
    
        ep_indices = get_ep_indices(self.obj_batch, self.reference_point)
        if len(ep_indices) == 0:
            print("Reference-point filter removed all candidates; falling back to all samples.")
            ep_indices = np.arange(len(self.obj_batch))
        self.index(ep_indices)
        if len(self.sample_batch) > self.policy_buffer_size:
            print(f"Sample batch size exceeded policy buffer size {self.policy_buffer_size}. Performing sampling.")
            crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
            sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance
            pareto_index = []
            for idx in sorted_indices:
                if len(pareto_index) < self.policy_buffer_size:
                    pareto_index.append(idx)
            self.sample_batch = self.sample_batch[pareto_index]
            self.obj_batch = self.obj_batch[pareto_index]
        print(f"Number of Pareto optimal solutions: {len(self.obj_batch)}")

        if select_after_update:
            self.filter_by_crowding_distance(self.num_select_solutions) #number of elite policies
    

    def calculate_crowding_distance(self, obj_batch, extreme):
        return calculate_crowding_distance(obj_batch, extreme)

    def filter_by_crowding_distance(self, num_select_solutions):
        crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
        sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance

        # Select the top num_select_solutions policies, ensuring we don't select duplicates based on objective values
        new_indices = []
        for idx in sorted_indices:
            # Convert objectives to a tuple to use as a hashable type for comparison
            obj_tuple = tuple(self.obj_batch[idx])

            if obj_tuple not in self.obj_hist and len(new_indices) < num_select_solutions:
                new_indices.append(idx)
                self.obj_hist.append(obj_tuple)

        if len(new_indices) < min(num_select_solutions, len(sorted_indices)):
            for idx in sorted_indices:
                if idx not in new_indices:
                    new_indices.append(idx)
                if len(new_indices) >= num_select_solutions:
                    break

        # Filter samples by these new indices using the existing self.index method
        self.crowding_distance_index(new_indices)

    def filter_by_dynamic_score(
        self,
        num_select_solutions,
        regime_embedding=None,
        drift_score=0.0,
        knee_lambda=0.75,
        dynamic_lambda=0.75,
        resilience_lambda=0.75,
        diversity_lambda=0.0,
        resilience_recovery_weight=0.30,
        resilience_regret_weight=0.20,
        resilience_latency_weight=0.15,
        resilience_utility_weight=0.10,
        resilience_gap_weight=0.10,
        resilience_bank_weight=0.15,
        dynamic_affinity_weight=0.65,
        dynamic_freshness_weight=0.20,
        dynamic_bank_weight=0.15,
    ):
        if len(self.obj_batch) == 0 or len(self.sample_batch) == 0:
            self.filter_by_crowding_distance(num_select_solutions)
            return
        new_indices, scores = select_samples_by_dynamic_score(
            self.obj_batch,
            self.sample_batch,
            num_select_solutions,
            regime_embedding=regime_embedding,
            drift_score=drift_score,
            knee_lambda=knee_lambda,
            dynamic_lambda=dynamic_lambda,
            resilience_lambda=resilience_lambda,
            diversity_lambda=diversity_lambda,
            obj_hist=self.obj_hist,
            resilience_recovery_weight=resilience_recovery_weight,
            resilience_regret_weight=resilience_regret_weight,
            resilience_latency_weight=resilience_latency_weight,
            resilience_utility_weight=resilience_utility_weight,
            resilience_gap_weight=resilience_gap_weight,
            resilience_bank_weight=resilience_bank_weight,
            dynamic_affinity_weight=dynamic_affinity_weight,
            dynamic_freshness_weight=dynamic_freshness_weight,
            dynamic_bank_weight=dynamic_bank_weight,
        )
        self.last_selection_scores = scores
        for idx in new_indices:
            obj_tuple = tuple(self.obj_batch[idx])
            if obj_tuple not in self.obj_hist:
                self.obj_hist.append(obj_tuple)
        self.crowding_distance_index(new_indices)
        
    def random_selection(self, num_select_solutions):
        """Randomly select num_select_solutions policies without considering crowding distance."""
        if len(self.obj_batch) < num_select_solutions:
            num_select_solutions = len(self.obj_batch)
        
        random_indices = np.random.choice(len(self.obj_batch), size=num_select_solutions, replace=False)
        self.crowding_distance_index(random_indices)
