import os, sys

# import python packages
import time
import json
from copy import deepcopy

# import third-party packages
import numpy as np
import torch
import torch.optim as optim
import multiprocessing as mp
import pickle
from typing import Any
from queue import Queue as ThreadQueue
from threading import Event as ThreadEvent

# import our packages
from .scalarization_methods import WeightedSumScalarization
from .sample import Sample
from .task import Task
from .ep import EP
from .ep import select_samples_by_dynamic_score
from .expert_bank import ContextExpertBank
from .metrics import (
    append_metrics_row,
    compute_trace_shift_metrics,
    evaluate_regime_metrics,
    shared_regime_seed_plan as build_shared_regime_seed_plan,
    summarize_regime_metrics,
    summarize_trace_metrics,
)
from .utils import generate_weights_batch_dfs, print_info, generate_w_batch_test, compute_eu, compute_sparsity
from .utils import get_ep_indices
from .warm_up import initialize_warm_up_batch
from .mopg import MOPG_worker, Extension_worker, evaluation, _deserialize_sample
from .temporal import RegimeAttention, build_trace_matrix, knee_scores, cosine_affinity
from .online_context import (
    ContextSnapshot,
    OnlineContextMemory,
    OnlineTraceWindow,
    context_volatility,
    detect_context_shift_points,
)
from .runtime_checkpoint import save_owcmorl_runtime_checkpoint
from pymoo.factory import get_performance_indicator
#from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def eval(eval_sample, ref, obj_num, eval_delta_weight):
    eval_sample = np.asarray(eval_sample, dtype=np.float64)
    if eval_sample.size == 0:
        return 0.0, 0.0, 0.0
    if eval_sample.ndim == 1:
        eval_sample = eval_sample.reshape(1, -1)
    perf_ind = get_performance_indicator("hv", ref_point = -ref)
    preferences = generate_w_batch_test(obj_num, eval_delta_weight)
    hv = perf_ind.do(-eval_sample)
    eu = compute_eu(eval_sample, preferences)
    sp = compute_sparsity(eval_sample)
    return hv, eu, sp


def _runtime_args_signature(args):
    """Record the runtime fields required to validate a resumed OOD update."""

    keys = (
        "env_name",
        "obj_num",
        "num_steps",
        "num_processes",
        "selection_method",
        "bank_retrieval_backend",
        "bank_query_slots",
        "bank_query_topk",
        "bank_enable",
        "context_encoder_enable",
        "strict_online_context",
        "ref_point",
        "eval_delta_weight",
        "raw",
    )
    return {key: deepcopy(getattr(args, key)) for key in keys if hasattr(args, key)}


def _cleanup_worker_handles(processes, results_queue):
    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        try:
            process.close()
        except Exception:
            pass
    try:
        results_queue.close()
        results_queue.join_thread()
    except Exception:
        pass


def _mp_context(args):
    env_name = str(getattr(args, "env_name", "")).lower()
    start_method = "spawn" if env_name.startswith("chlor_alkali") or ("chlor" in env_name and "alkali" in env_name) else "fork"
    try:
        return mp.get_context(start_method)
    except ValueError:
        return mp.get_context()


def _use_serial_workers(args) -> bool:
    if str(os.environ.get("OWCMORL_FORCE_PARALLEL", "")).strip().lower() in {"1", "true", "yes"}:
        return False
    # The outer RL loop already launches one worker per selected policy/sample.
    # On the current WSL host this nested multiprocessing has been unstable:
    # it caused SIGKILLs from memory pressure and leaked `torch_shm_manager`
    # / semaphore helpers after crashes. Default to serial outer scheduling for
    # stable long-running experiments; inner env parallelism remains configurable
    # through `args.num_processes`.
    return True


def _fit_regime_model(args, sample_batch):
    traces = []
    for sample in sample_batch:
        trace = sample.metadata.get("trace")
        if trace is None:
            continue
        trace_matrix = build_trace_matrix(trace, args.drift_window)
        if trace_matrix.ndim == 2 and len(trace_matrix) > 0 and trace_matrix.shape[1] > 0:
            traces.append(trace)
    if len(traces) == 0:
        return None, 0.0
    input_dim = int(build_trace_matrix(traces[0], args.drift_window).shape[1])
    model = RegimeAttention(input_dim=input_dim, window=args.drift_window)
    loss = model.fit(traces, steps=args.drift_steps, lr=1e-3)
    for sample in sample_batch:
        trace = sample.metadata.get("trace")
        if trace is not None:
            trace_matrix = build_trace_matrix(trace, args.drift_window)
            if trace_matrix.ndim == 2 and len(trace_matrix) > 0 and trace_matrix.shape[1] > 0:
                sample.metadata["context_embedding"] = model.embed(trace_matrix)
    return model, loss


def _aggregate_context_embedding(sample_batch):
    embeddings = [
        sample.metadata.get("context_embedding")
        for sample in sample_batch
        if sample.metadata.get("context_embedding") is not None
    ]
    if not embeddings:
        return None
    return np.mean(np.stack(embeddings, axis=0), axis=0)


def _select_online_trace_sources(sample_batch, max_samples):
    candidates = []
    for sample in sample_batch:
        trace = sample.metadata.get("trace")
        if trace is None:
            continue
        context = np.asarray(trace.get("context", []), dtype=np.float32)
        if context.ndim != 2 or len(context) == 0:
            continue
        candidates.append(sample)
    if not candidates:
        return []
    candidates.sort(
        key=lambda sample: float(sample.metadata.get("train_iteration", 0.0)),
        reverse=True,
    )
    limit = max(1, min(int(max_samples), len(candidates)))
    return [sample.metadata["trace"] for sample in candidates[:limit]]


def _estimate_context_support(sample_pool, target_embedding, obj_num, topk):
    if target_embedding is None:
        return {
            "gap_vector": np.zeros(obj_num, dtype=np.float64),
            "utility_mean": 0.0,
            "support_strength": 0.0,
            "matched_recovery": 0.0,
        }

    scored = []
    query = np.asarray(target_embedding, dtype=np.float64).reshape(-1)
    for sample in sample_pool:
        if sample is None:
            continue
        sample_embedding = sample.metadata.get("context_embedding")
        if sample_embedding is None:
            continue
        gap_vector = np.asarray(
            sample.metadata.get("shift_gap_vector", np.zeros(obj_num, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if gap_vector.shape != (obj_num,):
            gap_vector = np.zeros(obj_num, dtype=np.float64)
        trace_metrics = sample.metadata.get("trace_metrics", {})
        affinity = max(0.0, cosine_affinity(sample_embedding, query))
        recovery = max(0.0, float(trace_metrics.get("trace_recovery_score", 0.0)))
        regret = max(0.0, float(trace_metrics.get("trace_shift_regret", 0.0)))
        weight = affinity * (0.5 + 0.5 * recovery) / (1.0 + regret)
        if weight <= 0.0:
            continue
        scored.append(
            (
                float(weight),
                gap_vector,
                float(trace_metrics.get("trace_utility_mean", 0.0)),
                recovery,
            )
        )

    if not scored:
        return {
            "gap_vector": np.zeros(obj_num, dtype=np.float64),
            "utility_mean": 0.0,
            "support_strength": 0.0,
            "matched_recovery": 0.0,
        }

    scored.sort(key=lambda item: item[0], reverse=True)
    top_rows = scored[: max(1, min(int(topk), len(scored)))]
    weights = np.asarray([row[0] for row in top_rows], dtype=np.float64)
    weights /= max(weights.sum(), 1e-8)
    gap_vector = np.average(
        np.stack([row[1] for row in top_rows], axis=0),
        axis=0,
        weights=weights,
    )
    utility_mean = float(np.dot(weights, np.asarray([row[2] for row in top_rows], dtype=np.float64)))
    matched_recovery = float(np.dot(weights, np.asarray([row[3] for row in top_rows], dtype=np.float64)))
    return {
        "gap_vector": np.asarray(gap_vector, dtype=np.float64),
        "utility_mean": utility_mean,
        "support_strength": float(top_rows[0][0]),
        "matched_recovery": matched_recovery,
    }


def _update_online_trace_window(args, online_trace_window, sample_batch):
    traces = _select_online_trace_sources(sample_batch, args.context_probe_samples)
    if not traces:
        return 0
    return online_trace_window.append_aggregated_traces(
        traces,
        tail_steps=args.context_trace_steps,
        sample_mode=args.context_sample_mode,
        saliency_alpha=args.context_saliency_alpha,
    )


def _context_sample_pool(ep, expert_bank, extra_samples=None):
    pool = []
    if ep is not None:
        pool.extend(list(ep.sample_batch))
    if expert_bank is not None:
        pool.extend(expert_bank.export_samples())
    if extra_samples is not None:
        pool.extend([sample for sample in extra_samples if sample is not None])

    dedup = {}
    for sample in pool:
        if sample is None:
            continue
        sample_uid = sample.metadata.get("sample_uid", f"id_{id(sample)}")
        dedup[str(sample_uid)] = sample
    return list(dedup.values())


def _infer_online_context_state(
    args,
    regime_model,
    online_trace_window,
    context_memory,
    prev_target_embedding,
    iteration,
    sample_pool,
):
    if regime_model is None:
        return None
    trace = online_trace_window.as_trace()
    trace_matrix = build_trace_matrix(trace, args.drift_window)
    if len(trace_matrix) == 0 or trace_matrix.shape[1] == 0:
        return None

    current_embedding, forecast_embedding, _pred = regime_model.encode_with_forecast(
        trace_matrix
    )
    target_embedding = context_memory.build_target_embedding(current_embedding, forecast_embedding)
    support = _estimate_context_support(
        sample_pool,
        target_embedding,
        args.obj_num,
        topk=args.context_nearest_k,
    )
    gap_vector = np.asarray(support["gap_vector"], dtype=np.float64)
    history_gap = context_memory.nearest_gap_vector(target_embedding)
    if history_gap is not None:
        gap_vector = (
            (1.0 - args.context_history_lambda) * gap_vector
            + args.context_history_lambda * history_gap
        )

    reference_embedding = (
        prev_target_embedding
        if prev_target_embedding is not None
        else context_memory.reference_embedding()
    )
    if reference_embedding is None:
        drift_score = 0.0
    else:
        drift_score = max(0.0, 1.0 - cosine_affinity(reference_embedding, current_embedding))
    prediction_error = max(0.0, 1.0 - cosine_affinity(current_embedding, forecast_embedding))
    context_shift_points = detect_context_shift_points(
        trace_matrix,
        window=args.trace_recovery_window,
        shift_threshold=args.online_shift_threshold,
        min_shift_gap=args.online_shift_min_gap,
    )
    context_jitter = context_volatility(trace_matrix)
    snapshot = ContextSnapshot(
        iteration=int(iteration),
        current_embedding=np.asarray(current_embedding, dtype=np.float64),
        forecast_embedding=np.asarray(forecast_embedding, dtype=np.float64),
        target_embedding=np.asarray(target_embedding, dtype=np.float64),
        gap_vector=np.asarray(gap_vector, dtype=np.float64),
        utility_mean=float(support["utility_mean"]),
        drift_score=float(drift_score),
        prediction_error=float(prediction_error),
    )
    context_memory.update(snapshot)
    return {
        "current_embedding": snapshot.current_embedding,
        "forecast_embedding": snapshot.forecast_embedding,
        "target_embedding": snapshot.target_embedding,
        "gap_vector": snapshot.gap_vector,
        "drift_score": snapshot.drift_score,
        "prediction_error": snapshot.prediction_error,
        "probe_utility_mean": snapshot.utility_mean,
        "online_trace_steps": int(len(trace["context"])),
        "online_shift_count": float(len(context_shift_points)),
        "context_volatility": float(context_jitter),
        "context_support_strength": float(support["support_strength"]),
        "matched_recovery": float(support["matched_recovery"]),
    }


def _infer_current_token_context_state(
    args,
    regime_model,
    online_trace_window,
    context_memory,
    prev_target_embedding,
    iteration,
    sample_pool,
):
    """Ablation path that bypasses the temporal Transformer.

    The current environment token is mapped through the learned input and
    context heads solely to retain the expert bank's hidden dimensionality.
    Positional encoding, self-attention, forecast construction, window
    pooling, and history smoothing are deliberately not used.
    """
    if regime_model is None:
        return None
    trace = online_trace_window.as_trace()
    trace_matrix = build_trace_matrix(trace, args.drift_window)
    if len(trace_matrix) == 0 or trace_matrix.shape[1] == 0:
        return None

    device = next(regime_model.parameters()).device
    dtype = next(regime_model.parameters()).dtype
    token = torch.as_tensor(trace_matrix[-1], dtype=dtype, device=device).unsqueeze(0)
    with torch.no_grad():
        current_embedding = regime_model.context_head(regime_model.input_proj(token))
    current_embedding = current_embedding.squeeze(0).cpu().numpy()
    # There is no temporal forecast in this ablation, hence target=current.
    forecast_embedding = np.asarray(current_embedding, dtype=np.float64)
    target_embedding = np.asarray(current_embedding, dtype=np.float64)
    support = _estimate_context_support(
        sample_pool,
        target_embedding,
        args.obj_num,
        topk=args.context_nearest_k,
    )
    gap_vector = np.asarray(support["gap_vector"], dtype=np.float64)
    reference_embedding = (
        prev_target_embedding
        if prev_target_embedding is not None
        else context_memory.reference_embedding()
    )
    drift_score = (
        0.0
        if reference_embedding is None
        else max(0.0, 1.0 - cosine_affinity(reference_embedding, current_embedding))
    )
    context_shift_points = detect_context_shift_points(
        trace_matrix,
        window=args.trace_recovery_window,
        shift_threshold=args.online_shift_threshold,
        min_shift_gap=args.online_shift_min_gap,
    )
    snapshot = ContextSnapshot(
        iteration=int(iteration),
        current_embedding=np.asarray(current_embedding, dtype=np.float64),
        forecast_embedding=forecast_embedding,
        target_embedding=target_embedding,
        gap_vector=gap_vector,
        utility_mean=float(support["utility_mean"]),
        drift_score=float(drift_score),
        prediction_error=0.0,
    )
    context_memory.update(snapshot)
    return {
        "current_embedding": snapshot.current_embedding,
        "forecast_embedding": snapshot.forecast_embedding,
        "target_embedding": snapshot.target_embedding,
        "gap_vector": snapshot.gap_vector,
        "drift_score": snapshot.drift_score,
        "prediction_error": snapshot.prediction_error,
        "probe_utility_mean": snapshot.utility_mean,
        "online_trace_steps": int(len(trace["context"])),
        "online_shift_count": float(len(context_shift_points)),
        "context_volatility": float(context_volatility(trace_matrix)),
        "context_support_strength": float(support["support_strength"]),
        "matched_recovery": float(support["matched_recovery"]),
    }


def _select_policies(ep, args, current_regime_embedding, current_drift_score):
    if args.selection_method in {'dynamic-frontier', 'online-window'}:
        ep.filter_by_dynamic_score(
            args.num_select,
            current_regime_embedding,
            current_drift_score,
            knee_lambda=args.knee_lambda,
            dynamic_lambda=args.dynamic_lambda,
            resilience_lambda=args.resilience_lambda,
            diversity_lambda=args.diversity_lambda,
            resilience_recovery_weight=args.resilience_recovery_weight,
            resilience_regret_weight=args.resilience_regret_weight,
            resilience_latency_weight=args.resilience_latency_weight,
            resilience_utility_weight=args.resilience_utility_weight,
            resilience_gap_weight=args.resilience_gap_weight,
            resilience_bank_weight=args.resilience_bank_weight,
            dynamic_affinity_weight=args.dynamic_affinity_weight,
            dynamic_freshness_weight=args.dynamic_freshness_weight,
            dynamic_bank_weight=args.dynamic_bank_weight,
        )
    elif args.selection_method == 'random':
        ep.random_selection(args.num_select)
    else:
        ep.filter_by_crowding_distance(args.num_select)


def _select_online_candidates(ep, expert_bank, args, current_regime_embedding, current_drift_score):
    candidate_samples = list(ep.sample_batch)
    candidate_objs = [np.asarray(sample.objs, dtype=np.float64) for sample in candidate_samples]
    if expert_bank is not None and args.selection_method == "online-window" and int(args.bank_query_slots) > 0:
        if getattr(args, "bank_query_mode", "context") == "global-best":
            bank_samples = expert_bank.query_global(top_k=args.bank_query_topk)
        elif current_regime_embedding is not None:
            bank_samples = expert_bank.query(
                current_regime_embedding,
                top_slots=args.bank_query_slots,
                top_per_slot=args.bank_query_topk,
            )
        else:
            bank_samples = []
        for sample in bank_samples:
            candidate_samples.append(sample)
            candidate_objs.append(np.asarray(sample.objs, dtype=np.float64))

    if not candidate_samples:
        ep.selected_batch = np.array([])
        ep.selected_obj_batch = np.array([])
        return np.array([])

    candidate_obj_batch = np.stack(candidate_objs, axis=0)
    if args.selection_method in {'dynamic-frontier', 'online-window'}:
        indices, _scores = select_samples_by_dynamic_score(
            candidate_obj_batch,
            candidate_samples,
            args.num_select,
            regime_embedding=current_regime_embedding,
            drift_score=current_drift_score,
            knee_lambda=args.knee_lambda,
            dynamic_lambda=args.dynamic_lambda,
            resilience_lambda=args.resilience_lambda,
            diversity_lambda=args.diversity_lambda,
            obj_hist=ep.obj_hist,
            resilience_recovery_weight=args.resilience_recovery_weight,
            resilience_regret_weight=args.resilience_regret_weight,
            resilience_latency_weight=args.resilience_latency_weight,
            resilience_utility_weight=args.resilience_utility_weight,
            resilience_gap_weight=args.resilience_gap_weight,
            resilience_bank_weight=args.resilience_bank_weight,
            dynamic_affinity_weight=args.dynamic_affinity_weight,
            dynamic_freshness_weight=args.dynamic_freshness_weight,
            dynamic_bank_weight=args.dynamic_bank_weight,
        )
    elif args.selection_method == 'random':
        count = min(args.num_select, len(candidate_samples))
        indices = np.random.choice(len(candidate_samples), size=count, replace=False).tolist()
    else:
        crowding = ep.calculate_crowding_distance(candidate_obj_batch, True)
        indices = np.argsort(-crowding)[: min(args.num_select, len(candidate_samples))].tolist()

    selected_samples = np.array([candidate_samples[idx] for idx in indices], dtype=object)
    selected_objs = candidate_obj_batch[np.asarray(indices, dtype=np.int64)]
    ep.selected_batch = selected_samples
    ep.selected_obj_batch = selected_objs
    for idx in indices:
        obj_tuple = tuple(candidate_obj_batch[idx])
        if obj_tuple not in ep.obj_hist:
            ep.obj_hist.append(obj_tuple)
    return selected_samples


def _dedup_samples_by_obj(sample_list):
    dedup = {}
    for sample in sample_list:
        obj_vec = np.asarray(sample.objs, dtype=np.float64).reshape(-1)
        key = tuple(np.round(obj_vec, decimals=8).tolist())
        dedup[key] = sample
    return list(dedup.values())


def _minmax(values, *, invert=False):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(values) == 0 or np.ptp(values) <= 1e-12:
        scaled = np.ones_like(values)
    else:
        scaled = (values - values.min()) / (np.ptp(values) + 1e-12)
    return 1.0 - scaled if invert else scaled


def _resilient_archive_scores(samples, args):
    """Normalize ID trace signals before ranking final archive candidates."""
    trace_metrics = [sample.metadata.get("trace_metrics", {}) for sample in samples]
    recovery = [float(metrics.get("trace_recovery_score", 0.0)) for metrics in trace_metrics]
    regret = [abs(float(metrics.get("trace_shift_regret", 0.0))) for metrics in trace_metrics]
    latency = [max(0.0, float(metrics.get("trace_recovery_latency", 0.0))) for metrics in trace_metrics]
    bank_score = [float(sample.metadata.get("bank_score", 0.0)) for sample in samples]
    affinity = [float(sample.metadata.get("bank_query_affinity", 0.0)) for sample in samples]
    train_iter = [float(sample.metadata.get("train_iteration", 0.0)) for sample in samples]
    return (
        float(args.final_score_recovery_weight) * _minmax(recovery)
        + float(args.final_score_regret_weight) * _minmax(regret, invert=True)
        + float(args.final_score_latency_weight) * _minmax(latency, invert=True)
        + float(args.final_score_bank_weight) * _minmax(bank_score)
        + float(args.final_score_affinity_weight) * _minmax(affinity)
        + float(args.final_score_train_iter_weight) * _minmax(train_iter)
    )


def _context_distance_matrix(samples):
    embeddings = [sample.metadata.get("context_embedding") for sample in samples]
    if not embeddings or any(embedding is None for embedding in embeddings):
        return np.zeros((len(samples), len(samples)), dtype=np.float64)
    arrays = [np.asarray(embedding, dtype=np.float64).reshape(-1) for embedding in embeddings]
    if not arrays or any(array.shape != arrays[0].shape or array.size == 0 for array in arrays):
        return np.zeros((len(samples), len(samples)), dtype=np.float64)
    matrix = np.stack(arrays, axis=0)
    scale = np.maximum(np.ptp(matrix, axis=0), 1e-8)
    normalized = (matrix - matrix.min(axis=0, keepdims=True)) / scale
    distance = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    maximum = float(distance.max())
    return distance / maximum if maximum > 1e-12 else distance


def _select_resilient_diverse_final_archive(samples, args):
    """Keep robust candidates while preserving objective and context coverage."""
    if not samples:
        return []
    cap = min(len(samples), max(1, int(args.final_archive_max_samples)))
    obj_batch = np.stack([np.asarray(sample.objs, dtype=np.float64) for sample in samples], axis=0)
    obj_scale = np.maximum(np.ptp(obj_batch, axis=0, keepdims=True), 1e-8)
    obj_normalized = (obj_batch - obj_batch.min(axis=0, keepdims=True)) / obj_scale
    robustness = _resilient_archive_scores(samples, args)
    context_distance = _context_distance_matrix(samples)

    selected: list[int] = []
    # Preserve every objective extreme before trading coverage against robustness.
    for objective_idx in range(obj_normalized.shape[1]):
        selected.append(int(np.argmax(obj_normalized[:, objective_idx])))
    selected.append(int(np.argmax(robustness)))
    selected = list(dict.fromkeys(selected))[:cap]

    while len(selected) < cap:
        remaining = [idx for idx in range(len(samples)) if idx not in selected]
        if not remaining:
            break
        objective_distance = np.linalg.norm(
            obj_normalized[np.asarray(remaining)][:, None, :]
            - obj_normalized[np.asarray(selected)][None, :, :],
            axis=2,
        ).min(axis=1)
        context_coverage = context_distance[np.asarray(remaining)][:, np.asarray(selected)].min(axis=1)
        value = (
            robustness[np.asarray(remaining)]
            + float(args.final_objective_diversity_lambda) * objective_distance
            + float(args.final_context_diversity_lambda) * context_coverage
        )
        selected.append(int(remaining[int(np.argmax(value))]))

    for rank, idx in enumerate(selected):
        samples[idx].metadata["final_archive_mode"] = "resilient-diverse"
        samples[idx].metadata["final_archive_rank"] = int(rank)
        samples[idx].metadata["final_archive_robustness"] = float(robustness[idx])
    return [samples[idx] for idx in selected]


def _final_archive_samples(ep, expert_bank, args, selected_history=None):
    final_samples = list(ep.sample_batch)
    if expert_bank is not None:
        final_samples.extend(expert_bank.export_samples())
    if selected_history:
        final_samples.extend([Sample.copy_from(sample) for sample in selected_history])
    if not final_samples:
        return []
    final_samples = _dedup_samples_by_obj(final_samples)
    if getattr(args, "final_archive_mode", "pareto") == "resilient-diverse":
        return _select_resilient_diverse_final_archive(final_samples, args)
    if getattr(args, "final_archive_mode", "pareto") == "union":
        scores = np.asarray([_final_union_score(sample, args) for sample in final_samples], dtype=np.float64)
        order = np.argsort(-scores)
        cap = max(1, int(getattr(args, "final_archive_max_samples", len(final_samples))))
        return [final_samples[int(idx)] for idx in order[: min(cap, len(order))]]
    obj_batch = np.stack([np.asarray(sample.objs, dtype=np.float64) for sample in final_samples], axis=0)
    ep_indices = get_ep_indices(obj_batch, ep.reference_point)
    if len(ep_indices) == 0:
        ep_indices = np.arange(len(final_samples))
    return [final_samples[int(idx)] for idx in ep_indices]


def _write_obj_rows(path, obj_rows, obj_num):
    with open(path, 'w') as fp:
        for obj in obj_rows:
            obj = np.asarray(obj, dtype=np.float64).reshape(-1)
            fp.write(('{:5f}' + (obj_num - 1) * ',{:5f}' + '\n').format(*obj.tolist()))


def _final_policy_metadata(sample):
    metadata = sample.metadata
    trace_metrics = metadata.get("trace_metrics", {})
    return {
        "objectives": np.asarray(sample.objs, dtype=np.float64).reshape(-1).tolist(),
        "context_embedding": None
        if metadata.get("context_embedding") is None
        else np.asarray(metadata["context_embedding"], dtype=np.float64).reshape(-1).tolist(),
        "trace_metrics": {
            str(key): float(value)
            for key, value in dict(trace_metrics).items()
            if np.isscalar(value)
        },
        "bank_score": float(metadata.get("bank_score", 0.0)),
        "bank_query_affinity": float(metadata.get("bank_query_affinity", 0.0)),
        "train_iteration": int(metadata.get("train_iteration", 0)),
        "final_archive_mode": str(metadata.get("final_archive_mode", "pareto")),
        "final_archive_rank": metadata.get("final_archive_rank"),
        "final_archive_robustness": metadata.get("final_archive_robustness"),
    }


def _shared_regime_artifact_stem(row):
    regime_id = int(row.get("regime_id", -1))
    regime = str(row.get("regime", "regime")).replace("/", "_").replace(" ", "_")
    if regime_id >= 0:
        return f"regime_{regime_id:03d}_{regime}"
    return regime


def _save_shared_regime_artifacts(save_dir, env_name, seed, selection_method, regime_rows):
    shared_dir = os.path.join(save_dir, "shared_regime_returns")
    os.makedirs(shared_dir, exist_ok=True)
    for row in regime_rows:
        stem = _shared_regime_artifact_stem(row)
        payload = {
            "env_name": env_name,
            "seed": int(seed),
            "selection_method": selection_method,
            "regime": str(row.get("regime", "regime")),
            "regime_id": int(row.get("regime_id", -1)),
            "regime_meta": dict(row.get("regime_meta", {})),
            "num_solutions": int(row.get("num_solutions", 0)),
            "hv": float(row.get("hv", 0.0)),
            "eu": float(row.get("eu", 0.0)),
            "sp": float(row.get("sp", 0.0)),
            "front_points": row.get("front_points", []),
            "solution_points": row.get("solution_points", []),
        }
        with open(os.path.join(shared_dir, f"{stem}.json"), "w") as fp:
            json.dump(payload, fp, indent=2)
        np.savez_compressed(
            os.path.join(shared_dir, f"{stem}.npz"),
            solutions=np.asarray(row.get("solution_points", []), dtype=np.float32),
            pareto_front_points=np.asarray(row.get("front_points", []), dtype=np.float32),
            regime_id=np.asarray([int(row.get("regime_id", -1))], dtype=np.int32),
        )


def _run_shared_final_benchmark(args, final_samples):
    if not final_samples:
        return
    regime_seed_plan, shared_episode_seeds = build_shared_regime_seed_plan(args)
    regime_front_dir = os.path.join(args.save_dir, "regime_fronts")
    os.makedirs(regime_front_dir, exist_ok=True)

    def _persist_regime_rows(rows):
        _save_shared_regime_artifacts(
            args.save_dir,
            args.env_name,
            args.seed,
            args.selection_method,
            rows,
        )
        with open(os.path.join(regime_front_dir, "shared_final.json"), "w") as fp:
            json.dump(
                {
                    "stage": "shared_final",
                    "iteration": int(10**9),
                    "seed": int(args.seed),
                    "env_name": args.env_name,
                    "selection_method": args.selection_method,
                    "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                    "shared_regime_seed_plan": regime_seed_plan,
                    "regimes": rows,
                },
                fp,
                indent=2,
            )

    regime_rows = evaluate_regime_metrics(
        args,
        final_samples,
        evaluation,
        np.asarray(args.ref_point, dtype=np.float64),
        limit=None,
        episode_seeds=shared_episode_seeds,
        incremental_callback=_persist_regime_rows,
    )
    _persist_regime_rows(regime_rows)

    final_dir = os.path.join(args.save_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    def _write_shared_summary(front_points, trace_rows, trace_summary, dynamic_summary):
        with open(os.path.join(final_dir, "shared_eval_summary.json"), "w") as fp:
            json.dump(
                {
                    "stage": "shared_final",
                    "seed": int(args.seed),
                    "env_name": args.env_name,
                    "selection_method": args.selection_method,
                    "shared_eval_episode_seeds": [int(seed) for seed in shared_episode_seeds],
                    "shared_regime_seed_plan": regime_seed_plan,
                    "front_points": front_points,
                    "trace_metrics": trace_summary,
                    "per_sample_trace_metrics": trace_rows,
                    "regime_metrics": dynamic_summary,
                },
                fp,
                indent=2,
            )

    reevaluated_objs = []
    trace_rows = []
    dynamic_summary = summarize_regime_metrics(regime_rows)
    # Persist a minimal shared summary before the per-sample trace sweep so an
    # interrupted long run still retains the completed 20-regime artifacts and
    # can be re-scored for set-quality metrics.
    _write_shared_summary([], [], {}, dynamic_summary)

    for sample in final_samples:
        objs, trace = evaluation(
            args,
            sample,
            return_trace=True,
            episode_seeds=shared_episode_seeds,
        )
        reevaluated_objs.append(np.asarray(objs, dtype=np.float64))
        trace_rows.append(
            compute_trace_shift_metrics(
                trace,
                args.eval_delta_weight,
                recovery_window=args.trace_recovery_window,
            )
        )

    obj_array = np.stack(reevaluated_objs, axis=0) if reevaluated_objs else np.zeros((0, args.obj_num), dtype=np.float64)
    hv, eu, sp = eval(obj_array, np.asarray(args.ref_point, dtype=np.float64), args.obj_num, args.eval_delta_weight)
    trace_summary = summarize_trace_metrics(trace_rows)
    row = {
        "stage": "shared_final",
        "iteration": int(10**9),
        "seed": int(args.seed),
        "env_name": args.env_name,
        "selection_method": args.selection_method,
        "hv": hv,
        "eu": eu,
        "sp": sp,
        "drift_score": 0.0,
        "regime_loss": 0.0,
        "ep_size": int(len(obj_array)),
        "selected_size": int(len(obj_array)),
        "prediction_error": 0.0,
        "context_gap_norm": 0.0,
        "online_trace_steps": 0.0,
        "online_shift_count": 0.0,
        "context_volatility": 0.0,
        "context_support_strength": 0.0,
        "matched_recovery": 0.0,
        "shared_eval_episodes": int(len(shared_episode_seeds)),
    }
    row.update(dynamic_summary)
    row.update(trace_summary)
    append_metrics_row(os.path.join(args.save_dir, "metrics_history.csv"), row)

    _write_obj_rows(os.path.join(final_dir, "objs.txt"), obj_array, args.obj_num)
    _write_shared_summary(obj_array.tolist(), trace_rows, trace_summary, dynamic_summary)


def _dynamic_features_enabled(args):
    return (
        args.selection_method in {'dynamic-frontier', 'online-window'}
        and (
            args.dynamic_lambda > 0
            or args.knee_lambda > 0
            or args.resilience_lambda > 0
            or args.diversity_lambda > 0
            or args.shift_gap_lambda > 0
        )
    )


def _final_union_score(sample, args=None) -> float:
    trace_metrics = sample.metadata.get("trace_metrics", {})
    recovery = float(trace_metrics.get("trace_recovery_score", 0.0))
    regret = abs(float(trace_metrics.get("trace_shift_regret", 0.0)))
    latency = max(0.0, float(trace_metrics.get("trace_recovery_latency", 0.0)))
    bank_score = float(sample.metadata.get("bank_score", 0.0))
    query_aff = float(sample.metadata.get("bank_query_affinity", 0.0))
    train_iter = float(sample.metadata.get("train_iteration", 0.0))
    recovery_weight = 1.2 if args is None else float(getattr(args, "final_score_recovery_weight", 1.2))
    regret_weight = 0.35 if args is None else float(getattr(args, "final_score_regret_weight", 0.35))
    latency_weight = 0.25 if args is None else float(getattr(args, "final_score_latency_weight", 0.25))
    bank_weight = 0.15 if args is None else float(getattr(args, "final_score_bank_weight", 0.15))
    affinity_weight = 0.10 if args is None else float(getattr(args, "final_score_affinity_weight", 0.10))
    train_iter_weight = 0.05 if args is None else float(getattr(args, "final_score_train_iter_weight", 0.05))
    return float(
        recovery_weight * recovery
        + regret_weight / (1.0 + regret)
        + latency_weight / (1.0 + latency)
        + bank_weight * bank_score
        + affinity_weight * query_aff
        + train_iter_weight * train_iter
    )


def _assign_dynamic_objective_schedule(ep, selected_batch, args, context_gap_vector=None):
    if len(selected_batch) == 0 or not _dynamic_features_enabled(args):
        return
    frontier_ideal = ep.obj_batch.max(axis=0)
    topk = max(0, min(int(args.repeat_topk), int(args.obj_num)))
    context_gap_vector = (
        np.asarray(context_gap_vector, dtype=np.float64)
        if context_gap_vector is not None
        else None
    )
    for sample in selected_batch:
        frontier_gap = np.maximum(frontier_ideal - sample.objs, 0.0)
        shift_gap = np.asarray(
            sample.metadata.get("shift_gap_vector", np.zeros(args.obj_num, dtype=np.float64)),
            dtype=np.float64,
        )
        if shift_gap.shape != frontier_gap.shape:
            shift_gap = np.zeros_like(frontier_gap)

        frontier_priority = frontier_gap / (frontier_gap.max() + 1e-8) if frontier_gap.max() > 0 else frontier_gap
        shift_priority = shift_gap / (shift_gap.max() + 1e-8) if shift_gap.max() > 0 else shift_gap
        combined_shift = shift_priority
        if context_gap_vector is not None and context_gap_vector.shape == frontier_gap.shape:
            context_priority = (
                context_gap_vector / (context_gap_vector.max() + 1e-8)
                if context_gap_vector.max() > 0
                else context_gap_vector
            )
            combined_shift = 0.5 * shift_priority + 0.5 * context_priority
        combined_priority = frontier_priority + args.shift_gap_lambda * combined_shift
        if np.allclose(combined_priority.sum(), 0.0):
            combined_priority = frontier_priority

        order = list(np.argsort(-combined_priority))
        repeats = np.ones(args.obj_num, dtype=np.int64)
        for rank, obj_id in enumerate(order[:topk]):
            repeats[obj_id] += topk - rank
        sample.metadata["objective_priority"] = combined_priority
        sample.metadata["objective_order"] = order
        sample.metadata["objective_repeats"] = repeats


def _log_stage_metrics(args, stage, iteration, ep, selected_batch, drift_score, regime_loss, context_state=None):
    obj_array = np.asarray(ep.obj_batch, dtype=np.float64)
    hv, eu, sp = eval(obj_array, np.asarray(args.ref_point), args.obj_num, args.eval_delta_weight)
    regime_rows = evaluate_regime_metrics(
        args,
        selected_batch,
        evaluation,
        np.asarray(args.ref_point, dtype=np.float64),
        limit=args.regime_eval_samples,
    )
    dynamic_summary = summarize_regime_metrics(regime_rows)
    trace_rows = []
    trace_sample_batch = list(selected_batch)[: max(0, int(args.trace_eval_samples))]
    for sample in trace_sample_batch:
        _objs, trace = evaluation(args, sample, return_trace=True)
        trace_rows.append(
            compute_trace_shift_metrics(
                trace,
                args.eval_delta_weight,
                recovery_window=args.trace_recovery_window,
            )
        )
    trace_summary = summarize_trace_metrics(trace_rows)

    row = {
        "stage": stage,
        "iteration": int(iteration),
        "seed": int(args.seed),
        "env_name": args.env_name,
        "selection_method": args.selection_method,
        "hv": hv,
        "eu": eu,
        "sp": sp,
        "drift_score": float(drift_score),
        "regime_loss": float(regime_loss),
        "ep_size": int(len(ep.obj_batch)),
        "selected_size": int(len(selected_batch)),
        "prediction_error": float(context_state["prediction_error"]) if context_state is not None else 0.0,
        "context_gap_norm": float(np.linalg.norm(context_state["gap_vector"])) if context_state is not None else 0.0,
        "online_trace_steps": float(context_state.get("online_trace_steps", 0.0)) if context_state is not None else 0.0,
        "online_shift_count": float(context_state.get("online_shift_count", 0.0)) if context_state is not None else 0.0,
        "context_volatility": float(context_state.get("context_volatility", 0.0)) if context_state is not None else 0.0,
        "context_support_strength": float(context_state.get("context_support_strength", 0.0)) if context_state is not None else 0.0,
        "matched_recovery": float(context_state.get("matched_recovery", 0.0)) if context_state is not None else 0.0,
        "shared_eval_episodes": 0.0,
    }
    row.update(dynamic_summary)
    row.update(trace_summary)
    append_metrics_row(os.path.join(args.save_dir, "metrics_history.csv"), row)

    for regime_row in regime_rows:
        per_regime = {
            "stage": stage,
            "iteration": int(iteration),
            "seed": int(args.seed),
            "env_name": args.env_name,
            "selection_method": args.selection_method,
            "regime": regime_row["regime"],
            "regime_id": regime_row.get("regime_id", -1),
            "regime_meta": regime_row.get("regime_meta", {}),
            "hv": regime_row["hv"],
            "eu": regime_row["eu"],
            "sp": regime_row["sp"],
            "points": regime_row["points"],
            "num_solutions": regime_row.get("num_solutions", regime_row["points"]),
        }
        append_metrics_row(os.path.join(args.save_dir, "regime_metrics.csv"), per_regime)
    regime_front_dir = os.path.join(args.save_dir, "regime_fronts")
    os.makedirs(regime_front_dir, exist_ok=True)
    with open(os.path.join(regime_front_dir, f"{stage}_{int(iteration)}.json"), "w") as fp:
        json.dump(
            {
                "stage": stage,
                "iteration": int(iteration),
                "seed": int(args.seed),
                "env_name": args.env_name,
                "selection_method": args.selection_method,
                "regimes": regime_rows,
            },
            fp,
            indent=2,
        )
    return hv, eu, sp, {**dynamic_summary, **trace_summary}

def run(args):

    # --------------------> Preparation <-------------------- #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    start_time = time.time()

    # initialize ep and population and opt_graph
    ref = np.array(args.ref_point, dtype=np.float64)
    ep = EP(ref, args.num_select, args.policy_buffer)
    expert_bank = (
        ContextExpertBank(
            max_slots=args.bank_num_slots,
            experts_per_slot=args.bank_slot_size,
            merge_threshold=args.bank_merge_threshold,
            recovery_weight=args.expert_bank_recovery_weight,
            regret_weight=args.expert_bank_regret_weight,
            latency_weight=args.expert_bank_latency_weight,
            gap_weight=args.expert_bank_gap_weight,
            retrieval_backend=args.bank_retrieval_backend,
            hnsw_ef_search=args.hnsw_ef_search,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
        )
        if int(args.bank_enable) > 0
        else None
    )
    regime_model = None
    context_memory = OnlineContextMemory(
        max_size=args.context_history_size,
        history_lambda=args.context_history_lambda,
        forecast_lambda=args.context_forecast_lambda,
        nearest_k=args.context_nearest_k,
    )
    online_trace_window = OnlineTraceWindow(max_steps=args.context_buffer_steps)
    prev_target_embedding = None
    current_drift_score = 0.0
    current_context_state = None
    selected_history = []
    
    # Construct tasks for warm up
    selected_batch, scalarization_batch = initialize_warm_up_batch(args, device)
    if args.auto_ref_point and len(selected_batch) > 0:
        warmup_objs = np.asarray([sample.objs for sample in selected_batch], dtype=np.float64)
        lower = warmup_objs.min(axis=0)
        margin = 0.1 * np.maximum(np.abs(lower), 1.0)
        ref = lower - margin
        ep.reference_point = ref
        args.ref_point = ref.tolist()
        print(f"Auto reference point calibrated to {ref}.")
    
    print_info('\n------------------------------- Initialization Stage -------------------------------')    

    # --------------------> RL Optimization <-------------------- #
    # compose task for each selected solution
    iteration = 0
    task_batch = []
    for solutions, scalarization in \
            zip(selected_batch, scalarization_batch):
        task_batch.append(Task(solutions, scalarization)) # each task is a (policy, weight) pair
    worker_count = len(task_batch)
        
    # Compute update steps for initialization
    num_update = max(1, int(args.num_init_steps // max(len(task_batch), 1) // args.num_steps))

    # run MOPG for each task in parallel
    serial_workers = _use_serial_workers(args)
    if serial_workers:
        processes = []
        results_queue = ThreadQueue()
        done_event = ThreadEvent()
        done_event.set()
        all_offspring_batch = [[] for _ in range(len(task_batch))]
        for task_id, task in enumerate(task_batch):
            MOPG_worker(
                args,
                task_id,
                task,
                device,
                iteration,
                num_update,
                start_time,
                results_queue,
                done_event,
            )
            rl_results = results_queue.get()
            offsprings = rl_results["offspring_batch"]
            for sample in offsprings:
                restored = _deserialize_sample(sample) if isinstance(sample, dict) else sample
                all_offspring_batch[task_id].append(Sample.copy_from(restored))
    else:
        ctx = _mp_context(args)
        processes = []
        results_queue = ctx.Queue()
        done_event = ctx.Event()

        for task_id, task in enumerate(task_batch):
            p = ctx.Process(target = MOPG_worker, \
                args = (args, task_id, task, device, iteration, num_update, start_time, results_queue, done_event))
            p.start()
            processes.append(p)

        # collect MOPG results for offsprings and insert objs into objs buffer
        all_offspring_batch = [[] for _ in range(worker_count)]
        cnt_done_workers = 0
        while cnt_done_workers < len(processes):
            rl_results = results_queue.get()
            task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
            for sample in offsprings:
                restored = _deserialize_sample(sample) if isinstance(sample, dict) else sample
                all_offspring_batch[task_id].append(Sample.copy_from(restored))
            if rl_results['done']:
                cnt_done_workers += 1
        
    # put all intermidiate policies into all_sample_batch for EP update
    all_sample_batch = [] 
    for task_id in range(worker_count):
        offsprings = all_offspring_batch[task_id]
        for i, sample in enumerate(offsprings):
            all_sample_batch.append(sample)

    if not serial_workers:
        done_event.set()
        _cleanup_worker_handles(processes, results_queue)
    iteration += num_update

    # -----------------------> Update EP <----------------------- #
    # update EP and population
    ep.update(all_sample_batch, select_after_update=False)
    regime_model, regime_loss = (
        _fit_regime_model(args, all_sample_batch)
        if int(getattr(args, "context_encoder_enable", 1)) > 0
        else (None, 0.0)
    )
    if expert_bank is not None:
        expert_bank.update_batch(all_sample_batch)
    if regime_model is not None:
        if args.selection_method == "online-window":
            _update_online_trace_window(args, online_trace_window, all_sample_batch)
            next_context_state = _infer_online_context_state(
                args,
                regime_model,
                online_trace_window,
                context_memory,
                prev_target_embedding,
                iteration=0,
                sample_pool=_context_sample_pool(ep, expert_bank, all_sample_batch),
            )
            if next_context_state is not None:
                current_context_state = next_context_state
            current_regime_embedding = None if current_context_state is None else current_context_state["target_embedding"]
            current_drift_score = 0.0 if current_context_state is None else current_context_state["drift_score"]
            if current_context_state is not None:
                prev_target_embedding = current_context_state["target_embedding"]
        else:
            current_regime_embedding = _aggregate_context_embedding(all_sample_batch)
            if prev_target_embedding is not None and current_regime_embedding is not None:
                current_drift_score = 1.0 - cosine_affinity(prev_target_embedding, current_regime_embedding)
            else:
                current_drift_score = 0.0
            prev_target_embedding = current_regime_embedding
        print(f"Regime encoder loss: {regime_loss:.4f}, drift score: {current_drift_score:.4f}")
        _select_online_candidates(ep, expert_bank, args, current_regime_embedding, current_drift_score)
    else:
        current_context_state = None
        current_drift_score = 0.0
        _select_online_candidates(ep, expert_bank, args, current_regime_embedding=None, current_drift_score=0.0)
    all_sample_batch = []

    # ------------------- > Task Selection <--------------------- #
    selected_batch = ep.selected_batch
    selected_history.extend([Sample.copy_from(sample) for sample in selected_batch])
    _assign_dynamic_objective_schedule(
        ep,
        selected_batch,
        args,
        context_gap_vector=None if current_context_state is None else current_context_state["gap_vector"],
    )
        
    print_info('Selected Tasks:')
    for i in range(len(selected_batch)):
        print_info('objs = {}'.format(selected_batch[i].objs))
        
    # save
    hv, eu, sp, dynamic_summary = _log_stage_metrics(
        args,
        stage="init",
        iteration=0,
        ep=ep,
        selected_batch=selected_batch,
        drift_score=current_drift_score,
        regime_loss=regime_loss if regime_model is not None else 0.0,
        context_state=current_context_state,
    )
    print(f"Hyper Volume: {hv:.4f}, Expected Utility: {eu:.4f}, Sparsity: {sp:.4f}")
    if dynamic_summary:
        print(
            "Dynamic metrics: "
            + ", ".join(f"{k}={v:.4f}" for k, v in dynamic_summary.items())
        )
    
    # ----------------------> Save Results <---------------------- #
    # save ep
    ep_dir = os.path.join(args.save_dir, 'init', 'ep')
    os.makedirs(ep_dir, exist_ok = True)
    with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
        for obj in ep.obj_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))
    
    # save selections
    selected_dir = os.path.join(args.save_dir, 'init', 'solutions')
    os.makedirs(selected_dir, exist_ok = True)
    with open(os.path.join(selected_dir, 'solutions.txt'), 'w') as fp:
        for solutions in selected_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(solutions.objs)))
    with open(os.path.join(selected_dir, 'weights.txt'), 'w') as fp:
        for scalarization in scalarization_batch:
            fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
    

    print_info('\n------------------------------- Extension Stage -------------------------------') 
    
    total_num_update = max(
        1,
        int((args.num_time_steps - args.num_init_steps) // max(args.num_select, 1) // args.num_steps // max(args.obj_num, 1)),
    )
    extension_iter = total_num_update // args.update_iter
    beta = args.beta
    iteration = 0

    for iter_ in range (extension_iter+1):   
        remaining_updates = total_num_update - iteration
        if remaining_updates <= 0:
            break
        extension_start_time = time.time()
        dynamic_update = int(round(args.update_iter * (1.0 + 0.5 * current_drift_score)))
        num_update = min(max(1, dynamic_update), remaining_updates)
        print(f"Starting the extension from iteration {iteration} to {iteration+num_update}.")

        # run MOPG for each task in parallel
        serial_workers = _use_serial_workers(args)
        worker_count = len(selected_batch)
        if serial_workers:
            processes = []
            results_queue = ThreadQueue()
            done_event = ThreadEvent()
            done_event.set()
            all_offspring_batch = [[] for _ in range(len(selected_batch))]
            for sample_id, sample in enumerate(selected_batch):
                Extension_worker(
                    args,
                    sample_id,
                    sample,
                    device,
                    iteration,
                    num_update,
                    extension_start_time,
                    results_queue,
                    done_event,
                )
                rl_results = results_queue.get()
                offsprings = rl_results["offspring_batch"]
                for child_sample in offsprings:
                    restored = _deserialize_sample(child_sample) if isinstance(child_sample, dict) else child_sample
                    all_offspring_batch[sample_id].append(Sample.copy_from(restored))
        else:
            ctx = _mp_context(args)
            processes = []
            results_queue = ctx.Queue()
            done_event = ctx.Event()

            for sample_id, sample in enumerate(selected_batch):
                p = ctx.Process(target = Extension_worker, \
                    args = (args, sample_id, sample, device, iteration, num_update, extension_start_time, results_queue, done_event))
                p.start()
                processes.append(p)

            # collect MOPG results for offsprings and insert objs into objs buffer
            all_offspring_batch = [[] for _ in range(worker_count)]
            cnt_done_workers = 0
            while cnt_done_workers < len(processes):
                rl_results = results_queue.get()
                task_id, offsprings = rl_results['task_id'], rl_results['offspring_batch']
                for sample in offsprings:
                    restored = _deserialize_sample(sample) if isinstance(sample, dict) else sample
                    all_offspring_batch[task_id].append(Sample.copy_from(restored))
                if rl_results['done']:
                    cnt_done_workers += 1

        # put all intermidiate policies into all_sample_batch for EP update
        # all_sample_batch = [] 
        # store the last policy for each optimization weight for RA
        last_offspring_batch = [None] * worker_count
        # only the policies with iteration % update_iter = 0 are inserted into offspring_batch for population update
        # after warm-up stage, it's equivalent to the last_offspring_batch
        offspring_batch = [] 
        for task_id in range(worker_count):
            offsprings = all_offspring_batch[task_id]
            #prev_node_id = task_batch[task_id].sample.optgraph_id
            #opt_weights = deepcopy(task_batch[task_id].scalarization.weights).detach().numpy()
            for i, sample in enumerate(offsprings):
                all_sample_batch.append(sample)
                if (i + 1) % args.update_iter == 0:
                    #prev_node_id = opt_graph.insert(opt_weights, deepcopy(sample.objs), prev_node_id)
                    #sample.optgraph_id = prev_node_id
                    offspring_batch.append(sample)
            last_offspring_batch[task_id] = offsprings[-1]

        if not serial_workers:
            done_event.set()
            _cleanup_worker_handles(processes, results_queue)
        iteration += num_update

        # -----------------------> Update EP <----------------------- #
        # update EP and population
        ep.update(all_sample_batch, select_after_update=False)
        regime_model, regime_loss = (
            _fit_regime_model(args, all_sample_batch)
            if int(getattr(args, "context_encoder_enable", 1)) > 0
            else (None, 0.0)
        )
        if expert_bank is not None:
            expert_bank.update_batch(all_sample_batch)
        if regime_model is not None:
            if args.selection_method == "online-window":
                _update_online_trace_window(args, online_trace_window, last_offspring_batch)
                next_context_state = _infer_online_context_state(
                    args,
                    regime_model,
                    online_trace_window,
                    context_memory,
                    prev_target_embedding,
                    iteration=iteration,
                    sample_pool=_context_sample_pool(ep, expert_bank, last_offspring_batch),
                )
                if next_context_state is not None:
                    current_context_state = next_context_state
                current_regime_embedding = None if current_context_state is None else current_context_state["target_embedding"]
                current_drift_score = 0.0 if current_context_state is None else current_context_state["drift_score"]
                if current_context_state is not None:
                    prev_target_embedding = current_context_state["target_embedding"]
            else:
                current_regime_embedding = _aggregate_context_embedding(all_sample_batch)
                if prev_target_embedding is not None and current_regime_embedding is not None:
                    current_drift_score = 1.0 - cosine_affinity(prev_target_embedding, current_regime_embedding)
                else:
                    current_drift_score = 0.0
                prev_target_embedding = current_regime_embedding
            print(f"Regime encoder loss: {regime_loss:.4f}, drift score: {current_drift_score:.4f}")
            _select_online_candidates(ep, expert_bank, args, current_regime_embedding, current_drift_score)
        else:
            current_context_state = None
            current_drift_score = 0.0
            _select_online_candidates(ep, expert_bank, args, current_regime_embedding=None, current_drift_score=0.0)
        all_sample_batch = []

        # ------------------- > Task Selection <--------------------- #
        selected_batch = ep.selected_batch
        selected_history.extend([Sample.copy_from(sample) for sample in selected_batch])
        _assign_dynamic_objective_schedule(
            ep,
            selected_batch,
            args,
            context_gap_vector=None if current_context_state is None else current_context_state["gap_vector"],
        )
        print_info('Selected Tasks:')
        for i in range(len(selected_batch)):
            print_info('objs = {}'.format(selected_batch[i].objs))

        # save
        hv, eu, sp, dynamic_summary = _log_stage_metrics(
            args,
            stage="extension",
            iteration=iteration,
            ep=ep,
            selected_batch=selected_batch,
            drift_score=current_drift_score,
            regime_loss=regime_loss if regime_model is not None else 0.0,
            context_state=current_context_state,
        )
        print(f"Hyper Volume: {hv:.4f}, Expected Utility: {eu:.4f}, Sparsity: {sp:.4f}")
        if dynamic_summary:
            print(
                "Dynamic metrics: "
                + ", ".join(f"{k}={v:.4f}" for k, v in dynamic_summary.items())
            )

        # ----------------------> Save Results <---------------------- #
        # save ep
        ep_dir = os.path.join(args.save_dir, str(iteration), 'ep')
        os.makedirs(ep_dir, exist_ok = True)
        with open(os.path.join(ep_dir, 'objs.txt'), 'w') as fp:
            for obj in ep.obj_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*obj))

        # save selected solutions
        selected_dir = os.path.join(args.save_dir, str(iteration), 'solutions')
        os.makedirs(selected_dir, exist_ok = True)
        with open(os.path.join(selected_dir, 'solutions.txt'), 'w') as fp:
            for solutions in selected_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(solutions.objs)))
        with open(os.path.join(selected_dir, 'weights.txt'), 'w') as fp:
            for scalarization in scalarization_batch:
                fp.write(('{:5f}' + (args.obj_num - 1) * ',{:5f}' + '\n').format(*(scalarization.weights)))
        
    end_time = time.time()
    print('total time:', end_time - start_time)
    
     # ----------------------> Save Final Model <---------------------- 

    os.makedirs(os.path.join(args.save_dir, 'final'), exist_ok = True)

    final_samples = _final_archive_samples(ep, expert_bank, args, selected_history=selected_history)

    # save final policies & env_params
    for i, sample in enumerate(final_samples):
        torch.save(sample.actor_critic.state_dict(), os.path.join(args.save_dir, 'final', 'EP_policy_{}.pt'.format(i)))
        with open(os.path.join(args.save_dir, 'final', 'EP_env_params_{}.pkl'.format(i)), 'wb') as fp:
            pickle.dump(sample.env_params, fp)
        with open(os.path.join(args.save_dir, 'final', 'EP_metadata_{}.json'.format(i)), 'w') as fp:
            json.dump(_final_policy_metadata(sample), fp, indent=2)
    
    # save all final objectives
    _write_obj_rows(
        os.path.join(args.save_dir, 'final', 'objs.txt'),
        [sample.objs for sample in final_samples],
        args.obj_num,
    )

    # save all final env_params
    if args.obj_rms:
        with open(os.path.join(args.save_dir, 'final', 'env_params.txt'), 'w') as fp:
            for sample in final_samples:
                fp.write('obj_rms: mean: {} var: {}\n'.format(sample.env_params['obj_rms'].mean, sample.env_params['obj_rms'].var))

    # The final policy files support evaluation only.  A causal OOD continuation
    # additionally needs PPO optimizer state, the global archive, semantic
    # expert slots, and online context state.  HNSW itself is rebuilt from the
    # persisted slot prototypes when this checkpoint is loaded.
    save_owcmorl_runtime_checkpoint(
        os.path.join(args.save_dir, 'final', 'short_ood_runtime.pt'),
        ep=ep,
        expert_bank=expert_bank,
        regime_model=regime_model,
        context_memory=context_memory,
        online_trace_window=online_trace_window,
        prev_target_embedding=prev_target_embedding,
        current_drift_score=current_drift_score,
        current_context_state=current_context_state,
        selected_history=selected_history,
        selected_batch=selected_batch,
        final_samples=final_samples,
        iteration=iteration,
        args_signature=_runtime_args_signature(args),
    )
    if not getattr(args, "skip_final_shared_benchmark", False):
        _run_shared_final_benchmark(args, final_samples)
    else:
        print("Skipped shared-final benchmark after saving short-OOD runtime checkpoint.")
