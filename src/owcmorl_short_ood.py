"""Native OW-CMORL continuation used by the short-update OOD protocol."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from src.dynamic_morl.arguments import get_parser
from src.dynamic_morl.expert_bank import ContextExpertBank
from src.dynamic_morl.mopg import (
    _attach_dynamic_metadata,
    _collect_rollout,
    _finish_update,
    _make_rollouts,
    _make_sample,
    _restore_env_params,
    evaluation,
)
from src.dynamic_morl.envs import make_vec_envs
from src.dynamic_morl.morl import (
    _assign_dynamic_objective_schedule,
    _context_sample_pool,
    _final_archive_samples,
    _fit_regime_model,
    _infer_online_context_state,
    _select_online_candidates,
    _update_online_trace_window,
    eval as front_metrics,
)
from src.dynamic_morl.runtime_checkpoint import load_owcmorl_runtime_checkpoint
from src.dynamic_morl.sample import Sample
from src.short_ood_adapters import AdaptationAdapter


def _load_source_args(source_run_dir: Path):
    raw = ast.literal_eval((source_run_dir / "args.txt").read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Unexpected args.txt payload in {source_run_dir}")
    return get_parser().parse_args([str(value) for value in raw])


def _normalizer_payload(value: Any) -> Any:
    if value is None:
        return None
    if all(hasattr(value, key) for key in ("mean", "var", "count")):
        return {
            "mean": np.asarray(value.mean),
            "var": np.asarray(value.var),
            "count": float(value.count),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _sample_payload(sample: Any) -> dict[str, Any]:
    return {
        "objs": None if sample.objs is None else np.asarray(sample.objs, dtype=np.float64),
        "actor": sample.actor_critic.state_dict(),
        "optimizer": sample.agent.optimizer.state_dict(),
        "env_params": {
            key: _normalizer_payload(value) for key, value in dict(sample.env_params).items()
        },
        "metadata": deepcopy(sample.metadata),
    }


def _bank_payload(bank: ContextExpertBank | None) -> Any:
    if bank is None:
        return None
    return {
        "settings": {
            "max_slots": bank.max_slots,
            "experts_per_slot": bank.experts_per_slot,
            "merge_threshold": bank.merge_threshold,
            "backend": bank.retrieval_backend,
            "hnsw_ef_search": bank.hnsw_ef_search,
            "hnsw_m": bank.hnsw_m,
            "hnsw_ef_construction": bank.hnsw_ef_construction,
            "insert_count": bank._insert_count,
        },
        "slots": [
            {
                "prototype": np.asarray(slot.prototype, dtype=np.float64),
                "scores": list(slot.scores),
                "samples": [_sample_payload(sample) for sample in slot.samples],
            }
            for slot in bank.slots
        ],
    }


class OWCMORLAdapter(AdaptationAdapter):
    """Resumes the OW-CMORL archive, expert bank, and PPO snapshot state."""

    method = "dynamic"

    def __init__(self, source_run_dir: str | Path, *, local_lr_scale: float = 1.0):
        self.source_run_dir = Path(source_run_dir)
        runtime_path = self.source_run_dir / "final" / "short_ood_runtime.pt"
        self.runtime = load_owcmorl_runtime_checkpoint(runtime_path)
        self.args = _load_source_args(self.source_run_dir)
        self.device = torch.device("cpu")
        self.local_lr_scale = float(local_lr_scale)
        if self.local_lr_scale <= 0.0:
            raise ValueError("local_lr_scale must be positive")
        torch.set_default_dtype(torch.float64)
        self.runtime.setdefault("ood_refinement_samples", [])
        signature = self.runtime["args_signature"]
        if str(signature.get("env_name")) != str(self.args.env_name):
            raise ValueError("OW-CMORL runtime checkpoint and args.txt disagree on env_name.")
        if int(signature.get("obj_num")) != int(self.args.obj_num):
            raise ValueError("OW-CMORL runtime checkpoint and args.txt disagree on obj_num.")
        runtime_reference = signature.get("ref_point")
        if runtime_reference is None or len(runtime_reference) != int(self.args.obj_num):
            raise ValueError("OW-CMORL runtime checkpoint is missing a valid calibrated reference point.")
        # ``args.txt`` can retain the parser default after ``--auto-ref-point``
        # calibration.  The persisted runtime signature is the authoritative
        # calibrated value for any future direct front evaluation.
        self.args.ref_point = [float(value) for value in runtime_reference]
        if "id_evaluation_samples" not in self.runtime:
            self.runtime["id_evaluation_samples"] = self._retained_archive_samples()
        if int(self.args.num_processes) != 1:
            raise ValueError("Short OOD update accounting requires num_processes=1.")
        self._last_backend = "unqueried"
        self._last_retrieved = 0

    def set_trajectory_seed(self, seed: int) -> None:
        self.args.seed = int(seed)
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    def state_payload(self) -> Mapping[str, Any]:
        ep = self.runtime["ep"]
        memory = self.runtime["context_memory"]
        window = self.runtime["online_trace_window"]
        regime_model = self.runtime["regime_model"]
        return {
            "method": self.method,
            "local_lr_scale": self.local_lr_scale,
            "ep": {
                "reference_point": np.asarray(ep.reference_point, dtype=np.float64),
                "obj_batch": np.asarray(ep.obj_batch, dtype=np.float64),
                "obj_hist": list(ep.obj_hist),
                "samples": [_sample_payload(sample) for sample in ep.sample_batch],
                "selected": [_sample_payload(sample) for sample in self.runtime["selected_batch"]],
            },
            "expert_bank": _bank_payload(self.runtime["expert_bank"]),
            "regime_model": None if regime_model is None else regime_model.state_dict(),
            "context_memory": [
                {
                    "iteration": int(snapshot.iteration),
                    "current_embedding": np.asarray(snapshot.current_embedding),
                    "forecast_embedding": np.asarray(snapshot.forecast_embedding),
                    "target_embedding": np.asarray(snapshot.target_embedding),
                    "gap_vector": np.asarray(snapshot.gap_vector),
                    "utility_mean": float(snapshot.utility_mean),
                    "drift_score": float(snapshot.drift_score),
                    "prediction_error": float(snapshot.prediction_error),
                }
                for snapshot in memory.snapshots
            ],
            "online_trace_window": window.as_trace(),
            "prev_target_embedding": self.runtime.get("prev_target_embedding"),
            "current_drift_score": float(self.runtime.get("current_drift_score", 0.0)),
            "current_context_state": deepcopy(self.runtime.get("current_context_state")),
            "selected_history": [_sample_payload(sample) for sample in self.runtime["selected_history"]],
            "final_samples": [_sample_payload(sample) for sample in self.runtime["final_samples"]],
            "id_evaluation_samples": [
                _sample_payload(sample) for sample in self.runtime["id_evaluation_samples"]
            ],
            "ood_refinement_samples": [
                _sample_payload(sample) for sample in self.runtime["ood_refinement_samples"]
            ],
            "iteration": int(self.runtime["iteration"]),
        }

    def action(self, observation: Any, preference: Any = None) -> Any:
        del observation, preference
        raise RuntimeError("OW-CMORL evaluation operates on its persisted Pareto archive, not one scalar action.")

    def archive_size(self) -> int | None:
        return len(self._evaluation_samples())

    def _retained_archive_samples(self) -> list[Any]:
        ep_copy = deepcopy(self.runtime["ep"])
        bank = self.runtime["expert_bank"]
        bank_copy = None if bank is None else ContextExpertBank.from_state_dict(bank.state_dict())
        return _final_archive_samples(
            ep_copy,
            bank_copy,
            self.args,
            selected_history=deepcopy(self.runtime["selected_history"]),
        )

    def _evaluation_samples(self) -> list[Any]:
        """Keep refined OOD candidates through current-condition evaluation.

        The ID archive is reduced once before OOD begins and then frozen as
        the baseline candidate set. Every child produced by local OOD
        refinement is appended explicitly. All returned candidates are
        evaluated in the current ID or OOD condition, avoiding comparisons
        between a child's OOD objective vector and historical ID vectors.
        """

        retained = [Sample.copy_from(sample) for sample in self.runtime["id_evaluation_samples"]]
        refined = [Sample.copy_from(sample) for sample in self.runtime["ood_refinement_samples"]]
        return retained + refined

    def evaluate(self, env_kwargs: dict[str, Any], *, episode_seed: int) -> Mapping[str, Any]:
        # OOD adaptation consumes random numbers. Evaluation must remain a
        # fixed comparison of archive states rather than inherit that stream.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(int(episode_seed))
            np.random.seed(int(episode_seed))
            torch.manual_seed(int(episode_seed))
            samples = self._evaluation_samples()
            if not samples:
                raise RuntimeError("OW-CMORL runtime checkpoint contains no evaluable archive samples.")
            objectives = []
            for sample in samples:
                objective = evaluation(
                    self.args,
                    sample,
                    env_kwargs=env_kwargs,
                    episode_seeds=[int(episode_seed)],
                )
                objectives.append(np.asarray(objective, dtype=np.float64))
            front = np.stack(objectives, axis=0)
            hv, eu, _sp = front_metrics(
                front,
                np.asarray(self.args.ref_point, dtype=np.float64),
                self.args.obj_num,
                self.args.eval_delta_weight,
            )
            return {
                "HV": float(hv),
                "EU": float(eu),
                "front_size": int(len(front)),
                "front_points": front.tolist(),
                "hnsw_backend": self._last_backend,
                "retrieved_snapshots": int(self._last_retrieved),
            }
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def _seed_samples(
        self,
        env_kwargs: dict[str, Any],
        objective_schedule: list[int],
    ) -> list[Any]:
        bank = self.runtime["expert_bank"]
        context_state = self.runtime.get("current_context_state") or {}
        embedding = context_state.get("target_embedding")
        retrieved = []
        if bank is not None and embedding is not None:
            retrieved = bank.query(
                np.asarray(embedding, dtype=np.float64),
                top_slots=int(self.args.bank_query_slots),
                top_per_slot=int(self.args.bank_query_topk),
            )
            self._last_backend = bank.resolved_retrieval_backend
            if bank.retrieval_backend == "hnsw" and self._last_backend != "hnsw":
                raise RuntimeError("OW-CMORL OOD run requires HNSW, but the restored bank fell back to exact search.")
        self._last_retrieved = len(retrieved)
        candidates = retrieved or list(self.runtime["selected_batch"]) or list(self.runtime["final_samples"])
        if not candidates:
            raise RuntimeError("OW-CMORL runtime checkpoint contains no resumable snapshot.")

        # Once OOD adaptation is allowed, score the HNSW candidates in the
        # current OOD condition.  This is a causal one-episode probe, not a
        # zero-shot update, and assigns one context-relevant seed to each
        # objective region of the Pareto front.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(int(self.args.seed))
            np.random.seed(int(self.args.seed))
            torch.manual_seed(int(self.args.seed))
            objectives = np.asarray(
                [
                    evaluation(
                        self.args,
                        Sample.copy_from(sample),
                        env_kwargs=env_kwargs,
                        episode_seeds=[int(self.args.seed)],
                    )
                    for sample in candidates
                ],
                dtype=np.float64,
            )
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        return [
            Sample.copy_from(candidates[int(np.argmax(objectives[:, int(objective)]))])
            for objective in objective_schedule
        ]

    def _refine(
        self,
        samples: list[Any],
        *,
        objective_schedule: list[int],
        updates: int,
        env_kwargs: dict[str, Any],
    ) -> list[Any]:
        if len(samples) != len(objective_schedule):
            raise ValueError("each local-refinement objective requires one seed sample")
        base_budget, remainder = divmod(int(updates), len(objective_schedule))
        phase_budgets = [base_budget + int(index < remainder) for index in range(len(objective_schedule))]
        children: list[Any] = []
        completed_updates = 0

        for phase_index, (focus_objective, phase_updates, seed) in enumerate(
            zip(objective_schedule, phase_budgets, samples)
        ):
            if phase_updates <= 0:
                continue
            working = Sample.copy_from(seed)
            working.actor_critic = working.actor_critic.to(self.device).to(dtype=torch.get_default_dtype())
            working.link_policy_agent()
            for param_group in working.agent.optimizer.param_groups:
                param_group["lr"] = float(param_group["lr"]) * self.local_lr_scale
            envs = make_vec_envs(
                env_name=self.args.env_name,
                # Every objective phase refines the same observed OOD
                # condition. PPO update seeds below provide stochasticity;
                # changing the environment seed here would instead refit a
                # different chlor-alkali process surrogate per phase.
                seed=int(self.args.seed),
                num_processes=self.args.num_processes,
                gamma=self.args.gamma,
                log_dir=None,
                device=self.device,
                allow_early_resets=False,
                obj_rms=deepcopy(self.args.obj_rms),
                ob_rms=deepcopy(self.args.ob_rms),
                env_kwargs=env_kwargs,
            )
            try:
                _restore_env_params(envs, working.env_params)
                rollouts = _make_rollouts(self.args, envs, working.actor_critic, self.device)
                for local_update in range(phase_updates):
                    torch.manual_seed(int(self.runtime["iteration"]) + completed_updates + local_update)
                    _collect_rollout(self.args, envs, working.actor_critic, rollouts)
                    _finish_update(self.args, envs, working.actor_critic, rollouts)
                    obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None
                    working.agent.ipo_update(
                        rollouts,
                        int(focus_objective),
                        self.args.obj_num,
                        obj_rms_var,
                    )
                    rollouts.after_update()
                child = _make_sample(
                    envs,
                    working.actor_critic,
                    working.agent,
                    base_metadata=working.metadata,
                )
            finally:
                envs.close()
            objectives, trace = evaluation(
                self.args,
                child,
                return_trace=True,
                env_kwargs=env_kwargs,
                episode_seeds=[int(self.args.seed)],
            )
            child.objs = np.asarray(objectives, dtype=np.float64)
            _attach_dynamic_metadata(self.args, child, trace)
            child.metadata["train_iteration"] = int(
                self.runtime["iteration"] + completed_updates + phase_updates
            )
            child.metadata["phase_objective"] = int(focus_objective)
            child.metadata["phase_budget"] = int(phase_updates)
            child.metadata["seed_selection"] = "current_ood_objective_max"
            child.metadata["worker_mode"] = "short_ood_local_refinement"
            children.append(child)
            completed_updates += phase_updates
        return children

    def _update_online_state(self, children: list[Any]) -> None:
        if not children:
            return
        ep = self.runtime["ep"]
        bank = self.runtime["expert_bank"]
        ep.update(children, select_after_update=False)
        if bank is not None:
            bank.update_batch(children)
        regime_model, _loss = _fit_regime_model(self.args, children)
        self.runtime["regime_model"] = regime_model
        if regime_model is None:
            self.runtime["current_context_state"] = None
            self.runtime["current_drift_score"] = 0.0
            return
        _update_online_trace_window(self.args, self.runtime["online_trace_window"], children)
        context_state = _infer_online_context_state(
            self.args,
            regime_model,
            self.runtime["online_trace_window"],
            self.runtime["context_memory"],
            self.runtime.get("prev_target_embedding"),
            iteration=int(self.runtime["iteration"]),
            sample_pool=_context_sample_pool(ep, bank, children),
        )
        self.runtime["current_context_state"] = context_state
        if context_state is not None:
            self.runtime["prev_target_embedding"] = context_state["target_embedding"]
            self.runtime["current_drift_score"] = float(context_state["drift_score"])
        else:
            self.runtime["current_drift_score"] = 0.0
        _select_online_candidates(
            ep,
            bank,
            self.args,
            None if context_state is None else context_state["target_embedding"],
            float(self.runtime["current_drift_score"]),
        )
        self.runtime["selected_batch"] = list(ep.selected_batch)
        self.runtime["selected_history"].extend(Sample.copy_from(sample) for sample in ep.selected_batch)
        _assign_dynamic_objective_schedule(
            ep,
            ep.selected_batch,
            self.args,
            context_gap_vector=None if context_state is None else context_state["gap_vector"],
        )

    def adapt(self, env: Any, transitions: int) -> Mapping[str, float]:
        updates = int(transitions)
        if updates < 0:
            raise ValueError("update budget increment must be nonnegative")
        if updates == 0:
            return {"updates": 0.0, "transitions": 0.0}
        env_kwargs = dict(getattr(env, "env_kwargs"))
        objective_schedule = list(range(int(self.args.obj_num)))
        seeds = self._seed_samples(env_kwargs, objective_schedule)
        children = self._refine(
            seeds,
            objective_schedule=objective_schedule,
            updates=updates,
            env_kwargs=env_kwargs,
        )
        self.runtime["iteration"] += updates
        self.runtime["ood_refinement_samples"].extend(Sample.copy_from(child) for child in children)
        self._update_online_state(children)
        return {
            "updates": float(updates),
            "transitions": float(updates * int(self.args.num_steps) * int(self.args.num_processes)),
            "hnsw_retrieved": float(self._last_retrieved),
            "archive_refills": float(len(children)),
        }
