import os
import sys
from copy import deepcopy

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "externals", "baselines"))
sys.path.append(os.path.join(base_dir, "externals", "pytorch-a2c-ppo-acktr-gail"))

import mo_gymnasium as mo_gym
import numpy as np
import torch
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.storage import RolloutStorage

from .dynamic_building import BUILDING_REGIMES
from .chlor_alkali_env import ChlorAlkaliEnv
from .envs import DynamicBuildingEnv, make_vec_envs
from .sample import Sample
from .sustaingym_wrappers import (
    DEFAULT_BUILDING_WEATHERS,
    DEFAULT_COGEN_RENEWABLES,
    DEFAULT_EV_PERIODS,
    DynamicCogenEnv,
    DynamicEVChargingEnv,
    DynamicSustainBuildingEnv,
)
from .metrics import compute_trace_objective_gaps, compute_trace_shift_metrics
from ..ood_protocol import load_env_config_json
from .shared_eval import normalize_episode_seeds


def _policy_dtype(actor_critic):
    try:
        return next(actor_critic.parameters()).dtype
    except StopIteration:
        return torch.get_default_dtype()


def _dynamic_env_kwargs(args):
    if args.env_name.startswith("evcharging"):
        env_kwargs = {
            "site": args.ev_site,
            "periods": tuple(args.ev_periods),
            "schedule": args.regime_schedule,
            "episodes_per_regime": args.episodes_per_regime,
            "moer_forecast_steps": args.ev_moer_forecast_steps,
            "project_action_in_env": not args.ev_disable_projection,
        }
    elif args.env_name.startswith("cogen"):
        env_kwargs = {
            "renewables": tuple(args.cogen_renewables),
            "schedule": args.regime_schedule,
            "episodes_per_regime": args.episodes_per_regime,
            "forecast_horizon": args.cogen_forecast_horizon,
            "forecast_noise_std": args.cogen_forecast_noise_std,
        }
    elif args.env_name.startswith("sustaingym_building"):
        env_kwargs = {
            "weathers": tuple(args.sustaingym_building_weathers),
            "schedule": args.regime_schedule,
            "episodes_per_regime": args.episodes_per_regime,
        }
    elif args.env_name.startswith("chlor_alkali"):
        env_kwargs = {
            "dynamic_price": True,
            "aging_dynamics": True,
            "schedule": args.regime_schedule,
            "episode_length": args.chlor_episode_length,
        }
    else:
        env_kwargs = {
            "regimes": BUILDING_REGIMES,
            "schedule": args.regime_schedule,
            "episodes_per_regime": args.episodes_per_regime,
        }
    if getattr(args, "env_config_json", ""):
        env_kwargs.update(load_env_config_json(getattr(args, "env_config_json")))
    return env_kwargs


def _snapshot_env_params(envs):
    return {
        "ob_rms": deepcopy(envs.ob_rms) if envs.ob_rms is not None else None,
        "ret_rms": deepcopy(envs.ret_rms) if envs.ret_rms is not None else None,
        "obj_rms": deepcopy(envs.obj_rms) if envs.obj_rms is not None else None,
    }


def _restore_env_params(envs, env_params):
    if env_params["ob_rms"] is not None:
        envs.venv.ob_rms = deepcopy(env_params["ob_rms"])
    if env_params["ret_rms"] is not None:
        envs.venv.ret_rms = deepcopy(env_params["ret_rms"])
    if env_params["obj_rms"] is not None:
        envs.venv.obj_rms = deepcopy(env_params["obj_rms"])


def _make_rollouts(args, envs, actor_critic, device):
    rollouts = RolloutStorage(
        num_steps=args.num_steps,
        num_processes=args.num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
        obj_num=args.obj_num,
    )
    obs = envs.reset().to(dtype=_policy_dtype(actor_critic))
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    return rollouts


def _collect_rollout(args, envs, actor_critic, rollouts):
    dtype = rollouts.obs.dtype
    policy_dtype = _policy_dtype(actor_critic)
    for step in range(args.num_steps):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step].to(dtype=policy_dtype),
                rollouts.recurrent_hidden_states[step],
                rollouts.masks[step],
            )

        obs, _, done, infos = envs.step(action)
        obs = obs.to(dtype=dtype)
        obj_tensor = torch.zeros([args.num_processes, args.obj_num], dtype=dtype)

        for idx, info in enumerate(infos):
            obj_tensor[idx] = torch.from_numpy(np.asarray(info["obj"], dtype=np.float64)).to(dtype=dtype)

        masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=dtype)
        bad_masks = torch.tensor(
            [[0.0] if "bad_transition" in info else [1.0] for info in infos], dtype=dtype
        )
        rollouts.insert(
            obs,
            recurrent_hidden_states,
            action,
            action_log_prob,
            value,
            obj_tensor,
            masks,
            bad_masks,
        )


def _finish_update(args, envs, actor_critic, rollouts):
    policy_dtype = _policy_dtype(actor_critic)
    with torch.no_grad():
        next_value = actor_critic.get_value(
            rollouts.obs[-1].to(dtype=policy_dtype),
            rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1],
        ).detach()

    rollouts.compute_returns(
        next_value,
        args.use_gae,
        args.gamma,
        args.gae_lambda,
        args.use_proper_time_limits,
    )


def _should_evaluate(update_idx, total_updates, interval):
    if total_updates <= 0:
        return False
    return ((update_idx + 1) % interval == 0) or (update_idx == total_updates - 1)


def _make_sample(envs, actor_critic, agent, base_metadata=None):
    sample = Sample(
        _snapshot_env_params(envs),
        deepcopy(actor_critic),
        deepcopy(agent),
        metadata=deepcopy(base_metadata or {}),
    )
    return sample


def _serialize_sample(sample):
    return {
        "env_params": deepcopy(sample.env_params),
        "actor_critic": deepcopy(sample.actor_critic).cpu(),
        "agent": deepcopy(sample.agent),
        "objs": None if sample.objs is None else np.asarray(sample.objs, dtype=np.float64).tolist(),
        "metadata": deepcopy(getattr(sample, "metadata", {})),
    }


def _deserialize_sample(payload):
    sample = Sample(
        payload["env_params"],
        payload["actor_critic"],
        payload["agent"],
        objs=payload.get("objs"),
        metadata=payload.get("metadata"),
    )
    return sample


def _objective_schedule(sample, args):
    order = list(sample.metadata.get("objective_order", range(args.obj_num)))
    repeats = np.asarray(
        sample.metadata.get("objective_repeats", np.ones(args.obj_num, dtype=np.int64)),
        dtype=np.int64,
    )
    schedule = []
    for obj in order:
        repeat_count = int(max(1, repeats[obj]))
        schedule.extend([int(obj)] * repeat_count)
    if not schedule:
        schedule = list(range(args.obj_num))
    return schedule


def _phase_budgets(total_updates, num_phases):
    base, remainder = divmod(total_updates, max(num_phases, 1))
    return [base + (1 if idx < remainder else 0) for idx in range(num_phases)]


def _build_extension_metadata(sample, objective_schedule, focus_objective, phase_idx):
    metadata = deepcopy(getattr(sample, "metadata", {}))
    metadata["objective_sequence"] = list(objective_schedule)
    metadata["focus_objective"] = int(focus_objective)
    metadata["phase_idx"] = int(phase_idx)
    return metadata


def _attach_dynamic_metadata(args, sample, trace):
    trace_metrics = compute_trace_shift_metrics(
        trace,
        args.eval_delta_weight,
        recovery_window=args.trace_recovery_window,
        use_regime_id=not args.strict_online_context,
        shift_threshold=args.online_shift_threshold,
        min_shift_gap=args.online_shift_min_gap,
    )
    sample.metadata["trace"] = trace
    sample.metadata["context_feature_names"] = list(trace.get("context_feature_names", []))
    sample.metadata["trace_metrics"] = trace_metrics
    sample.metadata["shift_gap_vector"] = compute_trace_objective_gaps(
        trace,
        recovery_window=args.trace_recovery_window,
        eval_delta_weight=args.eval_delta_weight,
        use_regime_id=not args.strict_online_context,
        shift_threshold=args.online_shift_threshold,
        min_shift_gap=args.online_shift_min_gap,
    )
    sample.metadata["regime_returns"] = trace.get("regime_records", [])


def _make_eval_env(args, env_name=None, env_kwargs=None):
    env_name = env_name or args.env_name
    env_kwargs = dict(env_kwargs or {})
    if env_name in {"building_3d", "building_3d_static", "building_3d_dynamic"}:
        return DynamicBuildingEnv(
            dimension="3d",
            regimes=tuple(
                env_kwargs.get(
                    "regimes",
                    BUILDING_REGIMES if "dynamic" in env_name else (BUILDING_REGIMES[0],),
                )
            ),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "cyclic"
            ),
            episodes_per_regime=env_kwargs.get("episodes_per_regime", args.episodes_per_regime),
        )
    if env_name in {"building_9d", "building_9d_static", "building_9d_dynamic"}:
        return DynamicBuildingEnv(
            dimension="9d",
            regimes=tuple(
                env_kwargs.get(
                    "regimes",
                    BUILDING_REGIMES if "dynamic" in env_name else (BUILDING_REGIMES[0],),
                )
            ),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "cyclic"
            ),
            episodes_per_regime=env_kwargs.get("episodes_per_regime", args.episodes_per_regime),
        )
    if env_name in {"sustaingym_building_static", "sustaingym_building_dynamic"}:
        return DynamicSustainBuildingEnv(
            weathers=tuple(env_kwargs.get("weathers", args.sustaingym_building_weathers)),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "cyclic"
            ),
            episodes_per_regime=env_kwargs.get("episodes_per_regime", args.episodes_per_regime),
        )
    if env_name in {"evcharging_static", "evcharging_dynamic"}:
        return DynamicEVChargingEnv(
            site=env_kwargs.get("site", args.ev_site),
            periods=tuple(env_kwargs.get("periods", args.ev_periods)),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "cyclic"
            ),
            episodes_per_period=env_kwargs.get("episodes_per_regime", args.episodes_per_regime),
            moer_forecast_steps=env_kwargs.get(
                "moer_forecast_steps", args.ev_moer_forecast_steps
            ),
            project_action_in_env=env_kwargs.get(
                "project_action_in_env", not args.ev_disable_projection
            ),
        )
    if env_name in {"cogen_static", "cogen_dynamic"}:
        return DynamicCogenEnv(
            renewables_magnitudes=tuple(env_kwargs.get("renewables", args.cogen_renewables)),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "cyclic"
            ),
            episodes_per_regime=env_kwargs.get("episodes_per_regime", args.episodes_per_regime),
            forecast_horizon=env_kwargs.get(
                "forecast_horizon", args.cogen_forecast_horizon
            ),
            forecast_noise_std=env_kwargs.get(
                "forecast_noise_std", args.cogen_forecast_noise_std
            ),
        )
    if env_name in {"chlor_alkali_static", "chlor_alkali_dynamic"}:
        return ChlorAlkaliEnv(
            dataset_path=env_kwargs.get("dataset_path"),
            train_path=env_kwargs.get("train_path"),
            dynamic_price=bool(env_kwargs.get("dynamic_price", "dynamic" in env_name)),
            aging_dynamics=bool(env_kwargs.get("aging_dynamics", "dynamic" in env_name)),
            episode_length=int(env_kwargs.get("episode_length", args.chlor_episode_length)),
            schedule=env_kwargs.get(
                "schedule", args.regime_schedule if "dynamic" in env_name else "sequential"
            ),
            allowed_regimes=tuple(env_kwargs.get("allowed_regimes", ())),
            allowed_regime_ids=tuple(env_kwargs.get("allowed_regime_ids", ())),
            num_regime_clusters=int(env_kwargs.get("num_regime_clusters", getattr(args, "chlor_regime_clusters", 12))),
            price_multiplier=float(env_kwargs.get("price_multiplier", 1.0)),
            price_offset=float(env_kwargs.get("price_offset", 0.0)),
            surrogate_seed=int(env_kwargs.get("surrogate_seed", 0)),
        )
    if args.cost_objective:
        return mo_gym.make(env_name, cost_objective=True, max_episode_steps=500)
    return mo_gym.make(env_name, max_episode_steps=500)


def evaluation(args, sample, return_trace=False, env_name=None, env_kwargs=None, episode_seeds=None):
    eval_env = _make_eval_env(args, env_name=env_name, env_kwargs=env_kwargs)
    objs = np.zeros(args.obj_num)
    ob_rms = sample.env_params["ob_rms"]
    policy = sample.actor_critic
    trace = {
        "obs": [],
        "obj": [],
        "context": [],
        "regime": [],
        "regime_id": [],
        "step": [],
        "episode": [],
        "context_feature_names": [],
    }
    regime_records = []
    global_step = 0
    eval_discount = 1.0 if getattr(args, "raw", False) else float(args.eval_gamma)
    policy_dtype = _policy_dtype(policy)
    seeds = normalize_episode_seeds(
        env_name or args.env_name,
        base_seed=int(getattr(args, "seed", 0)),
        episode_seeds=episode_seeds,
        num_episodes=int(getattr(args, "eval_num", 1)),
        seed_offset=int(getattr(args, "shared_regime_seed_offset", 0)),
    )
    with torch.no_grad():
        for eval_id, eval_seed in enumerate(seeds):
            eval_env.seed = int(eval_seed)
            reset_out = eval_env.reset(seed=int(eval_seed))
            if isinstance(reset_out, tuple):
                ob, current_info = reset_out
            else:
                ob, current_info = reset_out, {}
            done = False
            gamma = 1.0
            episode_obj = np.zeros(args.obj_num, dtype=np.float64)
            regime_name = None
            while not done:
                if args.ob_rms:
                    ob = np.clip(
                        (ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8),
                        -10.0,
                        10.0,
                    )
                obs_before = np.asarray(ob, dtype=np.float32)
                obs_tensor = torch.as_tensor(ob, dtype=policy_dtype).unsqueeze(0)
                _, action, _, _ = policy.act(
                    obs_tensor, None, None, deterministic=True
                )
                action = np.squeeze(action.cpu().numpy())
                ob, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                obj_vec = info.get("obj_raw", info.get("obj", reward))
                objs += gamma * obj_vec
                episode_obj += gamma * np.asarray(obj_vec, dtype=np.float64)
                regime_name = (
                    current_info.get("regime_name")
                    or current_info.get("period_name")
                    or current_info.get("weather")
                    or current_info.get("renewables_magnitude")
                    or info.get("regime_name")
                    or info.get("period_name")
                    or info.get("weather")
                    or info.get("renewables_magnitude")
                )
                if return_trace:
                    context_vec = np.asarray(
                        current_info.get("context_vector", info.get("context_vector", [])),
                        dtype=np.float32,
                    ).reshape(-1)
                    if (
                        not trace["context_feature_names"]
                        and current_info.get("context_feature_names")
                    ):
                        trace["context_feature_names"] = list(
                            current_info.get("context_feature_names", [])
                        )
                    trace["obs"].append(obs_before)
                    trace["obj"].append(np.asarray(obj_vec, dtype=np.float32))
                    if len(context_vec) > 0:
                        trace["context"].append(context_vec)
                    trace["regime"].append(
                        current_info.get("regime_name")
                        or current_info.get("period_name")
                        or current_info.get("weather")
                        or str(current_info.get("renewables_magnitude"))
                        or info.get("regime_name")
                        or info.get("period_name")
                        or info.get("weather")
                        or str(info.get("renewables_magnitude"))
                    )
                    trace["regime_id"].append(
                        current_info.get(
                            "regime_id",
                            current_info.get(
                                "period_id",
                                current_info.get("renewables_magnitude", 0),
                            ),
                        )
                    )
                    trace["step"].append(global_step)
                    trace["episode"].append(eval_id)
                    global_step += 1
                current_info = info
                gamma *= eval_discount
            regime_records.append(
                {
                    "eval_id": eval_id,
                    "regime": str(regime_name),
                    "return": episode_obj.astype(np.float32),
                }
            )
    eval_env.close()
    objs /= max(len(seeds), 1)
    if return_trace:
        trace["obs"] = np.asarray(trace["obs"], dtype=np.float32)
        trace["obj"] = np.asarray(trace["obj"], dtype=np.float32)
        trace["context"] = np.asarray(trace["context"], dtype=np.float32)
        trace["regime_id"] = np.asarray(trace["regime_id"])
        trace["step"] = np.asarray(trace["step"], dtype=np.int64)
        trace["episode"] = np.asarray(trace["episode"], dtype=np.int64)
        trace["regime_records"] = regime_records
        return objs, trace
    return objs


def MOPG_worker(
    args,
    task_id,
    task,
    device,
    iteration,
    num_updates,
    start_time,
    results_queue,
    done_event,
):
    scalarization = task.scalarization
    env_params, actor_critic, agent = (
        task.sample.env_params,
        task.sample.actor_critic,
        task.sample.agent,
    )
    actor_critic = actor_critic.to(device).to(dtype=torch.get_default_dtype())

    envs = make_vec_envs(
        env_name=args.env_name,
        seed=args.seed,
        num_processes=args.num_processes,
        gamma=args.gamma,
        log_dir=None,
        device=device,
        allow_early_resets=False,
        obj_rms=args.obj_rms,
        ob_rms=args.ob_rms,
        env_kwargs=_dynamic_env_kwargs(args),
    )
    _restore_env_params(envs, env_params)
    rollouts = _make_rollouts(args, envs, actor_critic, device)

    offspring_batch = []
    start_iter, final_iter = iteration, num_updates

    for j in range(start_iter, final_iter):
        torch.manual_seed(j)
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer,
                j * args.lr_decay_ratio,
                final_iter,
                args.lr,
            )

        _collect_rollout(args, envs, actor_critic, rollouts)
        _finish_update(args, envs, actor_critic, rollouts)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None
        agent.update(rollouts, scalarization, obj_rms_var)
        rollouts.after_update()

        if _should_evaluate(j - start_iter, final_iter - start_iter, args.rl_eval_interval):
            sample = _make_sample(
                envs,
                actor_critic,
                agent,
                base_metadata=getattr(task.sample, "metadata", {}),
            )
            objs, trace = evaluation(args, sample, return_trace=True)
            sample.objs = objs
            _attach_dynamic_metadata(args, sample, trace)
            sample.metadata["train_iteration"] = int(j + 1)
            sample.metadata["worker_mode"] = "warmup"
            offspring_batch.append(sample)

    envs.close()
    results_queue.put(
        {
            "task_id": task_id,
            "offspring_batch": [_serialize_sample(sample) for sample in offspring_batch],
            "done": True,
        }
    )
    done_event.wait()


def Extension_worker(
    args,
    sample_id,
    sample,
    device,
    iteration,
    num_updates,
    start_time,
    results_queue,
    done_event,
):
    total_updates = int(max(1, num_updates * args.obj_num))
    objective_schedule = _objective_schedule(sample, args)
    phase_budgets = _phase_budgets(total_updates, len(objective_schedule))

    working_sample = Sample.copy_from(sample)
    working_sample.actor_critic = working_sample.actor_critic.to(device).to(dtype=torch.get_default_dtype())
    offspring_batch = []
    completed_updates = 0

    for phase_idx, (focus_objective, phase_updates) in enumerate(
        zip(objective_schedule, phase_budgets)
    ):
        if phase_updates <= 0:
            continue

        envs = make_vec_envs(
            env_name=args.env_name,
            seed=args.seed + phase_idx,
            num_processes=args.num_processes,
            gamma=args.gamma,
            log_dir=None,
            device=device,
            allow_early_resets=False,
            obj_rms=args.obj_rms,
            ob_rms=args.ob_rms,
            env_kwargs=_dynamic_env_kwargs(args),
        )
        _restore_env_params(envs, working_sample.env_params)
        rollouts = _make_rollouts(args, envs, working_sample.actor_critic, device)

        phase_metadata = _build_extension_metadata(
            sample, objective_schedule, focus_objective, phase_idx
        )
        for local_update in range(phase_updates):
            absolute_update = completed_updates + local_update
            torch.manual_seed(iteration + absolute_update)

            _collect_rollout(args, envs, working_sample.actor_critic, rollouts)
            _finish_update(args, envs, working_sample.actor_critic, rollouts)

            obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None
            working_sample.agent.ipo_update(
                rollouts,
                focus_objective,
                args.obj_num,
                obj_rms_var,
            )
            rollouts.after_update()

            if _should_evaluate(local_update, phase_updates, args.rl_eval_interval):
                eval_sample = _make_sample(
                    envs,
                    working_sample.actor_critic,
                    working_sample.agent,
                    base_metadata=phase_metadata,
                )
                objs, trace = evaluation(args, eval_sample, return_trace=True)
                eval_sample.objs = objs
                _attach_dynamic_metadata(args, eval_sample, trace)
                eval_sample.metadata["train_iteration"] = int(iteration + absolute_update + 1)
                eval_sample.metadata["phase_budget"] = int(phase_updates)
                eval_sample.metadata["worker_mode"] = "extension"
                offspring_batch.append(eval_sample)

        working_sample = _make_sample(
            envs,
            working_sample.actor_critic,
            working_sample.agent,
            base_metadata=phase_metadata,
        )
        envs.close()
        completed_updates += phase_updates

    results_queue.put(
        {
            "task_id": sample_id,
            "offspring_batch": [_serialize_sample(sample) for sample in offspring_batch],
            "done": True,
        }
    )
    done_event.wait()
