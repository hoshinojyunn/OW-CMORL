import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals', 'baselines'))
sys.path.append(os.path.join(base_dir, 'externals', 'pytorch-a2c-ppo-acktr-gail'))

from copy import deepcopy

import gym
import numpy as np
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy

from .sample import Sample
from .utils import generate_weights_batch_dfs
from .scalarization_methods import DynamicWeightedSumScalarization
from .mopg import evaluation, _make_eval_env, _dynamic_env_kwargs
from .envs import make_vec_envs

'''
initialize_warm_up_batch: method to generate tasks in the warm-up stage.
Each task is a pair of an initial random policy and an evenly distributed optimization weight.
The optimization weight is represented by a weighted-sum scalarization function.
'''
def initialize_warm_up_batch(args, device):
    # using evenly distributed weights for warm-up stage
    weights_batch = []
    generate_weights_batch_dfs(0, args.obj_num, args.min_weight, args.max_weight, args.delta_weight, [], weights_batch)
    sample_batch = []
    scalarization_batch = []

    #temp_env = gym.make(args.env_name) # temp_env is only used for initialization
    temp_env = _make_eval_env(args)
    temp_env.seed = seed=args.seed

    for weights in weights_batch:
        actor_critic = Policy(
            temp_env.observation_space.shape,
            temp_env.action_space,
            base_kwargs={'layernorm' : args.layernorm},
            obj_num=args.obj_num)

        actor_critic.to(device).double()

        if args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                args.t,
                args.beta,
                lr=args.lr,
                eps=1e-5,
                max_grad_norm=args.max_grad_norm)
        else:
            # NOTE: other algorithms are not supported yet
            raise NotImplementedError
    
        envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                            gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                            obj_rms=args.obj_rms, ob_rms=args.ob_rms, env_kwargs=_dynamic_env_kwargs(args))
        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None
        envs.close()

        scalarization = DynamicWeightedSumScalarization(num_objs = args.obj_num, weights = weights)

        sample = Sample(env_params, actor_critic, agent, optgraph_id = -1)
        # Warm-up only needs the initial frontier estimate. Full dynamic traces
        # are attached later on offspring policies during training/evaluation.
        objs = evaluation(args, sample, return_trace=False)
        sample.objs = objs
        sample.metadata["base_weights"] = np.asarray(weights, dtype=np.float64)

        sample_batch.append(sample)
        scalarization_batch.append(scalarization)
    
    temp_env.close()

    return sample_batch, scalarization_batch
