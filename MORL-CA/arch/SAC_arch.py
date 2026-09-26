import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.logger import configure

import matplotlib.pyplot as plt
from scipy.spatial import distance
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pyDOE2 import lhs
from arch.SAC_policy import *
from arch.Distributed_PER import *

import gym
import os
import re
from enum import Enum

class MOO_Problem:

    def __init__(self, objectives, constraints, bounds):

        self.objectives = objectives
        self.constraints = constraints
        self.bounds = bounds
        self.obj_dim = len(objectives)
        self.constraint_dim = len(constraints)
        self.state_dim = len(bounds)

    def evaluate(self, x):

        x = np.clip(x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
        fs = np.array([f(x) for f in self.objectives])
        constraint_vals = np.array([c(x) for c in self.constraints])
        return fs, constraint_vals

class PymooMOO(ElementwiseProblem):
    def __init__(self, moo_problem: MOO_Problem):
        super().__init__(n_var=moo_problem.state_dim,
                         n_obj=moo_problem.obj_dim,
                         n_constr=moo_problem.constraint_dim,
                         xl=np.array([b[0] for b in moo_problem.bounds]),
                         xu=np.array([b[1] for b in moo_problem.bounds]))
        self.moo = moo_problem

    def _evaluate(self, x, out, *args, **kwargs):
        f_vals, c_vals = self.moo.evaluate(x)
        out["F"] = f_vals
        out["G"] = c_vals

def extract_XO(filename, num=100):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    X_pattern = re.compile(r"X=\s*\[([^\]]+)\]")
    O_pattern = re.compile(r"F=\s*\[([^\]]+)\]")

    objective_matches = O_pattern.findall(content)

    objectives = []
    for match in objective_matches:
        obj_values = list(map(float, match.strip().split()))
        objectives.append(obj_values)
    obj_array = np.array(objectives)
    Xs = []
    X_matches = X_pattern.findall(content)
    for match in X_matches:
        X_values = list(map(float, match.strip().split()))
        Xs.append(X_values)
    X_array = np.array(Xs)

    if len(Xs) <= num:
        return X_array, obj_array

    idx = np.linspace(0, X_array.shape[0]-1, num).astype(int)

    return X_array[idx], obj_array[idx]

def archive_reduction(all_X, all_objs, target=2000):


    N = all_objs.shape[0]
    if N == 0:
        return all_X, all_objs

    if N <= target:
        return all_X.copy(), all_objs.copy()

    k = int(np.floor(np.sqrt(N)))
    k = max(1, min(k, N - 1))

    objs = np.asarray(all_objs, dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    objs_norm = (objs - mins) / ranges

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean', n_jobs=-1)
        nn.fit(objs_norm)
        distances, indices = nn.kneighbors(objs_norm, return_distance=True)

        sigma_k = distances[:, k]
    except Exception:

        print("Warning: sklearn not available or failed; using fallback O(N^2) method. This may be slow.")

        sigma_k = np.empty(N, dtype=float)
        for i in range(N):

            diff = objs_norm - objs_norm[i]
            dist = np.linalg.norm(diff, axis=1)

            dist[i] = np.inf

            kth = np.partition(dist, k)[k]
            sigma_k[i] = kth

    density = 1.0 / (sigma_k + 2.0)

    keep_count = target

    order = np.argsort(density)
    keep_idx = order[:keep_count]

    reducted_X = np.asarray(all_X)[keep_idx]
    reducted_objs = np.asarray(all_objs)[keep_idx]

    return reducted_X, reducted_objs

class SB3MOOEnv(gym.Env):

    metadata = {'render.modes': []}

    def __init__(self, moo_problem: MOO_Problem,
                 weight: np.ndarray = None,
                 weight_dist: dict = None,
                 max_steps: int = 1000,
                 penalty_coeff: float = 1e2):
        super().__init__()
        self.moo = moo_problem
        self.fixed_weight = np.array(weight, dtype=np.float32) if weight is not None else None
        self.weight_dist = weight_dist
        self.penalty_coeff = penalty_coeff
        self.state_low = np.array([b[0] for b in moo_problem.bounds])
        self.state_high = np.array([b[1] for b in moo_problem.bounds])

        self.observation_space = gym.spaces.Box(low=self.state_low, high=self.state_high, dtype=np.float32)
        ranges = np.array([b[1] - b[0] for b in moo_problem.bounds], dtype=np.float32)
        act_range = ranges / max_steps

        self.action_space = gym.spaces.Box(low=-act_range, high=act_range, shape=(moo_problem.state_dim, ), dtype=np.float32)

        self.x = None
        self.weight = None
        self.curr_step = 0
        self.max_steps = max_steps

    def sample_weight(self):
        if self.weight_dist and self.weight_dist.get('type') == 'dirichlet':
            alpha = np.array(self.weight_dist.get('alpha', [1]*self.moo.obj_dim))
            return np.random.dirichlet(alpha).astype(np.float32)
        return None

    def reset(self, start_point=None):
        self.curr_step = 0
        if self.fixed_weight is not None:
            self.weight = self.fixed_weight
        else:
            sampled = self.sample_weight()
            self.weight = sampled if sampled is not None else np.ones(self.moo.obj_dim, dtype=np.float32)/self.moo.obj_dim
        low = np.array([b[0] for b in self.moo.bounds], dtype=np.float32)
        high = np.array([b[1] for b in self.moo.bounds], dtype=np.float32)
        self.x = start_point if start_point is not None else np.random.uniform(low, high)
        return self.x

    def step(self, action):
        x_new = self.x + action
        x_new = np.clip(x_new, self.state_low, self.state_high)
        f_vals, c_vals = self.moo.evaluate(x_new)
        violation = np.sum(c_vals)

        reward_vec = -f_vals.astype(np.float32)

        reward = (-f_vals).sum()

        self.curr_step += 1
        self.x = x_new

        done = self.curr_step >= self.max_steps

        info = {'f_vals': f_vals, 'c_vals': c_vals, 'violation': violation, 'reward_vec': reward_vec}
        return self.x.astype(np.float32), reward, done, info

class BufferClass(Enum):
    DPER = 1
    FIFO = 2
    PER = 3

class SAC_Weighted_Arch:
    def __init__(self, problem: MOO_Problem,
                 weight_dist: dict = None,
                 sac_params: dict = None,
                 net_arch_hidden_dim=64,
                 aow_hidden_dim=64,
                 penalty_coeff: float = 1e3,
                 gamma=0.99,
                 tau=0.005,
                 buffer_class: BufferClass=BufferClass.DPER,
                 tensorboard_log='./tensorboard_logs/sac/',
                 aow_model_path=None):
        self.problem = problem
        self.weight_dist = weight_dist if weight_dist is not None else {'type':'dirichlet', 'alpha': [1] * problem.obj_dim}
        self.penalty_coeff = penalty_coeff
        self.tensorboard_log = tensorboard_log
        self.aow_model_path = aow_model_path

        self.train_env = SB3MOOEnv(self.problem, weight=None, weight_dist=self.weight_dist, max_steps=1000, penalty_coeff=self.penalty_coeff)
        self.policy_kwargs = {
            "state_dim": problem.state_dim,
            "feature_dim": net_arch_hidden_dim,
            "obj_dim": problem.obj_dim,
            "aow_model_path": aow_model_path,
            "aow_hidden_dim": aow_hidden_dim
        }
        self.buffer_class = buffer_class

        self.gamma = gamma
        self.tau = tau
        default_sac = {
            'learning_rate':3e-4,
            'buffer_size': 30000,
            'batch_size':64,
            'gamma':gamma,
            'train_freq':1,
            'verbose':1,
        }
        if sac_params:
            default_sac.update(sac_params)
        if self.buffer_class == BufferClass.DPER:
          replay_buffer_kwargs = {
            'obj_dim': problem.obj_dim,
            'clusters': 2**problem.obj_dim-1,
            'global_rank_update_freq': 500
          }
          self.model = SAC(
              policy=MO_SACPolicy,
              env=self.train_env,
              policy_kwargs=self.policy_kwargs,
              replay_buffer_class=DistributedPER,
              replay_buffer_kwargs=replay_buffer_kwargs,
              tensorboard_log=tensorboard_log,
              **default_sac
          )
        elif self.buffer_class == BufferClass.FIFO:
          self.model = SAC(
              policy=MO_SACPolicy,
              env=self.train_env,
              policy_kwargs=self.policy_kwargs,
              replay_buffer_class=FIFOReplayBuffer,
              tensorboard_log=tensorboard_log,
              **default_sac
          )
        elif self.buffer_class == BufferClass.PER:
            self.model = SAC(
              policy=MO_SACPolicy,
              env=self.train_env,
              policy_kwargs=self.policy_kwargs,
              tensorboard_log=tensorboard_log,
              **default_sac
          )
        self.target_entropy = -np.prod(self.problem.state_dim).item()
        self.log_ent_coef = torch.tensor(np.log(0.01), requires_grad=True)
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=3e-4)
        self.episode_count = 0

    def learn(self, initial_population_size=100, steps_per_episode=1000, batch_size=64, use_nsga2=False):
        if self.buffer_class == BufferClass.DPER:
            self._DPER_learn(initial_population_size, steps_per_episode, batch_size)
        elif self.buffer_class == BufferClass.FIFO:
            self._FIFOER_learn(initial_population_size, steps_per_episode, batch_size, use_nsga2)
        elif self.buffer_class == BufferClass.PER:
            self._PER_learn(initial_population_size, steps_per_episode, batch_size)

    def _DPER_learn(self, episodes=100, steps_per_episode=1000, batch_size=64):
        if not hasattr(self.model, "_logger"):
          self.model._logger = self.make_logger(tensorboard_log=self.tensorboard_log, name="SAC")

        self.train_env.max_steps = steps_per_episode
        samples = lhs(self.problem.state_dim, samples=episodes, criterion='center', random_state=42)
        bounds = np.array(self.problem.bounds)

        candidates = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
        ep_rewards = []
        ep_lengths = []
        for ep, candidate in enumerate(candidates):
          obs = self.train_env.reset(start_point=candidate)
          done = False
          ep_reward = 0
          ep_length = 0
          while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, info = self.train_env.step(action)

            self.model.replay_buffer.add(
                obs,
                new_obs,
                action,

                info['reward_vec'],
                done,
                info
            )
            obs = new_obs
            ep_reward += reward
            ep_length += 1
            if len(self.model.replay_buffer.buffer) >= batch_size:
              actor_loss, weighted_q, critic_losses = self.train(gamma=0.99, batch_size=batch_size)

              if not hasattr(self, '_train_metrics_buffer'):
                self._train_metrics_buffer = {'actor_loss': [], 'weighted_q': [], 'critic_losses': []}
              self._train_metrics_buffer['actor_loss'].append(actor_loss)
              self._train_metrics_buffer['weighted_q'].append(weighted_q)
              self._train_metrics_buffer['critic_losses'].append(critic_losses)

          ep_rewards.append(ep_reward)
          ep_lengths.append(ep_length)
          self.model.logger.record("rollout/ep_rew_mean", np.mean(ep_rewards))
          self.model.logger.record("rollout/ep_len_mean", np.mean(ep_lengths))

          if hasattr(self, '_train_metrics_buffer') and self._train_metrics_buffer['actor_loss']:
            buf = self._train_metrics_buffer
            self.model.logger.record("train/actor_loss", np.mean(buf['actor_loss']))
            self.model.logger.record("train/weighted_q", np.mean(buf['weighted_q']))
            n_critics = len(buf['critic_losses'][0])
            for i in range(n_critics):
              cl_mean = np.mean([cl[i] for cl in buf['critic_losses']])
              self.model.logger.record(f"train/critic_{i}_loss", cl_mean)
            self._train_metrics_buffer = {'actor_loss': [], 'weighted_q': [], 'critic_losses': []}
          self.model.logger.dump(step=ep)

    def _FIFOER_learn(self, initial_pop_size=100, steps_per_episode=1000, batch_size=64, use_nsga2=False):
        if not hasattr(self.model, "_logger"):
          self.model._logger = self.make_logger(tensorboard_log=self.tensorboard_log, name="SAC")

        self.train_env.max_steps = steps_per_episode
        candidates = None
        if use_nsga2:
            algo = NSGA2(pop_size=initial_pop_size)
            res = minimize(PymooMOO(self.problem),
                    algo,
                    termination=('n_gen', 100),
                    seed=1,
                    save_history=False,
                    verbose=False)
            candidates = res.X
        else:
            samples = lhs(self.problem.state_dim, samples=initial_pop_size, criterion='center', random_state=42)
            bounds = np.array(self.problem.bounds)

            candidates = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])

        ep_rewards = {}
        for i in range(self.problem.obj_dim):
            ep_rewards[f'reward_{i}'] = []
        ep_lengths = []
        ep_sum_rewards = []
        for ep, candidate in enumerate(candidates):
          obs = self.train_env.reset(start_point=candidate)
          done = False
          ep_length = 0
          ep_reward = {}
          ep_sum_reward = 0
          for i in range(self.problem.obj_dim):
            ep_reward[f'reward_{i}'] = 0
          while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, info = self.train_env.step(action)

            self.model.replay_buffer.add(
                obs,
                new_obs,
                action,

                info['reward_vec'],
                done,
                info
            )
            obs = new_obs
            for i, r in enumerate(info['reward_vec']):
                ep_reward[f'reward_{i}'] += r
            ep_length += 1
            ep_sum_reward += reward
            if len(self.model.replay_buffer.buffer) >= batch_size:
              actor_loss, weighted_q, critic_losses = self.train(gamma=self.gamma, tau=self.tau, batch_size=batch_size)
              if not hasattr(self, '_train_metrics_buffer'):
                self._train_metrics_buffer = {'actor_loss': [], 'weighted_q': [], 'critic_losses': []}
              self._train_metrics_buffer['actor_loss'].append(actor_loss)
              self._train_metrics_buffer['weighted_q'].append(weighted_q)
              self._train_metrics_buffer['critic_losses'].append(critic_losses)

          for i, reward in enumerate(info['reward_vec']):
            ep_rewards[f'reward_{i}'].append(ep_reward[f'reward_{i}'])
          ep_lengths.append(ep_length)
          ep_sum_rewards.append(ep_sum_reward)
          for k, v in ep_rewards.items():
            self.model.logger.record(f"rollout/{k}", v[-1])
          self.model.logger.record(f"rollout/ep_rew_mean", np.mean(ep_sum_rewards))
          self.model.logger.record("rollout/ep_len_mean", np.mean(ep_lengths))
          if hasattr(self, '_train_metrics_buffer') and self._train_metrics_buffer['actor_loss']:
            buf = self._train_metrics_buffer
            self.model.logger.record("train/actor_loss", np.mean(buf['actor_loss']))
            self.model.logger.record("train/weighted_q", np.mean(buf['weighted_q']))
            n_critics = len(buf['critic_losses'][0])
            for i in range(n_critics):
              cl_mean = np.mean([cl[i] for cl in buf['critic_losses']])
              self.model.logger.record(f"train/critic_{i}_loss", cl_mean)
            self._train_metrics_buffer = {'actor_loss': [], 'weighted_q': [], 'critic_losses': []}
          self.model.logger.dump(step=self.episode_count)
          self.episode_count += 1

    def _PER_learn(self, episodes=100, steps_per_episode=1000, batch_size=64, use_nsga2=False):
        if not hasattr(self.model, "_logger"):
          self.model._logger = self.make_logger(tensorboard_log=self.tensorboard_log, name="SAC")

        self.train_env.max_steps = steps_per_episode
        candidates = None
        if use_nsga2:
            algo = NSGA2(pop_size=episodes)
            res = minimize(PymooMOO(self.problem),
                    algo,
                    termination=('n_gen', 100),
                    seed=1,
                    save_history=False,
                    verbose=False)
            candidates = res.X
        else:
            samples = lhs(self.problem.state_dim, samples=episodes, criterion='center', random_state=42)
            bounds = np.array(self.problem.bounds)

            candidates = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
        ep_rewards = {}
        for i in range(self.problem.obj_dim):
            ep_rewards[f'reward_{i}'] = []
        ep_lengths = []
        ep_sum_rewards = []
        for ep, candidate in enumerate(candidates):
          obs = self.train_env.reset(start_point=candidate)
          done = False
          ep_length = 0
          ep_reward = {}
          ep_sum_reward = 0
          for i in range(self.problem.obj_dim):
            ep_reward[f'reward_{i}'] = 0
          while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            new_obs, reward, done, info = self.train_env.step(action)

            self.model.replay_buffer.add(
                obs,
                new_obs,
                action,
                reward,
                done,
                [{}]
            )
            obs = new_obs
            for i, r in enumerate(info['reward_vec']):
                ep_reward[f'reward_{i}'] += r
            ep_length += 1
            ep_sum_reward += reward
            if self.model.replay_buffer.size() >= batch_size:
              self.model.train(gradient_steps=1, batch_size=batch_size)

          for i, reward in enumerate(info['reward_vec']):
            ep_rewards[f'reward_{i}'].append(ep_reward[f'reward_{i}'])
          ep_lengths.append(ep_length)
          ep_sum_rewards.append(ep_sum_reward)
          for k, v in ep_rewards.items():
            self.model.logger.record(f"rollout/{k}", v[-1])
          self.model.logger.record(f"rollout/ep_rew_mean", np.mean(ep_sum_rewards))
          self.model.logger.record("rollout/ep_len_mean", np.mean(ep_lengths))
          if hasattr(self, '_train_metrics_buffer') and self._train_metrics_buffer['actor_loss']:
            buf = self._train_metrics_buffer
            self.model.logger.record("train/actor_loss", np.mean(buf['actor_loss']))
            self.model.logger.record("train/weighted_q", np.mean(buf['weighted_q']))
            n_critics = len(buf['critic_losses'][0])
            for i in range(n_critics):
              cl_mean = np.mean([cl[i] for cl in buf['critic_losses']])
              self.model.logger.record(f"train/critic_{i}_loss", cl_mean)
            self._train_metrics_buffer = {'actor_loss': [], 'weighted_q': [], 'critic_losses': []}
          self.model.logger.dump(step=ep)

    def save(self, path=None):
        if path is None:
            raise ValueError('In SAC_Weighted_Arch save: path should not be None')
        self.model.save(path)

    def load(self, path=None):
        if path is None:
            raise ValueError('In SAC_Weighted_Arch load: path should not be None')
        self.model = self.model.load(path)

    def make_logger(self, tensorboard_log: str, name: str = "SAC") -> Logger:

      existing = [
          f for f in os.listdir(tensorboard_log)
          if os.path.isdir(os.path.join(tensorboard_log, f)) and re.match(f"{re.escape(name)}_\\d+", f)
      ]
      run_id = len(existing) + 1
      full_path = os.path.join(tensorboard_log, f"{name}_{run_id}")
      os.makedirs(full_path, exist_ok=True)

      return configure(folder=full_path, format_strings=["tensorboard", "stdout"])

    def train(self, gamma=0.99, tau=0.005, batch_size=64):
        replay_data = self.model.replay_buffer.sample(batch_size=batch_size)
        obs, next_obs, actions, reward_vec, dones = replay_data.observations, replay_data.next_observations, replay_data.actions, replay_data.rewards, replay_data.dones
        critic_losses = self._update_critics(obs, next_obs, actions, reward_vec, dones, gamma=gamma)
        actor_loss, weighted_q = self._update_actor(obs)
        self._update_target_critic(tau)
        return actor_loss, weighted_q, [l.item() for l in critic_losses]

    def _update_critics(self, obs, next_obs, actions, reward_vec, dones, gamma=0.99):

        with torch.no_grad():
            next_actions, next_log_prob = self.model.policy.action_log_prob(next_obs)
            next_features = self.model.actor.features_extractor(next_obs)
            q_target_vals = []
            for qf_target in self.model.policy.qf_target_list:
                q_next = qf_target(torch.cat([next_features, next_actions], dim=1))
                q_next = q_next - self.log_ent_coef * next_log_prob.reshape(-1, 1)
                q_target_vals.append(q_next)
            q_target_vals = torch.cat(q_target_vals, dim=1)
            q_target = reward_vec + gamma * (1 - dones) * q_target_vals

        q_pred = []
        critic_losses = []
        for i, qf in enumerate(self.model.policy.qf_list):
            features = self.model.actor.features_extractor(obs).detach()
            q_val = qf(torch.cat([features, actions], dim=1))
            q_pred.append(q_val)
            loss = F.mse_loss(q_val, q_target[:, i:i+1])
            critic_losses.append(loss)
            self.model.policy.qf_optim_list[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.qf_list[i].parameters(), max_norm=1.0)
            self.model.policy.qf_optim_list[i].step()
        return critic_losses

    def _update_actor(self, obs):

        features = self.model.actor.features_extractor(obs)

        aow_weights = self.model.policy.aow_net(obs)

        action_pi, log_prob = self.model.policy.action_log_prob(obs)
        q_vals = [qf(torch.cat([features, action_pi], dim=1)) for qf in self.model.policy.qf_list]
        q_vals = torch.cat(q_vals, dim=1)

        weighted_q = -(q_vals * aow_weights).sum(dim=1)

        ent_coef = torch.exp(self.log_ent_coef.detach())

        actor_loss = (weighted_q + ent_coef * log_prob).mean()

        self.model.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.model.actor.optimizer.step()

        ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        return actor_loss.item(), weighted_q.mean().item()

    def _update_target_critic(self, tau):

        with torch.no_grad():
            for qf, qf_target in zip(self.model.policy.qf_list, self.model.policy.qf_target_list):
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                    target_param.data.mul_(1 - tau)
                    target_param.data.add_(tau * param.data)

    def optimize(self, start_points_sample=100, n_gen=200, max_steps=1000, GE_from_file=None, seed=1, max_size=30000):

        self.opt_env = SB3MOOEnv(self.problem, weight=None, weight_dist=self.weight_dist, max_steps=max_steps, penalty_coeff=self.penalty_coeff)

        if GE_from_file is None:
            algo = NSGA2(pop_size=start_points_sample)
            res = minimize(PymooMOO(self.problem),
                    algo,
                    termination=('n_gen', n_gen),
                    seed=seed,
                    save_history=False,
                    verbose=False)
            candidates, objs, constraints = res.X, res.F, res.G
        else:
            print(f'Load pareto solutions from {GE_from_file}')
            X, O = extract_XO(GE_from_file, num=start_points_sample)
            candidates, objs = X, O
            constraints = []
            for candidate in candidates:
                obj, constraint = self.problem.evaluate(candidate)
                constraints.append(constraint)
        ARCHIVE_X, ARCHIVE_O, ARCHIVE_C = [], [], []
        for i, x0 in enumerate(candidates):
          obs = self.opt_env.reset(x0)
          done = False
          traj_X, traj_O, traj_C = [], [], []
          obj0, c0 = self.problem.evaluate(x0)
          traj_X.append(x0.copy())
          traj_O.append(obj0)
          traj_C.append(c0)
          while not done:
              with torch.no_grad():

                action, _ = self.model.predict(obs, deterministic=True)
                new_obs, reward, done, info = self.opt_env.step(action)
                obs = new_obs

                reward_vec, constraint = info['reward_vec'], info['c_vals']

                traj_X.append(self.opt_env.x.copy()), traj_O.append(-reward_vec), traj_C.append(constraint)
          traj_X = np.asarray(traj_X)
          traj_O = np.asarray(traj_O)
          traj_C = np.asarray(traj_C)
          idx = self._filter_feasibility_first(traj_O, traj_C)
          ARCHIVE_X.append(traj_X[idx])
          ARCHIVE_O.append(traj_O[idx])
          ARCHIVE_C.append(traj_C[idx])
          print(f'Optimization schedule: {i+1}/{start_points_sample}')

        ARCHIVE_X = np.vstack(ARCHIVE_X)
        ARCHIVE_O = np.vstack(ARCHIVE_O)
        ARCHIVE_C = np.vstack(ARCHIVE_C)

        pareto_X, pareto_O, pareto_C = self._spea2_compress_archive(
            ARCHIVE_X, ARCHIVE_O, ARCHIVE_C, max_size
        )

        return pareto_X, pareto_O, pareto_C

    def optimize_only_nsga2(self, start_points_sample=100, n_gen=200):
      algo = NSGA2(pop_size=start_points_sample)
      res = minimize(PymooMOO(self.problem),
              algo,
              termination=('n_gen', n_gen),
              seed=1,
              save_history=False,
              verbose=False)
      candidates, objs, constraints = res.X, res.F, res.G
      return np.array(candidates), np.array(objs), np.array(constraints)

    def _filter_feasibility_first(self, objectives: np.ndarray, constraints: np.ndarray, infeasible_quantile: float = 0.25):
      feasible = np.all(constraints == 0, axis=1)
      if feasible.any():
          candidates = np.where(feasible)[0]
      else:

          violations = np.sum(np.maximum(constraints, 0), axis=1)
          thresh = np.quantile(violations, infeasible_quantile)
          candidates = np.where(violations <= thresh)[0]

      sub_objs = objectives[candidates]

      nds = NonDominatedSorting()
      front = nds.do(sub_objs, only_non_dominated_front=True)

      return candidates[front]

    def _spea2_compress_archive(self, X, O, C, max_size, infeasible_quantile: float = 0.25):

      from sklearn.neighbors import NearestNeighbors

      feasible = np.all(C == 0, axis=1)
      if feasible.any():
          idx = np.where(feasible)[0]
      else:
          violations = np.sum(np.maximum(C, 0), axis=1)
          thresh = np.quantile(violations, infeasible_quantile)
          idx = np.where(violations <= thresh)[0]

      Xf, Of, Cf = X[idx], O[idx], C[idx]
      N = len(Xf)

      if N <= max_size:
          return Xf, Of, Cf

      k_neighbors = int(np.sqrt(N))
      nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, N-1)).fit(Of)
      distances, _ = nbrs.kneighbors(Of)

      density = 1.0 / (distances[:, -1] + 1e-6)

      selected_idx = np.argsort(density)[:max_size]

      return Xf[selected_idx], Of[selected_idx], Cf[selected_idx]

    def _spea2_full(self, X, O, C, max_size, infeasible_quantile: float = 0.25):

      from sklearn.neighbors import NearestNeighbors
      from pymoo.algorithms.moo.spea2 import SPEA2Survival
      from pymoo.core.population import Population

      feasible = np.all(C == 0, axis=1)
      if feasible.any():
          idx = np.where(feasible)[0]
      else:
          violations = np.sum(np.maximum(C, 0), axis=1)
          thresh = np.quantile(violations, infeasible_quantile)
          idx = np.where(violations <= thresh)[0]
      Xf, Of, Cf = X[idx], O[idx], C[idx]
      pop = Population.new("X", Xf, "F", Of, "C", Cf)
      class DummyProblem(Problem):
        def __init__(self, n_var, n_obj, n_constr=0):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr,
                           xl=np.full(n_var, -np.inf), xu=np.full(n_var, np.inf))
        def _evaluate(self, X, out, *args, **kwargs):

            pass
      survival = SPEA2Survival()
      problem = DummyProblem(n_var=Xf.shape[1], n_obj=Of.shape[1], n_constr=C.shape[1])
      survivors = survival.do(problem=problem, pop=pop, n_survive=max_size)

      X_selected = np.array([ind.X for ind in survivors])
      O_selected = np.array([ind.F for ind in survivors])
      C_selected = np.array([ind.get("C") for ind in survivors])
      return X_selected, O_selected, C_selected

def _filter_feasibility_first(objectives: np.ndarray, constraints: np.ndarray, infeasible_quantile: float = 0.25):
    feasible = np.all(constraints == 0, axis=1)
    if feasible.any():
        candidates = np.where(feasible)[0]
    else:

        violations = np.sum(np.maximum(constraints, 0), axis=1)
        thresh = np.quantile(violations, infeasible_quantile)
        candidates = np.where(violations <= thresh)[0]

    sub_objs = objectives[candidates]

    nds = NonDominatedSorting()
    front = nds.do(sub_objs, only_non_dominated_front=True)

    return candidates[front]

def _spea2_compress_archive(X, O, C, max_size, infeasible_quantile=0.25):

    from sklearn.neighbors import NearestNeighbors

    feasible = np.all(C == 0, axis=1)
    if feasible.any():
        idx = np.where(feasible)[0]
    else:
        violations = np.sum(np.maximum(C, 0), axis=1)
        thresh = np.quantile(violations, infeasible_quantile)
        idx = np.where(violations <= thresh)[0]

    Xf, Of, Cf = X[idx], O[idx], C[idx]
    N = len(Xf)

    if N <= max_size:
        return Xf, Of, Cf

    k_neighbors = int(np.sqrt(N))
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, N-1)).fit(Of)
    distances, _ = nbrs.kneighbors(Of)

    density = 1.0 / (distances[:, -1] + 1e-6)

    selected_idx = np.argsort(density)[:max_size]

    return Xf[selected_idx], Of[selected_idx], Cf[selected_idx]

def random_optimize(problem: MOO_Problem, start_points_sample=100, n_gen=200, max_steps=1000, GE_from_file=None, seed=1, max_size=30000, infeasible_quantile=0.25):
    weight_dist = {'type':'dirichlet', 'alpha': [1] * problem.obj_dim}
    ranges = np.array([b[1] - b[0] for b in problem.bounds], dtype=np.float32)
    act_range = ranges / max_steps
    action_space = gym.spaces.Box(low=-act_range, high=act_range, shape=(problem.state_dim, ), dtype=np.float32)

    opt_env = SB3MOOEnv(problem, weight=None, weight_dist=weight_dist, max_steps=max_steps, penalty_coeff=1e3)
    if GE_from_file is None:
        algo = NSGA2(pop_size=start_points_sample)
        res = minimize(PymooMOO(problem),
                algo,
                termination=('n_gen', n_gen),
                seed=seed,
                save_history=False,
                verbose=False)
        candidates, objs, constraints = res.X, res.F, res.G
    else:
        print(f'Load pareto solutions from {GE_from_file}')
        X, O = extract_XO(GE_from_file, num=start_points_sample)
        candidates, objs = X, O
        constraints = []
        for candidate in candidates:
            obj, constraint = problem.evaluate(candidate)
            constraints.append(constraint)
    ARCHIVE_X, ARCHIVE_O, ARCHIVE_C = [], [], []
    for i, x0 in enumerate(candidates):
      obs = opt_env.reset(x0)
      done = False
      traj_X, traj_O, traj_C = [], [], []
      obj0, c0 = problem.evaluate(x0)
      traj_X.append(x0.copy())
      traj_O.append(obj0)
      traj_C.append(c0)
      while not done:
          with torch.no_grad():

            action = action_space.sample()
            new_obs, reward, done, info = opt_env.step(action)
            obs = new_obs

            reward_vec, constraint = info['reward_vec'], info['c_vals']

            traj_X.append(opt_env.x.copy()), traj_O.append(-reward_vec), traj_C.append(constraint)
      traj_X = np.asarray(traj_X)
      traj_O = np.asarray(traj_O)
      traj_C = np.asarray(traj_C)

      ARCHIVE_X.append(traj_X)
      ARCHIVE_O.append(traj_O)
      ARCHIVE_C.append(traj_C)

      print(f'Optimization schedule: {i+1}/{start_points_sample}')

    ARCHIVE_X = np.vstack(ARCHIVE_X)
    ARCHIVE_O = np.vstack(ARCHIVE_O)
    ARCHIVE_C = np.vstack(ARCHIVE_C)

    pareto_X, pareto_O, pareto_C = _spea2_compress_archive(
        ARCHIVE_X, ARCHIVE_O, ARCHIVE_C, max_size
    )
    idx = _filter_feasibility_first(pareto_O, pareto_C)

    return pareto_X[idx], pareto_O[idx], pareto_C[idx]

if __name__ == '__main__':
    from pymoo.problems import get_problem

    problem = get_problem("zdt1")
    print(problem.has_constraints())

    SAC_Weighted_Arch()
    env = SB3MOOEnv(problem=problem)

    policy_kwargs = {
        'hidden_dim': 64
    }

    model = SAC(
        policy=MO_SACPolicy,
        env=env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=15000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        tensorboard_log='./tensorboard_logs/zdt1/'
    )
    print(model.policy)

    model = model.load('sac_zdt1_hidden64')
    weight1 = [0.8, 0.2]
    weight2 = [0.2, 0.8]

    action1, _ = model.predict(weight1, deterministic=True)
    res1 = problem.evaluate(action1)
    action2, _ = model.predict(weight2, deterministic=True)
    res2 = problem.evaluate(action2)
    print(f'variable1={action1}, res1={res1}')
    print(f'variable1={action2}, res1={res2}')
