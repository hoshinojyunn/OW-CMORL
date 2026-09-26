import torch
import numpy as np
from collections import deque
import heapq
from sklearn.cluster import MiniBatchKMeans, KMeans
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class DistributedPER(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, obj_dim, device="cpu",  clusters=5, global_rank_update_freq=1000, **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, **kwargs)
        self.capacity = buffer_size
        self.obj_dim = obj_dim
        self.clusters = clusters
        self.buffer = []
        self.exp_to_index = {}
        self.cluster_buffers = [deque(maxlen=self.capacity//clusters) for _ in range(clusters)]
        self.priorities = []
        self.kmeans = MiniBatchKMeans(n_clusters=clusters)
        self.kmeans_fitted = False
        self.dirty = False
        self.global_ranks = np.zeros(self.capacity, dtype=np.float32)
        self.cluster_assignment = {}
        self.stability_threshold = 0.1
        self.last_trained_size = 0
        self.add_count = 0
        self.update_freq = global_rank_update_freq
        self.uid = 0

    def add(self, obs, next_obs, action, reward, done, infos):
        experience = (torch.FloatTensor(obs), torch.FloatTensor(next_obs), torch.FloatTensor(action), torch.FloatTensor(reward), torch.FloatTensor([done]), infos)
        state, next_state, action, reward , done, infos = experience
        exp_id = self._generate_exp_id(experience)
        if exp_id in self.cluster_assignment:
          cluster_id = self.cluster_assignment[exp_id]
        else:
          cluster_id = self._assign_stable_cluster(reward)
          self.cluster_assignment[exp_id] = cluster_id

        temp_priority = 0
        if len(self.buffer) > 0:
          temp_priority = self.estimate_priority(reward)
        else:
          temp_priority = 1.0

        if len(self.buffer) >= self.capacity:
            self.remove_lowest_priority()

        self.global_ranks[len(self.buffer)] = temp_priority

        self._add_to_buffers(experience, temp_priority, cluster_id)
        self.add_count += 1

        if self.add_count % self.update_freq == 0:
          self.update_global_ranks()

        if self._should_retrain():
            self.fit_kmeans()

        self.dirty = True

    def predict_cluster(self, reward):
        return self.kmeans.predict(np.array([reward.numpy()]))[0]

    def estimate_priority(self, reward):
        ideal_point = torch.max(torch.stack([e[3] for e in self.buffer]), dim=0)[0] if self.buffer else torch.zeros(self.obj_dim)
        distance = torch.norm(reward - ideal_point)
        return 1.0 / (1.0 + distance.item())

    def remove_lowest_priority(self):
        _, exp_id = heapq.heappop(self.priorities)
        remove_idx = self.exp_to_index.pop(exp_id)

        self.buffer.pop(remove_idx)
        for index in range(remove_idx, len(self.buffer)):
            self.exp_to_index[id(self.buffer[index])] = index

        for cluster_idx in range(len(self.cluster_buffers)):
          cluster_buf = self.cluster_buffers[cluster_idx]
          self.cluster_buffers[cluster_idx] = deque(
              [(e, p) for (e, p) in cluster_buf if id(e) != exp_id],
              maxlen=self.capacity // self.clusters
          )

        self.dirty = True

    def sample(self, batch_size, env=None):
        if len(self.buffer) < batch_size:
          return

        samples = []
        samples_per_cluster = max(1, batch_size // self.clusters)

        for cluster_buf in self.cluster_buffers:
            if not cluster_buf:
                continue

            priorities = np.array([p for _, p in cluster_buf])
            probs = priorities / priorities.sum()

            indices = np.random.choice(
                len(cluster_buf),
                min(samples_per_cluster, len(cluster_buf)),
                p=probs,
                replace=False
            )

            samples.extend([cluster_buf[i][0] for i in indices])

        remaining = batch_size - len(samples)
        if remaining > 0:
            global_priorities = np.array([self.global_ranks[i] for i in range(len(self.buffer))])
            probs = global_priorities / global_priorities.sum()

            indices = np.random.choice(
                len(self.buffer),
                remaining,
                p=probs,
                replace=False
            )

            samples.extend([self.buffer[i] for i in indices])

        obs, next_obs, actions, rewards, dones, infos = zip(*samples)
        return ReplayBufferSamples(
            torch.stack(obs),
            torch.stack(actions),
            torch.stack(next_obs),
            torch.stack(dones),
            torch.stack(rewards),
        )

    def update_global_ranks(self):
        if not self.buffer:
            raise RuntimeError("buffer size smaller than batch_size")
        all_rewards_min = torch.stack([-e[3] for e in self.buffer])

        nds = NonDominatedSorting()
        fronts = nds.do(all_rewards_min.numpy(), return_rank=False)

        for front_idx, front_indices in enumerate(fronts):
          self.global_ranks[front_indices] = 1.0 / (front_idx + 1)

        for cluster_idx, cluster_buf in enumerate(self.cluster_buffers):
            new_buf = []
            for exp, _ in cluster_buf:
              buf_idx = self.exp_to_index.get(id(exp), None)
              if buf_idx is not None:
                  new_buf.append((exp, self.global_ranks[buf_idx]))
            self.cluster_buffers[cluster_idx] = deque(new_buf, maxlen=self.capacity//self.clusters)

        self.dirty = False

    def _fast_non_dominated_sort(self, rewards):
        n = len(rewards)
        S = [[] for _ in range(n)]
        domination_count = torch.zeros(n, dtype=torch.int32)
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(rewards[i], rewards[j]):
                    S[i].append(j)
                elif self._dominates(rewards[j], rewards[i]):
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1

        return fronts

    def _dominates(self, a, b):
        not_worse = torch.all(a <= b)
        better = torch.any(a < b)
        return not_worse and better

    def _add_to_buffers(self, experience, priority, cluster_id):
      idx = len(self.buffer)
      if id(experience) in self.exp_to_index:
          print(f'{id(experience)}: {self.exp_to_index[id(experience)]}\n')
          print(f'repeated: {experience}\n')
          exit(-1)

      self.exp_to_index[id(experience)] = idx
      heapq.heappush(self.priorities, (priority, id(experience)))
      self.buffer.append(experience)
      self.cluster_buffers[cluster_id].append((experience, priority))

    def _assign_stable_cluster(self, reward):
        if self.kmeans_fitted:
            dists = np.linalg.norm(
                self.kmeans.cluster_centers_ - reward.numpy(),
                axis=1
            )
            min_dist = np.min(dists)
            if min_dist < self.stability_threshold:
                return np.argmin(dists)

        if len(self.buffer) < self.clusters * 5:
            cluster_id = len(self.buffer) % self.clusters
        else:
            sizes = [len(b) for b in self.cluster_buffers]
            cluster_id = np.argmin(sizes)

        return cluster_id

    def _should_retrain(self):
        return len(self.buffer) > self.clusters and (len(self.buffer) - self.last_trained_size > 1000 or
                len(self.buffer) > 2 * self.last_trained_size)

    def fit_kmeans(self):
        all_rewards = np.array([e[3].numpy() for e in self.buffer])

        self.kmeans.partial_fit(all_rewards)
        self.kmeans_fitted = True

        self.redistribute_clusters()
        self.last_trained_size = len(self.buffer)

    def redistribute_clusters(self):
        self.cluster_buffers = [deque(maxlen=self.capacity//self.clusters)
                              for _ in range(self.clusters)]

        for i, exp in enumerate(self.buffer):
            _, _, _, reward, _, _ = exp
            exp_id = self._generate_exp_id(exp)

            new_cluster = self.predict_cluster(reward)
            self.cluster_assignment[exp_id] = new_cluster

            priority = self.global_ranks[i]

            self.cluster_buffers[new_cluster].append((exp, priority))

    def _experiences_equal(self, exp1, exp2):
      return all(torch.equal(a, b) for a, b in zip(exp1, exp2))

    def _find_exp_index(self, experience):
      for idx, exp in enumerate(self.buffer):
        if self._experiences_equal(exp, experience):
            return idx
      raise ValueError("Experience not found in buffer.")

    def _generate_exp_id(self, exp):
        state, action, _, _, _, _ = exp
        return hash((
            tuple(state.numpy().ravel().round(4).tolist()),
            tuple(action.numpy().ravel().round(4).tolist())
        ))

class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cpu", **kwargs):
        super().__init__(buffer_size, observation_space, action_space, device, **kwargs)
        self.capacity = buffer_size
        self.buffer = deque(maxlen=self.capacity)

    def add(self, obs, next_obs, action, reward, done, infos):
        experience = (torch.FloatTensor(obs), torch.FloatTensor(next_obs), torch.FloatTensor(action), torch.FloatTensor(reward), torch.FloatTensor([done]), infos)
        self.buffer.append(experience)

    def sample(self, batch_size, env=None):
        samples = []

        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            replace=False
        )

        samples.extend([self.buffer[i] for i in indices])
        obs, next_obs, actions, rewards, dones, infos = zip(*samples)
        return ReplayBufferSamples(
            torch.stack(obs),
            torch.stack(actions),
            torch.stack(next_obs),
            torch.stack(dones),
            torch.stack(rewards),
        )
