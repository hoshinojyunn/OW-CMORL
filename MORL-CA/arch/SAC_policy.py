import torch
import torch.nn as nn
import math
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.type_aliases import Schedule
import os

class PreferenceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, state_dim, feature_dim=32):
        super().__init__(observation_space, features_dim=feature_dim)
        self.state_dim = state_dim
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.state_net(observations)

class AOWNet(nn.Module):

    def __init__(self, input_dim: int, n_obj: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_obj)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        logits = self.net(x)
        weights = torch.softmax(logits, dim=-1)
        return weights

class AOWActor(nn.Module):

    def __init__(self, feature_dim: int, action_dim: int, n_obj: int, hidden_dim: int = 64):
        super().__init__()
        self.n_obj = n_obj
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_obj, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )

    def forward(self, features: torch.Tensor, aow_weights: torch.Tensor) -> tuple:

        combined = torch.cat([features, aow_weights], dim=-1)
        output = self.net(combined)
        mean, log_std = output.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

class MO_SACPolicy(SACPolicy):
    def __init__(self, observation_space, action_space, lr_schedule: Schedule, state_dim, obj_dim,
                 feature_dim=32, aow_model_path=None, aow_hidden_dim=64, *args, **kwargs):
        self.action_dim = action_space.shape[0]
        self.state_dim = state_dim
        self.features_dim = feature_dim
        self.obj_dim = obj_dim
        self.aow_model_path = aow_model_path
        self.aow_hidden_dim = aow_hidden_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=PreferenceExtractor,
            features_extractor_kwargs={
                "state_dim": state_dim,
                "feature_dim": feature_dim
            },
            net_arch=dict(pi=[feature_dim], qf=[feature_dim]),
            *args,
            **kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)

        self.aow_net = AOWNet(
            input_dim=self.state_dim,
            n_obj=self.obj_dim,
            hidden_dim=self.aow_hidden_dim
        )

        self.aow_actor = AOWActor(
            feature_dim=self.features_dim,
            action_dim=self.action_dim,
            n_obj=self.obj_dim,
            hidden_dim=self.aow_hidden_dim
        )

        self.actor_optim = torch.optim.Adam(
            list(self.aow_net.parameters()) + list(self.aow_actor.parameters()),
            lr=lr_schedule(1.0)
        )

        if self.aow_model_path and os.path.exists(self.aow_model_path):
            self.load_aow_weights(self.aow_model_path)

        self.qf_list = nn.ModuleList()
        self.qf_target_list = nn.ModuleList()
        self.qf_optim_list = []
        for _ in range(self.obj_dim):
            qf = self.create_critic(self.features_dim, self.action_dim, self.net_arch["qf"])
            qf_target = self.create_critic(self.features_dim, self.action_dim, self.net_arch["qf"])
            qf_target.load_state_dict(qf.state_dict())
            self.qf_list.append(qf)
            self.qf_target_list.append(qf_target)
            self.qf_optim_list.append(torch.optim.Adam(qf.parameters(), lr=lr_schedule(1.0)))

    def create_critic(self, features_dim, action_dim, net_arch):
        return nn.Sequential(
            nn.Linear(features_dim + action_dim, net_arch[0]),
            nn.ReLU(),
            nn.Linear(net_arch[0], net_arch[0]),
            nn.ReLU(),
            nn.Linear(net_arch[0], 1)
        )

    def load_aow_weights(self, path):

        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.aow_net.load_state_dict(checkpoint['model_state_dict'])
            self.aow_net.eval()
            print(f"Loaded pre-trained AOW weights from {path}")
        except Exception as e:
            print(f"Warning: Could not load AOW weights from {path}: {e}")

    def get_aow_weights(self, obs: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            return self.aow_net(obs)

    def aow_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple:


        model_dtype = next(self.aow_net.parameters()).dtype
        if obs.dtype != model_dtype:
            obs = obs.to(model_dtype)
        features = self.actor.features_extractor(obs)
        aow_weights = self.aow_net(obs)

        mean, log_std = self.aow_actor(features, aow_weights)

        if deterministic:
            action_raw = mean
        else:
            action_raw = mean + torch.randn_like(mean) * torch.exp(log_std)

        action = torch.tanh(action_raw)

        action_low = torch.as_tensor(self.action_space.low, dtype=action.dtype, device=action.device)
        action_high = torch.as_tensor(self.action_space.high, dtype=action.dtype, device=action.device)
        action_scaled = action_low + (action + 1) / 2 * (action_high - action_low)

        log_prob = -0.5 * ((action_raw - mean) / (torch.exp(log_std) + 1e-6)).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action_scaled, log_prob, features, aow_weights

    def forward(self, obs, deterministic: bool = False):
        return self.aow_action(obs, deterministic)

    def _predict(self, observation, deterministic: bool = False):
        action, log_prob, features, aow_weights = self.aow_action(observation, deterministic)
        return action

    def action_log_prob(self, obs: torch.Tensor) -> tuple:

        action, log_prob, features, aow_weights = self.aow_action(obs)
        return action, log_prob