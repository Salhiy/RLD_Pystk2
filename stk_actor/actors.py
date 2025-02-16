import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn.functional as F
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agent, Agents, TemporalAgent, KWAgentWrapper
from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, make_env, record_video
from bbrl.utils.replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import sys
import math
import time
import bbrl_gymnasium
import numpy as np

import torch as th

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.distributions import Bernoulli, Categorical, Normal


class Distribution(ABC):

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(
        self, *args, **kwargs
    ) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
        self, latent_dim: int, log_std_init: float = 0.0
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = nn.Parameter(
            th.ones(self.action_dim) * log_std_init, requires_grad=True
        )
        return mean_actions, log_std

    def make_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return th.tanh(x)

    @staticmethod
    def atanh(x: th.Tensor) -> th.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: th.Tensor) -> th.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = th.finfo(y.dtype).eps
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clamp(min=-1.0 + eps, max=1.0 - eps))

    def log_prob_correction(self, x: th.Tensor) -> th.Tensor:
        # Squash correction (from original SAC implementation)
        return th.log(1.0 - th.tanh(x) ** 2 + self.epsilon)

class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> "SquashedDiagGaussianDistribution":
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(
        self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None
    ) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions**2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        raise Exception("Call to entropy in squashed Diag Gaussian distribution")
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(
        self, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob

    def get_ten_samples(self) -> List:
        action_list = []
        for i in range(10):
            action = self.sample()
            action_list.append(action)
        return action_list

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, horizon=2):
        super().__init__(env)
        self.horizon = horizon  # Nombre d'observations à stocker
        self.frames = {"discrete": [], "continuous": []}  # Historique des observations

        
        self.continuous_space = env.observation_space["continuous"]
        self.discrete_space = env.observation_space["discrete"]

        self.observation_space = gym.spaces.Dict({
            "discrete": gym.spaces.MultiDiscrete(np.tile(self.discrete_space.nvec, (horizon,))),
            "continuous": gym.spaces.Box(
                low=np.tile(self.continuous_space.low, (horizon,)),
                high=np.tile(self.continuous_space.high, (horizon,)),
                dtype=self.continuous_space.dtype
            )
        })

    def reset(self, seed=None, options=None):
            obs, info = self.env.reset(seed=seed, options=options)  

            self.frames = {
                "discrete": [obs["discrete"]] * self.horizon,
                "continuous": [obs["continuous"]] * self.horizon
            }

            return {
                "discrete": torch.tensor(np.concatenate(self.frames["discrete"]), dtype=torch.long),
                "continuous": torch.tensor(np.concatenate(self.frames["continuous"]), dtype=torch.float32)
            }, info


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        self.frames["discrete"].insert(0, obs["discrete"])
        self.frames["continuous"].insert(0, obs["continuous"])

        if len(self.frames["discrete"]) > self.horizon:
            self.frames["discrete"].pop()
        if len(self.frames["continuous"]) > self.horizon:
            self.frames["continuous"].pop()

        return {
            "discrete": torch.tensor(np.concatenate(self.frames["discrete"]), dtype=torch.long),
            "continuous": torch.tensor(np.concatenate(self.frames["continuous"]), dtype=torch.float32)
        }, reward, done, truncated, info


class BackbonePolicyLstm(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size,
                    projection_state_size, projection_action_size):
        super(BackbonePolicyLstm, self).__init__()
        self.proj_state_1 = nn.Linear(state_dim["discrete"], projection_state_size) # discret
        self.proj_state_2 = nn.Linear(state_dim["continuous"], projection_state_size) # continous

        self.proj_action_1 = nn.Linear(action_dim["discrete"], projection_action_size) # discret
        self.proj_action_2 = nn.Linear(action_dim["continuous"], projection_action_size) # continous


        self.linear1 = nn.Linear(projection_state_size * 2 , hidden_size)
        self.linear2 = nn.Linear((projection_state_size + projection_action_size)*2, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)


    def forward(self, state, last_action, hidden_in):

        state = torch.cat([self.proj_state_1(state["discrete"]), self.proj_state_2(state["continuous"])], -1)
        state = state.view(state.shape[0], -1) 

        last_action = torch.cat([self.proj_action_1(last_action["discrete"]), self.proj_action_2(last_action["continuous"])], -1)
        last_action = last_action.view(last_action.shape[0], -1)

        fc_branch = F.tanh(self.linear1(state))

        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.tanh(self.linear2(lstm_branch))

        lstm_branch = lstm_branch.unsqueeze(0) 
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)
        lstm_branch = lstm_branch.squeeze(0)

        merged_branch=torch.cat([fc_branch, lstm_branch], -1)
        x = F.tanh(self.linear3(merged_branch))
        x = F.tanh(self.linear4(x))

        return x, lstm_hidden

class BaseActor(Agent):

    def copy_parameters(self, other):
        for self_p, other_p in zip(self.parameters(), other.parameters()):
            self_p.data.copy_(other_p)


class SquashedGaussianActor(BaseActor):
    def __init__(self, state_dim, hidden_size, action_dim, action_space, n_env, projection_state_size, projection_action_size, action_range=1., init_w=3e-3):
        super().__init__()
        self.backbone = BackbonePolicyLstm(state_dim, action_dim, hidden_size, projection_state_size, projection_action_size)
        self.last_mean_layer = nn.Linear(hidden_size, action_dim["continuous"])#continous
        self.last_log_std_layer = nn.Linear(hidden_size, action_dim["continuous"])#continous

        self.last_discrete_action = nn.ModuleList([
            nn.Linear(hidden_size, n_choices) for n_choices in action_space["discrete"].nvec
        ])

        self.last_mean_layer.weight.data.uniform_(-init_w, init_w)
        self.last_mean_layer.bias.data.uniform_(-init_w, init_w)

        self.last_log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.last_log_std_layer.bias.data.uniform_(-init_w, init_w)

        self.action_dist = SquashedDiagGaussianDistribution(action_space["continuous"].shape[0])
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.n_env = n_env
        self.hidden_lstm = (torch.zeros([self.backbone.lstm1.num_layers, self.n_env, self.hidden_size], dtype=torch.float), \
                torch.zeros([self.backbone.lstm1.num_layers, self.n_env, self.hidden_size], dtype=torch.float))


    def get_distribution(self, obs, last_action, hidden_in):
        backbone_output, hidden_out = self.backbone(obs, last_action, hidden_in)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_log_std_layer(backbone_output)
        discrete_action_logits = [layer(backbone_output) for layer in self.last_discrete_action]

        std_out = std_out.clamp(-20, 2)
        std = torch.exp(std_out)

        return self.action_dist.make_distribution(mean, std), discrete_action_logits, hidden_out

    def forward(self, t, stochastic=False, predict_proba=False, write_lstm_state=True, **kwargs):

        obs = {
            "discrete" : self.get(("env/env_obs/discrete", t)).float(),
             "continuous" : self.get(("env/env_obs/continuous", t))
        }

        last_action, hidden_in = None, None
        if (t > 0):
          last_action = {
              "discrete" : self.get(("action/discrete", t-1)).float(),
              "continuous" : self.get(("action/continuous", t-1))
          }
        else:
          sampled_action = self.action_space.sample()

          action_discrete = torch.tensor(sampled_action["discrete"]).float().unsqueeze(0).expand(self.n_env, -1)

          action_continuous = torch.tensor(sampled_action["continuous"]).float().unsqueeze(0).expand(self.n_env, -1)

          last_action = {
              "discrete": action_discrete,
              "continuous": action_continuous
          }


        (h, c) = (self.hidden_lstm[0].detach(), self.hidden_lstm[1].detach())
        action_dist, discrete_action_logits, hidden_out = self.get_distribution(obs, last_action, (h, c))

        if (write_lstm_state):
          self.hidden_lstm = hidden_out

        if predict_proba:
            action = self.get(("action", t))
            log_prob = action_dist.log_prob(action)
            self.set(("logprob_predict", t), log_prob)
        else:
            discrete_scores = torch.stack([torch.softmax(logits, dim=-1) for logits in discrete_action_logits], dim=1)
            if stochastic:
                action = action_dist.sample()
                discrete_action = torch.distributions.Categorical(logits=discrete_scores).sample()

            else:
                action = action_dist.mode()
                discrete_action = discrete_scores.argmax(dim=-1)


            log_prob_continious = action_dist.log_prob(action)
            log_probs_discrete = torch.gather(discrete_scores.log(), dim=-1, index=discrete_action.unsqueeze(-1)).squeeze(-1)

            log_prob_total = log_probs_discrete.sum(-1) + log_prob_continious


            self.set(("action/discrete", t), discrete_action)
            self.set(("action/continuous", t), action)

            self.set(("action_logprobs", t), log_prob_total)

    def predict_action(self, obs, last_action, hidden_in, stochastic=False):
        """Predict just one action (without using the workspace)"""
        action_dist, _ = self.get_distribution(obs, last_action, hidden_in)
        return action_dist.sample() if stochastic else action_dist.mode()