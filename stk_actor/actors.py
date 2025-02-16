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
from bbrl.utils.distributions import SquashedDiagGaussianDistribution

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