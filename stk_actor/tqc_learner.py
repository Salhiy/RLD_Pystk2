import os
import sys
import math
import time
import torch
import bbrl_gymnasium
import copy
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, Optional, Iterator
from functools import partial
from omegaconf import OmegaConf
from abc import abstractmethod, ABC
from time import strftime
from bbrl import instantiate_class


def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class QNetworkLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_quantiles,
                 projection_state_size, projection_action_size, activation=F.relu):
        super(QNetworkLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.proj_state_1 = nn.Linear(state_dim["discrete"], projection_state_size) # discret
        self.proj_state_2 = nn.Linear(state_dim["continuous"], projection_state_size) # continous

        self.proj_action_1 = nn.Linear(action_dim["discrete"], projection_action_size) # discret
        self.proj_action_2 = nn.Linear(action_dim["continuous"], projection_action_size) # continous


        self.linear1 = nn.Linear((projection_state_size + projection_action_size) * 2, hidden_dim)
        self.linear2 = nn.Linear((projection_state_size + projection_action_size) * 2, hidden_dim)

        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, n_quantiles)
        self.linear4.apply(linear_weights_init)
        self.activation = activation

    def forward(self, state, action, last_action, hidden_in):


        state = torch.cat([self.proj_state_1(state["discrete"]), self.proj_state_2(state["continuous"])], -1)
        state = state.view(state.shape[0], -1) # concatenation des discret + continous

        last_action = torch.cat([self.proj_action_1(last_action["discrete"]), self.proj_action_2(last_action["continuous"])], -1)
        last_action = last_action.view(last_action.shape[0], -1)

        action = torch.cat([self.proj_action_1(action["discrete"]), self.proj_action_2(action["continuous"])], -1)
        action = action.view(action.shape[0], -1)

        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2

        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = self.activation(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim

        lstm_branch = lstm_branch.unsqueeze(0)  # (1, batch, hidden)
        (h, c) = (hidden_in[0].detach(), hidden_in[1].detach())
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, (h, c) )  # no activation after lstm
        lstm_branch = lstm_branch.squeeze(0)  # (batch, hidden)
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1)

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)


        return x, lstm_hidden

class TruncatedQuantileNetwork(Agent):
    def __init__(self, state_dim, hidden_dim, n_nets, action_dim, n_quantiles, action_space, n_env, projection_state_size, projection_action_size):
        super().__init__()
        self.is_q_function = True
        self.nets = []
        self.action_space = action_space
        self.n_env = n_env
        for i in range(n_nets):
            net = QNetworkLSTM(state_dim, action_dim, hidden_dim, n_quantiles, projection_state_size, projection_action_size)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)


    def forward(self, t, **kwargs):
        obs = {
            "discrete" : self.get(("env/env_obs/discrete", t)).float(),
             "continuous" : self.get(("env/env_obs/continuous", t))
        }

        action = {
             "discrete" : self.get(("action/discrete", t)).float(),
             "continuous" : self.get(("action/continuous", t))
        }

        last_action = None
        if (t>0):
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

        hidden_in = kwargs["hidden_lstm"]

        quantiles = torch.stack(tuple(net(obs, action ,last_action, hidden_in)[0] for net in self.nets), dim=1)


        self.set(("quantiles", t), quantiles)
        return quantiles

    def predict_value(self, obs, action, last_action, hidden_in):
        quantiles = torch.stack(tuple(net(obs, action, last_action, hidden_in)[0] for net in self.nets), dim=1)
        return quantiles