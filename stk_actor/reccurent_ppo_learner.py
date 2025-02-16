from .actors import LstmAgent
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
from bbrl.utils.functional import gae
import sys
import math
import time
import bbrl_gymnasium
import numpy as np
from bbrl_utils.algorithms import EpisodicAlgo, iter_partial_episodes
from bbrl_utils.nn import build_ortho_mlp
import torch as th
from bbrl_utils.nn import copy_parameters, ortho_init
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.distributions import Bernoulli, Categorical, Normal
import copy

class VAgent(Agent):
    def __init__(self, input_size, hidden_layers, head, name="critic"):
        super().__init__(name)
        self.is_q_function = False
        self.head = head
        self.model = build_ortho_mlp(
            [input_size] + list(hidden_layers) + [1], activation=nn.ReLU()
        )
        for layer in self.model:
          if isinstance(layer, nn.Linear):
            ortho_init(layer, std=1)

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        done = self.get(("env/terminated", t))

        observation = self.head.forward(t, observation, done,
                                          **kwargs)
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)

class PPOClip(EpisodicAlgo):
    def __init__(self, cfg):
        super().__init__(cfg, autoreset=True)
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.head_actor = LstmAgent(obs_size, cfg.algorithm.architecture.input_lstm_size,
                              cfg.algorithm.architecture.lstm_hidden_size,
                              cfg.algorithm.n_envs,
                              device)

        self.head_critic = copy.deepcopy(self.head_actor)

        self.train_policy = globals()[cfg.algorithm.policy_type](
            cfg.algorithm.architecture.lstm_hidden_size,
            cfg.algorithm.architecture.actor_hidden_size,
            act_size,
            self.head_actor
        ).with_prefix("current_policy/")

        self.eval_policy = KWAgentWrapper(
            self.train_policy,
            stochastic=False,
            update_lstm_state=True
        )

        self.critic_agent = VAgent(
            cfg.algorithm.architecture.lstm_hidden_size, cfg.algorithm.architecture.critic_hidden_size, self.head_critic
        ).with_prefix("critic/")
        self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")

        self.old_policy = copy.deepcopy(self.train_policy)
        self.old_policy.with_prefix("old_policy/")

        self.old_critic_agent.head = self.old_policy.head
        self.critic_agent.head = self.old_policy.head

        self.policy_optimizer = setup_optimizer(
            cfg.optimizer, self.train_policy
        )
        self.critic_optimizer = setup_optimizer(
            cfg.optimizer, self.critic_agent
        )



def run(ppo_clip: PPOClip):
    cfg = ppo_clip.cfg

    t_policy = TemporalAgent(ppo_clip.train_policy)
    t_old_policy = TemporalAgent(ppo_clip.old_policy)
    t_critic = TemporalAgent(ppo_clip.critic_agent)
    t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)

    num_updates = cfg.algorithm.total_timesteps // cfg.algorithm.batch_size


    for update in range(1, num_updates+1):
      #maj du lr
      inital_lstm_state = ppo_clip.old_policy.head.lstm_state

      if cfg.algorithm.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * cfg.optimizer.lr
        ppo_clip.policy_optimizer.param_groups[0]["lr"] = lrnow
        ppo_clip.critic_optimizer.param_groups[0]["lr"] = lrnow

      for train_workspace in iter_partial_episodes(
          ppo_clip, cfg.algorithm.n_steps
      ):
          with torch.no_grad():
              t_old_policy(
                  train_workspace,
                  t=0,
                  n_steps=cfg.algorithm.n_steps,
                  predict_proba=True,
                  compute_entropy=False,
                  update_lstm_state=True
              )

          # Compute the critic value over the whole workspace
          t_critic(train_workspace,
                   t=0,
                   n_steps=cfg.algorithm.n_steps,
                   lstm_state = ppo_clip.old_policy.head.lstm_state,
          )

          with torch.no_grad():
              t_old_critic(train_workspace,
                           t=0,
                           n_steps=cfg.algorithm.n_steps,
                           lstm_state = ppo_clip.old_policy.head.lstm_state,
              )

          ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
              "env/terminated",
              "env/reward",
              "critic/v_values",
              "old_critic/v_values",
          ]

          # the critic values are clamped to move not too far away from the values of the previous critic
          if cfg.algorithm.clip_range_vf > 0:
              # Clip the difference between old and new values
              # NOTE: this depends on the reward scaling
              ws_v_value = ws_old_v_value + torch.clamp(
                  ws_v_value - ws_old_v_value,
                  -cfg.algorithm.clip_range_vf,
                  cfg.algorithm.clip_range_vf,
              )

          # Compute the advantage using the (clamped) critic values
          with torch.no_grad():
              advantage = gae(
                  ws_reward[1:],
                  ws_v_value[1:],
                  ~ws_terminated[1:],
                  ws_v_value[:-1],
                  cfg.algorithm.discount_factor,
                  cfg.algorithm.gae,
              )

          ppo_clip.critic_optimizer.zero_grad()
          target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
          critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target) * cfg.algorithm.critic_coef
          critic_loss.backward()
          torch.nn.utils.clip_grad_norm_(
              ppo_clip.critic_agent.parameters(), cfg.algorithm.max_grad_norm
          )
          ppo_clip.critic_optimizer.step()

          # We store the advantage into the transition_workspace
          if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
              advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

          train_workspace.set_full("advantage", torch.cat(
              (advantage, torch.zeros(1, advantage.shape[1]))
          ))

          transition_workspace = train_workspace.get_transitions(no_final_state=True)

          num_transistions = transition_workspace.batch_size()

          envsperbatch = cfg.algorithm.n_envs // cfg.algorithm.num_minibatches
          envinds = np.arange(cfg.algorithm.n_envs)

          assert num_transistions % cfg.algorithm.n_envs == 0, f"{num_transistions} % {cfg.algorithm.n_envs} != 0"

          flatinds = np.arange(num_transistions).reshape(-1, cfg.algorithm.n_envs)

          for opt_epoch in range(cfg.algorithm.opt_epochs):
              np.random.shuffle(envinds)

              for start in range(0, cfg.algorithm.n_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = torch.from_numpy(flatinds[:, mbenvinds].ravel())

                sample_workspace = transition_workspace.select_batch(mb_inds)

                t_policy(
                        sample_workspace,
                        n_steps=1,#one step to get the probs
                        t=0,
                        predict_proba=True,
                        compute_entropy=True,
                        lstm_state = (inital_lstm_state[0][:, mbenvinds], inital_lstm_state[1][:, mbenvinds])
                )

                policy_probs = sample_workspace["current_policy/logprob_predict"]
                old_policy_probs = sample_workspace["old_policy/logprob_predict"]
                actions = sample_workspace["action"]
                policy_advantage = sample_workspace["advantage"]
                entropy = sample_workspace["current_policy/entropy"]

                ratio = (policy_probs / (old_policy_probs + 1e-8)).gather(1, actions.long())

                clipped_ratio = torch.clamp(
                    ratio, 1 - cfg.algorithm.clip_range, 1 + cfg.algorithm.clip_range
                )
                policy_loss = torch.min(
                    ratio * policy_advantage,
                    clipped_ratio * policy_advantage
                ).mean()

                loss_policy = -cfg.algorithm.policy_coef * policy_loss

                assert len(entropy) == 1, f"{entropy.shape}"
                entropy_loss = entropy[0].mean()
                loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

                # Store the losses for tensorboard display
                ppo_clip.logger.log_losses(
                    critic_loss, entropy_loss, policy_loss, ppo_clip.nb_steps
                )
                ppo_clip.logger.add_log(
                    "advantage", policy_advantage[0].mean(), ppo_clip.nb_steps
                )

                loss = loss_policy + loss_entropy

                loss.backward()
                ppo_clip.policy_optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(
                    ppo_clip.train_policy.parameters(), cfg.algorithm.max_grad_norm
                )

                ppo_clip.policy_optimizer.step()


          # Copy parameters
          lstm_state = ppo_clip.old_policy.head.lstm_state
          copy_parameters(ppo_clip.train_policy, ppo_clip.old_policy)
          copy_parameters(ppo_clip.critic_agent, ppo_clip.old_critic_agent)
          ppo_clip.old_policy.head.lstm_state = lstm_state

          #evaluation
          ppo_clip.evaluate()
