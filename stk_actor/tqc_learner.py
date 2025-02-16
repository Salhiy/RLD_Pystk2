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
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agent, Agents, TemporalAgent, KWAgentWrapper
from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, make_env, record_video
from bbrl.utils.replay_buffer import ReplayBuffer
import numpy as np
import gym
from pystk2_gymnasium import AgentSpec
import inspect
from .pystk_actor import get_wrappers

def setup_entropy_optimizers(cfg):
    if cfg.algorithm.target_entropy == "auto":
        entropy_coef_optimizer_args = get_arguments(cfg.entropy_coef_optimizer)
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = torch.log(
            torch.ones(1) * cfg.algorithm.entropy_coef
        ).requires_grad_(True)
        entropy_coef_optimizer = get_class(cfg.entropy_coef_optimizer)(
            [log_entropy_coef], **entropy_coef_optimizer_args
        )
    else:
        log_entropy_coef = 0
        entropy_coef_optimizer = None
    return entropy_coef_optimizer, log_entropy_coef

# Configure the optimizer
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer

# Create the TQC Agent
def create_tqc_agent(cfg, train_env_agent, eval_env_agent):
    obs_size = {
        "continuous": train_env_agent.observation_space["continuous"].shape[0],
        "discrete":train_env_agent.observation_space["discrete"].shape[0]
    }

    act_size = {
        "continuous": train_env_agent.action_space["continuous"].shape[0],
        "discrete":train_env_agent.action_space["discrete"].shape[0]
    }

    action_space = train_env_agent.get_action_space()

    # Actor
    actor = SquashedGaussianActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size, action_space
        , cfg.algorithm.n_envs, cfg.algorithm.projection_state_size, cfg.algorithm.projection_action_size
    )

    # Train/Test agents
    tr_agent = Agents(train_env_agent, actor)
    ev_agent = Agents(eval_env_agent, actor)

    # Builds the critics
    critic = TruncatedQuantileNetwork(
        obs_size, cfg.algorithm.architecture.critic_hidden_size,
        cfg.algorithm.architecture.n_nets, act_size,
        cfg.algorithm.architecture.n_quantiles, action_space, cfg.algorithm.n_envs,
        cfg.algorithm.projection_state_size, cfg.algorithm.projection_action_size
    )
    target_critic = copy.deepcopy(critic)

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    )

def get_env_agents(cfg, *, autoreset=True, include_last_state=True, wrappers=None) -> Tuple[GymAgent, GymAgent]:

    train_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, autoreset=autoreset, wrappers=wrappers, agent=AgentSpec(use_ai=False)),
        cfg.algorithm.n_envs,
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, wrappers=wrappers, agent=AgentSpec(use_ai=False)),
        cfg.algorithm.nb_evals,
        include_last_state=include_last_state
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent

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

def compute_critic_loss(
        cfg, reward, must_bootstrap,
        t_actor,
        q_agent,
        target_q_agent,
        rb_workspace,
        ent_coef, hidden_lstm
):
    # Compute quantiles from critic with the actions present in the buffer:
    # at t, we have Qu  ntiles(s,a) from the (s,a) in the RB
    q_agent(rb_workspace, t=0, n_steps=1,
            hidden_lstm = hidden_lstm)

    quantiles = rb_workspace["quantiles"].squeeze()
    with torch.no_grad():
        # Replay the current actor on the replay buffer to get actions of the
        # current policy
        t_actor(rb_workspace, t=1, n_steps=1, stochastic=True, write_lstm_state=False)
        action_logprobs_next = rb_workspace["action_logprobs"]

        # Compute target quantiles from the target critic: at t+1, we have
        # Quantiles(s+1,a+1) from the (s+1,a+1) where a+1 has been replaced in the RB

        target_q_agent(rb_workspace, t=1, n_steps=1, hidden_lstm=hidden_lstm)
        post_quantiles = rb_workspace["quantiles"][1]

        sorted_quantiles, _ = torch.sort(post_quantiles.reshape(quantiles.shape[0], -1))
        quantiles_to_drop_total = cfg.algorithm.top_quantiles_to_drop * cfg.algorithm.architecture.n_nets
        truncated_sorted_quantiles = sorted_quantiles[:,
                                     :quantiles.size(-1) * quantiles.size(-2) - quantiles_to_drop_total]


        # compute the target
        logprobs = (ent_coef * action_logprobs_next[1])

        y = (reward[0].unsqueeze(-1) + must_bootstrap.int().unsqueeze(-1) *
          cfg.algorithm.discount_factor * (truncated_sorted_quantiles - logprobs.unsqueeze(-1)))


    # computing the Huber loss
    pairwise_delta = y[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples

    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()

    return loss

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def compute_actor_loss(ent_coef, t_actor, q_agent, rb_workspace, hidden_lstm):

    t_actor(rb_workspace, t=0, n_steps=1, stochastic=True, write_lstm_state=False)
    action_logprobs_new = rb_workspace["action_logprobs"]

    q_agent(rb_workspace, t=0, n_steps=1,
            hidden_lstm = hidden_lstm)
    quantiles = rb_workspace["quantiles"][0]

    actor_loss = (ent_coef * action_logprobs_new[0] - quantiles.mean(2).mean(1))

    return actor_loss.mean()

class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string: float, loss: float, steps: int):
        self.logger.add_scalar(log_string, loss.item(), steps)

    # A specific function for RL algorithms having a critic, an actor and an
    # entropy losses
    def log_losses(
        self, critic_loss: float, entropy_loss: float, actor_loss: float, steps: int
    ):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards: torch.Tensor, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)

def run_tqc(cfg):
    logger = Logger(cfg)
    best_reward = float('-inf')
    ent_coef = cfg.algorithm.entropy_coef

    train_env_agent, eval_env_agent = get_env_agents(cfg, wrappers=[HistoryWrapper])

    (
        train_agent,
        eval_agent,
        actor,
        critic,
        target_critic
    ) = create_tqc_agent(cfg, train_env_agent, eval_env_agent)

    t_actor = TemporalAgent(actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    train_workspace = Workspace()


    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)
    nb_steps = 0
    tmp_steps = 0

    if cfg.algorithm.target_entropy == "auto":
            target_entropy = -np.prod((train_env_agent.action_space["continuous"].shape[0]
                                      +train_env_agent.action_space["discrete"].shape[0])).astype(np.float32)
    else:
            target_entropy = cfg.algorithm.target_entropy

    # Training loop
    pbar = tqdm(range(cfg.algorithm.nb_episodes))

    for episode in pbar:

        if episode > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps - 1,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                stochastic=True,
            )


        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action/discrete"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        if nb_steps > cfg.algorithm.learning_starts:
            # Get a sample from the workspace
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action_logprobs_rb = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action_logprobs"
            ]

            
            must_bootstrap = ~done[1]

            reward = cfg.algorithm.reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)

            critic_loss = compute_critic_loss(cfg, reward, must_bootstrap,
                                              t_actor, q_agent, target_q_agent,
                                              rb_workspace, ent_coef,
                                              actor.hidden_lstm
                                              )

            logger.add_log("critic_loss", critic_loss, nb_steps)

            actor_loss = compute_actor_loss(
                ent_coef, t_actor, q_agent, rb_workspace, actor.hidden_lstm
            )
            logger.add_log("actor_loss", actor_loss, nb_steps)

            if entropy_coef_optimizer is not None:
               
                ent_coef = torch.exp(log_entropy_coef.detach())
                entropy_coef_loss = -(
                        log_entropy_coef * (action_logprobs_rb + target_entropy)
                ).mean()
                entropy_coef_optimizer.zero_grad()
                
                entropy_coef_loss.backward(retain_graph=True)

                entropy_coef_optimizer.step()
                logger.add_log("entropy_coef_loss", entropy_coef_loss, nb_steps)
                logger.add_log("entropy_coef", ent_coef, nb_steps)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                critic.parameters(), cfg.algorithm.max_grad_norm
            )
            critic_optimizer.step()
            ####################################################

            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            soft_update_params(critic, target_critic, tau)
            # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                n_steps=cfg.algorithm.max_evals,
                stochastic=False,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(mean, nb_steps)

            pbar.set_description(f"nb_steps: {nb_steps}, reward: {mean:.3f}")
            
            if cfg.save_best and mean > best_reward:
                mod_path = Path(inspect.getfile(get_wrappers)).parent
                torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")