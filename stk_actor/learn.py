from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SamplingActor
from .pystk_actor import env_name, get_wrappers, player_name
from .tqc_learner import run_tqc
from omegaconf import OmegaConf

tqc_params = {
  "save_best": True,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tblogs/",
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,
    },

  "algorithm":{
    "imitation_learning_steps" : 1000,
    "projection_state_size": 128,
    "projection_action_size": 128,
    "nb_episodes" : 1000000,
    "reward_scale" : 10.,
    "seed": 1,
    "horizon" : 2,
    "n_envs": 8,
    "n_steps": 32,
    "n_updates": 32,
    "buffer_size": 1e6,
    "max_evals" : 1000,
    "batch_size": 8,
    "max_grad_norm": 0.5,
    "nb_evals":8,
    "eval_interval": 3000,
    "learning_starts": 100,
    "max_epochs": 8000,
    "discount_factor": 0.98,
    "entropy_coef": 1e-7,
    "target_entropy": "auto",
    "tau_target": 0.05,
    "top_quantiles_to_drop": 2,
    "architecture":{
      "actor_hidden_size": 128,
      "critic_hidden_size": 128,
      "n_nets": 4,
      "n_quantiles": 25,
    },
  },
  "gym_env":{
    "classname": "__main__.make_gym_env",
    "env_name": "supertuxkart/flattened-v0",
    },
  "actor_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    },
  "critic_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    },
  "entropy_coef_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
    }
}

ppo_params = params = {
    "base_dir": "${gym_env.env_name}/ppo-clip-S${algorithm.seed}_${current_time:}",
    "save_best": False,
    "logger": {
        "classname": "bbrl.utils.logger.TFLogger",
        "cache_size": 10000,
        "every_n_seconds": 10,
        "verbose": False,
    },
    "algorithm": {
        "anneal_lr" : True,
        "num_minibatches" : 4,
        "total_timesteps" : 1000,
        "seed": 12,
        "max_grad_norm": 0.5,
        "n_envs": 8,
        "n_steps": 32,
        "eval_interval": 1000,
        "nb_evals": 16,
        "gae": 0.8,
        "discount_factor": 0.98,
        "normalize_advantage": True,
        "max_epochs": 5_000,
        "opt_epochs": 10,
        "batch_size": 256,
        "clip_range": 0.2,
        "clip_range_vf": 0,
        "entropy_coef": 2e-7,
        "policy_coef": 1,
        "critic_coef": 1.0,
        "policy_type": "PPOPolicy",
        "architecture": {
            "lstm_hidden_size": 512,
            "input_lstm_size" : 128,
            "actor_hidden_size": [512, 64],
            "critic_hidden_size": [512, 64],
        },
    },
    "gym_env": {
        "env_name": "CartPole-v1",
    },
    "optimizer": {
        "classname": "torch.optim.AdamW",
        "lr": 1e-3,
        "eps": 1e-5,
    },
}

tqc_config = OmegaConf.create(tqc_params)
ppo_config = OmegaConf.create(ppo_params)

if __name__ == "__main__":
    run_tqc(tqc_config)