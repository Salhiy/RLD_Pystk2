from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SamplingActor
from .pystk_actor import env_name, get_wrappers, player_name
from omegaconf import OmegaConf

params = {
  "save_best": True,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tblogs/",
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,
    },

  "algorithm":{
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

config = OmegaConf.create(params)

if __name__ == "__main__":
    # (1) Setup the environment
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]

    actor = SamplingActor(env.action_space)

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    num_episodes = 10  
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = actor(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            loss = -torch.sum(torch.tensor(reward)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
    print(f"Actor state saved to {mod_path / 'pystk_actor.pth'}")
