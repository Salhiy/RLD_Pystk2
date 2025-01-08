from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import SamplingActor
from .pystk_actor import env_name, get_wrappers, player_name


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
