from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

from .actors import SquashedGaussianActor, HistoryWrapper

env_name = "supertuxkart/flattened-v0"

#: Player name
player_name = "TurboNebula"


#tqc_actor
def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    
    obs_size = {
        "continuous": observation_space["continuous"].shape[0],
        "discrete": observation_space["discrete"].shape[0]
    }

    act_size = {
        "continuous": action_space["continuous"].shape[0],
        "discrete":action_space["discrete"].shape[0]
    }

    actor = SquashedGaussianActor(
        obs_size, 128, act_size, action_space
        , 1, 128, 128
    )

    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return actor


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:

    return [
        lambda env: HistoryWrapper(env)
    ]
