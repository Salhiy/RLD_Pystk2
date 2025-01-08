import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn.functional as F


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # Transform the action based on the option parameter
        if self.option == 1:
            return action + 1  # Example: increment action by 1
        elif self.option == 2:
            return action * 2  # Example: double the action
        else:
            return action  # No modification


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, policy_net: torch.nn.Module):
        super().__init__()
        self.policy_net = policy_net

    def forward(self, t: int):
        # Use the policy network to compute probabilities over actions
        state = self.get(("state", t))  # Get the state at time t
        logits = self.policy_net(state)  # Compute logits
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        self.set(("probs", t), probs)  # Store probabilities for further use


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def __init__(self, policy_net: torch.nn.Module):
        super().__init__()
        self.policy_net = policy_net

    def forward(self, t: int):
        # Compute the action by selecting the highest probability
        state = self.get(("state", t))  # Get the state at time t
        logits = self.policy_net(state)  # Compute logits
        action = torch.argmax(logits, dim=-1)  # Select the action with max probability
        self.set(("action", t), action)  # Store the action


class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        # Sample a random action from the action space
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
