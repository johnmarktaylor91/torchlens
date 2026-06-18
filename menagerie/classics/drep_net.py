"""Dynamic REINFORCE error propagation net, 1989, Williams and Zipser era.

Paper: "A learning algorithm for continually running fully recurrent neural networks."
The module captures the differentiable model-based substrate: an encoded observation,
an action-conditioned recurrent transition, and reward/value heads. Reinforcement
backpropagation through a training objective is omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynamicREINFORCEErrorPropagationNet(nn.Module):
    """Model-based recurrent rollout network for imagined trajectories."""

    def __init__(self, obs_size: int = 8, action_size: int = 3, hidden_size: int = 12) -> None:
        """Initialize encoder, transition, reward, and value heads.

        Parameters
        ----------
        obs_size
            Number of observation features.
        action_size
            Number of action features per imagined step.
        hidden_size
            Latent recurrent state size.
        """
        super().__init__()
        self.action_size = action_size
        self.encoder = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.Tanh())
        self.policy = nn.Linear(hidden_size, action_size)
        self.transition = nn.GRUCell(action_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Roll the learned dynamics forward under an action sequence.

        Parameters
        ----------
        obs
            Initial observation tensor of shape ``(batch, 8)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Predicted rewards, values, and internally generated actions.
        """
        hidden = self.encoder(obs)
        rewards: list[Tensor] = []
        values: list[Tensor] = []
        actions: list[Tensor] = []
        for _step in range(5):
            action = torch.tanh(self.policy(hidden))
            hidden = self.transition(action, hidden)
            actions.append(action)
            rewards.append(self.reward_head(hidden))
            values.append(self.value_head(hidden))
        return torch.stack(rewards, dim=1), torch.stack(values, dim=1), torch.stack(actions, dim=1)


def build() -> nn.Module:
    """Build a DRE-style recurrent model substrate.

    Returns
    -------
    nn.Module
        Configured ``DynamicREINFORCEErrorPropagationNet`` instance.
    """
    return DynamicREINFORCEErrorPropagationNet()


def example_input() -> Tensor:
    """Create an example observation.

    Returns
    -------
    Tensor
        Observation tensor with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [
    ("Dynamic REINFORCE Error Propagation Net", "build", "example_input", "1989", "DD")
]
