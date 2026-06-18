"""Adaptive critic designs, 1990, Werbos.

Paper: "Backpropagation through time: what it does and how to do it."
This compact HDP/DHP-style substrate combines an action network, learned one-step
model, and critic head; optimal-control updates and value-gradient losses are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AdaptiveCriticDesign(nn.Module):
    """Neural action, model, and critic components for adaptive critic designs."""

    def __init__(
        self, state_size: int = 8, action_size: int = 3, hidden_size: int = 16, mode: str = "hdp"
    ) -> None:
        """Initialize action, model, and critic networks.

        Parameters
        ----------
        state_size
            Number of state features.
        action_size
            Number of action features.
        hidden_size
            Number of hidden units.
        mode
            Critic mode, ``"hdp"`` for scalar value or ``"dhp"`` for value gradient.
        """
        super().__init__()
        if mode not in {"hdp", "dhp"}:
            raise ValueError("mode must be 'hdp' or 'dhp'")
        self.mode = mode
        critic_out = 1 if mode == "hdp" else state_size
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),
        )
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, critic_out)
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute action, critic estimate, and predicted next state.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, 8)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Action, value or value-gradient estimate, and predicted next state.
        """
        action = self.actor(state)
        next_state = self.model(torch.cat((state, action), dim=-1))
        critic_estimate = self.critic(state)
        return action, critic_estimate, next_state


def build() -> nn.Module:
    """Build a compact HDP adaptive critic design.

    Returns
    -------
    nn.Module
        Configured ``AdaptiveCriticDesign`` instance.
    """
    return AdaptiveCriticDesign()


def example_input() -> Tensor:
    """Create an example state vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [("Adaptive Critic Designs (HDP/DHP)", "build", "example_input", "1990", "DD")]
