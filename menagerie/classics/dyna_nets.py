"""Dyna model and value networks, 1990, Sutton.

Paper: "Integrated architectures for learning, planning, and reacting based on approximating dynamic programming."
The forward pass exposes the learned model and value approximators used by Dyna;
real/simulated experience scheduling and updates are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DynaModelValueNetworks(nn.Module):
    """Paired next-state/reward model and value network."""

    def __init__(self, state_size: int = 4, action_size: int = 2, hidden_size: int = 16) -> None:
        """Initialize model and value MLPs.

        Parameters
        ----------
        state_size
            Number of state features.
        action_size
            Number of action features.
        hidden_size
            Number of hidden units.
        """
        super().__init__()
        self.state_size = state_size
        self.action = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, state_size + 1),
        )
        self.value = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict next state, reward, and current value.

        Parameters
        ----------
        state
            State tensor of shape ``(batch, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Predicted next state, reward, value, and model-query action.
        """
        action = torch.tanh(self.action(state))
        pred = self.model(torch.cat((state, action), dim=-1))
        next_state = pred[:, : self.state_size]
        reward = pred[:, self.state_size :]
        return next_state, reward, self.value(state), action


def build() -> nn.Module:
    """Build compact Dyna model/value networks.

    Returns
    -------
    nn.Module
        Configured ``DynaModelValueNetworks`` instance.
    """
    return DynaModelValueNetworks()


def example_input() -> Tensor:
    """Create an example state tensor.

    Returns
    -------
    Tensor
        State tensor with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [("Dyna Model+Value Networks", "build", "example_input", "1990", "DD")]
