"""Tolman-Eichenbaum Machine, 2020, Whittington and colleagues.

Paper: "The Tolman-Eichenbaum Machine: Unifying space and relational memory through
generalization in the hippocampal formation." This simplified substrate binds grid
state and sensory embedding, then retrieves through a fixed Hebbian memory.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Tolman-Eichenbaum Machine (TEM)", "build", "example_input", "2020", "DB")]


class TolmanEichenbaumMachine(nn.Module):
    """Simplified grid, place-binding, and Hebbian retrieval module."""

    def __init__(self, obs_dim: int = 8, grid_dim: int = 32) -> None:
        """Initialize transition, sensory, and memory tensors.

        Parameters
        ----------
        obs_dim
            Observation and action dimensionality for the example substrate.
        grid_dim
            Grid/place representation dimensionality.
        """
        super().__init__()
        self.grid_from_action = nn.Linear(obs_dim, grid_dim)
        self.sensory = nn.Linear(obs_dim, grid_dim)
        memory = torch.randn(grid_dim, grid_dim) * 0.05
        self.register_buffer("memory", memory)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Bind observation to grid state and retrieve a place vector.

        Parameters
        ----------
        obs
            Observation/action proxy tensor of shape ``(batch, obs_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Grid state, bound place state, and Hebbian retrieval.
        """
        grid = torch.tanh(self.grid_from_action(obs))
        sensory = torch.tanh(self.sensory(obs))
        place = grid * sensory
        retrieved = place @ self.memory
        return grid, place, retrieved


def build() -> nn.Module:
    """Build a small simplified TEM module.

    Returns
    -------
    nn.Module
        Configured ``TolmanEichenbaumMachine`` instance.
    """
    return TolmanEichenbaumMachine()


def example_input() -> Tensor:
    """Return an observation/action proxy example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)
