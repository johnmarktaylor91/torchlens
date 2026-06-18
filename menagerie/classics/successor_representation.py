"""Successor Representation predictive map, 1993, Dayan.

Paper: "Improving generalization for temporal difference learning: The successor
representation." A learned SR matrix maps one-hot states to expected discounted
future occupancy, and a linear reward readout gives value.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Successor Representation Predictive Map", "build", "example_input", "1993", "DB")
]


class SuccessorRepresentation(nn.Module):
    """State-to-successor map with value readout."""

    def __init__(self, n_states: int = 64) -> None:
        """Initialize successor matrix and reward weights.

        Parameters
        ----------
        n_states
            Number of discrete state features.
        """
        super().__init__()
        self.successor = nn.Parameter(torch.eye(n_states))
        self.reward = nn.Parameter(torch.randn(n_states, 1) * 0.05)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Compute successor features and value.

        Parameters
        ----------
        state
            One-hot or distributed state tensor of shape ``(batch, n_states)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Scalar value and successor feature row.
        """
        sr = state @ self.successor
        value = sr @ self.reward
        return value, sr


def build() -> nn.Module:
    """Build a small successor-representation module.

    Returns
    -------
    nn.Module
        Configured ``SuccessorRepresentation`` instance.
    """
    return SuccessorRepresentation()


def example_input() -> Tensor:
    """Return a one-hot state example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 64)``.
    """
    x = torch.zeros(1, 64)
    x[:, 3] = 1.0
    return x
