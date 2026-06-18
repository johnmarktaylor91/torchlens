"""Bidirectional Associative Memory, 1988, Bart Kosko.

Paper: "Bidirectional Associative Memories." BAM stores paired bipolar
patterns in a heteroassociative matrix and recalls them by alternating
threshold updates between the two layers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BidirectionalAssociativeMemory(nn.Module):
    """Bipolar heteroassociative recurrent memory."""

    def __init__(self, n_x: int = 6, n_y: int = 5, steps: int = 4) -> None:
        """Initialize the BAM.

        Parameters
        ----------
        n_x:
            Size of the first pattern layer.
        n_y:
            Size of the second pattern layer.
        steps:
            Number of alternating recall cycles.
        """
        super().__init__()
        x_patterns = torch.sign(torch.randn(4, n_x))
        y_patterns = torch.sign(torch.randn(4, n_y))
        self.register_buffer("weight", x_patterns.transpose(0, 1) @ y_patterns / 4.0)
        self.steps = steps

    @staticmethod
    def bipolar_sign(value: Tensor) -> Tensor:
        """Return a bipolar sign with zeros mapped to +1.

        Parameters
        ----------
        value:
            Input activations.

        Returns
        -------
        Tensor
            Bipolar activations in ``{-1, +1}``.
        """
        return torch.where(value >= 0.0, torch.ones_like(value), -torch.ones_like(value))

    def energy(self, x_state: Tensor, y_state: Tensor) -> Tensor:
        """Compute BAM Lyapunov energy.

        Parameters
        ----------
        x_state:
            Bipolar state for the first layer.
        y_state:
            Bipolar state for the second layer.

        Returns
        -------
        Tensor
            Per-example heteroassociative energy.
        """
        return -((x_state @ self.weight) * y_state).sum(dim=-1)

    def forward(self, x_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Recall a paired pattern by alternating layer updates.

        Parameters
        ----------
        x_state:
            Initial bipolar pattern with shape ``(B, n_x)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Recalled ``x`` pattern, recalled ``y`` pattern, and final energy.
        """
        x = x_state
        y = self.bipolar_sign(x @ self.weight)
        for _ in range(self.steps):
            x = self.bipolar_sign(y @ self.weight.transpose(0, 1))
            y = self.bipolar_sign(x @ self.weight)
        return x, y, self.energy(x, y)


def build() -> nn.Module:
    """Build a small random-init BAM.

    Returns
    -------
    nn.Module
        A traceable ``BidirectionalAssociativeMemory`` instance.
    """
    return BidirectionalAssociativeMemory()


def example_input() -> Tensor:
    """Return a bipolar query pattern.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 6)``.
    """
    return torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]])
