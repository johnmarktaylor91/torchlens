"""Brain-State-in-a-Box (1977), James Anderson, Jack Silverstein, and colleagues.

Paper: "Distinctive features, categorical perception, and probability learning."
A clipped-linear recurrent autoassociator integrates correlation-memory input
until states settle on corners of the bounded hypercube.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BrainStateInABox(nn.Module):
    """Clipped-linear recurrent autoassociative memory."""

    def __init__(self, n_units: int = 6, steps: int = 5, alpha: float = 0.35) -> None:
        """Initialize recurrent correlation weights.

        Parameters
        ----------
        n_units
            Number of state dimensions.
        steps
            Number of recurrent integration steps.
        alpha
            Recurrent update rate.
        """
        super().__init__()
        patterns = torch.eye(n_units) * 2.0 - 1.0
        weights = patterns.T @ patterns / n_units
        weights.fill_diagonal_(0.0)
        self.steps = steps
        self.alpha = alpha
        self.register_buffer("weights", weights)

    def forward(self, x: Tensor) -> Tensor:
        """Run clipped-linear BSB recurrence.

        Parameters
        ----------
        x
            Initial state of shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Bounded recalled state.
        """
        state = x
        for _ in range(self.steps):
            state = torch.clamp(state + self.alpha * (state @ self.weights), -1.0, 1.0)
        return state


def build() -> nn.Module:
    """Build a small Brain-State-in-a-Box module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return BrainStateInABox()


def example_input() -> Tensor:
    """Return an example bounded memory cue.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[0.8, -0.7, 0.2, -0.1, 0.4, -0.6]])
