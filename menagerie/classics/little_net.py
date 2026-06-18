"""Little persistent-state recurrent net (1974), William Little.

Paper: "The existence of persistent states in the brain."
A symmetric binary threshold recurrent network synchronously relaxes toward
persistent attractor states, anticipating the Hopfield formulation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LittleNet(nn.Module):
    """Binary recurrent threshold attractor with persistent-state dynamics."""

    def __init__(self, n_units: int = 6, steps: int = 4) -> None:
        """Initialize symmetric recurrent weights.

        Parameters
        ----------
        n_units
            Number of recurrent binary units.
        steps
            Number of synchronous relaxation steps.
        """
        super().__init__()
        weights = torch.randn(n_units, n_units)
        weights = 0.5 * (weights + weights.T)
        weights.fill_diagonal_(0.0)
        self.steps = steps
        self.register_buffer("weights", weights / n_units**0.5)
        self.register_buffer("bias", torch.zeros(n_units))

    def forward(self, x: Tensor) -> Tensor:
        """Relax an initial state by synchronous threshold updates.

        Parameters
        ----------
        x
            Initial bipolar state of shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Relaxed bipolar state.
        """
        state = x
        for _ in range(self.steps):
            field = state @ self.weights - self.bias
            state = torch.where(field >= 0.0, torch.ones_like(state), -torch.ones_like(state))
        return state

    def energy(self, state: Tensor) -> Tensor:
        """Compute Little/Hopfield-style quadratic energy.

        Parameters
        ----------
        state
            Bipolar state tensor.

        Returns
        -------
        Tensor
            Per-example energy.
        """
        interaction = (state @ self.weights * state).sum(dim=-1)
        return -0.5 * interaction + (state * self.bias).sum(dim=-1)


def build() -> nn.Module:
    """Build a small Little recurrent net.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return LittleNet()


def example_input() -> Tensor:
    """Return an example bipolar state.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
