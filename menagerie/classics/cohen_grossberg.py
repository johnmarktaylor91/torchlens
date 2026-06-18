"""Cohen-Grossberg competitive network, 1983.

Cohen and Grossberg described Lyapunov-stable recurrent competitive dynamics with
state-dependent gain, decay, activation, and recurrent interactions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CohenGrossbergNet(nn.Module):
    """Rate RNN following the Cohen-Grossberg competitive form."""

    def __init__(self, n_units: int = 12, steps: int = 8, dt: float = 0.1) -> None:
        """Initialize recurrent competitive dynamics.

        Parameters
        ----------
        n_units
            Number of rate units.
        steps
            Number of integration steps.
        dt
            Euler update size.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.recurrent = nn.Parameter(torch.randn(n_units, n_units) * 0.12)
        self.bias = nn.Parameter(torch.zeros(n_units))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Integrate competitive dynamics from an initial cue.

        Parameters
        ----------
        x
            Initial cue of shape ``(batch, n_units)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Final state and a quadratic energy proxy.
        """
        state = torch.relu(x)
        for _ in range(self.steps):
            gain = 1.0 + state.square()
            decay = state
            activation = torch.sigmoid(state)
            recurrent_drive = activation @ self.recurrent.t() + self.bias + x
            state = state + self.dt * gain * (-decay + recurrent_drive)
            state = torch.relu(state)
        energy = 0.5 * state.square().sum(dim=-1) - 0.5 * (
            activation @ self.recurrent * activation
        ).sum(dim=-1)
        return state, energy


def build() -> nn.Module:
    """Build a Cohen-Grossberg competitive network.

    Returns
    -------
    nn.Module
        Random-initialized Cohen-Grossberg module.
    """
    return CohenGrossbergNet()


def example_input() -> Tensor:
    """Return a float32 cue.

    Returns
    -------
    Tensor
        Cue of shape ``(2, 12)``.
    """
    return torch.rand(2, 12, dtype=torch.float32)
