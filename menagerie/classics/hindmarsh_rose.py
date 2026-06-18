"""Hindmarsh-Rose bursting neuron, 1984, Hindmarsh and Rose.

Paper: Hindmarsh and Rose 1984, "A model of neuronal bursting using three
coupled first order differential equations." Fast voltage and recovery variables
interact with slow adaptation to produce burst-like dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Hindmarsh-Rose bursting neuron", "build", "example_input", "1984", "CF")]


class HindmarshRose(nn.Module):
    """Euler-integrated three-variable Hindmarsh-Rose neuron."""

    def __init__(self, dt: float = 0.05) -> None:
        """Initialize canonical bursting parameters.

        Parameters
        ----------
        dt
            Euler integration step size.
        """
        super().__init__()
        self.dt = dt
        self.a = 1.0
        self.b = 3.0
        self.c = 1.0
        self.d = 5.0
        self.r = 0.006
        self.s = 4.0
        self.x_r = -1.6

    def forward(self, current: Tensor) -> Tensor:
        """Integrate the neuron for an injected-current sequence.

        Parameters
        ----------
        current
            Current tensor of shape ``(batch, time, 1)``.

        Returns
        -------
        Tensor
            Voltage trajectory of shape ``(batch, time, 1)``.
        """
        batch = current.shape[0]
        x = current.new_full((batch, 1), -1.2)
        y = current.new_full((batch, 1), -8.0)
        z = current.new_full((batch, 1), 3.0)
        outputs: list[Tensor] = []
        for step in range(current.shape[1]):
            i_t = current[:, step]
            dx = y - self.a * x**3 + self.b * x**2 + i_t - z
            dy = self.c - self.d * x**2 - y
            dz = self.r * (self.s * (x - self.x_r) - z)
            x = x + self.dt * dx
            y = y + self.dt * dy
            z = z + self.dt * dz
            outputs.append(x)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a Hindmarsh-Rose neuron.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return HindmarshRose()


def example_input() -> Tensor:
    """Return an injected-current sequence.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 1)``.
    """
    return torch.full((2, 8, 1), 2.5)
