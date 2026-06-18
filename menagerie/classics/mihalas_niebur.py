"""Mihalas-Niebur generalized integrate-and-fire neuron, 2009.

Paper: Mihalas and Niebur 2009, "A generalized linear integrate-and-fire neural
model produces diverse spiking behaviors." Adaptive threshold and internal
currents reproduce tonic, phasic, bursting, and adapting regimes. This minimal
version uses smooth surrogate spike/reset signals for traceable dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Mihalas-Niebur Generalized IF neuron", "build", "example_input", "2009", "CF")
]


class MihalasNieburNeuron(nn.Module):
    """Generalized integrate-and-fire neuron with two internal currents."""

    def __init__(self, dt: float = 0.1) -> None:
        """Initialize reduced generalized IF parameters.

        Parameters
        ----------
        dt
            Euler integration step size.
        """
        super().__init__()
        self.dt = dt
        self.e_l = -0.65
        self.theta_inf = -0.45
        self.v_reset = -0.7
        self.r_m = 1.0

    def forward(self, current: Tensor) -> Tensor:
        """Integrate adaptive-threshold membrane dynamics.

        Parameters
        ----------
        current
            Input current of shape ``(batch, time, 1)``.

        Returns
        -------
        Tensor
            Smooth spike-rate trajectory.
        """
        batch = current.shape[0]
        v = current.new_full((batch, 1), self.e_l)
        theta = current.new_full((batch, 1), self.theta_inf)
        i1 = current.new_zeros(batch, 1)
        i2 = current.new_zeros(batch, 1)
        spikes: list[Tensor] = []
        for step in range(current.shape[1]):
            v = v + self.dt * (-(v - self.e_l) + self.r_m * current[:, step] + i1 + i2)
            theta = theta + self.dt * (0.4 * (v - self.e_l) - 0.2 * (theta - self.theta_inf))
            i1 = i1 + self.dt * (-i1 / 2.0)
            i2 = i2 + self.dt * (-i2 / 6.0)
            spike = torch.sigmoid(30.0 * (v - theta))
            v = v * (1.0 - spike) + self.v_reset * spike
            theta = theta + 0.1 * spike
            i1 = i1 + 0.2 * spike
            i2 = i2 - 0.15 * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


def build() -> nn.Module:
    """Build a Mihalas-Niebur neuron.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return MihalasNieburNeuron()


def example_input() -> Tensor:
    """Return an input-current sequence.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 1)``.
    """
    return torch.randn(2, 8, 1) * 0.15 + 0.35
