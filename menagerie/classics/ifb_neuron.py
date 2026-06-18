"""Integrate-and-Fire-or-Burst neuron, 2000, Smith et al.

Paper: Smith et al. 2000, "Integrate-and-fire-or-burst neurons." A leaky
membrane couples to a slow T-type calcium inactivation gate, switching between
tonic and rebound burst regimes. This traceable version uses smooth surrogate
spike/reset signals instead of hard threshold events.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Integrate-and-Fire-or-Burst (IFB)", "build", "example_input", "2000", "CF")]


class IFBNeuron(nn.Module):
    """Smooth integrate-and-fire-or-burst membrane model."""

    def __init__(self, dt: float = 0.1) -> None:
        """Initialize reduced thalamic relay-cell parameters.

        Parameters
        ----------
        dt
            Euler integration step size.
        """
        super().__init__()
        self.dt = dt
        self.g_l = 0.1
        self.g_t = 0.25
        self.e_l = -0.7
        self.e_ca = 1.2
        self.v_h = -0.45
        self.v_thr = 0.25
        self.v_reset = -0.6

    def forward(self, current: Tensor) -> Tensor:
        """Integrate IFB dynamics over an input-current sequence.

        Parameters
        ----------
        current
            Input current of shape ``(batch, time, 1)``.

        Returns
        -------
        Tensor
            Smooth spike-rate trajectory of shape ``(batch, time, 1)``.
        """
        batch = current.shape[0]
        v = current.new_full((batch, 1), self.e_l)
        h = current.new_full((batch, 1), 0.7)
        spikes: list[Tensor] = []
        for step in range(current.shape[1]):
            i_t = current[:, step]
            m_inf = torch.sigmoid(30.0 * (v - self.v_h))
            dv = -self.g_l * (v - self.e_l) - self.g_t * h * m_inf * (v - self.e_ca) + i_t
            h_inf = torch.sigmoid(-20.0 * (v - self.v_h))
            h = h + self.dt * (h_inf - h) / 5.0
            v = v + self.dt * dv
            spike = torch.sigmoid(25.0 * (v - self.v_thr))
            v = v * (1.0 - spike) + self.v_reset * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


def build() -> nn.Module:
    """Build an IFB neuron module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return IFBNeuron()


def example_input() -> Tensor:
    """Return a current pulse sequence.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 8, 1)``.
    """
    return torch.randn(2, 8, 1) * 0.2 + 0.4
