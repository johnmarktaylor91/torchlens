"""Morris-Lecar neuron, 1981.

Morris and Lecar's barnacle muscle-fiber model integrates voltage and potassium
activation with voltage-dependent calcium, potassium, and leak conductances.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MorrisLecarNeuron(nn.Module):
    """Euler-unrolled Morris-Lecar conductance neuron."""

    def __init__(self, dt: float = 0.02) -> None:
        """Initialize Morris-Lecar parameters.

        Parameters
        ----------
        dt
            Euler step size.
        """
        super().__init__()
        self.dt = dt
        self.c = 20.0
        self.g_ca = 4.4
        self.g_k = 8.0
        self.g_l = 2.0
        self.v_ca = 120.0
        self.v_k = -84.0
        self.v_l = -60.0

    def forward(self, current: Tensor) -> Tensor:
        """Integrate Morris-Lecar voltage over time.

        Parameters
        ----------
        current
            Input current of shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Voltage trace.
        """
        v = current.new_full((current.shape[0],), -60.0)
        n = current.new_full((current.shape[0],), 0.02)
        voltage: list[Tensor] = []
        for t in range(current.shape[1]):
            m_inf = 0.5 * (1.0 + torch.tanh((v + 1.2) / 18.0))
            n_inf = 0.5 * (1.0 + torch.tanh((v - 2.0) / 30.0))
            lam_n = 0.04 * torch.cosh((v - 2.0) / 60.0)
            i_ca = self.g_ca * m_inf * (v - self.v_ca)
            i_k = self.g_k * n * (v - self.v_k)
            i_l = self.g_l * (v - self.v_l)
            v = v + self.dt * (current[:, t] - i_ca - i_k - i_l) / self.c
            n = n + self.dt * lam_n * (n_inf - n)
            voltage.append(v)
        return torch.stack(voltage, dim=1)


def build() -> nn.Module:
    """Build a Morris-Lecar neuron module.

    Returns
    -------
    nn.Module
        Morris-Lecar neuron.
    """
    return MorrisLecarNeuron()


def example_input() -> Tensor:
    """Return a float32 current sequence.

    Returns
    -------
    Tensor
        Input current of shape ``(2, 8)``.
    """
    return torch.full((2, 8), 80.0, dtype=torch.float32)
