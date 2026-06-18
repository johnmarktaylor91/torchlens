"""FitzHugh-Nagumo excitable neuron, 1961.

FitzHugh and Nagumo reduced Hodgkin-Huxley excitability to voltage and recovery
variables, yielding a traceable relaxation oscillator forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FitzHughNagumoNeuron(nn.Module):
    """Euler-unrolled FitzHugh-Nagumo neuron."""

    def __init__(self, dt: float = 0.05, a: float = 0.7, b: float = 0.8, tau: float = 12.5) -> None:
        """Initialize FHN constants.

        Parameters
        ----------
        dt
            Euler step size.
        a
            Recovery offset.
        b
            Recovery damping.
        tau
            Recovery time constant.
        """
        super().__init__()
        self.dt = dt
        self.a = a
        self.b = b
        self.tau = tau

    def forward(self, current: Tensor) -> Tensor:
        """Integrate voltage under input current.

        Parameters
        ----------
        current
            Input current of shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Voltage trace.
        """
        v = current.new_zeros(current.shape[0])
        w = current.new_zeros(current.shape[0])
        voltage: list[Tensor] = []
        for t in range(current.shape[1]):
            dv = v - v.pow(3) / 3.0 - w + current[:, t]
            dw = (v + self.a - self.b * w) / self.tau
            v = v + self.dt * dv
            w = w + self.dt * dw
            voltage.append(v)
        return torch.stack(voltage, dim=1)


def build() -> nn.Module:
    """Build a FitzHugh-Nagumo neuron module.

    Returns
    -------
    nn.Module
        FitzHugh-Nagumo neuron.
    """
    return FitzHughNagumoNeuron()


def example_input() -> Tensor:
    """Return a float32 current sequence.

    Returns
    -------
    Tensor
        Input current of shape ``(2, 24)``.
    """
    return torch.full((2, 24), 0.8, dtype=torch.float32)
