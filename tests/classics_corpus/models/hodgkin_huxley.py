"""Hodgkin-Huxley neuron, 1952.

Hodgkin and Huxley, "A quantitative description of membrane current..." Sodium,
potassium, and leak conductances are integrated as an unrolled differentiable ODE.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HodgkinHuxleyNeuron(nn.Module):
    """Differentiable Euler-unrolled Hodgkin-Huxley membrane."""

    def __init__(self, dt: float = 0.01) -> None:
        """Initialize conductance constants.

        Parameters
        ----------
        dt
            Euler step size in milliseconds.
        """
        super().__init__()
        self.dt = dt
        self.cm = 1.0
        self.g_na = 120.0
        self.g_k = 36.0
        self.g_l = 0.3
        self.e_na = 50.0
        self.e_k = -77.0
        self.e_l = -54.387

    def _safe_rate(self, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Return a numerically stable voltage-rate ratio.

        Parameters
        ----------
        numerator
            Rate numerator.
        denominator
            Rate denominator.

        Returns
        -------
        Tensor
            Stabilized ratio.
        """
        return numerator / denominator.clamp_min(1e-4)

    def forward(self, current: Tensor) -> Tensor:
        """Integrate membrane voltage for an input-current sequence.

        Parameters
        ----------
        current
            Input current of shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Voltage trace of shape ``(batch, time)``.
        """
        v = current.new_full((current.shape[0],), -65.0)
        m = current.new_full((current.shape[0],), 0.05)
        h = current.new_full((current.shape[0],), 0.6)
        n = current.new_full((current.shape[0],), 0.32)
        voltages: list[Tensor] = []
        for t in range(current.shape[1]):
            v_shift = v + 65.0
            alpha_m = self._safe_rate(
                0.1 * (25.0 - v_shift), torch.exp((25.0 - v_shift) / 10.0) - 1.0
            )
            beta_m = 4.0 * torch.exp(-v_shift / 18.0)
            alpha_h = 0.07 * torch.exp(-v_shift / 20.0)
            beta_h = 1.0 / (torch.exp((30.0 - v_shift) / 10.0) + 1.0)
            alpha_n = self._safe_rate(
                0.01 * (10.0 - v_shift), torch.exp((10.0 - v_shift) / 10.0) - 1.0
            )
            beta_n = 0.125 * torch.exp(-v_shift / 80.0)
            m = m + self.dt * (alpha_m * (1.0 - m) - beta_m * m)
            h = h + self.dt * (alpha_h * (1.0 - h) - beta_h * h)
            n = n + self.dt * (alpha_n * (1.0 - n) - beta_n * n)
            i_na = self.g_na * m.pow(3) * h * (v - self.e_na)
            i_k = self.g_k * n.pow(4) * (v - self.e_k)
            i_l = self.g_l * (v - self.e_l)
            v = v + self.dt * (current[:, t] - i_na - i_k - i_l) / self.cm
            voltages.append(v)
        return torch.stack(voltages, dim=1)


def build() -> nn.Module:
    """Build a small Hodgkin-Huxley neuron module.

    Returns
    -------
    nn.Module
        Hodgkin-Huxley neuron.
    """
    return HodgkinHuxleyNeuron()


def example_input() -> Tensor:
    """Return a float32 current sequence.

    Returns
    -------
    Tensor
        Input current of shape ``(2, 8)``.
    """
    return torch.full((2, 8), 10.0, dtype=torch.float32)
