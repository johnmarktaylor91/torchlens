"""Random Neural Network, 1989, Erol Gelenbe.

Paper: "Random Neural Networks with Negative and Positive Signals and Product Form Solution."
Positive and negative spike rates determine stationary excitation probabilities
through a fixed-point queueing recurrence rather than a deterministic RNN cell.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GelenbeRandomNeuralNetwork(nn.Module):
    """Fixed-point positive and negative signal-rate Random Neural Network."""

    def __init__(self, n_units: int = 32, steps: int = 8) -> None:
        """Initialize positive and negative spike-rate parameters.

        Parameters
        ----------
        n_units:
            Number of stochastic neurons.
        steps:
            Number of fixed-point iterations.
        """
        super().__init__()
        self.steps = steps
        self.register_buffer("w_pos", torch.rand(n_units, n_units) * 0.05)
        self.register_buffer("w_neg", torch.rand(n_units, n_units) * 0.04)
        self.register_buffer("external_pos", torch.rand(n_units) * 0.15 + 0.05)
        self.register_buffer("external_neg", torch.rand(n_units) * 0.10 + 0.03)
        self.register_buffer("rates", torch.rand(n_units) * 0.5 + 0.9)

    def forward(self, x: Tensor) -> Tensor:
        """Solve excitation probabilities from input-modulated arrival rates.

        Parameters
        ----------
        x:
            Nonnegative input rate tensor with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Stationary excitation probabilities.
        """
        q = torch.sigmoid(x)
        pos = self.external_pos + torch.relu(x)
        neg = self.external_neg
        for _ in range(self.steps):
            arrival_pos = pos + q @ self.w_pos
            arrival_neg = neg + q @ self.w_neg
            q = arrival_pos / (self.rates + arrival_neg).clamp_min(1.0e-5)
            q = q.clamp(0.0, 0.995)
        return q


def build() -> nn.Module:
    """Build a small Gelenbe Random Neural Network.

    Returns
    -------
    nn.Module
        Configured ``GelenbeRandomNeuralNetwork`` instance.
    """
    return GelenbeRandomNeuralNetwork()


def example_input() -> Tensor:
    """Create an input-rate example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.rand(1, 32) * 0.2


MENAGERIE_ENTRIES = [("Random Neural Network (Gelenbe)", "build", "example_input", "1989", "MB1")]
