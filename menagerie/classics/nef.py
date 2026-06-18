"""Neural Engineering Framework, 2003.

Eliasmith and Anderson's NEF represents vectors through heterogeneous encoder
populations with nonlinear tuning curves and decodes transforms by linear decoders.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NEFPopulation(nn.Module):
    """Small NEF population with LIF-rate tuning and decoded output."""

    def __init__(self, d: int = 3, n_neurons: int = 32) -> None:
        """Initialize encoders, gains, biases, and decoders.

        Parameters
        ----------
        d
            Represented vector dimension.
        n_neurons
            Number of neurons in the population.
        """
        super().__init__()
        enc = torch.randn(n_neurons, d)
        enc = enc / enc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("encoders", enc)
        self.gain = nn.Parameter(torch.rand(n_neurons) * 1.5 + 0.5)
        self.bias = nn.Parameter(torch.rand(n_neurons) * 0.2)
        self.decoder = nn.Linear(n_neurons, d, bias=False)

    def _lif_rate(self, current: Tensor) -> Tensor:
        """Compute rectified LIF-rate responses.

        Parameters
        ----------
        current
            Input current.

        Returns
        -------
        Tensor
            Firing rates.
        """
        j = torch.relu(current - 1.0) + 1.0
        return 1.0 / (0.002 + torch.log1p(1.0 / (j - 1.0 + 1e-3)))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a vector population and decode it back to vector space.

        Parameters
        ----------
        x
            Represented vectors of shape ``(batch, d)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Decoded vector and neural activities.
        """
        current = x @ self.encoders.t() * self.gain + self.bias
        activity = self._lif_rate(current)
        decoded = self.decoder(activity)
        return decoded, activity


def build() -> nn.Module:
    """Build a small NEF population.

    Returns
    -------
    nn.Module
        Random NEF population module.
    """
    return NEFPopulation()


def example_input() -> Tensor:
    """Return a float32 represented vector.

    Returns
    -------
    Tensor
        Input of shape ``(2, 3)``.
    """
    return torch.randn(2, 3, dtype=torch.float32)
