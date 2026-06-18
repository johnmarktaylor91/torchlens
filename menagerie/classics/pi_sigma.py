"""Pi-Sigma network, 1991, Shin and Ghosh.

Paper: "The Pi-Sigma Network: An Efficient Higher-Order Neural Network."
The network forms affine sums in several hidden channels and multiplies those
sums, yielding higher-order interactions with far fewer weights.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PiSigmaNetwork(nn.Module):
    """Higher-order product-of-sums network."""

    def __init__(self, n_in: int = 5, n_factors: int = 4, n_out: int = 2) -> None:
        """Initialize a Pi-Sigma network.

        Parameters
        ----------
        n_in:
            Number of input features.
        n_factors:
            Number of affine sums multiplied together.
        n_out:
            Number of independent product outputs.
        """
        super().__init__()
        self.sums = nn.Linear(n_in, n_factors * n_out)
        self.n_factors = n_factors
        self.n_out = n_out

    def forward(self, x: Tensor) -> Tensor:
        """Compute products of learned affine sums.

        Parameters
        ----------
        x:
            Input tensor with shape ``(B, n_in)``.

        Returns
        -------
        Tensor
            Tanh-compressed product outputs.
        """
        sums = self.sums(x).view(x.shape[0], self.n_out, self.n_factors)
        return torch.tanh(sums.prod(dim=-1))


def build() -> nn.Module:
    """Build a small random-init Pi-Sigma network.

    Returns
    -------
    nn.Module
        A traceable ``PiSigmaNetwork`` instance.
    """
    return PiSigmaNetwork()


def example_input() -> Tensor:
    """Return a continuous input batch.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 5)``.
    """
    return torch.tensor([[0.3, 0.7, -0.2, 0.4, 0.9], [-0.6, 0.2, 0.5, -0.1, 0.8]])
