"""Counterpropagation Network, 1987, Hecht-Nielsen.

Paper: "Counterpropagation Networks." A Kohonen competitive layer selects a
prototype or neighborhood code, and a Grossberg outstar maps that code to an
associated output pattern.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CounterpropagationNetwork(nn.Module):
    """Kohonen winner-take-all front end with Grossberg outstar readout."""

    def __init__(
        self, n_in: int = 6, n_units: int = 7, n_out: int = 3, temperature: float = 0.15
    ) -> None:
        """Initialize the CPN.

        Parameters
        ----------
        n_in:
            Input feature dimensionality.
        n_units:
            Number of competitive Kohonen units.
        n_out:
            Grossberg outstar output dimensionality.
        temperature:
            Soft neighborhood temperature for traceable WTA.
        """
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_units, n_in))
        self.outstar = nn.Linear(n_units, n_out, bias=False)
        self.temperature = temperature

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Select a competitive code and read out an associated pattern.

        Parameters
        ----------
        x:
            Input tensor with shape ``(B, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Outstar output, soft competitive code, and squared distances.
        """
        distances = torch.cdist(x, self.prototypes).pow(2.0)
        code = torch.softmax(-distances / self.temperature, dim=-1)
        return self.outstar(code), code, distances


def build() -> nn.Module:
    """Build a small random-init CPN.

    Returns
    -------
    nn.Module
        A traceable ``CounterpropagationNetwork`` instance.
    """
    return CounterpropagationNetwork()


def example_input() -> Tensor:
    """Return continuous CPN examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 6)``.
    """
    return torch.tensor([[0.2, -0.1, 0.7, 0.4, -0.3, 0.5], [-0.6, 0.8, 0.1, -0.2, 0.9, 0.3]])
