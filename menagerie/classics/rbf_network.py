"""Radial Basis Function Network, 1988/1989, Broomhead and Lowe.

Paper: Broomhead and Lowe 1988, "Multivariable functional interpolation and
adaptive networks." Gaussian radial units compare inputs to learned centers and
feed a linear output readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Radial Basis Function Network (RBF-Net)", "build", "example_input", "1988/1989", "CF")
]


class RBFNet(nn.Module):
    """Gaussian radial-basis hidden layer with linear readout."""

    def __init__(self, dim: int = 4, n_centers: int = 6, output_size: int = 2) -> None:
        """Initialize centers, widths, and readout.

        Parameters
        ----------
        dim
            Input dimensionality.
        n_centers
            Number of Gaussian basis functions.
        output_size
            Number of output features.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.randn(n_centers, dim))
        self.log_sigma = nn.Parameter(torch.zeros(n_centers))
        self.readout = nn.Linear(n_centers, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate radial basis functions and readout.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Readout tensor.
        """
        dist2 = torch.cdist(x, self.centers) ** 2
        sigma2 = self.log_sigma.exp().square().clamp_min(1.0e-6)
        hidden = torch.exp(-dist2 / (2.0 * sigma2))
        return self.readout(hidden)


def build() -> nn.Module:
    """Build a small RBF network.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return RBFNet()


def example_input() -> Tensor:
    """Return vector inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(3, 4)``.
    """
    return torch.randn(3, 4)
