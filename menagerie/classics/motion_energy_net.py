"""Adelson-Bergen motion energy model, 1985, Adelson and Bergen.

Paper: "Spatiotemporal energy models for the perception of motion." Quadrature
spatiotemporal filters are squared and summed, then opponent directions are
subtracted to produce direction-selective motion energy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Adelson-Bergen Motion Energy Model", "build", "example_input", "1985", "DA")]


class MotionEnergyNet(nn.Module):
    """Quadrature 3D convolutional motion-energy filters."""

    def __init__(self, n_pairs: int = 4) -> None:
        """Initialize even and odd spatiotemporal filters.

        Parameters
        ----------
        n_pairs
            Number of quadrature direction pairs.
        """
        super().__init__()
        self.even = nn.Conv3d(1, n_pairs, kernel_size=(5, 7, 7), padding=(2, 3, 3), bias=False)
        self.odd = nn.Conv3d(1, n_pairs, kernel_size=(5, 7, 7), padding=(2, 3, 3), bias=False)
        self.opponent = nn.Conv3d(n_pairs, n_pairs, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute motion energy and opponent direction signal.

        Parameters
        ----------
        x
            Movie tensor of shape ``(batch, 1, time, height, width)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Raw quadrature energy and opponent-normalized motion response.
        """
        energy = self.even(x).pow(2) + self.odd(x).pow(2)
        opponent = self.opponent(energy)
        return energy, opponent


def build() -> nn.Module:
    """Build a small motion-energy module.

    Returns
    -------
    nn.Module
        Configured ``MotionEnergyNet`` instance.
    """
    return MotionEnergyNet()


def example_input() -> Tensor:
    """Return a movie example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 16, 32, 32)``.
    """
    return torch.randn(1, 1, 16, 32, 32)
