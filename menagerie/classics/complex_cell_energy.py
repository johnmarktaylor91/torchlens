"""Complex cell energy model, 1982, Adelson and Bergen.

Paper: "Spatiotemporal energy models for the perception of motion." A quadrature
Gabor pair is squared and summed to produce phase-invariant complex-cell energy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Complex Cell Energy Model", "build", "example_input", "1982", "DA")]


class ComplexCellEnergy(nn.Module):
    """Learned quadrature-like filter pair with energy pooling."""

    def __init__(self, n_channels: int = 8) -> None:
        """Initialize cosine and sine response filters.

        Parameters
        ----------
        n_channels
            Number of complex-cell channels.
        """
        super().__init__()
        self.cos_filter = nn.Conv2d(1, n_channels, 9, padding=4, bias=False)
        self.sin_filter = nn.Conv2d(1, n_channels, 9, padding=4, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute phase-invariant complex-cell energy.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Complex-cell energy responses.
        """
        real = self.cos_filter(x)
        imag = self.sin_filter(x)
        return torch.sqrt(real.pow(2) + imag.pow(2) + 1.0e-6)


def build() -> nn.Module:
    """Build a small complex-cell energy module.

    Returns
    -------
    nn.Module
        Configured ``ComplexCellEnergy`` instance.
    """
    return ComplexCellEnergy()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
