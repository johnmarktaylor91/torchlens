"""Simoncelli-Freeman steerable pyramid, 1995, Simoncelli and Freeman.

Paper: "The steerable pyramid: A flexible architecture for multi-scale derivative
computation." Fixed oriented derivative filters at successive scales form a compact
trace-clean approximation of the steerable-pyramid decomposition.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Simoncelli-Freeman Steerable Pyramid", "build", "example_input", "1995", "DA")
]


class SteerablePyramid(nn.Module):
    """Multi-scale oriented derivative filter bank."""

    def __init__(self, n_orientations: int = 4, kernel_size: int = 7) -> None:
        """Initialize fixed oriented derivative filters.

        Parameters
        ----------
        n_orientations
            Number of orientations per scale.
        kernel_size
            Spatial filter width.
        """
        super().__init__()
        grid = torch.linspace(-1.0, 1.0, kernel_size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        filters = []
        for index in range(n_orientations):
            theta = math.pi * float(index) / float(n_orientations)
            coord = xx * math.cos(theta) + yy * math.sin(theta)
            filt = coord * torch.exp(-(xx.pow(2) + yy.pow(2)) / 0.5)
            filters.append(filt - filt.mean())
        self.register_buffer("filters", torch.stack(filters).unsqueeze(1))
        self.lowpass = nn.AvgPool2d(2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute two oriented subband scales and a residual lowpass.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 128, 128)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            First-scale bands, second-scale bands, and final lowpass image.
        """
        pad = self.filters.shape[-1] // 2
        bands_1 = F.conv2d(x, self.filters, padding=pad)
        low_1 = self.lowpass(x)
        bands_2 = F.conv2d(low_1, self.filters, padding=pad)
        low_2 = self.lowpass(low_1)
        return bands_1, bands_2, low_2


def build() -> nn.Module:
    """Build a small steerable-pyramid module.

    Returns
    -------
    nn.Module
        Configured ``SteerablePyramid`` instance.
    """
    return SteerablePyramid()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 128, 128)``.
    """
    return torch.randn(1, 1, 128, 128)
