"""Schwartz-Simoncelli wavelet gain control, 2001, Schwartz and Simoncelli.

Paper: "Natural signal statistics and sensory gain control." Oriented wavelet-like
coefficients are normalized by local and cross-channel energy pools, reducing
natural-image dependencies in a trace-clean feedforward substrate.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Schwartz-Simoncelli Wavelet Gain Control", "build", "example_input", "2001", "DA")
]


class WaveletGainControl(nn.Module):
    """Fixed oriented filter bank with local divisive gain control."""

    def __init__(self, n_orientations: int = 6, kernel_size: int = 9) -> None:
        """Initialize oriented derivative filters and pooling weights.

        Parameters
        ----------
        n_orientations
            Number of oriented subbands.
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
            envelope = torch.exp(-(xx.pow(2) + yy.pow(2)) / 0.4)
            filt = coord * envelope
            filters.append(filt - filt.mean())
        weight = torch.stack(filters).unsqueeze(1)
        self.register_buffer("weight", weight)
        self.pool = nn.Conv2d(
            n_orientations, n_orientations, 5, padding=2, groups=n_orientations, bias=False
        )
        nn.init.constant_(self.pool.weight, 1.0 / 25.0)
        self.channel_gain = nn.Parameter(torch.ones(1, n_orientations, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Compute gain-controlled oriented coefficients.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Normalized oriented coefficients.
        """
        coeff = F.conv2d(x, self.weight, padding=self.weight.shape[-1] // 2)
        pool = self.pool(coeff.pow(2))
        return coeff / torch.sqrt(torch.relu(self.channel_gain) * torch.relu(pool) + 1.0e-4)


def build() -> nn.Module:
    """Build a small wavelet gain-control module.

    Returns
    -------
    nn.Module
        Configured ``WaveletGainControl`` instance.
    """
    return WaveletGainControl()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)
