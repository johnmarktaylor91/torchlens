"""Gabor Jet dynamic-link architecture, 1991, Lades, von der Malsburg, and colleagues.

Paper: "Distortion invariant object recognition in the dynamic link architecture."
Images are represented by complex Gabor filter responses sampled at graph nodes; this
minimal version returns amplitude and phase jets for a fixed node grid.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ENTRIES = [
    ("Gabor Jet / Dynamic Link Architecture", "build", "example_input", "1991", "DA")
]


class GaborJetNet(nn.Module):
    """Fixed complex Gabor bank sampled on graph nodes."""

    def __init__(self, n_orientations: int = 4, kernel_size: int = 11, grid_size: int = 4) -> None:
        """Initialize quadrature filters and graph-node coordinates.

        Parameters
        ----------
        n_orientations
            Number of Gabor orientations.
        kernel_size
            Spatial filter width.
        grid_size
            Number of sampled graph nodes per spatial axis.
        """
        super().__init__()
        grid = torch.linspace(-1.5, 1.5, kernel_size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        cos_filters = []
        sin_filters = []
        for index in range(n_orientations):
            theta = math.pi * float(index) / float(n_orientations)
            coord = xx * math.cos(theta) + yy * math.sin(theta)
            envelope = torch.exp(-(xx.pow(2) + yy.pow(2)) / 1.2)
            cos_filters.append(envelope * torch.cos(5.0 * coord))
            sin_filters.append(envelope * torch.sin(5.0 * coord))
        self.register_buffer("cos_weight", torch.stack(cos_filters).unsqueeze(1))
        self.register_buffer("sin_weight", torch.stack(sin_filters).unsqueeze(1))
        coords = torch.linspace(-0.75, 0.75, grid_size)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer(
            "nodes", torch.stack((gx, gy), dim=-1).view(1, grid_size * grid_size, 1, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return sampled amplitude and phase jets.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 128, 128)``.

        Returns
        -------
        Tensor
            Jet tensor of shape ``(batch, nodes, 2 * orientations)``.
        """
        padding = self.cos_weight.shape[-1] // 2
        real = F.conv2d(x, self.cos_weight, padding=padding)
        imag = F.conv2d(x, self.sin_weight, padding=padding)
        amplitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1.0e-6)
        phase = torch.atan2(imag, real + 1.0e-6)
        jets = torch.cat((amplitude, phase), dim=1)
        sampled = F.grid_sample(
            jets, self.nodes.expand(x.shape[0], -1, -1, -1), align_corners=False
        )
        return sampled.squeeze(-1).transpose(1, 2)


def build() -> nn.Module:
    """Build a small Gabor jet module.

    Returns
    -------
    nn.Module
        Configured ``GaborJetNet`` instance.
    """
    return GaborJetNet()


def example_input() -> Tensor:
    """Return a grayscale face-like image tensor.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 128, 128)``.
    """
    return torch.randn(1, 1, 128, 128)
