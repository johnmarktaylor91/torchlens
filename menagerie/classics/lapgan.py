"""LAPGAN, 2015, Emily Denton et al.

Paper: Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks.
A generator cascade samples a coarse image, upsamples it, and predicts
conditional residuals at each finer Laplacian-pyramid level.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualGenerator(nn.Module):
    """Conditional residual generator for one Laplacian level."""

    def __init__(self, in_channels: int, noise_channels: int, hidden_channels: int = 12) -> None:
        """Initialize a conditional residual block.

        Parameters
        ----------
        in_channels:
            Conditioning image channels.
        noise_channels:
            Per-level noise channels.
        hidden_channels:
            Hidden convolution channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + noise_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    def forward(self, image: Tensor, noise: Tensor) -> Tensor:
        """Generate a residual conditioned on an upsampled image.

        Parameters
        ----------
        image:
            Upsampled conditioning image.
        noise:
            Noise map at the same resolution.

        Returns
        -------
        Tensor
            Residual image.
        """
        return self.net(torch.cat([image, noise], dim=1))


class LAPGAN(nn.Module):
    """Small Laplacian-pyramid generator cascade."""

    def __init__(self, noise_dim: int = 16, channels: int = 3, levels: int = 3) -> None:
        """Initialize coarse and residual generators.

        Parameters
        ----------
        noise_dim:
            Flat input noise dimension.
        channels:
            Generated image channels.
        levels:
            Number of pyramid levels.
        """
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.noise_channels = 4
        self.coarse = nn.Linear(noise_dim, channels * 8 * 8)
        self.noise_maps = nn.ModuleList(
            [
                nn.Linear(noise_dim, self.noise_channels * 8 * (2**level) * 8 * (2**level))
                for level in range(1, levels)
            ]
        )
        self.generators = nn.ModuleList(
            [ResidualGenerator(channels, self.noise_channels) for _ in range(1, levels)]
        )

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image through a Laplacian pyramid.

        Parameters
        ----------
        z:
            Flat noise tensor ``(B, noise_dim)``.

        Returns
        -------
        Tensor
            Generated image tensor.
        """
        batch = z.shape[0]
        image = self.coarse(z).view(batch, self.channels, 8, 8)
        for level, generator in enumerate(self.generators, start=1):
            size = 8 * (2**level)
            image = F.interpolate(image, size=(size, size), mode="nearest")
            noise = self.noise_maps[level - 1](z).view(batch, self.noise_channels, size, size)
            image = image + generator(image, noise)
        return torch.tanh(image)


def build() -> nn.Module:
    """Build a compact LAPGAN generator.

    Returns
    -------
    nn.Module
        Random-initialized LAPGAN.
    """
    return LAPGAN()


def example_input() -> Tensor:
    """Return a traceable noise batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)
