"""Heeger divisive normalization, 1992, David Heeger.

Paper: "Normalization of cell responses in cat striate cortex." Linear cortical
drive is divided by pooled local contrast plus semi-saturation, implementing the
canonical V1 gain-control computation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Heeger Divisive Normalization", "build", "example_input", "1992", "DA")]


class HeegerDivisiveNormalization(nn.Module):
    """Convolutional drive with local contrast divisive gain control."""

    def __init__(self, n_channels: int = 8) -> None:
        """Initialize front-end and fixed pooling kernels.

        Parameters
        ----------
        n_channels
            Number of response channels.
        """
        super().__init__()
        self.front_end = nn.Conv2d(1, n_channels, kernel_size=5, padding=2, bias=False)
        self.pool = nn.Conv2d(
            n_channels, n_channels, kernel_size=5, padding=2, groups=n_channels, bias=False
        )
        nn.init.constant_(self.pool.weight, 1.0 / 25.0)
        self.sigma = nn.Parameter(torch.full((1, n_channels, 1, 1), 0.2))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize linear drive by local pooled contrast.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Divisively normalized responses.
        """
        drive = self.front_end(x)
        pooled = self.pool(drive.pow(2))
        return drive / torch.sqrt(torch.relu(self.sigma) + pooled + 1.0e-6)


def build() -> nn.Module:
    """Build a small Heeger normalization module.

    Returns
    -------
    nn.Module
        Configured ``HeegerDivisiveNormalization`` instance.
    """
    return HeegerDivisiveNormalization()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
