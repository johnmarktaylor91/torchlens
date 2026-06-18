"""Carandini-Heeger-Movshon normalization, 1997, Carandini, Heeger, and Movshon.

Paper: "Linearity and normalization in simple cells of the macaque primary visual
cortex." Simple-cell filter responses are divided by a suppressive pool, capturing
contrast saturation and cross-orientation suppression.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Carandini-Heeger-Movshon Normalization", "build", "example_input", "1997", "DA")
]


class CHMNormalization(nn.Module):
    """Filter bank responses normalized by shared suppressive drive."""

    def __init__(self, n_channels: int = 8) -> None:
        """Initialize numerator and suppressive-pool filters.

        Parameters
        ----------
        n_channels
            Number of simple-cell channels.
        """
        super().__init__()
        self.filter_bank = nn.Conv2d(1, n_channels, kernel_size=5, padding=2, bias=False)
        self.pool = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False)
        nn.init.constant_(self.pool.weight, 1.0 / (9.0 * n_channels))
        self.sigma = nn.Parameter(torch.full((1, n_channels, 1, 1), 0.5))
        self.p = nn.Parameter(torch.tensor(2.0))
        self.q = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        """Apply suppressive-pool divisive normalization.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Normalized simple-cell responses.
        """
        numerator = self.filter_bank(x)
        exponent_p = torch.relu(self.p) + 1.0e-3
        exponent_q = torch.relu(self.q) + 1.0e-3
        pool = self.pool(torch.abs(numerator).pow(exponent_p))
        denom = torch.relu(self.sigma) + torch.relu(pool) + 1.0e-6
        return numerator / denom.pow(exponent_q)


def build() -> nn.Module:
    """Build a small CHM normalization module.

    Returns
    -------
    nn.Module
        Configured ``CHMNormalization`` instance.
    """
    return CHMNormalization()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
