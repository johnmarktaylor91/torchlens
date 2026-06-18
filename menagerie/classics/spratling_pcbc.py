"""Spratling predictive-coding/biased-competition network, 2008, Michael Spratling.

Paper: "Predictive coding as a model of biased competition in visual attention."
Top-down predictions divisively suppress explained input while residual evidence
drives recurrent representation updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Spratling Predictive-Coding/Biased-Competition Net", "build", "example_input", "2008", "DA")
]


class SpratlingPCBC(nn.Module):
    """Divisive predictive-coding recurrent visual module."""

    def __init__(self, n_channels: int = 8, steps: int = 4) -> None:
        """Initialize bottom-up and top-down convolutional weights.

        Parameters
        ----------
        n_channels
            Representation-channel count.
        steps
            Number of recurrent update steps.
        """
        super().__init__()
        self.bottom_up = nn.Conv2d(1, n_channels, 3, padding=1)
        self.top_down = nn.Conv2d(n_channels, 1, 3, padding=1)
        self.steps = steps

    def forward(self, x: Tensor) -> Tensor:
        """Infer a representation from divisively suppressed residual input.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 32, 32)``.

        Returns
        -------
        Tensor
            Recurrent representation map.
        """
        r = torch.relu(self.bottom_up(x))
        for _ in range(self.steps):
            prediction = torch.relu(self.top_down(r)) + 1.0e-3
            residual = x / prediction
            r = torch.relu(0.6 * r + self.bottom_up(residual))
        return r


def build() -> nn.Module:
    """Build a small Spratling PC/BC module.

    Returns
    -------
    nn.Module
        Configured ``SpratlingPCBC`` instance.
    """
    return SpratlingPCBC()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.rand(1, 1, 32, 32)
