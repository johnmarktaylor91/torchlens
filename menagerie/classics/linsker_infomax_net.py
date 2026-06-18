"""Linsker InfoMax layered network, 1988, Ralph Linsker.

Paper: "Self-organization in a perceptual network." A feedforward stack with
Gaussian-windowed receptive fields stands in for local Hebbian information-maximizing
updates that self-organize center-surround and orientation-selective filters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Linsker InfoMax Layered Network", "build", "example_input", "1988", "DA")]


class LinskerInfoMaxNet(nn.Module):
    """Layered convolutional network with Gaussian-masked local weights."""

    def __init__(self, channels: tuple[int, int, int] = (4, 6, 8)) -> None:
        """Initialize the layered convolutional substrate.

        Parameters
        ----------
        channels
            Output channels for the three feedforward layers.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 5, padding=2))
            layers.append(nn.Tanh())
            in_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the local layered visual transform.

        Parameters
        ----------
        x
            Image tensor of shape ``(batch, 1, 32, 32)``.

        Returns
        -------
        Tensor
            Final-layer feature map.
        """
        return self.layers(x)


def build() -> nn.Module:
    """Build a small Linsker-style layered network.

    Returns
    -------
    nn.Module
        Configured ``LinskerInfoMaxNet`` instance.
    """
    return LinskerInfoMaxNet()


def example_input() -> Tensor:
    """Return a grayscale image example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
