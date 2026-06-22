"""SkipNet-ResNet compact faithful reconstruction.

Wang et al. 2018, "SkipNet: Learning Dynamic Routing in Convolutional Networks".

SkipNet augments a residual network with gates that choose per-input whether to
execute or bypass each residual block. This compact traceable version uses soft
gates so both routes remain visible in the graph.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """Small residual convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize block layers.

        Parameters
        ----------
        channels:
            Feature channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Residual-activated feature map.
        """
        return torch.relu(x + self.net(x))


class SkipGate(nn.Module):
    """Lightweight recurrent-style routing gate."""

    def __init__(self, channels: int) -> None:
        """Initialize gate projection.

        Parameters
        ----------
        channels:
            Feature channel count.
        """
        super().__init__()
        self.proj = nn.Linear(channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict an execute probability.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Soft execution gate with broadcastable shape.
        """
        pooled = x.mean(dim=(2, 3))
        return torch.sigmoid(self.proj(pooled)).view(x.shape[0], 1, 1, 1)


class SkipNetResNetCompact(nn.Module):
    """Compact gated ResNet."""

    def __init__(self, channels: int = 24, blocks: int = 4, classes: int = 10) -> None:
        """Initialize stem, gated blocks, and classifier.

        Parameters
        ----------
        channels:
            Feature channel count.
        blocks:
            Number of gated residual blocks.
        classes:
            Class count.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        self.gates = nn.ModuleList([SkipGate(channels) for _ in range(blocks)])
        self.head = nn.Linear(channels, classes)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image with soft dynamic routing.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = self.stem(image)
        for block, gate in zip(self.blocks, self.gates, strict=True):
            routed = block(x)
            g = gate(x)
            x = g * routed + (1.0 - g) * x
        return self.head(x.mean(dim=(2, 3)))


def build() -> nn.Module:
    """Build compact random-init SkipNet-ResNet.

    Returns
    -------
    nn.Module
        Compact SkipNet.
    """
    return SkipNetResNetCompact()


def example_input() -> Tensor:
    """Return a small RGB image.

    Returns
    -------
    Tensor
        Image tensor.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("SkipNet-ResNet", "build", "example_input", "2018", "E7")]
