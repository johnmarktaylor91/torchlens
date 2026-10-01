"""CBAM: Convolutional Block Attention Module.

Paper: CBAM: Convolutional Block Attention Module.
Woo, Park, Lee, and Kweon, ECCV 2018.

CBAM refines an intermediate CNN feature map with two sequential gates:
channel attention from shared-MLP average/max pooled descriptors, then spatial
attention from channelwise average/max maps and a convolutional mask.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """CBAM channel attention with shared MLP over average and max descriptors."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize the shared descriptor MLP.

        Parameters
        ----------
        channels:
            Number of feature channels.
        reduction:
            Bottleneck reduction ratio for the shared MLP.
        """
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply channel attention.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Channel-refined feature map with the same shape as ``x``.
        """
        avg_gate = self.mlp(self.avg_pool(x))
        max_gate = self.mlp(self.max_pool(x))
        return x * torch.sigmoid(avg_gate + max_gate)


class SpatialAttention(nn.Module):
    """CBAM spatial attention over channel average and max maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        """Initialize the spatial attention convolution.

        Parameters
        ----------
        kernel_size:
            Odd convolution kernel size for the spatial gate.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spatial attention.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Spatially refined feature map with the same shape as ``x``.
        """
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate


class CBAMBlock(nn.Module):
    """Sequential channel-then-spatial CBAM block."""

    def __init__(self, channels: int) -> None:
        """Initialize CBAM submodules.

        Parameters
        ----------
        channels:
            Number of input and output channels.
        """
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        """Refine a feature map with CBAM.

        Parameters
        ----------
        x:
            Feature map with shape ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            Refined feature map.
        """
        return self.spatial(self.channel(x))


class CBAMDemoNet(nn.Module):
    """Compact CNN that inserts CBAM after a convolutional block."""

    def __init__(self, channels: int = 12, num_classes: int = 5) -> None:
        """Initialize the demonstration classifier.

        Parameters
        ----------
        channels:
            Width of the convolutional trunk.
        num_classes:
            Number of output logits.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.cbam = CBAMBlock(channels)
        self.head = nn.Sequential(nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits through a CBAM-refined residual block.

        Parameters
        ----------
        x:
            RGB image tensor with shape ``(B, 3, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        y = self.stem(x)
        y = y + self.cbam(self.conv(y))
        return self.classifier(self.head(y))


def build() -> nn.Module:
    """Build a compact CBAM demonstration network.

    Returns
    -------
    nn.Module
        Random-initialized CBAM demo network.
    """
    return CBAMDemoNet()


def example_input() -> Tensor:
    """Return a small traceable image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("CBAM (Convolutional Block Attention Module)", "build", "example_input", "2018", "DC")
]
