"""SPAN: Swift Parameter-free Attention Network for efficient SR.

Paper: "Swift Parameter-free Attention Network for Efficient
Super-Resolution", Wan et al., CVPRW 2024.

SPAN replaces learned attention modules with a parameter-free symmetric
activation attention map computed directly from convolutional features.  The
compact reconstruction keeps stacked SPAB blocks and x4 pixel-shuffle SR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPAB(nn.Module):
    """Swift parameter-free attention block."""

    def __init__(self, channels: int) -> None:
        """Initialize SPAB.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free symmetric attention.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Attended feature tensor.
        """

        feat = F.silu(self.conv1(x))
        feat = self.conv2(feat)
        attention = torch.sigmoid(feat) * torch.tanh(feat)
        refined = self.conv3(feat * attention)
        return x + refined


class SPANCompact(nn.Module):
    """Compact SPAN x4 super-resolution model."""

    def __init__(self, channels: int = 24, blocks: int = 4) -> None:
        """Initialize compact SPAN.

        Parameters
        ----------
        channels:
            Feature width.
        blocks:
            Number of SPAB blocks.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([SPAB(channels) for _ in range(blocks)])
        self.collect = nn.Conv2d(channels * blocks, channels, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by 4x.

        Parameters
        ----------
        x:
            Low-resolution RGB image.

        Returns
        -------
        torch.Tensor
            Reconstructed RGB image.
        """

        feat = self.head(x)
        outs = []
        current = feat
        for block in self.blocks:
            current = block(current)
            outs.append(current)
        return self.up(self.collect(torch.cat(outs, dim=1)) + feat)


def build_span_x4() -> nn.Module:
    """Build compact SPAN x4.

    Returns
    -------
    nn.Module
        Random-init SPAN reconstruction.
    """

    return SPANCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Example image tensor.
    """

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "SPAN x4 (swift parameter-free attention SR)",
        "build_span_x4",
        "example_input",
        "2024",
        "E7",
    )
]
