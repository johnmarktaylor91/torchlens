"""EDSR: enhanced deep residual network for single-image super-resolution.

Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution",
Lim et al., CVPRW 2017.

EDSR removes batch normalization from SRResNet-style residual blocks, widens the
residual trunk, uses residual scaling for stability, and applies late pixel
shuffle upsampling.  The compact model keeps those choices with reduced width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EDSRBlock(nn.Module):
    """Batch-norm-free EDSR residual block with residual scaling."""

    def __init__(self, channels: int, scale: float = 0.1) -> None:
        """Initialize an EDSR residual block."""

        super().__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""

        return x + self.conv2(F.relu(self.conv1(x))) * self.scale


class EDSRCompact(nn.Module):
    """Compact EDSR x4 super-resolution model."""

    def __init__(self, channels: int = 32, blocks: int = 4) -> None:
        """Initialize compact EDSR."""

        super().__init__()
        self.sub_mean = nn.Conv2d(3, 3, 1, bias=False)
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[EDSRBlock(channels) for _ in range(blocks)])
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by 4x."""

        normed = self.sub_mean(x)
        feat = self.head(normed)
        body = self.body_tail(self.body(feat)) + feat
        return self.up(body)


def build_basicsr_edsr_x4() -> nn.Module:
    """Build compact EDSR x4."""

    return EDSRCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "EDSR x4 (batch-norm-free residual scaling SR)",
        "build_basicsr_edsr_x4",
        "example_input",
        "2017",
        "E5",
    )
]
