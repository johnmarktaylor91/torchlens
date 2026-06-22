"""RCAN x4: residual channel attention network for super-resolution.

Paper: "Image Super-Resolution Using Very Deep Residual Channel Attention
Networks", Zhang et al., ECCV 2018.

RCAN's distinctive primitive is residual-in-residual structure with channel
attention blocks that rescale feature channels from global pooled statistics.
This compact BasicSR-style reconstruction keeps residual groups, residual
channel attention blocks, long skip addition, and late pixel-shuffle x4
upsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention used by RCAN."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """Initialize channel attention."""

        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale channels by global context."""

        return x * self.net(x)


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, channels: int) -> None:
        """Initialize RCAB."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual channel attention."""

        return x + self.ca(self.conv2(F.relu(self.conv1(x))))


class ResidualGroup(nn.Module):
    """RCAN residual group containing RCABs and a group skip."""

    def __init__(self, channels: int, blocks: int = 2) -> None:
        """Initialize a residual group."""

        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(channels) for _ in range(blocks)], nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply group residual connection."""

        return x + self.body(x)


class RCANCompact(nn.Module):
    """Compact RCAN x4 super-resolution model."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact RCAN."""

        super().__init__()
        self.sub_mean = nn.Conv2d(3, 3, 1, bias=False)
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.groups = nn.Sequential(ResidualGroup(channels), ResidualGroup(channels))
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

        feat = self.head(self.sub_mean(x))
        body = self.body_tail(self.groups(feat)) + feat
        return self.up(body)


def build() -> nn.Module:
    """Build compact RCAN x4."""

    return RCANCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("BasicSR RCAN x4 (residual channel attention groups)", "build", "example_input", "2018", "E5")
]
