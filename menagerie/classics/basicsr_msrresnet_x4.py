"""MSRResNet x4: modified SRResNet with residual blocks and pixel shuffle.

Paper: "Photo-Realistic Single Image Super-Resolution Using a Generative
Adversarial Network", Ledig et al., CVPR 2017; BasicSR MSRResNet removes batch
normalization and uses residual blocks plus late pixel-shuffle upsampling.

This compact reconstruction preserves the modified SRResNet generator backbone:
shallow feature extraction, BN-free residual trunk, trunk skip connection, and
two x2 pixel-shuffle stages for x4 super-resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSRResidualBlock(nn.Module):
    """Batch-norm-free modified SRResNet residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual convolutional refinement."""

        return x + self.conv2(F.leaky_relu(self.conv1(x), negative_slope=0.2))


class MSRResNetCompact(nn.Module):
    """Compact MSRResNet x4 generator."""

    def __init__(self, channels: int = 32, blocks: int = 4) -> None:
        """Initialize compact MSRResNet."""

        super().__init__()
        self.conv_first = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[MSRResidualBlock(channels) for _ in range(blocks)])
        self.trunk = nn.Conv2d(channels, channels, 3, padding=1)
        self.up1 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.hr = nn.Conv2d(channels, channels, 3, padding=1)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by 4x."""

        feat = self.conv_first(x)
        feat = self.trunk(self.body(feat)) + feat
        feat = F.leaky_relu(F.pixel_shuffle(self.up1(feat), 2), negative_slope=0.2)
        feat = F.leaky_relu(F.pixel_shuffle(self.up2(feat), 2), negative_slope=0.2)
        return self.out(F.leaky_relu(self.hr(feat), negative_slope=0.2))


def build() -> nn.Module:
    """Build compact MSRResNet x4."""

    return MSRResNetCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "BasicSR MSRResNet x4 (BN-free SRResNet pixel-shuffle generator)",
        "build",
        "example_input",
        "2017",
        "E5",
    )
]
