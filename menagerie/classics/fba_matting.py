"""FBA Matting -- F, B, Alpha Matte Prediction.

Forte and Pitie, CVPR 2020.
Paper: https://arxiv.org/abs/2003.07711
Source: https://github.com/MarcoForte/FBA_Matting

FBA Matting jointly predicts foreground, background, and alpha from a 11-channel
input: RGB image (3), trimap (2 one-hot), foreground hint (3), background hint (3).
GroupNorm is used throughout instead of BatchNorm to handle the small effective
batch sizes typical in matting pipelines.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """GroupNorm residual block used in FBA Matting."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(4, channels)
        self.conv = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class FBAMatting(nn.Module):
    """Compact FBA Matting network.

    Input: (B, 11, H, W) -- image(3) + trimap(2) + fg(3) + bg(3).
    Output: (B, 1, H, W) alpha matte.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(11, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = ResBlock(64)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64x64
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.alpha_head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc1(x)
        e = self.enc2(e)
        d = self.dec1(e)
        return torch.sigmoid(self.alpha_head(d))


def build_fba_matting() -> nn.Module:
    """Build compact FBA Matting network."""
    return FBAMatting()


def example_input() -> torch.Tensor:
    """Example 11-channel matting input tensor ``(1, 11, 64, 64)``."""
    return torch.randn(1, 11, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "FBA Matting (F/B/alpha trimap-guided matting)",
        "build_fba_matting",
        "example_input",
        "2020",
        "DC",
    ),
]
