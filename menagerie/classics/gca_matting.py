"""GCA Matting -- Guided Contextual Attention for Natural Image Matting.

Li and Lu, AAAI 2020.
Paper: https://arxiv.org/abs/2001.04069
Source: https://github.com/Yaoyi-Li/GCA-Matting

GCA Matting introduces a guided contextual attention module that propagates
information from known (definite) regions to unknown (transition) regions.
The attention is computed on compact spatial features (downsampled 4x) to
keep the attention matrix tractable.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCAModule(nn.Module):
    """Guided contextual attention: softmax-weighted spatial attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        k = self.key(x).view(B, C, -1)  # (B, C, HW)
        v = self.value(x).view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        attn = torch.softmax(torch.bmm(q, k) / (C**0.5), dim=-1)  # (B, HW, HW)
        out = torch.bmm(attn, v).permute(0, 2, 1).view(B, C, H, W)
        return x + self.out(out)


class GCAMatting(nn.Module):
    """Compact GCA Matting network.

    Input: (B, 4, H, W) -- RGB(3) + trimap(1).
    Output: (B, 1, H, W) alpha matte.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Attention at 16x16 (256 positions)
        self.gca = GCAModule(32)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.enc1(x)
        e = self.gca(e)
        return torch.sigmoid(self.dec(e))


def build_gca_matting() -> nn.Module:
    """Build compact GCA Matting network."""
    return GCAMatting()


def example_input() -> torch.Tensor:
    """Example 4-channel matting input tensor ``(1, 4, 64, 64)``."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "GCA Matting (guided contextual attention matting)",
        "build_gca_matting",
        "example_input",
        "2020",
        "DC",
    ),
]
