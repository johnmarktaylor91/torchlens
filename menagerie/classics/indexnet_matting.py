"""IndexNet Matting -- Indices Matter for Guided Image Matting.

Lu et al., ICCV 2019.
Paper: https://arxiv.org/abs/1908.00672
Source: https://github.com/poppinace/indexnet_matting

IndexNet Matting uses learned index functions (depthwise conv with sigmoid)
to produce guided upsampling weights in the decoder, analogous to how max-pool
indices guide unpooling but in a learned, content-adaptive way.
This is a faithful compact random-init reimplementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexBlock(nn.Module):
    """Learned index function: depthwise conv + sigmoid."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.bn(self.dw_conv(x)))


class IndexNetMatting(nn.Module):
    """Compact IndexNet Matting network.

    Input: (B, 4, H, W) -- RGB(3) + trimap(1).
    Output: (B, 1, H, W) alpha matte.
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Index block produces guided upsampling weights at enc1 resolution
        self.idx1 = IndexBlock(32)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # (B, 32, 32, 32)
        idx = self.idx1(e1)  # (B, 32, 32, 32) index weights
        e2 = self.enc2(e1)  # (B, 64, 16, 16)

        # Decode 16x16 -> 32x32 with guided scaling
        d2 = self.dec2(e2)  # (B, 32, 16, 16)
        d2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = d2 * idx  # guided element-wise (B, 32, 32, 32)

        # Decode 32x32 -> 64x64
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.dec1(d1)  # (B, 1, 64, 64)
        return torch.sigmoid(out)


def build_indexnet_matting() -> nn.Module:
    """Build compact IndexNet Matting network."""
    return IndexNetMatting()


def example_input() -> torch.Tensor:
    """Example 4-channel matting input tensor ``(1, 4, 64, 64)``."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "IndexNet Matting (index-guided decoder matting)",
        "build_indexnet_matting",
        "example_input",
        "2019",
        "DC",
    ),
]
