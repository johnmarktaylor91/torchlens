"""HINet: half instance normalization network for image restoration.

Paper: "HINet: Half Instance Normalization Network for Image Restoration",
Chen et al., CVPRW 2021.

HINet introduced the Half Instance Normalization (HIN) block for low-level
restoration: only half of the intermediate channels are instance-normalized,
preserving channel statistics in the other half.  The compact version keeps the
two-stage U-Net-like restoration pattern with cross-stage feature fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HINBlock(nn.Module):
    """Half Instance Normalization residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize a HIN block."""

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(channels // 2, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply half-normalized residual transformation."""

        feat = self.conv1(x)
        left, right = feat.chunk(2, dim=1)
        feat = torch.cat([self.norm(left), right], dim=1)
        return x + self.conv2(F.leaky_relu(feat, negative_slope=0.2))


class HINetStage(nn.Module):
    """Small encoder-decoder restoration stage with HIN blocks."""

    def __init__(self, channels: int) -> None:
        """Initialize one HINet stage."""

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.enc = HINBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.mid = HINBlock(channels * 2)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec = HINBlock(channels)
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(
        self, x: torch.Tensor, guide: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore an image and return stage features."""

        feat = self.head(x)
        if guide is not None:
            feat = feat + guide
        skip = self.enc(feat)
        bottleneck = self.mid(self.down(skip))
        decoded = self.dec(self.up(bottleneck) + skip)
        return x + self.tail(decoded), decoded


class HINetCompact(nn.Module):
    """Two-stage compact HINet with cross-stage feature fusion."""

    def __init__(self, channels: int = 20) -> None:
        """Initialize compact HINet."""

        super().__init__()
        self.stage1 = HINetStage(channels)
        self.csff = nn.Conv2d(channels, channels, 1)
        self.stage2 = HINetStage(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an RGB image."""

        out1, feat1 = self.stage1(x)
        out2, _ = self.stage2(out1, self.csff(feat1))
        return out2


def build() -> nn.Module:
    """Build compact HINet."""

    return HINetCompact()


def example_input() -> torch.Tensor:
    """Return a small degraded RGB image."""

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    (
        "HINet image restoration (two-stage half instance normalization)",
        "build",
        "example_input",
        "2021",
        "E5",
    )
]
