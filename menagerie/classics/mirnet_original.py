"""MIRNet: Learning Enriched Features for Real Image Restoration.

Zamir et al., 2020.
Paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/2415_ECCV_2020_paper.php

MIRNet's distinctive unit is the multi-scale residual block: parallel
multi-resolution streams exchange information, use dual attention units, and
selectively fuse scales with SKFF.  This compact random-init implementation
keeps those mechanisms and a residual RGB restoration head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttention(nn.Module):
    """MIRNet dual attention unit with channel and spatial gates."""

    def __init__(self, channels: int) -> None:
        """Initialize channel and spatial attention branches."""

        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual dual-attention refinement."""

        y = self.body(x)
        return x + y * self.channel(y) + y * self.spatial(y)


class SKFF(nn.Module):
    """Selective kernel feature fusion across MIRNet scales."""

    def __init__(self, channels: int, num_scales: int = 3) -> None:
        """Initialize scale-selection layers."""

        super().__init__()
        self.num_scales = num_scales
        self.squeeze = nn.Conv2d(channels, channels // 2, 1)
        self.excite = nn.Conv2d(channels // 2, channels * num_scales, 1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Fuse same-resolution scale tensors with learned soft selection."""

        pooled = torch.stack(xs, dim=1).sum(dim=1).mean((2, 3), keepdim=True)
        weights = self.excite(F.relu(self.squeeze(pooled)))
        weights = weights.view(weights.shape[0], self.num_scales, xs[0].shape[1], 1, 1)
        weights = torch.softmax(weights, dim=1)
        stacked = torch.stack(xs, dim=1)
        return (stacked * weights).sum(dim=1)


class MultiScaleResidualBlock(nn.Module):
    """Three-stream MIRNet block with exchange and selective fusion."""

    def __init__(self, channels: int) -> None:
        """Initialize stream attention and cross-scale exchange modules."""

        super().__init__()
        self.high = DualAttention(channels)
        self.mid = DualAttention(channels)
        self.low = DualAttention(channels)
        self.skff = SKFF(channels)
        self.out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run parallel multi-resolution streams and fuse them."""

        high = self.high(x)
        mid = self.mid(F.avg_pool2d(x, 2))
        low = self.low(F.avg_pool2d(x, 4))
        mid_up = F.interpolate(mid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.skff([high, mid_up, low_up])
        return x + self.out(fused)


class MIRNet(nn.Module):
    """Compact MIRNet restoration model."""

    def __init__(self, channels: int = 18, blocks: int = 2) -> None:
        """Initialize shallow features, MRB trunk, and residual head."""

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[MultiScaleResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore an RGB image through multi-scale residual blocks."""

        return x + self.head(self.blocks(self.stem(x)))


def build_mirnet_original() -> nn.Module:
    """Build compact MIRNet original."""

    return MIRNet(channels=18, blocks=2)


def build_mirnet_v2_enhancement() -> nn.Module:
    """Build compact MIRNet-v2 enhancement variant."""

    return MIRNet(channels=16, blocks=2)


def build_mirnet_v2_super_resolution_x4() -> nn.Module:
    """Build compact MIRNet-v2 x4 super-resolution variant."""

    return nn.Sequential(
        MIRNet(channels=16, blocks=2), nn.Conv2d(3, 48, 3, padding=1), nn.PixelShuffle(4)
    )


def example_input() -> torch.Tensor:
    """Return a small RGB restoration input."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "MIRNet original (multi-scale residual block with SKFF)",
        "build_mirnet_original",
        "example_input",
        "2020",
        "image-restoration/enhancement",
    ),
]
