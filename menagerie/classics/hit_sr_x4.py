"""HiT-SR: Hierarchical Transformer for Efficient Image Super-Resolution.

Zhang, Zhang, and Yu, 2024.
Paper: https://arxiv.org/abs/2407.05878

HiT-SR replaces fixed small-window SR attention with expanding hierarchical
windows and a spatial-channel correlation (S-SC/C-SC) attention surrogate that
keeps cost linear in window size.  This compact random-init version preserves
the load-bearing primitive: parallel local and larger pooled windows produce
spatial correlation maps, channel correlation gates, and residual SR features
before pixel-shuffle x4 reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelCorrelation(nn.Module):
    """Hierarchical spatial-channel correlation block used by HiT-SR."""

    def __init__(self, channels: int, pooled_size: int) -> None:
        """Initialize projections and a hierarchical pooled window size.

        Parameters
        ----------
        channels:
            Feature width.
        pooled_size:
            Output side length for the expanded hierarchical window branch.
        """

        super().__init__()
        self.pooled_size = pooled_size
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear spatial correlation plus channel correlation.

        Parameters
        ----------
        x:
            Feature tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Correlation-mixed feature tensor with the same shape.
        """

        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(F.adaptive_avg_pool2d(x, self.pooled_size)).flatten(2)
        v = self.v(F.adaptive_avg_pool2d(x, self.pooled_size)).flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k) / (c**0.5), dim=-1)
        spatial = torch.matmul(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return self.proj(spatial * self.channel_gate(x))


class HiTBlock(nn.Module):
    """Residual HiT-SR block with expanding hierarchical windows."""

    def __init__(self, channels: int, pooled_size: int) -> None:
        """Initialize a compact hierarchical transformer block.

        Parameters
        ----------
        channels:
            Feature width.
        pooled_size:
            Side length for pooled hierarchical attention keys/values.
        """

        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.ssc = SpatialChannelCorrelation(channels, pooled_size)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run hierarchical correlation and feed-forward residual paths."""

        x = x + self.ssc(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class HiTSRx4(nn.Module):
    """Compact HiT-SR x4 super-resolution network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize shallow SR stem, two HiT blocks, and x4 upsampler."""

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [HiTBlock(channels, pooled_size=4), HiTBlock(channels, pooled_size=8)]
        )
        self.recon = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 3 * 16, 3, padding=1),
            nn.PixelShuffle(4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve an RGB image by a factor of four."""

        feat = self.stem(x)
        for block in self.blocks:
            feat = block(feat)
        return self.recon(feat) + F.interpolate(x, scale_factor=4, mode="bilinear")


def build_hit_sr_x4() -> nn.Module:
    """Build a compact random-init HiT-SR x4 model."""

    return HiTSRx4()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB input."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "HiT-SR x4 (hierarchical spatial-channel correlation transformer)",
        "build_hit_sr_x4",
        "example_input",
        "2024",
        "image-restoration/super-resolution",
    ),
]
