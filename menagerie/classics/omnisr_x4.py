"""OmniSR: omni-scale aggregation network for lightweight super-resolution.

Paper: "Omni Aggregation Networks for Lightweight Image Super-Resolution",
Wang et al., CVPR 2023.

OmniSR combines Omni Self-Attention (OSA), which jointly models spatial and
channel correlations, with an Omni-Scale Aggregation Group (OSAG) that cascades
local convolutional, meso-window, and global/channel aggregation paths.  This
compact random-init reconstruction keeps those primitives and an x4 pixel-shuffle
tail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OmniSelfAttention(nn.Module):
    """Joint spatial-token and channel-correlation attention."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        """Initialize OSA.

        Parameters
        ----------
        channels:
            Feature channel count.
        heads:
            Number of spatial attention heads.
        """

        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.spatial_proj = nn.Linear(channels, channels)
        self.channel_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply OSA to image features.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Attended features with the same shape.
        """

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        head_dim = c // self.heads
        q = q.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5), dim=-1)
        spatial = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, c)
        spatial = self.spatial_proj(spatial).transpose(1, 2).view(b, c, h, w)

        cq, ck, cv = self.channel_qkv(x).chunk(3, dim=1)
        cq = F.normalize(cq.flatten(2), dim=-1)
        ck = F.normalize(ck.flatten(2), dim=-1)
        cv = cv.flatten(2)
        channel = torch.softmax(torch.matmul(cq, ck.transpose(-1, -2)) * self.temperature, dim=-1)
        channel = torch.matmul(channel, cv).view(b, c, h, w)
        return self.fuse(torch.cat([spatial, channel], dim=1))


class OSAGBlock(nn.Module):
    """Omni-scale aggregation block with local, meso, and global paths."""

    def __init__(self, channels: int) -> None:
        """Initialize an OSAG block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.meso = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.osa = OmniSelfAttention(channels)
        self.mix = nn.Conv2d(channels * 3, channels, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run omni-scale aggregation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Aggregated feature map.
        """

        y = self.norm(x)
        aggregated = self.mix(torch.cat([self.local(y), self.meso(y), self.osa(y)], dim=1))
        y = x + aggregated
        return y + self.ffn(y)


class OmniSRCompact(nn.Module):
    """Compact OmniSR x4 super-resolution model."""

    def __init__(self, channels: int = 24, blocks: int = 2) -> None:
        """Initialize compact OmniSR.

        Parameters
        ----------
        channels:
            Feature width.
        blocks:
            Number of OSAG blocks.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[OSAGBlock(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU(),
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
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
        return self.up(self.tail(self.body(feat)) + feat)


def build_omnisr_x4() -> nn.Module:
    """Build compact OmniSR x4.

    Returns
    -------
    nn.Module
        Random-init OmniSR reconstruction.
    """

    return OmniSRCompact()


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image.

    Returns
    -------
    torch.Tensor
        Example image tensor.
    """

    return torch.randn(1, 3, 12, 12)


MENAGERIE_ENTRIES = [
    (
        "OmniSR x4 (OSA omni-scale aggregation SR)",
        "build_omnisr_x4",
        "example_input",
        "2023",
        "E7",
    )
]
