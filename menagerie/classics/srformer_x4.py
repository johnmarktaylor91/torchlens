"""SRFormer: permuted self-attention transformer for image super-resolution.

Paper: "SRFormer: Permuted Self-Attention for Single Image Super-Resolution",
Zhou et al., ICCV 2023.

SRFormer keeps large-window context affordable by permuting channels into
groups before window self-attention, balancing channel and spatial information.
This module provides compact full and light x4 reconstructions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PermutedSelfAttention(nn.Module):
    """Compact permuted self-attention (PSA)."""

    def __init__(self, channels: int, heads: int = 2, groups: int = 2) -> None:
        """Initialize PSA.

        Parameters
        ----------
        channels:
            Feature channel count.
        heads:
            Number of attention heads.
        groups:
            Number of channel permutation groups.
        """

        super().__init__()
        self.heads = heads
        self.groups = groups
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-permuted spatial attention.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Attended feature tensor.
        """

        b, c, h, w = x.shape
        grouped = x.view(b, self.groups, c // self.groups, h, w)
        permuted = grouped.permute(0, 3, 4, 1, 2).reshape(b, h * w, c)
        q, k, v = self.qkv(permuted).chunk(3, dim=-1)
        head_dim = c // self.heads
        q = q.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, h * w, self.heads, head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5), dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, c)
        out = self.proj(out).view(b, h, w, self.groups, c // self.groups)
        return out.permute(0, 3, 4, 1, 2).reshape(b, c, h, w)


class SRFormerBlock(nn.Module):
    """SRFormer residual transformer block with PSA."""

    def __init__(self, channels: int) -> None:
        """Initialize an SRFormer block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = PermutedSelfAttention(channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run PSA and feed-forward residual branches.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output feature tensor.
        """

        y = x + self.attn(self.norm1(x))
        return y + self.ffn(self.norm2(y))


class SRFormerCompact(nn.Module):
    """Compact SRFormer x4 super-resolution model."""

    def __init__(self, channels: int = 24, blocks: int = 3) -> None:
        """Initialize compact SRFormer.

        Parameters
        ----------
        channels:
            Feature width.
        blocks:
            Number of PSA blocks.
        """

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*[SRFormerBlock(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 16, 3, padding=1),
            nn.PixelShuffle(4),
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


def build_srformer_x4() -> nn.Module:
    """Build compact SRFormer x4.

    Returns
    -------
    nn.Module
        Random-init SRFormer reconstruction.
    """

    return SRFormerCompact()


def build_srformer_light_x4() -> nn.Module:
    """Build compact SRFormer-light x4.

    Returns
    -------
    nn.Module
        Random-init SRFormer-light reconstruction.
    """

    return SRFormerCompact(channels=16, blocks=2)


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
        "SRFormer x4 (permuted self-attention SR)",
        "build_srformer_x4",
        "example_input",
        "2023",
        "E7",
    ),
    (
        "SRFormer-light x4 (permuted self-attention SR)",
        "build_srformer_light_x4",
        "example_input",
        "2023",
        "E7",
    ),
]
