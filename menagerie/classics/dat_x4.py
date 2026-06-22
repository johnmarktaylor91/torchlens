"""DAT: Dual Aggregation Transformer for image super-resolution.

Paper: "Dual Aggregation Transformer for Image Super-Resolution", Chen et al.,
ICCV 2023.

DAT alternates spatial and channel self-attention, uses an adaptive interaction
module to pass information between the two attention views, and uses an SGFN
feed-forward network with spatial gating.  The compact model keeps those
load-bearing primitives with a small x4 pixel-shuffle tail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Window-free compact spatial self-attention."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        """Initialize spatial attention."""

        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to ``(B, C, H, W)`` features."""

        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        q = q.view(b, h * w, self.heads, c // self.heads).transpose(1, 2)
        k = k.view(b, h * w, self.heads, c // self.heads).transpose(1, 2)
        v = v.view(b, h * w, self.heads, c // self.heads).transpose(1, 2)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / ((c // self.heads) ** 0.5), dim=-1
        )
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, h * w, c)
        return self.proj(out).transpose(1, 2).view(b, c, h, w)


class ChannelAttention(nn.Module):
    """Channel self-attention used in DAT's alternate blocks."""

    def __init__(self, channels: int) -> None:
        """Initialize channel attention."""

        super().__init__()
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to feature maps."""

        b, c, h, w = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = F.normalize(q.flatten(2), dim=-1)
        k = F.normalize(k.flatten(2), dim=-1)
        v = v.flatten(2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.temperature, dim=-1)
        return self.proj(torch.matmul(attn, v).view(b, c, h, w))


class SGFN(nn.Module):
    """Spatial-gate feed-forward network."""

    def __init__(self, channels: int) -> None:
        """Initialize SGFN."""

        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward processing."""

        a, gate = self.in_proj(x).chunk(2, dim=1)
        return self.out(F.gelu(a) * torch.sigmoid(self.spatial(gate)))


class DATBlock(nn.Module):
    """Dual Aggregation Transformer block."""

    def __init__(self, channels: int, channel_mode: bool) -> None:
        """Initialize one alternate spatial/channel block."""

        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.attn = ChannelAttention(channels) if channel_mode else SpatialAttention(channels)
        self.aim = nn.Conv2d(channels * 2, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = SGFN(channels)

    def forward(self, x: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run attention, AIM interaction, and SGFN."""

        attended = self.attn(self.norm1(x))
        mixed = self.aim(torch.cat([attended, carry], dim=1))
        y = x + mixed
        return y + self.ffn(self.norm2(y)), attended


class DATCompact(nn.Module):
    """Compact DAT x4 super-resolution network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize compact DAT."""

        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.ModuleList([DATBlock(channels, False), DATBlock(channels, True)])
        self.body = nn.Conv2d(channels, channels, 3, padding=1)
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

        feat = self.head(x)
        carry = torch.zeros_like(feat)
        out = feat
        for block in self.blocks:
            out, carry = block(out, carry)
        out = self.body(out) + feat
        return self.up(out)


def build_dat_x4() -> nn.Module:
    """Build compact DAT x4."""

    return DATCompact()


def build_dat_light_x4() -> nn.Module:
    """Build compact DAT-light x4."""

    return DATCompact(channels=16)


def example_input() -> torch.Tensor:
    """Return a small low-resolution RGB image."""

    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "DAT x4 (dual spatial/channel aggregation transformer SR)",
        "build_dat_x4",
        "example_input",
        "2023",
        "E7",
    ),
    (
        "DAT-light x4 (dual aggregation transformer SR)",
        "build_dat_light_x4",
        "example_input",
        "2023",
        "E7",
    ),
]
