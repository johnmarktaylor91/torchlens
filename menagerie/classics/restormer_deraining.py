"""Restormer deraining: efficient transformer for high-resolution restoration.

Paper: "Restormer: Efficient Transformer for High-Resolution Image Restoration",
Zamir et al., CVPR 2022.

Restormer uses an encoder-decoder transformer whose distinctive pieces are
Multi-DConv Head Transposed Attention (MDTA), which attends across channels
instead of all spatial tokens, and a Gated-DConv Feed-Forward Network (GDFN).
This compact deraining model preserves those primitives and the residual
image-to-image restoration form.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasFreeLayerNorm2d(nn.Module):
    """Bias-free layer norm over channels for image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        """Initialize channel layer normalization."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each spatial position over channels."""

        var = x.var(dim=1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(var + self.eps) * self.weight


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        """Initialize MDTA projections."""

        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.dw = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-transposed attention with local depthwise context."""

        batch, channels, height, width = x.shape
        q, k, v = self.dw(self.qkv(x)).chunk(3, dim=1)
        head_dim = channels // self.heads
        q = F.normalize(q.view(batch, self.heads, head_dim, height * width), dim=-1)
        k = F.normalize(k.view(batch, self.heads, head_dim, height * width), dim=-1)
        v = v.view(batch, self.heads, head_dim, height * width)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.temperature, dim=-1)
        out = torch.matmul(attn, v).view(batch, channels, height, width)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated depthwise-convolution feed-forward network."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """Initialize GDFN."""

        super().__init__()
        hidden = channels * expansion
        self.project_in = nn.Conv2d(channels, hidden * 2, 1)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.project_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation."""

        left, right = self.dw(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class RestormerBlock(nn.Module):
    """Restormer block with MDTA and GDFN."""

    def __init__(self, channels: int) -> None:
        """Initialize the block."""

        super().__init__()
        self.norm1 = BiasFreeLayerNorm2d(channels)
        self.attn = MDTA(channels)
        self.norm2 = BiasFreeLayerNorm2d(channels)
        self.ffn = GDFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a Restormer residual block."""

        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RestormerCompact(nn.Module):
    """Compact Restormer deraining model."""

    def __init__(self, channels: int = 24, in_channels: int = 3) -> None:
        """Initialize compact Restormer."""

        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.enc = RestormerBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.mid = RestormerBlock(channels * 2)
        self.up = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.dec = RestormerBlock(channels)
        self.skip = nn.Conv2d(in_channels, 3, 1)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove rain streaks from an RGB image."""

        enc = self.enc(self.patch_embed(x))
        mid = self.mid(self.down(enc))
        dec = self.dec(self.up(mid) + enc)
        return self.skip(x) + self.out(dec)


def build() -> nn.Module:
    """Build compact Restormer deraining."""

    return RestormerCompact()


def example_input() -> torch.Tensor:
    """Return a small rainy RGB image tensor."""

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [
    ("Restormer deraining (MDTA + GDFN transformer)", "build", "example_input", "2022", "E5")
]
