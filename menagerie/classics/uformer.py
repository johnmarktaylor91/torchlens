"""Uformer: A General U-Shaped Transformer for Image Restoration.

Wang et al., CVPR 2022.
Paper: https://arxiv.org/abs/2106.03106
Source: https://github.com/ZhendongWang6/Uformer

Uformer is a U-shaped hierarchical encoder-decoder whose every stage is a stack
of LeWin (Locally-enhanced Window) Transformer blocks:

  W-MSA  -- window-based multi-head self-attention over non-overlapping
            ``win_size x win_size`` windows (the local self-attention that makes
            high-resolution restoration tractable), with a learned relative
            position bias.
  LeFF   -- Locally-enhanced Feed-Forward: Linear -> (reshape to spatial) ->
            depthwise 3x3 Conv -> (reshape back) -> Linear, injecting local
            context into the token MLP.

Downsampling between stages is a strided 4x4 conv (channels x2, resolution /2);
upsampling is a 2x2 transposed conv; skip connections concatenate the matching
encoder tokens onto the decoder tokens.

Two catalog variants share one parametric module and differ only by width:
  uformer_tiny_b16 -- embed_dim 16
  uformer_b32      -- embed_dim 32

win_size is kept at 8 and depths are modest (a 2-level U with one LeWin block per
stage) so the captured graph is small and ``draw`` finishes quickly, while the
architecture (W-MSA windows, LeFF depthwise FFN, strided/transposed-conv resampling,
U-skip concatenation) is reproduced faithfully.  The forward returns a single
restored image tensor.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Window partition helpers
# ============================================================


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, win_size, win_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows: torch.Tensor, win_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, win_size, win_size, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ============================================================
# Window-based multi-head self-attention (W-MSA)
# ============================================================


class WindowAttention(nn.Module):
    """Non-overlapping window multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, win_size: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias table for the (2*win-1) x (2*win-1) offsets.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads)
        )
        coords_h = torch.arange(win_size)
        coords_w = torch.arange(win_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += win_size - 1
        relative_coords[:, :, 1] += win_size - 1
        relative_coords[:, :, 0] *= 2 * win_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (num_windows*B, N, C) where N = win_size*win_size
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ============================================================
# Locally-enhanced Feed-Forward (LeFF)
# ============================================================


class LeFF(nn.Module):
    """Linear -> depthwise 3x3 conv (spatial) -> Linear, with GELU activations."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim
            ),
            nn.GELU(),
        )
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C) with N = H*W
        B, N, C = x.shape
        x = self.linear1(x)
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, hidden, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)
        x = self.linear2(x)
        return x


# ============================================================
# LeWin Transformer block
# ============================================================


class LeWinTransformerBlock(nn.Module):
    """Window-attention (W-MSA) + LeFF, each with a residual and LayerNorm."""

    def __init__(self, dim: int, num_heads: int, win_size: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, win_size=win_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = LeFF(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Partition windows -> W-MSA -> merge windows.
        x_windows = window_partition(x, self.win_size)  # (nW*B, win, win, C)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        x = window_reverse(attn_windows, self.win_size, H, W)  # (B, H, W, C)
        x = x.view(B, H * W, C)
        x = shortcut + x

        # LeFF.
        x = x + self.mlp(self.norm2(x), H, W)
        return x


# ============================================================
# Projections + resampling
# ============================================================


class InputProj(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H*W, out_channels)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channels: int = 32, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # (B, H*W, C) -> (B, out_channels, H, W)
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """Strided 4x4 conv: channels x2, resolution /2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Upsample(nn.Module):
    """2x2 transposed conv: channels /2, resolution x2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.deconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================================
# Full Uformer (U-shaped LeWin transformer)
# ============================================================


class Uformer(nn.Module):
    """U-shaped LeWin-transformer image-restoration network (random-init reimpl).

    A 2-level encoder/decoder around a bottleneck, every stage a stack of LeWin
    blocks, strided-conv downsampling, transposed-conv upsampling, and U-skip
    concatenation.  Returns ``input + residual`` (a restored image).
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 32,
        depths: List[int] = None,
        num_heads: List[int] = None,
        win_size: int = 8,
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [1, 1, 1, 1, 1]  # enc0, enc1, bottleneck, dec1, dec0
        if num_heads is None:
            num_heads = [1, 2, 4, 2, 1]
        self.win_size = win_size

        self.input_proj = InputProj(in_chans, embed_dim)

        # ----- Encoder -----
        self.encoder0 = nn.ModuleList(
            [LeWinTransformerBlock(embed_dim, num_heads[0], win_size) for _ in range(depths[0])]
        )
        self.down0 = Downsample(embed_dim, embed_dim * 2)
        self.encoder1 = nn.ModuleList(
            [LeWinTransformerBlock(embed_dim * 2, num_heads[1], win_size) for _ in range(depths[1])]
        )
        self.down1 = Downsample(embed_dim * 2, embed_dim * 4)

        # ----- Bottleneck -----
        self.bottleneck = nn.ModuleList(
            [LeWinTransformerBlock(embed_dim * 4, num_heads[2], win_size) for _ in range(depths[2])]
        )

        # ----- Decoder -----
        self.up1 = Upsample(embed_dim * 4, embed_dim * 2)
        # after concat with encoder1 skip -> 2*(embed_dim*2) channels, fuse back to embed_dim*2
        self.fuse1 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.decoder1 = nn.ModuleList(
            [LeWinTransformerBlock(embed_dim * 2, num_heads[3], win_size) for _ in range(depths[3])]
        )
        self.up0 = Upsample(embed_dim * 2, embed_dim)
        self.fuse0 = nn.Linear(embed_dim * 2, embed_dim)
        self.decoder0 = nn.ModuleList(
            [LeWinTransformerBlock(embed_dim, num_heads[4], win_size) for _ in range(depths[4])]
        )

        self.output_proj = OutputProj(embed_dim, in_chans)

    @staticmethod
    def _run(blocks: nn.ModuleList, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for blk in blocks:
            x = blk(x, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        B, _, H, W = x.shape

        y = self.input_proj(x)  # (B, H*W, embed_dim)

        # Encoder level 0
        e0 = self._run(self.encoder0, y, H, W)
        y = self.down0(e0, H, W)
        H1, W1 = H // 2, W // 2

        # Encoder level 1
        e1 = self._run(self.encoder1, y, H1, W1)
        y = self.down1(e1, H1, W1)
        H2, W2 = H1 // 2, W1 // 2

        # Bottleneck
        y = self._run(self.bottleneck, y, H2, W2)

        # Decoder level 1 (upsample, skip-concat with e1, fuse, transform)
        y = self.up1(y, H2, W2)
        y = torch.cat([y, e1], dim=-1)
        y = self.fuse1(y)
        y = self._run(self.decoder1, y, H1, W1)

        # Decoder level 0
        y = self.up0(y, H1, W1)
        y = torch.cat([y, e0], dim=-1)
        y = self.fuse0(y)
        y = self._run(self.decoder0, y, H, W)

        # Reconstruct image, residual connection.
        out = self.output_proj(y, H, W)
        return inp + out


def build(variant: str = "b") -> nn.Module:
    """Build a Uformer variant (returns one restored image tensor).

    Args:
        variant: "tiny" (embed_dim 16) or "b" (embed_dim 32).
    """
    if variant == "tiny":
        return Uformer(in_chans=3, embed_dim=16, depths=[1, 1, 1, 1, 1], num_heads=[1, 2, 4, 2, 1])
    return Uformer(in_chans=3, embed_dim=32, depths=[1, 1, 1, 1, 1], num_heads=[1, 2, 4, 2, 1])


# ============================================================
# Menagerie wiring: zero-arg builders + example input + entries.
# ============================================================


def build_uformer_tiny() -> nn.Module:
    """Build Uformer-Tiny (uformer_tiny_b16, embed_dim 16)."""
    return build("tiny")


def build_uformer_b() -> nn.Module:
    """Build Uformer-B (uformer_b32, embed_dim 32)."""
    return build("b")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 128, 128)`` for Uformer."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "Uformer (uformer_tiny_b16, U-shaped LeWin transformer restoration)",
        "build_uformer_tiny",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "Uformer (uformer_b32, U-shaped LeWin transformer restoration)",
        "build_uformer_b",
        "example_input",
        "2022",
        "DC",
    ),
]
