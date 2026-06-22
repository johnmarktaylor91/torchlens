"""SwinIR: Image Restoration Using Swin Transformer.

Liang et al., ICCV Workshops 2021.
Paper: https://arxiv.org/abs/2108.10257
Source: https://github.com/JingyunLiang/SwinIR

SwinIR is an image-restoration transformer built from three stages:

  1. Shallow feature extraction -- 3x3 conv head.
  2. Deep feature extraction -- stack of RSTB (Residual Swin Transformer Blocks).
     Each RSTB = several Swin Transformer Layers (window-MSA / shifted-window-MSA
     alternating, with relative position bias) followed by a 3x3 conv, and a
     residual connection from RSTB input to RSTB output.
  3. Reconstruction head -- task-specific:
       * 'pixelshuffle'      -- Conv + PixelShuffle per upscale factor (classical SR x2/3/4/8)
       * 'pixelshuffledirect'-- single Conv + PixelShuffle (lightweight SR x4)
       * 'nearest+conv'      -- Upsample (nearest) + Conv (real-world SR x4)
       * None                -- Conv head only, residual output (denoising / CAR)

Distinctive primitives shown:
  - Window partition / merge (W-MSA over non-overlapping win_size x win_size windows)
  - Cyclic shift + mask trick for SW-MSA (shifted window attention)
  - RSTB residual (transformer block stack + conv + skip)
  - Task-specific upsampler (pixel-shuffle, nearest+conv, or passthrough)

Compact config: embed_dim 30-60, 2 RSTBs x 2 Swin layers, window_size 8.
Input sizes kept small (32x32 LR or 32x32 for restoration).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ---------------------------------------------------------------------------
# Window Attention (W-MSA / SW-MSA)
# ---------------------------------------------------------------------------


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with optional cyclic shift (SW-MSA)."""

    def __init__(self, dim: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        )
        coords_flat = torch.flatten(coords, 1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # relative position bias
        rp_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rp_bias = rp_bias.view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + rp_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Swin Transformer Layer
# ---------------------------------------------------------------------------


class SwinTransformerLayer(nn.Module):
    """One Swin Transformer layer: W-MSA or SW-MSA + MLP, with LayerNorm & residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(
        self, x: torch.Tensor, H: int, W: int, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# RSTB: Residual Swin Transformer Block
# ---------------------------------------------------------------------------


class RSTB(nn.Module):
    """Residual Swin Transformer Block: num_layers Swin layers + 3x3 conv + residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int = 2,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        shift = window_size // 2

        self.layers = nn.ModuleList(
            [
                SwinTransformerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else shift,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(num_layers)
            ]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def _compute_mask(self, H: int, W: int, shift_size: int, device: torch.device) -> torch.Tensor:
        """Compute attention mask for SW-MSA."""
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        residual = x  # (B, H*W, C)
        B, L, C = x.shape

        shift_size = self.window_size // 2
        attn_mask = self._compute_mask(H, W, shift_size, x.device)

        for i, layer in enumerate(self.layers):
            mask = attn_mask if (i % 2 == 1) else None
            x = layer(x, H, W, mask)

        # Conv on spatial form + residual
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x + residual


# ---------------------------------------------------------------------------
# Upsamplers
# ---------------------------------------------------------------------------


class UpsamplerPixelShuffle(nn.Module):
    """Pixel-shuffle upsampler (classical SR). Handles x2, x3, x4, x8."""

    def __init__(self, scale: int, num_feat: int, out_chans: int = 3) -> None:
        super().__init__()
        layers = []
        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                layers += [
                    nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
        elif scale == 3:
            layers += [
                nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1),
                nn.PixelShuffle(3),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Conv2d(num_feat, out_chans, 3, 1, 1))
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpsamplerPixelShuffleDirect(nn.Module):
    """Single-conv pixel-shuffle (lightweight SR x4)."""

    def __init__(self, scale: int, num_feat: int, out_chans: int = 3) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(num_feat, out_chans * (scale**2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class UpsamplerNearestConv(nn.Module):
    """Nearest-neighbour + conv upsampler (real-world SR x4)."""

    def __init__(self, scale: int, num_feat: int, out_chans: int = 3) -> None:
        super().__init__()
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log2(scale))):
                layers += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
        layers.append(nn.Conv2d(num_feat, out_chans, 3, 1, 1))
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


# ---------------------------------------------------------------------------
# Full SwinIR model
# ---------------------------------------------------------------------------


class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer (random-init reimpl).

    Args:
        upscale:      SR scale factor (1 for restoration tasks).
        in_chans:     Input channels (3 for colour, 1 for grayscale).
        embed_dim:    Feature channels.
        num_rstb:     Number of RSTB blocks.
        layers_per_rstb: Swin layers per RSTB.
        num_heads:    Attention heads.
        window_size:  Window size for W-MSA.
        upsampler:    'pixelshuffle', 'pixelshuffledirect', 'nearest+conv', or None.
    """

    def __init__(
        self,
        upscale: int = 4,
        in_chans: int = 3,
        embed_dim: int = 30,
        num_rstb: int = 2,
        layers_per_rstb: int = 2,
        num_heads: int = 3,
        window_size: int = 8,
        upsampler: str | None = "pixelshuffle",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Deep feature extraction: stack of RSTB
        self.rstb_layers = nn.ModuleList(
            [RSTB(embed_dim, num_heads, layers_per_rstb, window_size) for _ in range(num_rstb)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Reconstruction head
        if upsampler == "pixelshuffle":
            self.upsample = UpsamplerPixelShuffle(upscale, embed_dim, in_chans)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsamplerPixelShuffleDirect(upscale, embed_dim, in_chans)
        elif upsampler == "nearest+conv":
            self.upsample = UpsamplerNearestConv(upscale, embed_dim, in_chans)
        else:
            # Denoising / CAR: residual output at same resolution
            self.upsample = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Shallow features
        feat = self.conv_first(x)

        # Deep features: flatten tokens
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        for rstb in self.rstb_layers:
            tokens = rstb(tokens, H, W)
        tokens = self.norm(tokens)

        # Back to spatial
        deep = tokens.transpose(1, 2).view(B, -1, H, W)
        deep = self.conv_after_body(deep) + feat

        # Reconstruction
        if self.upsampler in ("pixelshuffle", "pixelshuffledirect", "nearest+conv"):
            out = self.upsample(deep)
        else:
            # Residual image restoration
            out = self.upsample(deep) + x
        return out


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_swinir_classical_sr_x2() -> nn.Module:
    return SwinIR(
        upscale=2,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler="pixelshuffle",
    )


def build_swinir_classical_sr_x3() -> nn.Module:
    return SwinIR(
        upscale=3,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler="pixelshuffle",
    )


def build_swinir_classical_sr_x4() -> nn.Module:
    return SwinIR(
        upscale=4,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler="pixelshuffle",
    )


def build_swinir_classical_sr_x8() -> nn.Module:
    return SwinIR(
        upscale=8,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler="pixelshuffle",
    )


def build_swinir_lightweight_sr_x4() -> nn.Module:
    return SwinIR(
        upscale=4,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler="pixelshuffledirect",
    )


def build_swinir_real_sr_x4() -> nn.Module:
    return SwinIR(
        upscale=4,
        in_chans=3,
        embed_dim=60,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=6,
        window_size=8,
        upsampler="nearest+conv",
    )


def build_swinir_gray_denoising() -> nn.Module:
    return SwinIR(
        upscale=1,
        in_chans=1,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler=None,
    )


def build_swinir_color_jpeg_car() -> nn.Module:
    return SwinIR(
        upscale=1,
        in_chans=3,
        embed_dim=30,
        num_rstb=2,
        layers_per_rstb=2,
        num_heads=3,
        window_size=8,
        upsampler=None,
    )


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------


def example_input_sr() -> torch.Tensor:
    """RGB LR image (1,3,32,32) for SR variants."""
    return torch.randn(1, 3, 32, 32)


def example_input_gray() -> torch.Tensor:
    """Grayscale image (1,1,32,32) for gray denoising."""
    return torch.randn(1, 1, 32, 32)


def example_input_color() -> torch.Tensor:
    """RGB image (1,3,32,32) for color JPEG CAR."""
    return torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Menagerie entries
# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    (
        "SwinIR Classical SR x2 (window/shifted-window MSA, RSTB, pixel-shuffle x2)",
        "build_swinir_classical_sr_x2",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Classical SR x3 (window/shifted-window MSA, RSTB, pixel-shuffle x3)",
        "build_swinir_classical_sr_x3",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Classical SR x4 (window/shifted-window MSA, RSTB, pixel-shuffle x4)",
        "build_swinir_classical_sr_x4",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Classical SR x8 (window/shifted-window MSA, RSTB, pixel-shuffle x8)",
        "build_swinir_classical_sr_x8",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Lightweight SR x4 (fewer RSTBs, pixelshuffledirect upsampler)",
        "build_swinir_lightweight_sr_x4",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Real-World SR x4 (nearest+conv upsampler for real degradation)",
        "build_swinir_real_sr_x4",
        "example_input_sr",
        "2021",
        "DC",
    ),
    (
        "SwinIR Gray Denoising (1-ch input, residual output, no upsample)",
        "build_swinir_gray_denoising",
        "example_input_gray",
        "2021",
        "DC",
    ),
    (
        "SwinIR Color JPEG CAR (3-ch, residual output, artifact removal)",
        "build_swinir_color_jpeg_car",
        "example_input_color",
        "2021",
        "DC",
    ),
]
