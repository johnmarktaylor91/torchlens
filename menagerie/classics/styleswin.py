"""StyleSwin: Rethinking the Backbone Architecture for Diffusion-based Image Generation.

Zhang et al., CVPR 2022.  arXiv:2112.10762.
Source: https://github.com/microsoft/StyleSwin

StyleSwin combines a **Swin Transformer** backbone with **StyleGAN2-style modulation**:

Distinctive primitives:
  - **Swin window attention with style modulation**: instead of standard LayerNorm,
    each Swin block uses an affine modulation (from the style code w) to scale and
    shift the feature map before window self-attention. This is the signature op:
    style-modulated window attention.
  - **Shifted window attention**: alternating regular (W-MSA) and shifted (SW-MSA)
    windows for cross-window information flow (inherited from Swin Transformer).
  - **To-RGB projection at each scale**: learned 1x1 conv to RGB at each resolution,
    summed via skip connections (StyleGAN-style multi-scale output).
  - Mapping network (z -> w) feeds all Swin-style blocks.
  - Double Identity (DI): two consecutive identical Swin blocks per resolution (here 1 for compactness).
  - StyleGAN2 discriminator structure (residual conv blocks).

Compact: z_dim=64, w_dim=64, base_ch=32, win_size=4, output 32x32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Mapping network
# ============================================================


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 64, w_dim: int = 64, n_layers: int = 4) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.pixel_norm(z))


# ============================================================
# Style-modulated affine (replaces LayerNorm in Swin blocks)
# ============================================================


class StyleAffine(nn.Module):
    """Affine modulation from style code w -> per-channel scale and shift.

    StyleSwin's signature: replaces LayerNorm with style modulation.
    """

    def __init__(self, dim: int, w_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale_proj = nn.Linear(w_dim, dim)
        self.shift_proj = nn.Linear(w_dim, dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale = self.scale_proj(w).unsqueeze(1)  # (B, 1, dim)
        shift = self.shift_proj(w).unsqueeze(1)
        return x * (1 + scale) + shift


# ============================================================
# Window partition helpers
# ============================================================


def window_partition(x: torch.Tensor, win_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (nW*B, win*win, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size * win_size, C)
    return windows


def window_reverse(windows: torch.Tensor, win_size: int, H: int, W: int) -> torch.Tensor:
    """(nW*B, win*win, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# ============================================================
# Window attention module
# ============================================================


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA or SW-MSA)."""

    def __init__(self, dim: int, win_size: int, num_heads: int) -> None:
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        # Relative position bias
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * win_size - 1) * (2 * win_size - 1), num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        nh = self.num_heads
        hd = C // nh
        qkv = self.qkv(x).reshape(B_, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


# ============================================================
# Style-Swin block (style-modulated window attention + MLP)
# ============================================================


class StyleSwinBlock(nn.Module):
    """StyleSwin block: style-modulated affine -> W-MSA -> style-modulated affine -> MLP.

    Key difference from standard Swin: LayerNorm replaced by style modulation.
    Optionally shifted for SW-MSA.
    """

    def __init__(
        self,
        dim: int,
        w_dim: int,
        num_heads: int,
        win_size: int = 4,
        shift: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.shift = shift
        self.shift_size = win_size // 2 if shift else 0
        self.norm1 = StyleAffine(dim, w_dim)
        self.attn = WindowAttention(dim, win_size, num_heads)
        self.norm2 = StyleAffine(dim, w_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x, w)
        x = x.view(B, H, W, C)
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # Window partition
        x_windows = window_partition(shifted_x, self.win_size)  # (nW*B, win*win, C)
        attn_windows = self.attn(x_windows)
        x_merged = window_reverse(attn_windows, self.win_size, H, W)
        # Reverse cyclic shift
        if self.shift_size > 0:
            x_merged = torch.roll(x_merged, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x_merged.view(B, H * W, C)
        x = shortcut + x
        # MLP
        x = x + self.mlp(self.norm2(x, w))
        return x


# ============================================================
# StyleSwin synthesis stage (one resolution level)
# ============================================================


class StyleSwinStage(nn.Module):
    """One upsampling stage of StyleSwin.

    Upsample -> linear projection -> StyleSwin block(s) -> to-RGB.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        win_size: int = 4,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.proj_in = nn.Linear(in_ch, out_ch)
        # Two blocks: regular + shifted (StyleSwin Double Identity)
        self.block1 = StyleSwinBlock(out_ch, w_dim, num_heads, win_size, shift=False)
        self.block2 = StyleSwinBlock(out_ch, w_dim, num_heads, win_size, shift=True)
        self.to_rgb = nn.Conv2d(out_ch, 3, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.upsample(x)
        H2, W2 = H * 2, W * 2
        # Flatten to tokens
        x_tok = x.flatten(2).permute(0, 2, 1)  # (B, H2*W2, C)
        x_tok = self.proj_in(x_tok)  # (B, H2*W2, out_ch)
        x_tok = self.block1(x_tok, w, H2, W2)
        x_tok = self.block2(x_tok, w, H2, W2)
        # Reshape back
        out_ch = x_tok.size(-1)
        x_out = x_tok.permute(0, 2, 1).view(B, out_ch, H2, W2)
        rgb = self.to_rgb(x_out)
        return x_out, rgb


# ============================================================
# StyleSwin Generator
# ============================================================


class StyleSwinGenerator(nn.Module):
    """StyleSwin generator (compact).

    Mapping (z->w) -> constant 4x4 tensor -> 3 StyleSwin stages (each upsample x2)
    -> skip RGB sum. Output: 32x32.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        base_ch: int = 32,
        win_size: int = 4,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        ch = [base_ch * 4, base_ch * 2, base_ch]
        self.stage0 = StyleSwinStage(ch[0], ch[0], w_dim, win_size, num_heads=2)  # 4->8
        self.stage1 = StyleSwinStage(ch[0], ch[1], w_dim, win_size, num_heads=2)  # 8->16
        self.stage2 = StyleSwinStage(ch[1], ch[2], w_dim, win_size, num_heads=2)  # 16->32
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x, rgb0 = self.stage0(x, w)
        x, rgb1 = self.stage1(x, w)
        x, rgb2 = self.stage2(x, w)
        out = self.upsample(self.upsample(rgb0)) + self.upsample(rgb1) + rgb2
        return torch.tanh(out)


def build_styleswin_generator() -> nn.Module:
    return StyleSwinGenerator()


def example_input_styleswin_generator() -> torch.Tensor:
    return torch.randn(1, 64)


# ============================================================
# StyleSwin Discriminator (residual conv, same as StyleGAN2)
# ============================================================


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2)
        self.act2 = nn.LeakyReLU(0.2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act2(self.conv2(self.act1(self.conv1(x))))
        return (h + self.skip(x)) / math.sqrt(2)


class StyleSwinDiscriminator(nn.Module):
    """StyleSwin discriminator: standard residual blocks, same as StyleGAN2."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.from_rgb = nn.Sequential(nn.Conv2d(3, base_ch, 1), nn.LeakyReLU(0.2))
        self.block0 = ResBlock(base_ch, base_ch * 2)
        self.block1 = ResBlock(base_ch * 2, base_ch * 4)
        self.block2 = ResBlock(base_ch * 4, base_ch * 4)
        self.fc = nn.Linear(base_ch * 4 * 4 * 4, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(img)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1))


def build_styleswin_discriminator() -> nn.Module:
    return StyleSwinDiscriminator()


def example_input_styleswin_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "styleswin_generator",
        "build_styleswin_generator",
        "example_input_styleswin_generator",
        "2022",
        "DC",
    ),
    (
        "styleswin_discriminator",
        "build_styleswin_discriminator",
        "example_input_styleswin_discriminator",
        "2022",
        "DC",
    ),
]
