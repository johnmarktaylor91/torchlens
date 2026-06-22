"""Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation.

Cao et al., ECCV Workshop 2021.
Paper: https://arxiv.org/abs/2105.05537
Source: https://github.com/HuCaoFighting/Swin-Unet

Swin-Unet is a U-Net with Swin Transformer blocks replacing convolutions:
  Encoder: patch partition + patch merging downsampling stages with
    Swin Transformer blocks (window-based shifted-window self-attention)
  Bottleneck: Swin Transformer blocks at the deepest scale
  Decoder: patch expanding upsampling (linear expand + LayerNorm) with skip
    connections from encoder stages + Swin Transformer blocks
  Final: linear projection to num_classes at full resolution

Key Swin primitives faithfully reproduced:
  - Window partition / reverse for local self-attention within windows
  - Shifted-window attention (cyclic shift + masking)
  - Patch merging (4x spatial -> 2x channel)
  - Patch expanding (0.5x spatial -> 0.5x channel)

Architecture notes / faithful-core simplifications:
  - Compact dims: embed_dim=32, window_size=4, img_size=64x64
  - 2 Swin blocks per stage (published tiny: 2/2/6/2)
  - depth=4 (encoder: 3 stages + bottleneck)
  - Input: (1, 1, 64, 64) -- single channel for medical imaging convention
  - trace+draw verified 2026-06-21
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Window helpers
# ============================================================


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (num_windows*B, window_size, window_size, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition: (num_windows*B, Wh, Ww, C) -> (B, H, W, C)."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ============================================================
# Swin Transformer Block
# ============================================================


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA / SW-MSA).

    Faithfully includes: relative position bias table (simplified: learned offset
    per shift pair), cyclic-shift masking for shifted windows.
    """

    def __init__(self, dim: int, window_size: int, n_heads: int, shift: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.shift = shift
        head_dim = dim // n_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias
        self.rel_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, n_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)
        # Build relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flat = torch.flatten(coords, 1)  # (2, Wh*Ww)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_idx = rel.sum(-1)  # (N, N)
        self.register_buffer("rel_idx", rel_idx)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Relative position bias
        bias = self.rel_bias_table[self.rel_idx.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional cyclic shift.

    Automatically clamps window_size to min(H, W) for small feature maps,
    and disables shifting when the spatial size is too small.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple,
        n_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.H, self.W = input_resolution
        # Clamp window size and shift to spatial dimensions
        self.window_size = min(window_size, self.H, self.W)
        self.shift_size = (
            0 if (self.H <= self.window_size or self.W <= self.window_size) else shift_size
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, n_heads, shift=self.shift_size > 0)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        # Compute attention mask for SW-MSA
        if self.shift_size > 0:
            img_mask = torch.zeros(1, self.H, self.W, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            windows_m = window_partition(img_mask, self.window_size).view(
                -1, self.window_size * self.window_size
            )
            attn_mask = windows_m.unsqueeze(1) - windows_m.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        windows = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        attn_out = self.attn(windows, mask=self.attn_mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C) + shortcut
        x = x + self.mlp(self.norm2(x))
        return x


def _swin_stage(
    dim: int, res: tuple, n_heads: int, window_size: int, n_blocks: int
) -> nn.Sequential:
    """Build alternating W-MSA / SW-MSA blocks."""
    blocks = []
    for i in range(n_blocks):
        shift = (window_size // 2) if (i % 2 == 1) else 0
        blocks.append(SwinTransformerBlock(dim, res, n_heads, window_size, shift))
    return nn.Sequential(*blocks)


# ============================================================
# Patch Merging (encoder downsampling)
# ============================================================


class PatchMerging(nn.Module):
    """Swin patch merging: 2x spatial down, 2x channel up."""

    def __init__(self, dim: int, input_resolution: tuple) -> None:
        super().__init__()
        self.H, self.W = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


# ============================================================
# Patch Expanding (decoder upsampling)
# ============================================================


class PatchExpand(nn.Module):
    """Swin patch expanding: 2x spatial up, halve channels."""

    def __init__(self, dim: int, input_resolution: tuple) -> None:
        super().__init__()
        self.H, self.W = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        B, L, C = x.shape
        x = self.expand(x)  # (B, HW, 2C)
        x = x.view(B, H, W, 2 * C)
        # Rearrange: split 2C into 2x2 spatial blocks
        x = x.view(B, H, W, 2, 2, C // 2)
        # permute -> (B, 2H, 2W, C//2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, 2 * H, 2 * W, C // 2)
        x = x.view(B, 4 * H * W, C // 2)
        return self.norm(x)


class FinalPatchExpand(nn.Module):
    """Final patch expand to recover full resolution (patch_size x upscale)."""

    def __init__(self, dim: int, input_resolution: tuple, patch_size: int = 4) -> None:
        super().__init__()
        self.H, self.W = input_resolution
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.H, self.W
        p = self.patch_size
        B, L, C = x.shape
        x = self.expand(x)  # (B, HW, p*p*C)
        x = x.view(B, H, W, p, p, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * p, W * p, C)
        x = x.view(B, -1, C)
        return self.norm(x)


# ============================================================
# Swin-Unet
# ============================================================


class SwinUnetTiny(nn.Module):
    """Swin-Unet: hierarchical Swin Transformer U-Net.

    Encoder: patch_partition -> linear_embed -> 3 stages (patch merging)
    Bottleneck: Swin blocks at deepest scale
    Decoder: 3 stages (patch expand + skip concatenation + linear project + Swin blocks)
    Head: final_patch_expand -> linear projection to num_classes
    """

    def __init__(
        self,
        img_size: int = 64,
        in_ch: int = 1,
        num_classes: int = 9,
        embed_dim: int = 32,
        patch_size: int = 4,
        window_size: int = 4,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        H = W = img_size // patch_size  # patch grid size after initial embed

        # Initial patch embed
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.LayerNorm(embed_dim)

        # Encoder stage dims and resolutions
        dims = [embed_dim * (2**i) for i in range(4)]
        resolutions = [(H // (2**i), W // (2**i)) for i in range(4)]
        n_heads = [max(1, d // 32) for d in dims]

        # Encoder blocks
        self.enc0 = _swin_stage(dims[0], resolutions[0], n_heads[0], window_size, n_blocks)
        self.down0 = PatchMerging(dims[0], resolutions[0])
        self.enc1 = _swin_stage(dims[1], resolutions[1], n_heads[1], window_size, n_blocks)
        self.down1 = PatchMerging(dims[1], resolutions[1])
        self.enc2 = _swin_stage(dims[2], resolutions[2], n_heads[2], window_size, n_blocks)
        self.down2 = PatchMerging(dims[2], resolutions[2])

        # Bottleneck
        self.bottleneck = _swin_stage(dims[3], resolutions[3], n_heads[3], window_size, n_blocks)

        # Decoder (skip connection: concat + project back to dims[i])
        self.up2 = PatchExpand(dims[3], resolutions[3])
        # after up2: dim=dims[3]//2=dims[2], concat with enc2 output -> dims[2]*2 -> dims[2]
        self.skip2_proj = nn.Linear(dims[2] * 2, dims[2], bias=False)
        self.dec2 = _swin_stage(dims[2], resolutions[2], n_heads[2], window_size, n_blocks)

        self.up1 = PatchExpand(dims[2], resolutions[2])
        self.skip1_proj = nn.Linear(dims[1] * 2, dims[1], bias=False)
        self.dec1 = _swin_stage(dims[1], resolutions[1], n_heads[1], window_size, n_blocks)

        self.up0 = PatchExpand(dims[1], resolutions[1])
        self.skip0_proj = nn.Linear(dims[0] * 2, dims[0], bias=False)
        self.dec0 = _swin_stage(dims[0], resolutions[0], n_heads[0], window_size, n_blocks)

        # Final expand back to full resolution
        self.final_expand = FinalPatchExpand(dims[0], resolutions[0], patch_size=patch_size)
        self.head = nn.Linear(dims[0], num_classes, bias=False)

        self.H0, self.W0 = resolutions[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_in, W_in = x.shape[2], x.shape[3]

        # Patch embed
        feat = self.patch_embed(x)  # (B, embed_dim, H0, W0)
        B, C, H0, W0 = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H0*W0, C)
        tokens = self.patch_norm(tokens)

        # Encoder
        e0 = self.enc0(tokens)  # (B, H0*W0, d0)
        e1 = self.enc1(self.down0(e0))  # (B, H1*W1, d1)
        e2 = self.enc2(self.down1(e1))  # (B, H2*W2, d2)

        # Bottleneck
        b = self.bottleneck(self.down2(e2))  # (B, H3*W3, d3)

        # Decoder with skip connections
        d2 = self.up2(b)  # (B, H2*W2, d2)
        d2 = self.skip2_proj(torch.cat([d2, e2], dim=-1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # (B, H1*W1, d1)
        d1 = self.skip1_proj(torch.cat([d1, e1], dim=-1))
        d1 = self.dec1(d1)

        d0 = self.up0(d1)  # (B, H0*W0, d0)
        d0 = self.skip0_proj(torch.cat([d0, e0], dim=-1))
        d0 = self.dec0(d0)

        # Final expand + head
        out_tokens = self.final_expand(d0)  # (B, H_in*W_in, d0)
        logits = self.head(out_tokens)  # (B, H_in*W_in, num_classes)
        B2, L, K = logits.shape
        logits = logits.view(B2, H_in, W_in, K).permute(0, 3, 1, 2)
        return logits


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_swin_unet_tiny() -> nn.Module:
    return SwinUnetTiny(
        img_size=64,
        in_ch=1,
        num_classes=9,
        embed_dim=32,
        patch_size=4,
        window_size=4,
        n_blocks=2,
    )


def example_input() -> torch.Tensor:
    """Single-channel medical image (1, 1, 64, 64) for fast tracing."""
    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Swin-Unet Tiny (Swin Transformer U-Net: window-attn encoder + patch-expand decoder + skips)",
        "build_swin_unet_tiny",
        "example_input",
        "2021",
        "DC",
    ),
]
