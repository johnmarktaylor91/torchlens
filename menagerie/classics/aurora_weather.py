"""Aurora: a foundation model for the Earth system (3D Swin Transformer U-Net).

Bodnar et al., "A foundation model for the Earth system", Nature 2025
(arXiv:2405.13063).
Source: https://github.com/microsoft/aurora (aurora/model/swin3d.py, encoder.py,
        decoder.py, perceiver.py).

Aurora "ceilings" in the source catalog because the real ``forward`` takes an
``aurora.Batch`` dataclass (surface-variable dict, static-variable dict,
atmospheric-variable dict at pressure levels, plus ``Metadata``) -- never a plain
tensor (``tl.trace(model, tensor)`` fails with "'Tensor' has no attribute
'normalise'").  This is a faithful pure-torch reimplementation of the *core* 3D
Swin Transformer U-Net backbone consuming the token grid directly as a plain 5D
tensor ``(1, C, D, H, W)`` -- bypassing the Perceiver encoder/decoder (which only
map weather fields <-> the token grid) and replacing the time-conditioned
AdaptiveLayerNorm with plain LayerNorm.

Faithful-core architecture (source-verified):
  - 3 encoder stages of ``Swin3D`` blocks (windowed 3D self-attention with the
    W-MSA / shifted-window SW-MSA alternation) + ``PatchMerging3D`` downsampling.
  - a symmetric 3-stage decoder with ``PatchSplitting3D`` upsampling and additive
    U-Net skip connections from the matching encoder stage.
  - block order (pre-norm): LN -> (cyclic shift) -> window partition -> windowed
    MHSA -> window reverse -> (un-shift) -> +residual -> LN -> MLP -> +residual.

Note (source-verified): Aurora's ``WindowAttention`` uses masked scaled-dot-
product attention with NO learned relative-position-bias table; positional
information comes from Fourier encodings injected in the (here bypassed) encoder
and the SW-MSA attention mask.  This reimplementation keeps the masked SW-MSA
form faithfully and omits the rel-pos table accordingly.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn


def _window_partition(x: torch.Tensor, ws: Tuple[int, int, int]) -> torch.Tensor:
    # x: (B, D, H, W, C) -> (num_windows*B, wd*wh*ww, C)
    B, D, H, W, C = x.shape
    wd, wh, ww = ws
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, C)


def _window_reverse(
    windows: torch.Tensor, ws: Tuple[int, int, int], dims: Tuple[int, int, int]
) -> torch.Tensor:
    wd, wh, ww = ws
    D, H, W = dims
    B = int(windows.shape[0] / (D * H * W / wd / wh / ww))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


def _compute_mask(
    dims: Tuple[int, int, int],
    ws: Tuple[int, int, int],
    shift: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    D, H, W = dims
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    d_slices = (slice(-ws[0]), slice(-ws[0], -shift[0]), slice(-shift[0], None))
    h_slices = (slice(-ws[1]), slice(-ws[1], -shift[1]), slice(-shift[1], None))
    w_slices = (slice(-ws[2]), slice(-ws[2], -shift[2]), slice(-shift[2], None))
    for d, h, w in itertools.product(d_slices, h_slices, w_slices):
        img_mask[:, d, h, w, :] = cnt
        cnt += 1
    mask_windows = _window_partition(img_mask, ws).squeeze(-1)  # (nW, wd*wh*ww)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask  # (nW, wsz, wsz)


class _WindowAttention3D(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (nW*B, n, C)
        Bn, n, C = x.shape
        qkv = self.qkv(x).reshape(Bn, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (Bn, H, n, n)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.n_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(Bn, n, C)
        return self.proj(out)


class _Swin3DBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        window_size: Tuple[int, int, int],
        shift: bool,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(w // 2 for w in window_size) if shift else (0, 0, 0)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention3D(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        B, D, H, W, C = x.shape
        ws = self.window_size
        shortcut = x
        x = self.norm1(x)
        if any(self.shift_size):
            x = torch.roll(x, shifts=tuple(-s for s in self.shift_size), dims=(1, 2, 3))
            mask = _compute_mask((D, H, W), ws, self.shift_size, x.device)
        else:
            mask = None
        windows = _window_partition(x, ws)
        attn = self.attn(windows, mask)
        x = _window_reverse(attn, ws, (D, H, W))
        if any(self.shift_size):
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchMerging3D(nn.Module):
    """Gather 2x2 spatial neighbours -> concat 4*dim -> LayerNorm -> Linear(4dim->2dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, H, W, C)
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduction(self.norm(x))


class _PatchSplitting3D(nn.Module):
    """Inverse of PatchMerging: Linear(dim->2dim) -> LayerNorm -> scatter to 2x2 -> dim/2."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        x = self.expand(x)  # (B,D,H,W,2C)
        x = x.view(B, D, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, D, H * 2, W * 2, C // 2)
        return self.norm(x)


class _BasicLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _Swin3DBlock(dim, n_heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class Aurora3DSwinUNet(nn.Module):
    """Aurora core 3D Swin Transformer U-Net backbone (token-grid in, token-grid out)."""

    def __init__(
        self,
        embed_dim: int = 64,
        encoder_depths: Tuple[int, ...] = (2, 2, 2),
        encoder_heads: Tuple[int, ...] = (2, 4, 8),
        window_size: Tuple[int, int, int] = (2, 4, 4),
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.n_stages = len(encoder_depths)
        # encoder stages + downsamples between them
        self.enc_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dims = [embed_dim * (2**i) for i in range(self.n_stages)]
        for i in range(self.n_stages):
            self.enc_layers.append(
                _BasicLayer3D(dims[i], encoder_depths[i], encoder_heads[i], window_size, mlp_ratio)
            )
            if i < self.n_stages - 1:
                self.downsamples.append(_PatchMerging3D(dims[i]))
        # decoder stages (mirror) + upsamples
        self.dec_layers = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(self.n_stages - 1, 0, -1):
            self.upsamples.append(_PatchSplitting3D(dims[i]))
            self.dec_layers.append(
                _BasicLayer3D(
                    dims[i - 1], encoder_depths[i - 1], encoder_heads[i - 1], window_size, mlp_ratio
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) token grid -> (B, D, H, W, C) channels-last for attention
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        skips: List[torch.Tensor] = []
        for i, layer in enumerate(self.enc_layers):
            x = layer(x)
            if i < self.n_stages - 1:
                skips.append(x)
                x = self.downsamples[i](x)
        for up, layer in zip(self.upsamples, self.dec_layers):
            x = up(x)
            x = x + skips.pop()  # additive U-Net skip
            x = layer(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()  # back to (B, C, D, H, W)


def build_aurora() -> nn.Module:
    return Aurora3DSwinUNet(
        embed_dim=64,
        encoder_depths=(2, 2, 2),
        encoder_heads=(2, 4, 8),
        window_size=(2, 4, 4),
        mlp_ratio=4.0,
    )


def example_input_aurora() -> torch.Tensor:
    """Token grid ``(1, 64, 4, 16, 16)`` = (B, embed_dim, latent_levels, H', W')."""
    return torch.randn(1, 64, 4, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "Aurora (Swin-3D-U-Net weather foundation model)",
        "build_aurora",
        "example_input_aurora",
        "2024",
        "DC",
    ),
]
