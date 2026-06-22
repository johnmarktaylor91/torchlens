"""NeRV: Neural Representations for Videos.

Hao Chen, Bo He, Hanyu Wang, Yixuan Hou, Jui-Hsin Lai, Jia-Bin Huang, Matthew
Tran, Song Wang, Abhinav Gupta, Deepak Kachhadiya & Alex Schwing, NeurIPS 2021.
Paper: https://arxiv.org/abs/2110.13903
Source: https://github.com/haochen-rye/NeRV

NeRV is an implicit neural network that represents a VIDEO as a function of frame
index -> decoded frame.  Given a normalised time-step t in [0, 1], the network:

  1. Embeds t via a POSITIONAL ENCODING (sinusoidal frequencies), expanding the
     scalar index into a high-dimensional vector.
  2. Passes the encoding through a STEM MLP (1-2 fully-connected layers) producing
     an initial feature tensor that is reshaped into a small (H0, W0) spatial grid.
  3. Passes through a cascade of NeRV BLOCKS:
       - Conv2d upsampling layer (e.g. kernel 3x3 with stride=1, then pixel-shuffle 2x upsampling)
       - GELU activation + layer norm (or batch norm)
       - Output: spatially upsampled feature map
  4. A final head conv (1x1 or 3x3) mapping to 3-channel RGB, with sigmoid.

The distinctive primitives are:
  - Scalar frame-index -> positional encoding (no spatial input!)
  - NeRV block = Conv2d + PixelShuffle (sub-pixel upsampling) + activation
  - The entire video is implicitly decoded without any 2D input.

Simplifications in this reimplementation:
  - Positional encoding: 6 frequency octaves (sin+cos) of the scalar t -> 13-dim vector.
  - Stem MLP: linear 13 -> 64 -> 64, reshaped to (1, 64, 4, 4).
  - 3 NeRV blocks with upscale_factor=2 (4x4 -> 8x8 -> 16x16 -> 32x32).
  - Output: (1, 3, 32, 32) = one 32x32 RGB frame.
  - 1 video frame at a time (batch=1, scalar input).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding for NeRV (scalar time index -> high-dim vector)
# ---------------------------------------------------------------------------


class NeRVPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for a scalar time-step.

    Expands t (scalar) into [t, sin(pi*b^0*t), cos(pi*b^0*t), ..., sin(pi*b^(L-1)*t), cos(pi*b^(L-1)*t)]
    following the NeRV paper (frequency base b = 1.25 by default, L=6 octaves).

    Output dimension: 1 + 2 * L.
    """

    def __init__(self, L: int = 6, b: float = 1.25) -> None:
        super().__init__()
        self.L = L
        self.b = b
        self.d_out = 1 + 2 * L

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1) scalar time steps.  Returns (B, 1+2L)."""
        if t.dim() == 2:
            t = t.squeeze(-1)
        out = [t.unsqueeze(-1)]  # (B, 1)
        for i in range(self.L):
            freq = math.pi * (self.b**i)
            out.append(torch.sin(freq * t).unsqueeze(-1))
            out.append(torch.cos(freq * t).unsqueeze(-1))
        return torch.cat(out, dim=-1)  # (B, 1+2L)


# ---------------------------------------------------------------------------
# NeRV Block: the distinctive upsampling decoder block
# ---------------------------------------------------------------------------


class NeRVBlock(nn.Module):
    """A single NeRV decoder block.

    Architecture (faithful to NeRV source):
      1. Conv2d (in_channels -> out_channels * upscale^2, kernel 3x3, padding 1)
      2. PixelShuffle(upscale_factor)  -- rearranges (C*r^2, H, W) -> (C, H*r, W*r)
      3. GELU activation

    The PixelShuffle (sub-pixel convolution) upsampling is what gives NeRV
    its distinctive coarse-to-fine spatial decoding structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        # Conv outputs (out_channels * upscale^2) features for PixelShuffle
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.activation = nn.GELU()
        self.norm = nn.GroupNorm(1, out_channels)  # InstanceNorm2d equivalent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, H, W) -> (B, C_out, H*r, W*r)."""
        x = self.conv(x)  # (B, C_out * r^2, H, W)
        x = self.pixel_shuffle(x)  # (B, C_out, H*r, W*r)
        x = self.norm(x)
        x = self.activation(x)
        return x


# ---------------------------------------------------------------------------
# NeRV model
# ---------------------------------------------------------------------------


class NeRV(nn.Module):
    """NeRV: Neural Representation for Video.

    Maps a scalar frame index t -> decoded RGB frame via:
      positional_encoding(t) -> stem_mlp -> reshape -> NeRV_blocks -> head_conv -> RGB.

    Distinctive topology: NO spatial input; the entire frame is decoded from t alone.
    """

    def __init__(
        self,
        pe_L: int = 6,
        pe_b: float = 1.25,
        stem_dim: int = 64,
        stem_H: int = 4,
        stem_W: int = 4,
        block_channels: List[int] = None,
        upscale_factors: List[int] = None,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        if block_channels is None:
            block_channels = [64, 32, 16]
        if upscale_factors is None:
            upscale_factors = [2, 2, 2]
        assert len(block_channels) == len(upscale_factors)

        self.stem_H = stem_H
        self.stem_W = stem_W
        self.stem_dim = stem_dim

        # Positional encoding for scalar t
        self.pe = NeRVPositionalEncoding(L=pe_L, b=pe_b)
        d_pe = self.pe.d_out  # 1 + 2*L

        # Stem MLP: PE -> flat feature vector, then reshape to (C, H0, W0)
        self.stem_mlp = nn.Sequential(
            nn.Linear(d_pe, 256),
            nn.GELU(),
            nn.Linear(256, stem_dim * stem_H * stem_W),
            nn.GELU(),
        )

        # NeRV decoder blocks (cascade of upsampling blocks)
        in_ch = stem_dim
        nerv_blocks = []
        for out_ch, upscale in zip(block_channels, upscale_factors):
            nerv_blocks.append(NeRVBlock(in_ch, out_ch, upscale_factor=upscale))
            in_ch = out_ch
        self.nerv_blocks = nn.ModuleList(nerv_blocks)

        # Output head: final conv -> 3-channel RGB
        self.head_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t: (B,) or (B, 1) normalised time indices in [0, 1].

        Returns:
            (B, 3, H_out, W_out) decoded RGB frames with values in [0, 1].
        """
        # 1. Positional encoding
        enc = self.pe(t)  # (B, d_pe)

        # 2. Stem MLP -> reshape to spatial grid
        stem_feat = self.stem_mlp(enc)  # (B, stem_dim * H0 * W0)
        B = stem_feat.shape[0]
        x = stem_feat.view(B, self.stem_dim, self.stem_H, self.stem_W)  # (B, C, H0, W0)

        # 3. NeRV blocks: cascade of PixelShuffle upsamplings
        for block in self.nerv_blocks:
            x = block(x)

        # 4. Head conv -> RGB
        x = self.head_conv(x)
        return torch.sigmoid(x)  # (B, 3, H_out, W_out)


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_nerv() -> nn.Module:
    """Build a compact NeRV model (3 blocks: 4x4 -> 8x8 -> 16x16 -> 32x32)."""
    return NeRV(
        pe_L=6,
        pe_b=1.25,
        stem_dim=64,
        stem_H=4,
        stem_W=4,
        block_channels=[64, 32, 16],
        upscale_factors=[2, 2, 2],
        out_channels=3,
    )


def example_input() -> torch.Tensor:
    """Example frame index (1,) scalar for NeRV -- a single normalised time step."""
    return torch.tensor([0.5])  # (1,) scalar, represents mid-video frame


MENAGERIE_ENTRIES = [
    (
        "NeRV (Neural Representation for Video: frame-index -> PixelShuffle decoder -> RGB frame)",
        "build_nerv",
        "example_input",
        "2021",
        "DC",
    ),
]
