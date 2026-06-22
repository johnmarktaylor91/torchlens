"""MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis.

Belousov 2021.  arXiv:2104.04767.
Source: https://github.com/bes-dev/MobileStyleGAN.pytorch

MobileStyleGAN's distinctive primitives:
  - **Depthwise-separable modulated conv**: replaces the standard StyleGAN2 modulated
    conv with a factored depthwise (grouped) + pointwise (1x1) sequence, halving
    compute while maintaining the per-sample style modulation.
  - **Inverse Discrete Wavelet Transform (IDWT) upsampling**: instead of bilinear
    upsampling or transposed convolutions, the model uses an IDWT block to upsample
    feature maps from frequency sub-bands (four channels: LL, LH, HL, HH) to a
    spatially 2x larger feature map.  This gives the characteristic frequency-domain
    upsampling path in the synthesis network.
  - **StyleGAN2 mapping network**: shared z -> W latent space.

Here we reproduce:
  - Mapping network z -> w.
  - Three IDWT synthesis blocks (8->16->32): each block performs modulated
    depthwise-separable conv, then IDWT upsampling (conceptually: 4 sub-band channels
    are recombined to produce spatial 2x resolution).
  - RGB head.

IDWT is approximated as a simple Haar wavelet IDWT (sum-fold 2x2) for compact tracing.

Random init, CPU, small channels for clean tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IDWT upsampling (Haar wavelet approximation)
# ---------------------------------------------------------------------------


class HaarIDWT(nn.Module):
    """Haar IDWT: maps (B, 4*C, H, W) -> (B, C, 2H, 2W).

    Treats every 4 channels as sub-bands [LL, LH, HL, HH] for each output channel.
    Inverse transform: reconstruct 2x2 spatial block from four sub-bands.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H, W = x.shape
        assert C4 % 4 == 0, "channels must be divisible by 4 for Haar IDWT"
        C = C4 // 4
        # Split into 4 sub-bands: each (B, C, H, W)
        ll = x[:, 0 * C : 1 * C]
        lh = x[:, 1 * C : 2 * C]
        hl = x[:, 2 * C : 3 * C]
        hh = x[:, 3 * C : 4 * C]
        # Reconstruct 2x2 pixels (normalized Haar inverse)
        top_left = (ll + lh + hl + hh) * 0.5
        top_right = (ll - lh + hl - hh) * 0.5
        bot_left = (ll + lh - hl - hh) * 0.5
        bot_right = (ll - lh - hl + hh) * 0.5
        # Interleave: (B, C, 2H, 2W)
        out = torch.stack([top_left, top_right, bot_left, bot_right], dim=2)
        out = out.view(B, C, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C, 2 * H, 2 * W)
        return out


# ---------------------------------------------------------------------------
# Depthwise-separable modulated conv
# ---------------------------------------------------------------------------


class DWSModConv(nn.Module):
    """Depthwise-separable modulated conv: depthwise + pointwise, style from w."""

    def __init__(self, in_c: int, out_c: int, k: int, w_dim: int) -> None:
        super().__init__()
        # Modulation: per-input-channel scale from style
        self.style = nn.Linear(w_dim, in_c)
        # Depthwise 3x3
        self.dw_weight = nn.Parameter(torch.randn(in_c, 1, k, k))
        # Pointwise 1x1
        self.pw_weight = nn.Parameter(torch.randn(out_c, in_c, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_c))
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.pad = k // 2

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Per-sample modulation of input channels
        scale = (self.style(w) + 1.0).view(B, self.in_c, 1, 1)  # (B, in_c, 1, 1)
        x = x * scale
        # Depthwise: process each sample independently
        # Use grouped conv for depthwise -- fold B into groups
        x_flat = x.reshape(1, B * self.in_c, x.shape[2], x.shape[3])
        dw_w = self.dw_weight.repeat(B, 1, 1, 1)  # (B*in_c, 1, k, k)
        x_flat = F.conv2d(x_flat, dw_w, padding=self.pad, groups=B * self.in_c)
        x = x_flat.view(B, self.in_c, x_flat.shape[2], x_flat.shape[3])
        # Pointwise
        x = F.conv2d(x, self.pw_weight)
        return x + self.bias.view(1, -1, 1, 1)


class IDWTSynthBlock(nn.Module):
    """MobileStyleGAN synthesis block: DWSModConv -> IDWT upsampling."""

    def __init__(self, in_c: int, out_c: int, w_dim: int) -> None:
        super().__init__()
        # Conv maps to 4*out_c channels (4 sub-bands for IDWT)
        self.conv = DWSModConv(in_c, out_c * 4, 3, w_dim)
        self.idwt = HaarIDWT()
        self.noise_w = nn.Parameter(torch.zeros(1, out_c, 1, 1))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.conv(x, w)  # (B, 4*out_c, H, W)
        x = self.idwt(x)  # (B, out_c, 2H, 2W)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return F.leaky_relu(x + self.noise_w * noise, 0.2)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MobileStyleGANSynthesis(nn.Module):
    """MobileStyleGAN synthesis network: const -> IDWT blocks -> RGB."""

    def __init__(self, z_dim: int = 32, w_dim: int = 64, nf: int = 8) -> None:
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        # Start at 8x8 to keep graph small
        self.const = nn.Parameter(torch.randn(1, nf * 4, 8, 8))
        # Three IDWT blocks: 8->16->32->64 -- but only go to 32 here
        self.blk1 = IDWTSynthBlock(nf * 4, nf * 2, w_dim)  # 8->16
        self.blk2 = IDWTSynthBlock(nf * 2, nf, w_dim)  # 16->32
        self.to_rgb = nn.Conv2d(nf, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        w = self.mapping(z)
        x = self.const.expand(B, -1, -1, -1)
        x = self.blk1(x, w)
        x = self.blk2(x, w)
        return torch.tanh(self.to_rgb(x))


def build_mobilestylegan_synthesis() -> nn.Module:
    return MobileStyleGANSynthesis()


def example_input() -> torch.Tensor:
    return torch.randn(1, 32)


MENAGERIE_ENTRIES = [
    (
        "MobileStyleGAN Synthesis (depthwise-separable modulated conv + Haar IDWT upsampling)",
        "build_mobilestylegan_synthesis",
        "example_input",
        "2021",
        "DC",
    ),
]
