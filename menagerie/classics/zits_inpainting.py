"""ZITS: Incremental Transformer Structure Synthesis for Image Inpainting.

Dong et al., CVPR 2022.
Paper: https://arxiv.org/abs/2203.00867
Source: https://github.com/DQiaole/ZITS_inpainting

ZITS uses a two-branch coarse-to-fine inpainting strategy:

  1. TSR (Transformer Structure Restorer) -- a transformer-based module that
     predicts wireframe structure (edges and lines) from a masked image.  It
     operates on a downsampled version and uses a sparse transformer on top
     of a small CNN encoder.

  2. FTN (Fourier CNN Texture Network / inpainting generator) -- a CNN
     encoder-decoder with:
       * Masked convolutions for partial-conv style masking.
       * FFT-based Fourier blocks (FFC: Fast Fourier Convolution) that give
         the network global receptive field for filling large holes.
       * Skip connections from encoder to decoder.
       * The restored structure map from TSR is concatenated to the input.

Compact version: tiny CNN encoder-decoder + 2-layer transformer bottleneck
for TSR, and a small Fourier-CNN generator with 2 FFC blocks.
Input: (1, 4, 64, 64) -- 3-ch image + 1-ch mask concatenated.
Outputs: structure map (edges) + inpainted RGB.

Distinctive primitive: Fourier convolution block that mixes local conv +
global FFT spectral conv in the inpainting network.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fourier Convolution (FFC) block
# ---------------------------------------------------------------------------


class SpectralConv(nn.Module):
    """Global spectral convolution via 2D real FFT (compact FFC global branch)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Learn weights in the spectral domain -- simple linear on real/imag stacked
        self.weight_real = nn.Conv2d(channels, channels, 1)
        self.weight_imag = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        xf = torch.fft.rfft2(x, norm="ortho")
        # xf: (B, C, H, W//2+1) complex
        xf_real = self.weight_real(xf.real)
        xf_imag = self.weight_imag(xf.imag)
        xf_out = torch.complex(xf_real, xf_imag)
        return torch.fft.irfft2(xf_out, s=x.shape[-2:], norm="ortho")


class FFCBlock(nn.Module):
    """Fast Fourier Convolution block: local conv + global spectral conv, summed."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.global_conv = SpectralConv(channels)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_out = self.local_conv(x)
        global_out = self.global_conv(x)
        return self.act(self.norm(local_out + global_out))


# ---------------------------------------------------------------------------
# TSR: Transformer Structure Restorer
# ---------------------------------------------------------------------------


class TSREncoder(nn.Module):
    """Small CNN encoder for TSR: produces a sequence of tokens."""

    def __init__(self, in_ch: int = 4, dim: int = 32) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, 2, 1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 3, 2, 1),  # /4
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)  # (B, dim*2, H/4, W/4)


class TSRTransformer(nn.Module):
    """Lightweight transformer bottleneck (2 layers)."""

    def __init__(self, dim: int, num_heads: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer(tokens)


class TSR(nn.Module):
    """Transformer Structure Restorer: predicts edge/structure map."""

    def __init__(self, in_ch: int = 4, enc_dim: int = 32, out_ch: int = 1) -> None:
        super().__init__()
        feat_dim = enc_dim * 2
        self.encoder = TSREncoder(in_ch, enc_dim)
        self.transformer = TSRTransformer(feat_dim, num_heads=2, num_layers=2)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, enc_dim, 4, 2, 1),  # x2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(enc_dim, out_ch, 4, 2, 1),  # x2
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W) -- image+mask
        feat = self.encoder(x)  # (B, feat_dim, H/4, W/4)
        B, C, h, w = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, h*w, C)
        tokens = self.transformer(tokens)
        feat2 = tokens.permute(0, 2, 1).view(B, C, h, w)
        return self.head(feat2)  # (B, 1, H, W)


# ---------------------------------------------------------------------------
# FTN: Fourier CNN inpainting generator
# ---------------------------------------------------------------------------


class FTN(nn.Module):
    """Fourier Texture Network: CNN encoder-decoder with FFC blocks."""

    def __init__(self, in_ch: int = 5, base_ch: int = 32) -> None:
        """in_ch = 4 (img+mask) + 1 (structure map) = 5."""
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base_ch, 3, 1, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, 2, 1), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 2, 1), nn.ReLU(inplace=True)
        )
        # FFC bottleneck
        self.ffc1 = FFCBlock(base_ch * 4)
        self.ffc2 = FFCBlock(base_ch * 4)
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.ffc1(e3)
        b = self.ffc2(b)
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out


# ---------------------------------------------------------------------------
# Full ZITS model
# ---------------------------------------------------------------------------


class ZITS(nn.Module):
    """ZITS inpainting: TSR structure restorer + FTN Fourier-CNN generator.

    Input: (B, 4, H, W) -- masked RGB image (3ch) + binary mask (1ch).
    Output: tuple (structure_map, inpainted_rgb).
    """

    def __init__(self, enc_dim: int = 32, base_ch: int = 32) -> None:
        super().__init__()
        self.tsr = TSR(in_ch=4, enc_dim=enc_dim, out_ch=1)
        self.ftn = FTN(in_ch=5, base_ch=base_ch)  # img+mask+structure

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 4, H, W) = image_rgb + mask
        struct_map = self.tsr(x)  # (B, 1, H, W) edge prediction
        ftn_input = torch.cat([x, struct_map], dim=1)  # (B, 5, H, W)
        inpainted = self.ftn(ftn_input)  # (B, 3, H, W)
        return struct_map, inpainted


# ---------------------------------------------------------------------------
# Builder + example
# ---------------------------------------------------------------------------


def build_zits_inpainting() -> nn.Module:
    """Build compact ZITS inpainting model."""
    return ZITS(enc_dim=32, base_ch=32)


def example_input_zits() -> torch.Tensor:
    """(1, 4, 64, 64): 3-ch image + 1-ch binary mask."""
    return torch.randn(1, 4, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "ZITS Inpainting (TSR structure transformer + Fourier-CNN inpainting generator)",
        "build_zits_inpainting",
        "example_input_zits",
        "2022",
        "DC",
    ),
]
