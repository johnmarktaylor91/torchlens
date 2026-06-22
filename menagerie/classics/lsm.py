"""LSM: Latent Spectral Models for PDE Operator Learning.

Wu et al., 2023.
Paper: https://arxiv.org/abs/2306.00048
Source: https://github.com/thuml/Latent-Spectral-Models

LSM's distinctive architecture:
  1. Hierarchical encoder: projects the input function from physical space to a
     small set of latent tokens (latent_dim << grid_size). This is done via
     a learned PROJECTION from the spatial grid to a compact latent space
     (similar to a learned spatial pooling or a CLS token bank).
  2. Latent Spectral Processing: the latent tokens interact via a spectral
     (Fourier-type) attention operator. Since the latent space is small,
     this is computationally cheap and acts as a global spectral mixer.
     Specifically, the latent tokens are mixed using a learnable spectral
     operator (dense matrix over the latent dimension).
  3. Hierarchical decoder: the processed latent tokens are projected back
     to the physical grid via interpolation/transpose-projection.
  4. Multiple scales: several encoder-decoder pairs at different resolutions
     (similar to a UNet in latent space).

The key innovation: the LATENT SPECTRAL BLOCK which projects N physical tokens
down to L latent tokens (L << N), applies a spectral operator (dense L×L matrix
in frequency domain), and reprojects back.

Simplifications: 2 encoder stages, 2 latent spectral layers, d_model=64,
L=16 latent tokens, input grid 1D N=64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentProjection(nn.Module):
    """Project N physical tokens to L latent tokens (learned aggregation).

    This is the hierarchical encoder: a learned linear projection over the
    spatial dimension. Physical shape: (B, N, C) -> Latent: (B, L, C).
    """

    def __init__(self, n_phys: int, n_latent: int, d_model: int) -> None:
        super().__init__()
        # Learned projection matrix: L x N (applied over spatial dim)
        self.proj = nn.Linear(n_phys, n_latent, bias=False)
        # Channel mixing after projection
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x_t = x.transpose(1, 2)  # (B, C, N)
        latent_t = self.proj(x_t)  # (B, C, L)
        latent = latent_t.transpose(1, 2)  # (B, L, C)
        return self.channel_mix(latent)


class SpectralLatentOperator(nn.Module):
    """Spectral operator over latent tokens.

    For L latent tokens: computes rfft over the L dimension, applies a learned
    complex linear operator (truncated to n_modes modes), then irfft.
    This is the 'spectral' in LSM -- global mixing in frequency space.
    """

    def __init__(self, n_latent: int, d_model: int, n_modes: int = 8) -> None:
        super().__init__()
        self.n_latent = n_latent
        self.n_modes = min(n_modes, n_latent // 2 + 1)
        # Complex weights: (n_modes, d_model, d_model) -- separate real/imag
        self.W_re = nn.Parameter(torch.randn(self.n_modes, d_model, d_model) * 0.02)
        self.W_im = nn.Parameter(torch.randn(self.n_modes, d_model, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        B, L, D = x.shape

        # rfft over latent dim
        x_f = torch.fft.rfft(x, dim=1)  # (B, L//2+1, D) complex

        # Truncate to n_modes
        x_f_trunc = x_f[:, : self.n_modes, :]  # (B, n_modes, D)

        # Apply complex linear operator: (n_modes, D, D) x (B, n_modes, D)
        x_f_re = x_f_trunc.real
        x_f_im = x_f_trunc.imag

        out_re = torch.einsum("bmd,mde->bme", x_f_re, self.W_re) - torch.einsum(
            "bmd,mde->bme", x_f_im, self.W_im
        )
        out_im = torch.einsum("bmd,mde->bme", x_f_re, self.W_im) + torch.einsum(
            "bmd,mde->bme", x_f_im, self.W_re
        )
        out_f_trunc = torch.complex(out_re, out_im)

        # Pad back to full spectrum
        out_f = torch.zeros(B, L // 2 + 1, D, dtype=torch.complex64, device=x.device)
        out_f[:, : self.n_modes, :] = out_f_trunc

        # irfft
        out = torch.fft.irfft(out_f, n=L, dim=1)  # (B, L, D)
        return out


class LatentSpectralBlock(nn.Module):
    """Full LSM latent spectral block: project -> spectral op -> reproject."""

    def __init__(self, n_phys: int, n_latent: int, d_model: int, n_modes: int = 8) -> None:
        super().__init__()
        self.enc = LatentProjection(n_phys, n_latent, d_model)
        self.spectral = SpectralLatentOperator(n_latent, d_model, n_modes)
        self.norm = nn.LayerNorm(d_model)
        # Reproject from latent back to physical
        self.dec_proj = nn.Linear(n_latent, n_phys, bias=False)
        self.channel_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        B, N, D = x.shape

        # Encode to latent
        z = self.enc(x)  # (B, L, D)

        # Spectral mixing in latent space
        z = z + self.spectral(self.norm(z))

        # Decode back to physical
        z_t = z.transpose(1, 2)  # (B, D, L)
        x_out = self.dec_proj(z_t).transpose(1, 2)  # (B, N, D)
        return x + self.channel_out(x_out)


class LSM(nn.Module):
    """Latent Spectral Model for 1D PDE operator learning.

    Stacks LatentSpectralBlocks with a pointwise local branch (FNO-like bypass).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        n_phys: int = 64,  # spatial grid size
        n_latent: int = 16,  # latent token count (L << N)
        n_layers: int = 2,
        n_modes: int = 8,
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList(
            [LatentSpectralBlock(n_phys, n_latent, d_model, n_modes) for _ in range(n_layers)]
        )
        # Local pointwise bypass (channel mix without spatial interaction)
        self.bypass = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_channels)
        h = self.lift(x)  # (B, N, d_model)
        for blk, bypass in zip(self.blocks, self.bypass):
            h = blk(h) + bypass(h)
        return self.project(h)


def build_lsm() -> nn.Module:
    return LSM(
        in_channels=1, out_channels=1, d_model=64, n_phys=64, n_latent=16, n_layers=2, n_modes=8
    )


def example_input_lsm() -> torch.Tensor:
    # (B=1, N=64, in_channels=1): 1D PDE input function on 64 grid points
    return torch.randn(1, 64, 1)


MENAGERIE_ENTRIES = [
    ("LSM (Latent Spectral Model)", "build_lsm", "example_input_lsm", "2023", "DC"),
]
