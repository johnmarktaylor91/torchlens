"""Compact DDPM U-Net denoiser with timestep embeddings and attention.

Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.  The
standard DDPM denoiser is a U-Net that predicts noise from a noisy sample and a
diffusion timestep, using sinusoidal timestep embeddings injected into residual
blocks and self-attention at lower spatial resolutions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal diffusion timestep embeddings.

    Parameters
    ----------
    t:
        Timestep tensor ``(B,)``.
    dim:
        Embedding dimension.

    Returns
    -------
    torch.Tensor
        Sinusoidal embeddings.
    """

    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
    emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeResBlock(nn.Module):
    """DDPM residual block with additive timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        """Initialize convolutions and timestep projection.

        Parameters
        ----------
        in_channels:
            Input channels.
        out_channels:
            Output channels.
        time_dim:
            Timestep embedding width.
        """

        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply timestep-conditioned residual update.

        Parameters
        ----------
        x:
            Feature map.
        emb:
            Timestep embedding.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = self.conv1(F.silu(self.norm1(x)))
        y = y + self.time(emb).unsqueeze(-1).unsqueeze(-1)
        y = self.conv2(F.silu(self.norm2(y)))
        return y + self.skip(x)


class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial pixels."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        channels:
            Feature width.
        heads:
            Attention head count.
        """

        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Attended feature map.
        """

        bsz, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return x + out.transpose(1, 2).reshape(bsz, channels, height, width)


class DDPMUNetCompact(nn.Module):
    """Small timestep-conditioned U-Net denoiser."""

    def __init__(self, channels: int = 3, width: int = 32, time_dim: int = 64) -> None:
        """Initialize DDPM U-Net blocks.

        Parameters
        ----------
        channels:
            Image channel count.
        width:
            Base feature width.
        time_dim:
            Timestep embedding width.
        """

        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.in_conv = nn.Conv2d(channels, width, 3, padding=1)
        self.down1 = TimeResBlock(width, width, time_dim)
        self.down2 = TimeResBlock(width, width * 2, time_dim)
        self.mid = TimeResBlock(width * 2, width * 2, time_dim)
        self.attn = SpatialAttention(width * 2)
        self.up1 = TimeResBlock(width * 3, width, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(4, width), nn.SiLU(), nn.Conv2d(width, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise from a noisy image and timestep.

        Parameters
        ----------
        x:
            Noisy image tensor.
        t:
            Diffusion timestep tensor.

        Returns
        -------
        torch.Tensor
            Predicted noise.
        """

        emb = self.time_mlp(timestep_embedding(t, 64))
        h1 = self.down1(self.in_conv(x), emb)
        h2 = self.down2(F.avg_pool2d(h1, 2), emb)
        h2 = self.attn(self.mid(h2, emb))
        up = F.interpolate(h2, size=h1.shape[-2:], mode="nearest")
        return self.out(self.up1(torch.cat([up, h1], dim=1), emb))


def build() -> nn.Module:
    """Build compact DDPM U-Net.

    Returns
    -------
    nn.Module
        DDPM U-Net denoiser.
    """

    return DDPMUNetCompact()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy image and timestep inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image and timestep.
    """

    return torch.randn(1, 3, 32, 32), torch.tensor([10])


MENAGERIE_ENTRIES = [
    (
        "ddpm_unet (timestep-conditioned attention U-Net denoiser)",
        "build",
        "example_input",
        "2020",
        "DC",
    ),
]
