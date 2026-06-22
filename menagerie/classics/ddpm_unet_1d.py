"""Compact one-dimensional DDPM U-Net denoiser.

Audio and sequence diffusion models adapt DDPM's timestep-conditioned residual
U-Net to 1D signals, using Conv1d down/up paths, timestep embeddings, and
sequence self-attention at the bottleneck.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Parameters
    ----------
    t:
        Timestep tensor.
    dim:
        Embedding width.

    Returns
    -------
    torch.Tensor
        Sinusoidal embeddings.
    """

    half = dim // 2
    freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
    emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeResBlock1D(nn.Module):
    """1D residual block with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        """Initialize the block.

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
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Apply the timestep-conditioned update.

        Parameters
        ----------
        x:
            Sequence feature map.
        emb:
            Timestep embedding.

        Returns
        -------
        torch.Tensor
            Updated sequence features.
        """

        y = F.silu(self.conv1(x))
        y = y + self.time(emb).unsqueeze(-1)
        y = self.conv2(F.silu(y))
        return y + self.skip(x)


class DDPMUNet1DCompact(nn.Module):
    """Small 1D diffusion U-Net with bottleneck sequence attention."""

    def __init__(self, channels: int = 2, width: int = 24, time_dim: int = 64) -> None:
        """Initialize the 1D DDPM network.

        Parameters
        ----------
        channels:
            Signal channel count.
        width:
            Base width.
        time_dim:
            Timestep embedding width.
        """

        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.in_conv = nn.Conv1d(channels, width, 3, padding=1)
        self.down = TimeResBlock1D(width, width * 2, time_dim)
        self.attn = nn.MultiheadAttention(width * 2, 4, batch_first=True)
        self.up = TimeResBlock1D(width * 3, width, time_dim)
        self.out = nn.Conv1d(width, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict sequence noise.

        Parameters
        ----------
        x:
            Noisy sequence ``(B, C, L)``.
        t:
            Diffusion timestep.

        Returns
        -------
        torch.Tensor
            Predicted noise.
        """

        emb = self.time_mlp(timestep_embedding(t, 64))
        h1 = self.in_conv(x)
        h2 = self.down(F.avg_pool1d(h1, 2), emb)
        tok = h2.transpose(1, 2)
        attn, _ = self.attn(tok, tok, tok, need_weights=False)
        h2 = h2 + attn.transpose(1, 2)
        up = F.interpolate(h2, size=h1.shape[-1], mode="nearest")
        return self.out(self.up(torch.cat([up, h1], dim=1), emb))


def build() -> nn.Module:
    """Build compact 1D DDPM U-Net.

    Returns
    -------
    nn.Module
        One-dimensional diffusion denoiser.
    """

    return DDPMUNet1DCompact()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy sequence and timestep inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Sequence and timestep.
    """

    return torch.randn(1, 2, 64), torch.tensor([10])


MENAGERIE_ENTRIES = [
    (
        "ddpm_unet_1d (timestep-conditioned 1D diffusion U-Net)",
        "build",
        "example_input",
        "2020",
        "DC",
    ),
]
