"""Parallel WaveGAN generator.

Yamamoto et al. (ICASSP 2020), "Parallel WaveGAN: A fast waveform generation
model based on generative adversarial networks with multi-resolution
spectrogram."  The generator is a non-autoregressive WaveNet-like stack: random
noise at waveform rate is processed by dilated residual convolution blocks while
upsampled mel-spectrogram features condition every block.  This compact version
keeps the noise input, mel upsampling, gated residual blocks, and accumulated
skip projection with fewer layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PWGResidualBlock(nn.Module):
    """Gated residual block from a non-autoregressive WaveNet generator."""

    def __init__(self, channels: int, aux_channels: int, dilation: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Residual channel count.
        aux_channels:
            Conditioning channel count.
        dilation:
            Temporal convolution dilation.
        """

        super().__init__()
        self.conv = nn.Conv1d(channels, channels * 2, 3, padding=dilation, dilation=dilation)
        self.aux = nn.Conv1d(aux_channels, channels * 2, 1)
        self.res_skip = nn.Conv1d(channels, channels * 2, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gated residual conditioning.

        Parameters
        ----------
        x:
            Residual waveform features.
        cond:
            Upsampled conditioning features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated residual features and skip contribution.
        """

        gated = self.conv(x) + self.aux(cond)
        gate, value = gated.chunk(2, dim=1)
        h = torch.tanh(value) * torch.sigmoid(gate)
        residual, skip = self.res_skip(h).chunk(2, dim=1)
        return (x + residual) * (2.0**-0.5), skip


class ParallelWaveGANGenerator(nn.Module):
    """Compact Parallel WaveGAN mel-conditioned noise generator."""

    def __init__(self, mel_channels: int = 80, channels: int = 32) -> None:
        """Initialize the compact generator.

        Parameters
        ----------
        mel_channels:
            Number of mel bins.
        channels:
            Residual and skip channel width.
        """

        super().__init__()
        self.noise_proj = nn.Conv1d(1, channels, 1)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(mel_channels, channels, 8, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.ConvTranspose1d(channels, channels, 8, stride=4, padding=2),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.blocks = nn.ModuleList(
            [PWGResidualBlock(channels, channels, dilation) for dilation in (1, 2, 4, 8, 1, 2)]
        )
        self.post = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=False),
            nn.Conv1d(channels, 1, 1),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate a waveform from mel conditioning and deterministic noise.

        Parameters
        ----------
        mel:
            Mel-spectrogram tensor of shape ``(B, 80, T)``.

        Returns
        -------
        torch.Tensor
            Waveform tensor.
        """

        cond = self.upsample(mel)
        time = cond.shape[-1]
        noise = torch.linspace(-1.0, 1.0, time, device=mel.device, dtype=mel.dtype).view(1, 1, time)
        x = self.noise_proj(noise.expand(mel.shape[0], -1, -1))
        skip_total = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x, cond)
            skip_total = skip_total + skip
        return self.post(skip_total * (len(self.blocks) ** -0.5))


def build() -> nn.Module:
    """Build the compact Parallel WaveGAN generator.

    Returns
    -------
    nn.Module
        Random-init generator in evaluation mode.
    """

    return ParallelWaveGANGenerator().eval()


def example_input() -> torch.Tensor:
    """Return a compact mel-spectrogram.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 80, 10)``.
    """

    return torch.randn(1, 80, 10)


MENAGERIE_ENTRIES = [
    ("ParallelWaveGANGenerator", "build", "example_input", "2020", "DC"),
]
