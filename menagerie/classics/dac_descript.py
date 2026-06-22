"""Descript Audio Codec compact random-init reconstruction.

Paper: High-Fidelity Audio Compression with Improved RVQGAN
(Kumar et al., 2023).

DAC uses a strided convolutional encoder, residual vector quantizer codebooks,
Snake nonlinearities, and a transposed-convolution decoder.  This compact model
keeps that encoder-RVQ-decoder path with straight-through residual quantization.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Snake(nn.Module):
    """Snake activation used by neural audio codecs."""

    def __init__(self, channels: int) -> None:
        """Initialize per-channel periodic slope."""

        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply periodic Snake nonlinearity."""

        return x + torch.sin(self.alpha * x).pow(2) / self.alpha.clamp_min(1e-4)


class ResidualVectorQuantizer(nn.Module):
    """Straight-through residual vector quantizer with multiple codebooks."""

    def __init__(self, dim: int = 32, codebooks: int = 3, codes: int = 8) -> None:
        """Initialize residual codebooks."""

        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(codebooks, codes, dim) * 0.05)

    def forward(self, z: Tensor) -> Tensor:
        """Quantize latents by recursively encoding residuals."""

        residual = z.transpose(1, 2)
        quantized = torch.zeros_like(residual)
        for book in self.codebooks:
            dist = (residual[:, :, None, :] - book[None, None, :, :]).pow(2).sum(dim=-1)
            codes = torch.softmax(-dist, dim=-1)
            chosen = torch.matmul(codes, book)
            quantized = quantized + chosen
            residual = residual - chosen
        return quantized.transpose(1, 2)


class DescriptDAC(nn.Module):
    """Compact DAC encoder-RVQ-decoder audio codec."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize encoder, RVQ, decoder, and multi-scale discriminator head."""

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            Snake(channels),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            Snake(channels),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
        )
        self.rvq = ResidualVectorQuantizer(channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            Snake(channels),
            nn.ConvTranspose1d(channels, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.discriminator = nn.Conv1d(1, 4, 15, padding=7)

    def forward(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        """Encode, residual-quantize, decode, and score audio."""

        z = self.encoder(audio)
        q = self.rvq(z)
        recon = self.decoder(q)
        return recon, self.discriminator(recon)


def build() -> nn.Module:
    """Build a compact random-init Descript DAC model."""

    return DescriptDAC().eval()


def example_input() -> Tensor:
    """Return a short mono waveform."""

    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    ("DAC_descript", "build", "example_input", "2023", "DC"),
]
