"""FishSpeech DAC compact reconstruction.

Fish Speech uses a Descript Audio Codec (DAC)-style neural audio tokenizer with a
convolutional encoder, residual vector quantization codebooks, and convolutional decoder.
This compact random-init implementation preserves that codec topology.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Compact residual vector quantizer with learned codebooks."""

    def __init__(self, codebooks: int = 3, codes: int = 16, dim: int = 32) -> None:
        """Initialize RVQ codebooks.

        Parameters
        ----------
        codebooks:
            Number of residual codebooks.
        codes:
            Entries per codebook.
        dim:
            Latent dimension.
        """

        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(codebooks, codes, dim) * 0.02)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize latents with residual codebooks.

        Parameters
        ----------
        z:
            Latent tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Quantized latents and integer code indices.
        """

        residual = z
        quantized = torch.zeros_like(z)
        indices = []
        for book in self.codebooks:
            dist = torch.cdist(residual.reshape(-1, residual.shape[-1]), book)
            idx = dist.argmin(dim=-1)
            code = book[idx].view_as(residual)
            quantized = quantized + code
            residual = residual - code
            indices.append(idx.view(z.shape[0], z.shape[1]))
        return quantized, torch.stack(indices, dim=-1)


class CompactFishSpeechDAC(nn.Module):
    """Compact DAC-style audio autoencoder."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact DAC.

        Parameters
        ----------
        channels:
            Latent channel count.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ELU(),
            nn.Conv1d(16, channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
        )
        self.quantizer = ResidualVectorQuantizer(dim=channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose1d(channels, 16, 4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv1d(16, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, quantize, and reconstruct audio.

        Parameters
        ----------
        audio:
            Waveform tensor of shape ``(batch, 1, samples)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed waveform and RVQ indices.
        """

        latent = self.encoder(audio).transpose(1, 2)
        quantized, indices = self.quantizer(latent)
        recon = self.decoder(quantized.transpose(1, 2))
        return recon[..., : audio.shape[-1]], indices


def build_FishSpeech_DAC() -> nn.Module:
    """Build compact FishSpeech DAC.

    Returns
    -------
    nn.Module
        Random-init compact DAC.
    """

    return CompactFishSpeechDAC()


def example_input() -> torch.Tensor:
    """Create compact audio input.

    Returns
    -------
    torch.Tensor
        Waveform tensor of shape ``(1, 1, 64)``.
    """

    return torch.randn(1, 1, 64).clamp(-1.0, 1.0)


build = build_FishSpeech_DAC

MENAGERIE_ENTRIES = [("FishSpeech_DAC", "build_FishSpeech_DAC", "example_input", "2023", "E5")]
