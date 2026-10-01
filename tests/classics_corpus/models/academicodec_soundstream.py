"""AcademiCodec SoundStream: convolutional encoder, RVQ, decoder audio codec.

Source: SoundStream (Zeghidour et al., 2021) and the AcademiCodec toolkit. The
architecture is an end-to-end neural audio codec with a fully convolutional
encoder/decoder and residual vector quantizer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualUnit(nn.Module):
    """SoundStream-style residual dilated convolution unit."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a residual unit.

        Parameters
        ----------
        channels:
            Number of feature channels.
        dilation:
            Dilation factor for the temporal convolution.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual temporal filtering.

        Parameters
        ----------
        x:
            Audio feature tensor ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Filtered tensor with the same shape.
        """
        return x + self.net(x)


class ResidualVectorQuantizer(nn.Module):
    """Small residual vector quantizer with straight-through assignments."""

    def __init__(self, dim: int = 32, codebooks: int = 3, entries: int = 16) -> None:
        """Initialize additive codebooks.

        Parameters
        ----------
        dim:
            Latent channel dimension.
        codebooks:
            Number of residual quantization stages.
        entries:
            Number of entries per codebook.
        """
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(codebooks, entries, dim) * 0.02)

    def forward(self, z: Tensor) -> Tensor:
        """Quantize latents by sequential residual nearest-neighbor lookup.

        Parameters
        ----------
        z:
            Latent tensor with shape ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Straight-through quantized latent tensor.
        """
        flat = z.transpose(1, 2)
        residual = flat
        quantized = torch.zeros_like(flat)
        for codebook in self.codebooks:
            distances = (residual.unsqueeze(-2) - codebook).pow(2).sum(dim=-1)
            indices = distances.argmin(dim=-1)
            chosen = F.embedding(indices, codebook)
            quantized = quantized + chosen
            residual = residual - chosen
        quantized = flat + (quantized - flat).detach()
        return quantized.transpose(1, 2)


class GroupResidualVectorQuantizer(nn.Module):
    """HiFi-Codec group-residual vector quantizer."""

    def __init__(self, dim: int = 32, groups: int = 4, stages: int = 2, entries: int = 16) -> None:
        """Initialize group-specific residual codebooks.

        Parameters
        ----------
        dim:
            Latent channel dimension.
        groups:
            Number of channel groups.
        stages:
            Residual quantizers per group.
        entries:
            Entries per codebook.
        """
        super().__init__()
        self.groups = groups
        self.group_dim = dim // groups
        self.codebooks = nn.Parameter(torch.randn(groups, stages, entries, self.group_dim) * 0.02)

    def forward(self, z: Tensor) -> Tensor:
        """Quantize each latent group through residual quantizer stages.

        Parameters
        ----------
        z:
            Latent tensor with shape ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Straight-through group-residual quantized tensor.
        """
        grouped = z.transpose(1, 2).reshape(z.shape[0], z.shape[-1], self.groups, self.group_dim)
        quantized_groups: list[Tensor] = []
        for group_idx in range(self.groups):
            residual = grouped[:, :, group_idx]
            quantized = torch.zeros_like(residual)
            for codebook in self.codebooks[group_idx]:
                distances = (residual.unsqueeze(-2) - codebook).pow(2).sum(dim=-1)
                indices = distances.argmin(dim=-1)
                chosen = F.embedding(indices, codebook)
                quantized = quantized + chosen
                residual = residual - chosen
            quantized_groups.append(quantized)
        quantized = torch.stack(quantized_groups, dim=2).reshape(
            z.shape[0], z.shape[-1], z.shape[1]
        )
        quantized = z.transpose(1, 2) + (quantized - z.transpose(1, 2)).detach()
        return quantized.transpose(1, 2)


class SoundStreamCodec(nn.Module):
    """Compact SoundStream/AcademiCodec reconstruction."""

    def __init__(self, channels: int = 24, latent: int = 32) -> None:
        """Initialize convolutional encoder, RVQ, and decoder.

        Parameters
        ----------
        channels:
            Base channel width.
        latent:
            Latent channel width.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            ResidualUnit(channels, 1),
            ResidualUnit(channels, 3),
            nn.ELU(),
            nn.Conv1d(channels, latent, 4, stride=2, padding=1),
            ResidualUnit(latent, 1),
        )
        self.quantizer = ResidualVectorQuantizer(latent)
        self.decoder = nn.Sequential(
            ResidualUnit(latent, 1),
            nn.ELU(),
            nn.ConvTranspose1d(latent, channels, 4, stride=2, padding=1),
            ResidualUnit(channels, 1),
            ResidualUnit(channels, 3),
            nn.ELU(),
            nn.Conv1d(channels, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, wav: Tensor) -> Tensor:
        """Encode, quantize, and reconstruct waveform audio.

        Parameters
        ----------
        wav:
            Waveform tensor with shape ``(batch, 1, time)``.

        Returns
        -------
        Tensor
            Reconstructed waveform with the same shape.
        """
        return self.decoder(self.quantizer(self.encoder(wav)))


class HiFiCodec(nn.Module):
    """Compact HiFi-Codec with group-residual vector quantization."""

    def __init__(self, channels: int = 24, latent: int = 32) -> None:
        """Initialize encoder, GRVQ bottleneck, and decoder.

        Parameters
        ----------
        channels:
            Base channel width.
        latent:
            Latent channel width.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            ResidualUnit(channels, 1),
            nn.ELU(),
            nn.Conv1d(channels, latent, 4, stride=2, padding=1),
            ResidualUnit(latent, 3),
        )
        self.quantizer = GroupResidualVectorQuantizer(latent)
        self.decoder = nn.Sequential(
            ResidualUnit(latent, 1),
            nn.ELU(),
            nn.ConvTranspose1d(latent, channels, 4, stride=2, padding=1),
            ResidualUnit(channels, 3),
            nn.ELU(),
            nn.Conv1d(channels, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, wav: Tensor) -> Tensor:
        """Encode, group-residual quantize, and reconstruct audio.

        Parameters
        ----------
        wav:
            Waveform tensor with shape ``(batch, 1, time)``.

        Returns
        -------
        Tensor
            Reconstructed waveform.
        """
        return self.decoder(self.quantizer(self.encoder(wav)))


def build() -> nn.Module:
    """Build a compact AcademiCodec SoundStream model.

    Returns
    -------
    nn.Module
        Random-initialized audio codec.
    """
    return SoundStreamCodec()


def build_hificodec() -> nn.Module:
    """Build a compact AcademiCodec HiFi-Codec model.

    Returns
    -------
    nn.Module
        Random-initialized HiFi-Codec.
    """
    return HiFiCodec()


def example_input() -> Tensor:
    """Return a short mono waveform.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 1, 256)``.
    """
    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    ("academicodec_soundstream", "build", "example_input", "2021", "DE"),
    ("academicodec_hificodec", "build_hificodec", "example_input", "2023", "DE"),
]
