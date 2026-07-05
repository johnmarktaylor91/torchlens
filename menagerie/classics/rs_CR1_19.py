# FAITHFUL REIMPLEMENTATION from https://doi.org/10.3390/pr11020384 (no public code)
"""Toy 1D convolutional adversarial autoencoder for process monitoring."""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class ConvProcessAAE(nn.Module):
    """One-dimensional convolutional AAE for fault detection."""

    def __init__(self, channels: int = 3, latent_dim: int = 6) -> None:
        """Initialize the process-monitoring AAE.

        Parameters
        ----------
        channels:
            Number of process-variable channels.
        latent_dim:
            Latent dimension.
        """
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(channels, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 8, 3, padding=1),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(8 * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 8 * 16)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv1d(8, channels, 3, padding=1)
        )
        self.discriminator = nn.Sequential(nn.Linear(latent_dim, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, reconstruct, and adversarially score latent features.

        Parameters
        ----------
        x:
            Process time series of shape ``(batch, channels, 16)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Reconstruction, latent code, and discriminator logits.
        """
        encoded = self.encoder_conv(x)
        z = self.to_latent(encoded.flatten(1))
        decoded = self.from_latent(z).view(x.shape[0], 8, 16)
        reconstruction = self.decoder_conv(decoded)
        return reconstruction, z, self.discriminator(z)


def build_process_aae() -> ConvProcessAAE:
    """Build a tiny process-monitoring AAE.

    Returns
    -------
    ConvProcessAAE
        Model instance.
    """
    return ConvProcessAAE()


def example_input_process_aae() -> Tensor:
    """Create example process time series.

    Returns
    -------
    Tensor
        Time-series tensor.
    """
    return torch.randn(2, 3, 16)


MENAGERIE_ENTRIES = [
    (
        "Adversarial Autoencoder Process Monitor",
        build_process_aae,
        example_input_process_aae,
        2023,
        "REIMPL",
    )
]
