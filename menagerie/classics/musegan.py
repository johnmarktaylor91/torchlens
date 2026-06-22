"""MuseGAN: multi-track sequential GAN for symbolic music.

Paper: Dong et al. 2017/2018, "MuseGAN: Multi-Track Sequential Generative
Adversarial Networks for Symbolic Music Generation and Accompaniment".

The compact module implements the hybrid generator idea: a shared temporal
composer branch creates inter-track phrase structure while per-track jamming
branches add instrument-specific dynamics before piano-roll synthesis.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MuseGANGenerator(nn.Module):
    """Compact hybrid MuseGAN generator for five piano-roll tracks."""

    def __init__(self, latent: int = 24, tracks: int = 5, bars: int = 4) -> None:
        """Initialize shared composer and per-track jamming branches.

        Parameters
        ----------
        latent:
            Latent vector width.
        tracks:
            Number of instrument tracks.
        bars:
            Number of generated bars.
        """

        super().__init__()
        self.tracks = tracks
        self.bars = bars
        self.composer = nn.GRU(latent, 32, batch_first=True)
        self.track_style = nn.ModuleList([nn.Linear(latent, 32) for _ in range(tracks)])
        self.to_roll = nn.ModuleList([nn.Linear(64, 16 * 12) for _ in range(tracks)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate multi-track piano-roll bars.

        Parameters
        ----------
        z:
            Latent phrase tensor with shape ``(batch, bars, latent)``.

        Returns
        -------
        torch.Tensor
            Piano-roll probabilities with shape ``(batch, tracks, bars, 16, 12)``.
        """

        phrase, _ = self.composer(z)
        rolls = []
        for style, head in zip(self.track_style, self.to_roll, strict=True):
            local = torch.tanh(style(z))
            roll = torch.sigmoid(head(torch.cat([phrase, local], dim=-1)))
            rolls.append(roll.view(z.shape[0], self.bars, 16, 12))
        return torch.stack(rolls, dim=1)


def build() -> nn.Module:
    """Build the compact MuseGAN generator.

    Returns
    -------
    nn.Module
        Random-initialized MuseGAN generator.
    """

    return MuseGANGenerator()


def example_input() -> torch.Tensor:
    """Create latent phrase noise.

    Returns
    -------
    torch.Tensor
        Latent tensor with shape ``(1, 4, 24)``.
    """

    return torch.randn(1, 4, 24)


MENAGERIE_ENTRIES = [
    ("MuseGAN", "build", "example_input", "2017", "E5"),
]
