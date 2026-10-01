"""DGMR: Deep Generative Model of Radar.

Paper: "Skillful precipitation nowcasting using deep generative models of
radar", Ravuri et al., Nature 2021.

The compact reconstruction keeps DGMR's load-bearing structure: a context
conditioning stack, a latent conditioning stack, a recurrent ConvGRU generator,
and separate spatial/temporal discriminator heads for adversarial nowcasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell used by the DGMR generator."""

    def __init__(self, channels: int) -> None:
        """Initialize gate and candidate convolutions.

        Parameters
        ----------
        channels:
            Number of hidden and input channels.
        """

        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Advance one recurrent spatial state.

        Parameters
        ----------
        x:
            Input feature map.
        h:
            Previous hidden state.

        Returns
        -------
        torch.Tensor
            Updated hidden state.
        """

        reset, update = self.gates(torch.cat([x, h], dim=1)).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        proposal = torch.tanh(self.candidate(torch.cat([x, reset * h], dim=1)))
        return (1.0 - update) * h + update * proposal


class DGMRCompact(nn.Module):
    """Compact DGMR generator with spatial and temporal critics."""

    def __init__(self, channels: int = 16, horizon: int = 3) -> None:
        """Initialize DGMR components.

        Parameters
        ----------
        channels:
            Feature width.
        horizon:
            Number of future frames to generate.
        """

        super().__init__()
        self.horizon = horizon
        self.context = nn.Sequential(
            nn.Conv3d(1, channels, (2, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(channels, channels, (2, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
        )
        self.latent = nn.Sequential(
            nn.Linear(8, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.cell = ConvGRUCell(channels)
        self.frame_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.spatial_disc = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )
        self.temporal_disc = nn.Sequential(
            nn.Conv3d(1, channels, (3, 3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, 1),
        )

    def forward(self, radar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate future radar frames and critic scores.

        Parameters
        ----------
        radar:
            Radar history with shape ``(batch, time, 1, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Generated frames, spatial critic score, and temporal critic score.
        """

        batch = radar.shape[0]
        x = radar.transpose(1, 2)
        context = self.context(x).mean(dim=2)
        noise = torch.zeros(batch, 8, device=radar.device, dtype=radar.dtype)
        h = context + self.latent(noise).view(batch, -1, 1, 1)
        frames = []
        for _ in range(self.horizon):
            h = self.cell(context, h)
            frames.append(torch.sigmoid(self.frame_head(h)))
        generated = torch.stack(frames, dim=1)
        spatial_score = self.spatial_disc(generated[:, -1])
        temporal_score = self.temporal_disc(generated.transpose(1, 2))
        return generated, spatial_score, temporal_score


def build() -> nn.Module:
    """Build compact DGMR."""

    return DGMRCompact()


def example_input() -> torch.Tensor:
    """Return a short radar history."""

    return torch.randn(1, 4, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DGMR",
        "build",
        "example_input",
        "2021",
        "E7",
    )
]
