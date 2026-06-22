"""Asteroid ConvTasNet: time-domain encoder-masker-decoder separator.

Asteroid documents ConvTasNet as the Luo & Mesgarani 2019 architecture:
learned time-domain filterbank encoder, temporal convolutional mask network
with residual/skip connections, and waveform decoder from masked
representations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalBlock(nn.Module):
    """Depthwise-separable temporal convolution block."""

    def __init__(self, channels: int, hidden: int, dilation: int) -> None:
        """Initialize bottleneck, depthwise, residual, and skip paths.

        Parameters
        ----------
        channels:
            Bottleneck channel count.
        hidden:
            Hidden channel count.
        dilation:
            Temporal dilation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 1),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden),
            nn.PReLU(),
            nn.GroupNorm(1, hidden),
        )
        self.residual = nn.Conv1d(hidden, channels, 1)
        self.skip = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one TCN block.

        Parameters
        ----------
        x:
            Bottleneck features.

        Returns
        -------
        tuple[Tensor, Tensor]
            Residual-updated features and skip contribution.
        """
        h = self.net(x)
        return x + self.residual(h), self.skip(h)


class ConvTasNetMini(nn.Module):
    """Compact Asteroid ConvTasNet source separator."""

    def __init__(self, sources: int = 2, filters: int = 32, bottleneck: int = 24) -> None:
        """Initialize learned filterbank, TCN masker, and decoder.

        Parameters
        ----------
        sources:
            Number of separated sources.
        filters:
            Encoder filter count.
        bottleneck:
            TCN bottleneck width.
        """
        super().__init__()
        self.sources = sources
        self.encoder = nn.Conv1d(1, filters, 16, stride=8, bias=False)
        self.bottleneck = nn.Sequential(nn.GroupNorm(1, filters), nn.Conv1d(filters, bottleneck, 1))
        self.blocks = nn.ModuleList(
            [TemporalBlock(bottleneck, 64, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.mask = nn.Conv1d(bottleneck, sources * filters, 1)
        self.decoder = nn.ConvTranspose1d(filters, 1, 16, stride=8, bias=False)

    def forward(self, wav: Tensor) -> Tensor:
        """Separate a mono waveform into source estimates.

        Parameters
        ----------
        wav:
            Waveform tensor with shape ``(batch, 1, time)``.

        Returns
        -------
        Tensor
            Source estimates with shape ``(batch, sources, time)``.
        """
        encoded = torch.relu(self.encoder(wav))
        features = self.bottleneck(encoded)
        skip_sum = torch.zeros_like(features)
        for block in self.blocks:
            features, skip = block(features)
            skip_sum = skip_sum + skip
        masks = torch.softmax(
            self.mask(skip_sum).view(wav.shape[0], self.sources, -1, encoded.shape[-1]), dim=1
        )
        masked = masks * encoded.unsqueeze(1)
        decoded = []
        for source in range(self.sources):
            decoded.append(self.decoder(masked[:, source]).squeeze(1))
        return torch.stack(decoded, dim=1)


def build() -> nn.Module:
    """Build a compact Asteroid ConvTasNet model.

    Returns
    -------
    nn.Module
        Random-initialized source separator.
    """
    return ConvTasNetMini()


def example_input() -> Tensor:
    """Return a short mono mixture waveform.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 1, 256)``.
    """
    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    ("asteroid_ConvTasNet", "build", "example_input", "2019", "DE"),
]
