"""PhaseNet compact faithful reconstruction.

Zhu and Beroza 2019, "PhaseNet: a deep-neural-network-based seismic arrival-time
picking method".

PhaseNet treats P/S arrival picking as one-dimensional semantic segmentation:
a U-Net maps three-component waveforms to per-sample probabilities for P, S, and
noise. This compact version keeps the encoder-decoder skip topology.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Two Conv1d-ReLU layers for the PhaseNet U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the convolution pair.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolution pair.

        Parameters
        ----------
        x:
            Input sequence.

        Returns
        -------
        Tensor
            Output sequence.
        """
        return self.net(x)


class PhaseNetCompact(nn.Module):
    """Compact one-dimensional U-Net for seismic phase probabilities."""

    def __init__(self, base: int = 16) -> None:
        """Initialize encoder, decoder, and probability head.

        Parameters
        ----------
        base:
            Base channel count.
        """
        super().__init__()
        self.down1 = DoubleConv(3, base)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = DoubleConv(base * 2, base * 4)
        self.up2 = nn.ConvTranspose1d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose1d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv1d(base, 3, kernel_size=1)

    def forward(self, wave: Tensor) -> Tensor:
        """Predict P, S, and noise probabilities for each sample.

        Parameters
        ----------
        wave:
            Three-component waveform tensor with shape ``(batch, 3, time)``.

        Returns
        -------
        Tensor
            Per-sample class probabilities.
        """
        skip1 = self.down1(wave)
        skip2 = self.down2(self.pool1(skip1))
        x = self.bottleneck(self.pool2(skip2))
        x = self.up2(x)
        x = self.dec2(torch.cat((x, skip2), dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat((x, skip1), dim=1))
        return torch.softmax(self.head(x), dim=1)


def build() -> nn.Module:
    """Build a compact random-init PhaseNet.

    Returns
    -------
    nn.Module
        Compact PhaseNet reconstruction.
    """
    return PhaseNetCompact()


def example_input() -> Tensor:
    """Return a short three-component waveform.

    Returns
    -------
    Tensor
        Waveform tensor.
    """
    return torch.randn(1, 3, 128)


MENAGERIE_ENTRIES = [("PhaseNet", "build", "example_input", "2019", "E7")]
