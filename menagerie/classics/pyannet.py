"""PyanNet speaker diarization network from pyannote.audio.

Paper: pyannote.audio: neural building blocks for speaker diarization, Bredin et al. 2019.

The original PyanNet is an end-to-end sequence-labeling model over raw audio:
SincNet-like learnable band-pass filters, temporal convolutional context, recurrent
sequence modeling, and frame-level speaker/activity logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    """Learnable SincNet-style band-pass convolution front-end."""

    def __init__(
        self, out_channels: int = 8, kernel_size: int = 31, sample_rate: int = 16000
    ) -> None:
        """Initialize the analytic band-pass filter bank.

        Parameters
        ----------
        out_channels:
            Number of learnable filters.
        kernel_size:
            Odd filter length.
        sample_rate:
            Audio sample rate used to normalize cutoffs.
        """

        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        low = torch.linspace(80.0, 3000.0, out_channels)
        band = torch.full((out_channels,), 500.0)
        self.low_hz = nn.Parameter(low)
        self.band_hz = nn.Parameter(band)
        n = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        window = 0.54 - 0.46 * torch.cos(
            2 * math.pi * torch.arange(kernel_size) / (kernel_size - 1)
        )
        self.register_buffer("n", n)
        self.register_buffer("window", window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter a raw waveform tensor.

        Parameters
        ----------
        x:
            Raw waveform of shape ``(batch, 1, samples)``.

        Returns
        -------
        torch.Tensor
            Filter-bank activations.
        """

        low = self.low_hz.abs() + 30.0
        high = (low + self.band_hz.abs() + 50.0).clamp(max=float(self.sample_rate // 2 - 1))
        n = self.n.unsqueeze(0)
        low_norm = low.unsqueeze(1) / self.sample_rate
        high_norm = high.unsqueeze(1) / self.sample_rate
        band = 2 * high_norm * torch.sinc(2 * high_norm * n) - 2 * low_norm * torch.sinc(
            2 * low_norm * n
        )
        band = band * self.window.unsqueeze(0)
        band = band / band.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        return F.conv1d(x, band.unsqueeze(1), stride=4, padding=self.kernel_size // 2)


class PyanNet(nn.Module):
    """Compact PyanNet with SincNet front-end, TDNN context, LSTM, and frame logits."""

    def __init__(self, num_speakers: int = 3) -> None:
        """Initialize the compact diarization network.

        Parameters
        ----------
        num_speakers:
            Number of frame-level output speaker/activity logits.
        """

        super().__init__()
        self.sinc = SincConv1d()
        self.tdnn = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(16, 24, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(48, 24), nn.Tanh(), nn.Linear(24, num_speakers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frame-level diarization.

        Parameters
        ----------
        x:
            Raw waveform of shape ``(batch, 1, samples)``.

        Returns
        -------
        torch.Tensor
            Per-frame speaker/activity logits.
        """

        feats = torch.log1p(self.sinc(x).pow(2))
        feats = self.tdnn(feats).transpose(1, 2)
        seq, _ = self.rnn(feats)
        return self.classifier(seq)


def build() -> nn.Module:
    """Build a compact random-init PyanNet."""

    return PyanNet()


def example_input() -> torch.Tensor:
    """Return a short raw-audio window."""

    return torch.randn(1, 1, 512)


MENAGERIE_ENTRIES = [
    (
        "PyanNet",
        "build",
        "example_input",
        "2019",
        "audio/diarization",
    ),
]
