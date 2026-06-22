"""EQTransformer compact faithful reconstruction.

Mousavi et al. 2020, "Earthquake transformer: an attentive deep-learning model
for simultaneous earthquake detection and phase picking".

EQTransformer uses a deep convolutional/recurrent encoder with attention and
three task-specific decoder branches for detection, P picking, and S picking.
This compact random-init version preserves that multi-task attentive hierarchy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvBlock(nn.Module):
    """Convolution, normalization, and activation block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            Temporal convolution stride.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolution block.

        Parameters
        ----------
        x:
            Input waveform features.

        Returns
        -------
        Tensor
            Output features.
        """
        return self.net(x)


class AttentiveDecoder(nn.Module):
    """Task-specific EQTransformer decoder with temporal attention."""

    def __init__(self, channels: int, hidden: int) -> None:
        """Initialize decoder layers.

        Parameters
        ----------
        channels:
            Encoder channel count.
        hidden:
            Recurrent hidden size.
        """
        super().__init__()
        self.query = nn.Linear(hidden * 2, hidden)
        self.key = nn.Linear(hidden * 2, hidden)
        self.energy = nn.Linear(hidden, 1)
        self.up = nn.ConvTranspose1d(hidden * 4, channels, kernel_size=4, stride=4)
        self.out = nn.Conv1d(channels, 1, kernel_size=5, padding=2)

    def forward(self, encoded: Tensor) -> Tensor:
        """Decode one probability sequence from encoded features.

        Parameters
        ----------
        encoded:
            Encoded features with shape ``(batch, time, hidden * 2)``.

        Returns
        -------
        Tensor
            Probability sequence with shape ``(batch, time * 4)``.
        """
        pooled = encoded.mean(dim=1, keepdim=True)
        scores = self.energy(torch.tanh(self.key(encoded) + self.query(pooled))).transpose(1, 2)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoded).expand(-1, encoded.shape[1], -1)
        fused = torch.cat((encoded, context), dim=-1).transpose(1, 2)
        return torch.sigmoid(self.out(self.up(fused))).squeeze(1)


class EQTransformerCompact(nn.Module):
    """Compact multi-task EQTransformer."""

    def __init__(self, channels: int = 32, hidden: int = 32) -> None:
        """Initialize encoder and three decoder heads.

        Parameters
        ----------
        channels:
            Convolutional channel count.
        hidden:
            Bidirectional recurrent hidden size.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, channels),
            ConvBlock(channels, channels, stride=2),
            ConvBlock(channels, channels, stride=2),
        )
        self.lstm = nn.LSTM(channels, hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.detector = AttentiveDecoder(channels, hidden)
        self.p_picker = AttentiveDecoder(channels, hidden)
        self.s_picker = AttentiveDecoder(channels, hidden)

    def forward(self, wave: Tensor) -> Tensor:
        """Run simultaneous detection and phase picking.

        Parameters
        ----------
        wave:
            Three-component waveform tensor with shape ``(batch, 3, time)``.

        Returns
        -------
        Tensor
            Stacked detection, P, and S probabilities.
        """
        features = self.encoder(wave).transpose(1, 2)
        encoded, _ = self.lstm(features)
        return torch.stack(
            (self.detector(encoded), self.p_picker(encoded), self.s_picker(encoded)), dim=1
        )


def build() -> nn.Module:
    """Build a compact random-init EQTransformer.

    Returns
    -------
    nn.Module
        Compact EQTransformer reconstruction.
    """
    return EQTransformerCompact()


def example_input() -> Tensor:
    """Return a short three-component waveform.

    Returns
    -------
    Tensor
        Waveform tensor.
    """
    return torch.randn(1, 3, 128)


MENAGERIE_ENTRIES = [("EQTransformer", "build", "example_input", "2020", "E7")]
