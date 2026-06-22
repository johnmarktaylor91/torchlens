"""CBHG sequence block from Tacotron.

Paper: "Tacotron: Towards End-to-End Speech Synthesis", Wang et al.,
Interspeech 2017.

CBHG is a Tacotron text/speech representation block: a bank of 1-D convolutions
with multiple n-gram widths, max-pooling, projection convolutions, highway
layers, and a bidirectional GRU. This compact version preserves those stages.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HighwayLayer(nn.Module):
    """Highway transform used in CBHG."""

    def __init__(self, channels: int) -> None:
        """Initialize transform and carry gates.

        Parameters
        ----------
        channels:
            Feature width.
        """
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.gate = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a highway layer.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Highway output with the same shape as ``x``.
        """
        transform = torch.relu(self.proj(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * transform + (1.0 - gate) * x


class CBHG(nn.Module):
    """Compact convolution-bank highway bidirectional-GRU block."""

    def __init__(self, channels: int = 32, bank_channels: int = 8, max_width: int = 6) -> None:
        """Initialize convolution bank, projections, highways, and BiGRU.

        Parameters
        ----------
        channels:
            Input and projected feature width.
        bank_channels:
            Number of filters for each convolution-bank width.
        max_width:
            Largest convolution width in the bank.
        """
        super().__init__()
        self.max_width = max_width
        self.bank = nn.ModuleList(
            [nn.Conv1d(channels, bank_channels, kernel_size=k) for k in range(1, max_width + 1)]
        )
        self.proj1 = nn.Conv1d(max_width * bank_channels, channels, kernel_size=3, padding=1)
        self.proj2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.highways = nn.ModuleList([HighwayLayer(channels) for _ in range(2)])
        self.bigru = nn.GRU(channels, channels // 2, batch_first=True, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        """Extract CBHG sequence features.

        Parameters
        ----------
        x:
            Float tensor with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Contextual sequence tensor with shape ``(batch, time, channels)``.
        """
        seq = x.transpose(1, 2)
        convs = []
        for width, conv in enumerate(self.bank, start=1):
            left = (width - 1) // 2
            right = width - 1 - left
            convs.append(torch.relu(conv(F.pad(seq, (left, right)))))
        banked = torch.cat(convs, dim=1)
        pooled = F.max_pool1d(F.pad(banked, (1, 0)), kernel_size=2, stride=1)
        projected = torch.relu(self.norm1(self.proj1(pooled)))
        projected = self.norm2(self.proj2(projected)) + seq
        out = projected.transpose(1, 2)
        for highway in self.highways:
            out = highway(out)
        out, _ = self.bigru(out)
        return out


def build() -> nn.Module:
    """Build a compact CBHG block.

    Returns
    -------
    nn.Module
        Random-initialized CBHG model.
    """
    return CBHG()


def example_input() -> Tensor:
    """Return example acoustic/text sequence features.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 18, 32)``.
    """
    return torch.randn(1, 18, 32)


MENAGERIE_ENTRIES = [
    ("CBHG (convolution bank highway GRU)", "build", "example_input", "2017", "DE")
]
