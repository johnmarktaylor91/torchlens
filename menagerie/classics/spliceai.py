"""SpliceAI, 2019, Jaganathan et al., "Predicting Splicing from Primary Sequence".

Deep residual dilated convolutions over one-hot pre-mRNA produce per-nucleotide
splice acceptor, donor, or neither probabilities. This compact version keeps
the residual dilation mechanism with fewer channels and blocks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpliceAIBlock(nn.Module):
    """Residual dilated-convolution SpliceAI block."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a residual dilated block.

        Parameters
        ----------
        channels:
            Number of convolution channels.
        dilation:
            Dilation factor.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual dilated convolutions.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, channels, length)``.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x``.
        """
        return x + self.net(x)


class SpliceAI(nn.Module):
    """Compact per-nucleotide splice-site predictor."""

    def __init__(self, channels: int = 48, n_classes: int = 3) -> None:
        """Initialize input projection, residual stack, and classifier.

        Parameters
        ----------
        channels:
            Residual channel count.
        n_classes:
            Number of splice classes per nucleotide.
        """
        super().__init__()
        self.input_proj = nn.Conv1d(4, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [SpliceAIBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16)]
        )
        self.classifier = nn.Conv1d(channels, n_classes, kernel_size=1)

    def forward(self, onehot: Tensor) -> Tensor:
        """Predict per-nucleotide splice class probabilities.

        Parameters
        ----------
        onehot:
            One-hot pre-mRNA sequence with shape ``(batch, 4, length)``.

        Returns
        -------
        Tensor
            Class probabilities with shape ``(batch, length, n_classes)``.
        """
        hidden = self.input_proj(onehot)
        for block in self.blocks:
            hidden = block(hidden)
        return torch.softmax(self.classifier(hidden).transpose(1, 2), dim=-1)


def build() -> nn.Module:
    """Build a compact SpliceAI model.

    Returns
    -------
    nn.Module
        Random-initialized SpliceAI.
    """
    return SpliceAI()


def example_input() -> Tensor:
    """Return an example one-hot pre-mRNA sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 4, 400)``.
    """
    ids = torch.randint(0, 4, (1, 400), dtype=torch.long)
    return torch.nn.functional.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [
    ("SpliceAI Dilated Residual Splice Predictor", "build", "example_input", "2019", "DE")
]
