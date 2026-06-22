"""Pangolin splice-site strength predictor.

Zeng and Li (Genome Biology 2022) introduced Pangolin for variant-aware RNA
splice prediction across tissues and species.  Public descriptions identify it
as a deep model for splice-site strength from DNA sequence, closely related to
SpliceAI-style dilated sequence convolutions and tissue-specific outputs.  This
compact reconstruction keeps the faithful core: one-hot DNA input, residual
dilated 1-D sequence modeling over long context, a lightweight transformer for
context mixing, and per-tissue acceptor/donor splice-strength heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedConv(nn.Module):
    """Residual dilated convolution block for genomic sequence context."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        channels:
            Number of sequence channels.
        dilation:
            Temporal dilation for the convolution.
        """

        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.conv = nn.Conv1d(channels, channels, 11, padding=5 * dilation, dilation=dilation)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual dilated convolution.

        Parameters
        ----------
        x:
            Sequence feature map.

        Returns
        -------
        torch.Tensor
            Updated sequence feature map.
        """

        h = F.gelu(self.conv(F.gelu(self.norm(x))))
        return x + self.proj(h)


class CompactPangolin(nn.Module):
    """Compact Pangolin-like splice predictor."""

    def __init__(self, tissues: int = 4, channels: int = 32) -> None:
        """Initialize the predictor.

        Parameters
        ----------
        tissues:
            Number of tissue-specific output heads.
        channels:
            Hidden sequence width.
        """

        super().__init__()
        self.stem = nn.Conv1d(4, channels, 1)
        self.dilated = nn.Sequential(
            ResidualDilatedConv(channels, 1),
            ResidualDilatedConv(channels, 2),
            ResidualDilatedConv(channels, 4),
            ResidualDilatedConv(channels, 8),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 2,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Conv1d(channels, tissues * 2, 1)

    def forward(self, dna: torch.Tensor) -> torch.Tensor:
        """Predict acceptor/donor splice strength for each tissue.

        Parameters
        ----------
        dna:
            One-hot DNA tensor of shape ``(B, 4, L)``.

        Returns
        -------
        torch.Tensor
            Splice probabilities of shape ``(B, tissues, 2, L)``.
        """

        x = self.dilated(self.stem(dna))
        x = self.context(x.transpose(1, 2)).transpose(1, 2)
        logits = self.head(x)
        return torch.sigmoid(logits.view(dna.shape[0], -1, 2, dna.shape[-1]))


def build() -> nn.Module:
    """Build the compact Pangolin model.

    Returns
    -------
    nn.Module
        Random-init splice predictor in evaluation mode.
    """

    return CompactPangolin().eval()


def example_input() -> torch.Tensor:
    """Return a compact one-hot DNA sequence.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 4, 96)``.
    """

    ids = torch.randint(0, 4, (1, 96))
    return F.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [
    ("Pangolin", "build", "example_input", "2022", "DC"),
]
