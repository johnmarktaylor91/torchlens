"""AlphaGenome: compact long-context DNA regulatory prediction model.

DeepMind AlphaGenome (2025) is described as a unified DNA sequence model that
takes long one-hot DNA context, uses convolutional motif extraction,
transformers for long-range communication, and task heads for base-resolution
regulatory tracks/splicing/contact-style outputs. This random-init classic keeps
that CNN + transformer + multi-head prediction structure at small sequence size.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AlphaGenomeMini(nn.Module):
    """Compact AlphaGenome-style DNA track predictor."""

    def __init__(self, dim: int = 48, tracks: int = 6) -> None:
        """Initialize motif stem, transformer trunk, and modality heads.

        Parameters
        ----------
        dim:
            Hidden channel dimension.
        tracks:
            Number of dense regulatory tracks.
        """
        super().__init__()
        self.motif = nn.Sequential(
            nn.Conv1d(4, dim, 11, padding=5),
            nn.GELU(),
            nn.Conv1d(dim, dim, 5, padding=4, dilation=2),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=4 * dim, batch_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=2)
        self.track_head = nn.Conv1d(dim, tracks, 1)
        self.splice_head = nn.Conv1d(dim, 3, 1)
        self.contact_left = nn.Linear(dim, dim)
        self.contact_right = nn.Linear(dim, dim)

    def forward(self, dna: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict regulatory tracks, splice logits, and contact scores.

        Parameters
        ----------
        dna:
            One-hot DNA tensor with shape ``(batch, 4, length)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Track logits, splice logits, and compact contact map.
        """
        features = self.motif(dna)
        tokens = self.trunk(features.transpose(1, 2))
        channels = tokens.transpose(1, 2)
        tracks = self.track_head(channels)
        splice = self.splice_head(channels)
        pooled = F.avg_pool1d(channels, kernel_size=4, stride=4).transpose(1, 2)
        left = self.contact_left(pooled)
        right = self.contact_right(pooled)
        contacts = torch.matmul(left, right.transpose(-1, -2)) / (left.shape[-1] ** 0.5)
        return tracks, splice, contacts


def build() -> nn.Module:
    """Build the compact AlphaGenome reconstruction.

    Returns
    -------
    nn.Module
        Random-initialized DNA sequence model.
    """
    return AlphaGenomeMini()


def example_input() -> Tensor:
    """Return a one-hot encoded DNA segment.

    Returns
    -------
    Tensor
        One-hot DNA tensor with shape ``(1, 4, 64)``.
    """
    ids = torch.randint(0, 4, (1, 64))
    return F.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [
    ("AlphaGenome", "build", "example_input", "2025", "DC"),
]
