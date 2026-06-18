"""DeepBind, 2015, Alipanahi et al., "Predicting the sequence specificities".

A convolutional motif scanner over one-hot DNA or RNA sequence uses global max
pooling and a small classifier to predict binding affinity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepBind(nn.Module):
    """One-dimensional motif-scanning binding predictor."""

    def __init__(self, n_motifs: int = 32, motif_width: int = 24) -> None:
        """Initialize motif convolution and binding head.

        Parameters
        ----------
        n_motifs:
            Number of learned motif filters.
        motif_width:
            Motif convolution width.
        """
        super().__init__()
        self.conv = nn.Conv1d(4, n_motifs, kernel_size=motif_width)
        self.head = nn.Sequential(nn.Linear(n_motifs, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, onehot: Tensor) -> Tensor:
        """Predict binding score from one-hot sequence.

        Parameters
        ----------
        onehot:
            Tensor with shape ``(batch, 4, length)``.

        Returns
        -------
        Tensor
            Binding score tensor with shape ``(batch, 1)``.
        """
        motifs = torch.relu(self.conv(onehot))
        pooled = motifs.amax(dim=-1)
        return self.head(pooled)


def build() -> nn.Module:
    """Build a compact DeepBind model.

    Returns
    -------
    nn.Module
        Random-initialized DeepBind.
    """
    return DeepBind()


def example_input() -> Tensor:
    """Return an example one-hot nucleotide sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 4, 101)``.
    """
    ids = torch.randint(0, 4, (1, 101), dtype=torch.long)
    return torch.nn.functional.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [("DeepBind DNA/RNA Binding Predictor", "build", "example_input", "2015", "DE")]
