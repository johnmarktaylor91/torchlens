"""DanQ, 2016, Quang and Xie, "DanQ".

A convolutional motif layer feeds a bidirectional LSTM that captures regulatory
grammar before multi-label chromatin feature prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DanQ(nn.Module):
    """Compact DanQ CNN-BiLSTM sequence model."""

    def __init__(self, n_outputs: int = 919, hidden_size: int = 64) -> None:
        """Initialize motif convolution, BiLSTM, and classifier.

        Parameters
        ----------
        n_outputs:
            Number of chromatin feature outputs.
        hidden_size:
            LSTM hidden size per direction.
        """
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 64, kernel_size=26), nn.ReLU(), nn.MaxPool1d(13))
        self.blstm = nn.LSTM(64, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * hidden_size * 75, n_outputs))

    def forward(self, onehot: Tensor) -> Tensor:
        """Predict chromatin feature probabilities from one-hot DNA.

        Parameters
        ----------
        onehot:
            One-hot DNA tensor with shape ``(batch, 4, 1000)``.

        Returns
        -------
        Tensor
            Sigmoid probabilities with shape ``(batch, n_outputs)``.
        """
        motifs = self.conv(onehot).transpose(1, 2)
        encoded, _ = self.blstm(motifs)
        return torch.sigmoid(self.classifier(encoded.flatten(1)))


def build() -> nn.Module:
    """Build a compact DanQ model.

    Returns
    -------
    nn.Module
        Random-initialized DanQ.
    """
    return DanQ()


def example_input() -> Tensor:
    """Return an example one-hot DNA sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 4, 1000)``.
    """
    ids = torch.randint(0, 4, (1, 1000), dtype=torch.long)
    return torch.nn.functional.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [("DanQ CNN-BiLSTM Sequence Model", "build", "example_input", "2016", "DE")]
