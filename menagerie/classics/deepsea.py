"""DeepSEA, 2015, Zhou and Troyanskaya.

Paper: "Predicting effects of noncoding variants with deep learning-based
sequence model." A three-layer convolutional tower predicts many chromatin
features from one-hot DNA sequence.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeepSEA(nn.Module):
    """Compact DeepSEA-style chromatin feature predictor."""

    def __init__(self, n_outputs: int = 919) -> None:
        """Initialize convolutional tower and multi-label head.

        Parameters
        ----------
        n_outputs:
            Number of chromatin feature outputs.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1),
            nn.Conv1d(64, 96, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.1),
            nn.Conv1d(96, 128, kernel_size=8),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(128 * 53, n_outputs)

    def forward(self, onehot: Tensor) -> Tensor:
        """Predict chromatin feature probabilities.

        Parameters
        ----------
        onehot:
            One-hot DNA sequence with shape ``(batch, 4, 1000)``.

        Returns
        -------
        Tensor
            Sigmoid probabilities with shape ``(batch, n_outputs)``.
        """
        feats = self.features(onehot)
        return torch.sigmoid(self.classifier(feats.flatten(1)))


def build() -> nn.Module:
    """Build a compact DeepSEA model.

    Returns
    -------
    nn.Module
        Random-initialized DeepSEA.
    """
    return DeepSEA()


def example_input() -> Tensor:
    """Return an example one-hot DNA sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 4, 1000)``.
    """
    ids = torch.randint(0, 4, (1, 1000), dtype=torch.long)
    return torch.nn.functional.one_hot(ids, num_classes=4).float().transpose(1, 2)


MENAGERIE_ENTRIES = [
    ("DeepSEA Chromatin Feature Predictor", "build", "example_input", "2015", "DE")
]
