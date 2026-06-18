"""Multi-Column Deep Neural Network, 2012, Dan Ciresan et al.

Paper: Multi-column deep neural networks for image classification.
Several convolutional columns process differently normalized views of the same
image and average their class posterior probabilities.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Multi-Column Deep Neural Network (MCDNN/DanNet)", "build", "example_input", "2012", "DC")
]


class MultiColumnDeepNeuralNetwork(nn.Module):
    """Small ensemble CNN with averaged softmax posteriors."""

    def __init__(self, num_columns: int = 3, num_classes: int = 10) -> None:
        """Initialize multiple CNN columns and preprocessing offsets.

        Parameters
        ----------
        num_columns
            Number of ensemble columns.
        num_classes
            Number of classifier outputs.
        """
        super().__init__()
        self.columns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(16 * 8 * 8, num_classes),
                )
                for _ in range(num_columns)
            ]
        )
        offsets = torch.linspace(-0.2, 0.2, num_columns).view(num_columns, 1, 1, 1)
        self.register_buffer("offsets", offsets)

    def forward(self, x: Tensor) -> Tensor:
        """Average posterior probabilities from all columns.

        Parameters
        ----------
        x
            Grayscale image tensor with shape ``(B, 1, 32, 32)``.

        Returns
        -------
        Tensor
            Mean class probabilities.
        """
        probs = []
        for idx, column in enumerate(self.columns):
            adjusted = x + self.offsets[idx].view(1, 1, 1, 1)
            probs.append(torch.softmax(column(adjusted), dim=-1))
        return torch.stack(probs, dim=0).mean(dim=0)


def build() -> nn.Module:
    """Build a compact MCDNN ensemble.

    Returns
    -------
    nn.Module
        Random-initialized multi-column classifier.
    """
    return MultiColumnDeepNeuralNetwork()


def example_input() -> Tensor:
    """Return a traceable grayscale image batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)
