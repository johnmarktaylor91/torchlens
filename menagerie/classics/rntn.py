"""Recursive Neural Tensor Network, 2013, Socher et al., "Recursive Deep Models for Semantic Compositionality".

Paper: Socher 2013, "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank."
The module composes a fixed balanced binary tree with bilinear tensor slices and
emits a sentiment distribution from the root representation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RecursiveNeuralTensorNetwork(nn.Module):
    """Balanced-tree RNTN composition module."""

    def __init__(self, dim: int = 5, n_classes: int = 3) -> None:
        """Initialize tensor composition and sentiment head.

        Parameters
        ----------
        dim
            Node representation width.
        n_classes
            Number of sentiment classes.
        """
        super().__init__()
        self.tensor = nn.Parameter(torch.randn(2 * dim, 2 * dim, dim) * 0.08)
        self.compose = nn.Linear(2 * dim, dim)
        self.sentiment = nn.Linear(dim, n_classes)

    def _compose(self, left: Tensor, right: Tensor) -> Tensor:
        """Compose two child representations with a bilinear tensor.

        Parameters
        ----------
        left
            Left child tensor of shape ``(batch, dim)``.
        right
            Right child tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Parent representation.
        """
        joined = torch.cat((left, right), dim=-1)
        bilinear = torch.einsum("bi,ijd,bj->bd", joined, self.tensor, joined)
        return torch.tanh(bilinear + self.compose(joined))

    def forward(self, leaves: Tensor) -> Tensor:
        """Compose four leaves in a fixed balanced binary tree.

        Parameters
        ----------
        leaves
            Leaf embeddings of shape ``(batch, 4, dim)``.

        Returns
        -------
        Tensor
            Root sentiment probabilities.
        """
        left = self._compose(leaves[:, 0], leaves[:, 1])
        right = self._compose(leaves[:, 2], leaves[:, 3])
        root = self._compose(left, right)
        return torch.softmax(self.sentiment(root), dim=-1)


MENAGERIE_ENTRIES = [
    ("Recursive Neural Tensor Network (RNTN)", "build", "example_input", "2013", "CD")
]


def build() -> nn.Module:
    """Build a small RNTN.

    Returns
    -------
    nn.Module
        Configured RNTN module.
    """
    return RecursiveNeuralTensorNetwork()


def example_input() -> Tensor:
    """Create leaf embedding examples.

    Returns
    -------
    Tensor
        Example leaves with shape ``(2, 4, 5)``.
    """
    return torch.randn(2, 4, 5)
