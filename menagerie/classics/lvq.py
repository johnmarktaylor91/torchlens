"""Learning Vector Quantization, 1990, Kohonen, "The Self-Organizing Map".

A labeled prototype codebook classifies by nearest prototype; class scores are
negative nearest distances within each label group.
"""

import torch
from torch import Tensor, nn


class LVQ(nn.Module):
    """Nearest-labeled-prototype classifier."""

    def __init__(self, dim: int = 4, n_classes: int = 3, prototypes_per_class: int = 2) -> None:
        """Initialize labeled prototypes.

        Parameters
        ----------
        dim:
            Feature dimension.
        n_classes:
            Number of prototype labels.
        prototypes_per_class:
            Number of prototypes for each class.
        """
        super().__init__()
        labels = torch.arange(n_classes).repeat_interleave(prototypes_per_class)
        self.prototypes = nn.Parameter(torch.randn(labels.numel(), dim))
        self.register_buffer("proto_labels", labels)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        """Return negative nearest-prototype distances per class.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, n_classes)``.
        """
        d2 = torch.cdist(x, self.prototypes).pow(2)
        logits = []
        for class_index in range(self.n_classes):
            mask = self.proto_labels == class_index
            logits.append(-d2[:, mask].min(dim=1).values)
        return torch.stack(logits, dim=-1)

    def predict(self, x: Tensor) -> Tensor:
        """Return the label of the nearest prototype.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Integer predicted labels.
        """
        d2 = torch.cdist(x, self.prototypes).pow(2)
        return self.proto_labels[d2.argmin(dim=1)]


def build() -> nn.Module:
    """Build a small LVQ classifier.

    Returns
    -------
    nn.Module
        Configured ``LVQ`` instance.
    """
    return LVQ()


def example_input() -> Tensor:
    """Create a feature-vector example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4)``.
    """
    return torch.randn(2, 4)
