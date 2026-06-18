"""Generalized Learning Vector Quantization, 1995, Sato and Yamada.

GLVQ keeps labeled prototypes but exposes a differentiable relative-distance
margin between nearest correct-class and nearest wrong-class prototypes.
"""

import torch
from torch import Tensor, nn


class GLVQ(nn.Module):
    """Differentiable prototype classifier with GLVQ margin output."""

    def __init__(self, dim: int = 4, n_classes: int = 3, prototypes_per_class: int = 2) -> None:
        """Initialize labeled prototypes.

        Parameters
        ----------
        dim:
            Feature dimension.
        n_classes:
            Number of classes.
        prototypes_per_class:
            Number of prototypes for each class.
        """
        super().__init__()
        labels = torch.arange(n_classes).repeat_interleave(prototypes_per_class)
        self.prototypes = nn.Parameter(torch.randn(labels.numel(), dim))
        self.register_buffer("proto_labels", labels)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return LVQ logits and a label-free GLVQ-style closest-class margin.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Class logits and differentiable margin between closest and runner-up classes.
        """
        d2 = torch.cdist(x, self.prototypes).pow(2)
        class_distances = []
        for class_index in range(self.n_classes):
            mask = self.proto_labels == class_index
            class_distances.append(d2[:, mask].min(dim=1).values)
        distances = torch.stack(class_distances, dim=-1)
        two_best = distances.topk(k=2, dim=-1, largest=False).values
        margin = (two_best[:, 0] - two_best[:, 1]) / (two_best[:, 0] + two_best[:, 1] + 1.0e-6)
        return -distances, torch.sigmoid(margin).unsqueeze(-1)


def build() -> nn.Module:
    """Build a small GLVQ classifier.

    Returns
    -------
    nn.Module
        Configured ``GLVQ`` instance.
    """
    return GLVQ()


def example_input() -> Tensor:
    """Create a feature-vector example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4)``.
    """
    return torch.randn(2, 4)
