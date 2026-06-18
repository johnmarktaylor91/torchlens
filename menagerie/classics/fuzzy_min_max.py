"""Fuzzy Min-Max Neural Network, 1992, Patrick Simpson.

Paper: Simpson 1992, "Fuzzy min-max neural networks. I. Classification."
Hyperbox membership measures compare each feature to learned lower and upper
corners, then aggregate boxes by class.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Fuzzy Min-Max Neural Network (Simpson)", "build", "example_input", "1992", "CF")
]


class FuzzyMinMax(nn.Module):
    """Differentiable Simpson hyperbox classifier."""

    def __init__(self, n_features: int = 4, n_boxes: int = 6, n_classes: int = 3) -> None:
        """Initialize hyperbox corners and class assignments.

        Parameters
        ----------
        n_features
            Number of input features.
        n_boxes
            Number of fuzzy hyperboxes.
        n_classes
            Number of output classes.
        """
        super().__init__()
        centers = torch.rand(n_boxes, n_features)
        spans = torch.rand(n_boxes, n_features) * 0.25 + 0.05
        self.v = nn.Parameter((centers - spans).clamp(0.0, 1.0))
        self.w = nn.Parameter((centers + spans).clamp(0.0, 1.0))
        self.gamma = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("classes", torch.arange(n_boxes) % n_classes)
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        """Compute class memberships from hyperbox memberships.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_features)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Class membership scores of shape ``(batch, n_classes)``.
        """
        lower = torch.minimum(self.v, self.w)
        upper = torch.maximum(self.v, self.w)
        x_box = x.unsqueeze(1)
        above = torch.clamp(self.gamma * (x_box - upper), 0.0, 1.0)
        below = torch.clamp(self.gamma * (lower - x_box), 0.0, 1.0)
        box_membership = (1.0 - above - below).mean(dim=-1)
        class_scores = []
        for cls in range(self.n_classes):
            mask = (self.classes == cls).to(x.dtype)
            score = (box_membership * mask).amax(dim=-1)
            class_scores.append(score)
        return torch.stack(class_scores, dim=-1)


def build() -> nn.Module:
    """Build a small fuzzy min-max classifier.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return FuzzyMinMax()


def example_input() -> Tensor:
    """Return bounded feature inputs.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 4)``.
    """
    return torch.rand(2, 4)
