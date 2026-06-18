"""Pandemonium architecture (1959), Oliver Selfridge.

Paper: "Pandemonium: A paradigm for learning."
Feature demons score evidence in parallel, cognitive demons pool related evidence,
and a decision demon selects the strongest interpretation by competition.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Pandemonium(nn.Module):
    """Feature-detector committee with competitive decision demons."""

    def __init__(self, n_in: int = 12, n_features: int = 8, n_classes: int = 4) -> None:
        """Initialize feature, cognitive, and decision demons.

        Parameters
        ----------
        n_in
            Number of input features.
        n_features
            Number of feature demons.
        n_classes
            Number of decision demons.
        """
        super().__init__()
        self.feature_demons = nn.Linear(n_in, n_features)
        self.cognitive_demons = nn.Linear(n_features, n_classes, bias=False)
        with torch.no_grad():
            self.cognitive_demons.weight.copy_(torch.randn(n_classes, n_features).abs())

    def forward(self, x: Tensor) -> Tensor:
        """Score classes through a demon hierarchy.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Competitive class probabilities.
        """
        feature_cries = torch.relu(self.feature_demons(x))
        class_cries = self.cognitive_demons(feature_cries)
        return torch.softmax(class_cries, dim=-1)


def build() -> nn.Module:
    """Build a small Pandemonium module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return Pandemonium()


def example_input() -> Tensor:
    """Return an example continuous pattern.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(2, 12)
