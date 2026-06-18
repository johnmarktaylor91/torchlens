"""Fuzzy ARTMAP, 1991, Carpenter, Grossberg, and Reynolds.

Paper: "ARTMAP: Supervised Real-Time Learning and Classification of
Nonstationary Data." ARTMAP links input ART categories to label ART categories
through a map field with match tracking for supervised resonance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FuzzyARTMAP(nn.Module):
    """Supervised Fuzzy ART category search with a map field."""

    def __init__(
        self,
        n_features: int = 6,
        n_categories: int = 5,
        n_classes: int = 3,
        vigilance: float = 0.72,
    ) -> None:
        """Initialize Fuzzy ARTMAP components.

        Parameters
        ----------
        n_features:
            Raw analog input dimensionality.
        n_categories:
            Number of ARTa categories.
        n_classes:
            Number of supervised label categories.
        vigilance:
            Baseline ARTa vigilance.
        """
        super().__init__()
        self.n_features = n_features
        self.vigilance = vigilance
        self.alpha = 1.0e-3
        self.register_buffer("prototypes", torch.rand(n_categories, 2 * n_features))
        self.register_buffer(
            "map_field",
            torch.nn.functional.one_hot(torch.arange(n_categories) % n_classes, n_classes).float(),
        )

    def complement_code(self, x: Tensor) -> Tensor:
        """Complement-code raw analog input.

        Parameters
        ----------
        x:
            Raw or complement-coded input.

        Returns
        -------
        Tensor
            Complement-coded tensor.
        """
        if x.shape[-1] == 2 * self.n_features:
            return x
        clipped = x.clamp(0.0, 1.0)
        return torch.cat((clipped, 1.0 - clipped), dim=-1)

    def forward(self, x: Tensor, label: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Classify with map-field consistency and optional match tracking.

        Parameters
        ----------
        x:
            Raw input tensor with shape ``(B, n_features)``.
        label:
            Optional integer labels with shape ``(B,)`` used to mask categories
            whose map-field class disagrees.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Category one-hot codes, class logits from the map field, and ART
            match scores.
        """
        coded = self.complement_code(x)
        fuzzy = torch.minimum(coded[:, None, :], self.prototypes[None, :, :])
        fuzzy_mass = fuzzy.sum(dim=-1)
        match = fuzzy_mass / coded.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        choice = fuzzy_mass / (self.alpha + self.prototypes.sum(dim=-1)[None, :])
        masked = torch.where(match >= self.vigilance, choice, torch.full_like(choice, -1.0))
        if label is not None:
            class_match = self.map_field[:, label].transpose(0, 1) > 0.5
            masked = torch.where(class_match, masked, torch.full_like(masked, -1.0))
        winner = masked.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(winner, self.prototypes.shape[0]).to(x.dtype)
        return one_hot, one_hot @ self.map_field, match


def build() -> nn.Module:
    """Build a small random-init Fuzzy ARTMAP module.

    Returns
    -------
    nn.Module
        A traceable ``FuzzyARTMAP`` instance.
    """
    return FuzzyARTMAP()


def example_input() -> Tensor:
    """Return analog ARTMAP examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 6)``.
    """
    return torch.tensor(
        [[0.1, 0.7, 0.2, 0.9, 0.4, 0.3], [0.8, 0.2, 0.6, 0.1, 0.5, 0.9]], dtype=torch.float32
    )
