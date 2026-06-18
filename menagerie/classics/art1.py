"""ART1, 1987, Carpenter and Grossberg.

Paper: "A Massively Parallel Architecture for a Self-Organizing Neural Pattern
Recognition Machine." ART1 performs binary category search with bottom-up
choice, top-down expectation, vigilance, reset, and resonance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ART1(nn.Module):
    """Binary Adaptive Resonance Theory category search module."""

    def __init__(self, n_features: int = 8, n_categories: int = 5, vigilance: float = 0.7) -> None:
        """Initialize fixed ART1 category prototypes.

        Parameters
        ----------
        n_features:
            Binary input dimensionality.
        n_categories:
            Number of initialized F2 categories.
        vigilance:
            Minimum normalized match required for resonance.
        """
        super().__init__()
        prototypes = (torch.rand(n_categories, n_features) > 0.3).float()
        self.register_buffer("prototypes", prototypes)
        self.vigilance = vigilance
        self.alpha = 1.0e-3

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run ART1 choice and vigilance-gated search.

        Parameters
        ----------
        x:
            Binary input tensor with shape ``(B, n_features)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Resonant category one-hot codes, winning prototype, and match
            scores for all categories.
        """
        intersection = torch.minimum(x[:, None, :], self.prototypes[None, :, :])
        match_mass = intersection.sum(dim=-1)
        prototype_mass = self.prototypes.sum(dim=-1)
        choice = match_mass / (self.alpha + prototype_mass[None, :])
        input_mass = x.sum(dim=-1, keepdim=True).clamp_min(1.0)
        match = match_mass / input_mass
        masked_choice = torch.where(match >= self.vigilance, choice, torch.full_like(choice, -1.0))
        winner = masked_choice.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(winner, self.prototypes.shape[0]).to(x.dtype)
        expectation = one_hot @ self.prototypes
        return one_hot, expectation, match


def build() -> nn.Module:
    """Build a small random-init ART1 module.

    Returns
    -------
    nn.Module
        A traceable ``ART1`` instance.
    """
    return ART1()


def example_input() -> Tensor:
    """Return binary ART1 examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 8)``.
    """
    return torch.tensor(
        [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]]
    )
