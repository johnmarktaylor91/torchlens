"""Fuzzy ART, 1991, Carpenter, Grossberg, and Rosen.

Paper: "Fuzzy ART: Fast Stable Learning and Categorization of Analog Patterns."
Fuzzy ART uses complement coding plus fuzzy-min choice and match functions to
cluster analog inputs under a vigilance criterion.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FuzzyART(nn.Module):
    """Complement-coded fuzzy-min ART category search module."""

    def __init__(self, n_features: int = 4, n_categories: int = 5, vigilance: float = 0.78) -> None:
        """Initialize Fuzzy ART prototypes.

        Parameters
        ----------
        n_features:
            Raw analog feature count before complement coding.
        n_categories:
            Number of fixed category prototypes.
        vigilance:
            Minimum fuzzy match for resonance.
        """
        super().__init__()
        self.n_features = n_features
        self.alpha = 1.0e-3
        self.vigilance = vigilance
        self.register_buffer("prototypes", torch.rand(n_categories, 2 * n_features))

    def complement_code(self, x: Tensor) -> Tensor:
        """Apply complement coding to analog inputs.

        Parameters
        ----------
        x:
            Input tensor. If it already has ``2 * n_features`` columns it is
            treated as complement-coded.

        Returns
        -------
        Tensor
            Complement-coded input.
        """
        if x.shape[-1] == 2 * self.n_features:
            return x
        clipped = x.clamp(0.0, 1.0)
        return torch.cat((clipped, 1.0 - clipped), dim=-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run fuzzy-min choice and vigilance search.

        Parameters
        ----------
        x:
            Raw or complement-coded analog input.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Category one-hot codes, winning fuzzy expectation, and matches.
        """
        coded = self.complement_code(x)
        fuzzy = torch.minimum(coded[:, None, :], self.prototypes[None, :, :])
        fuzzy_mass = fuzzy.sum(dim=-1)
        choice = fuzzy_mass / (self.alpha + self.prototypes.sum(dim=-1)[None, :])
        match = fuzzy_mass / coded.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        masked = torch.where(match >= self.vigilance, choice, torch.full_like(choice, -1.0))
        winner = masked.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(winner, self.prototypes.shape[0]).to(x.dtype)
        return one_hot, one_hot @ self.prototypes, match


def build() -> nn.Module:
    """Build a small random-init Fuzzy ART module.

    Returns
    -------
    nn.Module
        A traceable ``FuzzyART`` instance.
    """
    return FuzzyART()


def example_input() -> Tensor:
    """Return complement-coded analog examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 8)``.
    """
    raw = torch.tensor([[0.2, 0.7, 0.4, 0.9], [0.8, 0.1, 0.6, 0.3]], dtype=torch.float32)
    return torch.cat((raw, 1.0 - raw), dim=-1)
