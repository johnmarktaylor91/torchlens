"""ART2, 1987, Carpenter and Grossberg.

Paper: "ART 2: Self-Organization of Stable Category Recognition Codes for
Analog Input Patterns." ART2 normalizes analog F1 activity with gain control
before vigilance-gated category resonance in F2.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ART2(nn.Module):
    """Analog-input ART module with normalized F1 preprocessing."""

    def __init__(self, n_features: int = 8, n_categories: int = 5, vigilance: float = 0.82) -> None:
        """Initialize ART2 prototypes.

        Parameters
        ----------
        n_features:
            Analog input dimensionality.
        n_categories:
            Number of F2 categories.
        vigilance:
            Cosine-match vigilance threshold.
        """
        super().__init__()
        prototypes = torch.rand(n_categories, n_features)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.register_buffer("prototypes", prototypes)
        self.vigilance = vigilance

    @staticmethod
    def normalize(x: Tensor) -> Tensor:
        """Normalize F1 activity with ART2-style gain control.

        Parameters
        ----------
        x:
            Analog input activity.

        Returns
        -------
        Tensor
            Unit-normalized activity.
        """
        rectified = torch.relu(x)
        return rectified / rectified.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run normalized analog category search.

        Parameters
        ----------
        x:
            Analog input tensor with shape ``(B, n_features)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Category one-hot codes, winning prototype, and cosine matches.
        """
        f1 = self.normalize(x)
        matches = f1 @ self.prototypes.transpose(0, 1)
        masked = torch.where(matches >= self.vigilance, matches, torch.full_like(matches, -1.0))
        winner = masked.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(winner, self.prototypes.shape[0]).to(x.dtype)
        return one_hot, one_hot @ self.prototypes, matches


def build() -> nn.Module:
    """Build a small random-init ART2 module.

    Returns
    -------
    nn.Module
        A traceable ``ART2`` instance.
    """
    return ART2()


def example_input() -> Tensor:
    """Return analog ART2 examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 8)``.
    """
    return torch.tensor(
        [[0.3, 0.8, 0.1, 0.0, 0.4, 0.2, 0.7, 0.5], [0.9, 0.1, 0.4, 0.6, 0.0, 0.2, 0.3, 0.8]]
    )
