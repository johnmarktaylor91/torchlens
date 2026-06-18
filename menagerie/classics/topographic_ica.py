"""Topographic ICA, 2001, Hyvarinen, Hoyer, and Inki.

Paper: "Topographic independent component analysis." Linear filters feed squared
responses into a fixed local pooling graph, producing complex-cell-like topographic
energies while omitting the training objective.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Topographic ICA / Independent Subspace Analysis", "build", "example_input", "2001", "DA")
]


class TopographicICA(nn.Module):
    """Linear filters with fixed neighborhood energy pooling."""

    def __init__(self, n_input: int = 256, n_components: int = 32) -> None:
        """Initialize filters and circular topographic pooling neighborhoods.

        Parameters
        ----------
        n_input
            Input feature count.
        n_components
            Number of independent components.
        """
        super().__init__()
        self.filters = nn.Linear(n_input, n_components, bias=False)
        positions = torch.arange(n_components)
        distance = torch.minimum(
            (positions[:, None] - positions[None, :]).abs(),
            n_components - (positions[:, None] - positions[None, :]).abs(),
        )
        neighborhood = torch.exp(-(distance.float().pow(2)) / 4.0)
        self.register_buffer("neighborhood", neighborhood / neighborhood.sum(dim=-1, keepdim=True))

    def forward(self, x: Tensor) -> Tensor:
        """Compute pooled topographic component energies.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        Tensor
            Neighborhood-pooled energies.
        """
        squared = self.filters(x).pow(2)
        return squared @ self.neighborhood.T


def build() -> nn.Module:
    """Build a small topographic ICA module.

    Returns
    -------
    nn.Module
        Configured ``TopographicICA`` instance.
    """
    return TopographicICA()


def example_input() -> Tensor:
    """Return a float input example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 256)``.
    """
    return torch.randn(1, 256)
