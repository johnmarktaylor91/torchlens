"""Dentate gyrus pattern separation, 1994, Treves and Rolls.

Paper: "Computational analysis of the role of the hippocampus in memory." Entorhinal
input expands into a much larger dentate representation with k-winner-take-all
inhibition to orthogonalize similar patterns.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Dentate Gyrus Pattern Separation", "build", "example_input", "1994", "DB")]


class DentateGyrusSeparation(nn.Module):
    """Sparse expansion recoding for pattern separation."""

    def __init__(self, ec_dim: int = 50, dg_dim: int = 256, k: int = 12) -> None:
        """Initialize entorhinal-to-dentate expansion weights.

        Parameters
        ----------
        ec_dim
            Entorhinal input dimensionality.
        dg_dim
            Dentate expansion dimensionality.
        k
            Number of active dentate winners.
        """
        super().__init__()
        self.expand = nn.Linear(ec_dim, dg_dim)
        self.k = k

    def forward(self, ec: Tensor) -> Tensor:
        """Compute sparse dentate pattern-separation code.

        Parameters
        ----------
        ec
            Entorhinal input tensor of shape ``(batch, ec_dim)``.

        Returns
        -------
        Tensor
            Sparse dentate code.
        """
        activity = torch.relu(self.expand(ec))
        winners, _ = torch.topk(activity, self.k, dim=-1)
        threshold = winners[..., -1:].expand_as(activity)
        return activity * (activity >= threshold).to(activity.dtype)


def build() -> nn.Module:
    """Build a small dentate pattern-separation module.

    Returns
    -------
    nn.Module
        Configured ``DentateGyrusSeparation`` instance.
    """
    return DentateGyrusSeparation()


def example_input() -> Tensor:
    """Return an entorhinal input example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 50)``.
    """
    return torch.randn(1, 50)
