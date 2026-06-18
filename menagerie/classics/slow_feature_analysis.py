"""Slow Feature Analysis, 2002, Wiskott and Sejnowski.

Paper: "Slow feature analysis: Unsupervised learning of invariances." A quadratic
expansion followed by a projection gives the differentiable substrate whose training
criterion minimizes temporal derivatives under whitening constraints.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Wiskott-Sejnowski Slow Feature Analysis", "build", "example_input", "2002", "DA")
]


class SlowFeatureAnalysis(nn.Module):
    """Quadratic feature expansion with learned slow-feature projection."""

    def __init__(self, n_input: int = 64, n_output: int = 16) -> None:
        """Initialize the projection from expanded features.

        Parameters
        ----------
        n_input
            Number of input features.
        n_output
            Number of output slow features.
        """
        super().__init__()
        self.project = nn.Linear(n_input * 2, n_output)

    def forward(self, x_seq: Tensor) -> Tensor:
        """Project an input sequence after nonlinear expansion.

        Parameters
        ----------
        x_seq
            Sequence tensor of shape ``(batch, time, n_input)`` or ``(batch, n_input)``.

        Returns
        -------
        Tensor
            Projected slow-feature activations.
        """
        expanded = torch.cat((x_seq, x_seq.pow(2)), dim=-1)
        return self.project(expanded)


def build() -> nn.Module:
    """Build a small SFA module.

    Returns
    -------
    nn.Module
        Configured ``SlowFeatureAnalysis`` instance.
    """
    return SlowFeatureAnalysis()


def example_input() -> Tensor:
    """Return a short feature sequence example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 4, 64)``.
    """
    return torch.randn(1, 4, 64)
