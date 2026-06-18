"""SATNet, 2019, Wang et al., "SATNet: Bridging deep learning and logical reasoning".

Paper: Wang 2019, "SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver."
This compact layer uses unrolled projected-gradient updates on relaxed Boolean
variables against a learnable clause matrix, standing in for the full SDP solver.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SATLayer(nn.Module):
    """Relaxed differentiable MAXSAT layer with unrolled updates."""

    def __init__(self, n_vars: int = 6, n_clauses: int = 8, n_steps: int = 5) -> None:
        """Initialize learnable signed clause matrix.

        Parameters
        ----------
        n_vars
            Number of relaxed Boolean variables.
        n_clauses
            Number of soft clauses.
        n_steps
            Number of unrolled optimization steps.
        """
        super().__init__()
        self.n_steps = n_steps
        self.clauses = nn.Parameter(torch.randn(n_clauses, n_vars) * 0.2)
        self.bias = nn.Parameter(torch.zeros(n_clauses))

    def forward(self, bits: Tensor) -> Tensor:
        """Refine relaxed bits by minimizing soft clause violations.

        Parameters
        ----------
        bits
            Relaxed bit probabilities of shape ``(batch, n_vars)``.

        Returns
        -------
        Tensor
            Refined bit probabilities.
        """
        z = bits * 2.0 - 1.0
        for _ in range(self.n_steps):
            margins = z @ self.clauses.T + self.bias
            violation = torch.sigmoid(-margins)
            grad = violation @ self.clauses
            z = torch.tanh(z + 0.25 * grad)
        return torch.sigmoid(z)


MENAGERIE_ENTRIES = [
    ("SATNet (differentiable MAXSAT layer)", "build", "example_input", "2019", "CD")
]


def build() -> nn.Module:
    """Build a tiny SATNet-style layer.

    Returns
    -------
    nn.Module
        Configured SAT layer.
    """
    return SATLayer()


def example_input() -> Tensor:
    """Create relaxed bit examples.

    Returns
    -------
    Tensor
        Example relaxed bits with shape ``(2, 6)``.
    """
    return torch.rand(2, 6)
