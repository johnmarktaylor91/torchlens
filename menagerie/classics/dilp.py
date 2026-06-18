"""Differentiable ILP, 2018, Evans and Grefenstette, "Learning Explanatory Rules".

Paper: Evans 2018, "Learning Explanatory Rules from Noisy Data."
This simplified delta-ILP layer enumerates fixed candidate clauses over atoms,
softly weights them, and performs fuzzy forward-chaining deduction steps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentiableILP(nn.Module):
    """Fuzzy forward-chaining layer over candidate binary clauses."""

    def __init__(self, n_atoms: int = 8, n_clauses: int = 6, n_steps: int = 3) -> None:
        """Initialize candidate clause tables and weights.

        Parameters
        ----------
        n_atoms
            Number of ground atoms.
        n_clauses
            Number of candidate clauses.
        n_steps
            Number of deduction iterations.
        """
        super().__init__()
        lhs = torch.arange(n_clauses) % n_atoms
        rhs = (torch.arange(n_clauses) * 2 + 1) % n_atoms
        head = (torch.arange(n_clauses) * 3 + 2) % n_atoms
        self.register_buffer("lhs", lhs)
        self.register_buffer("rhs", rhs)
        self.register_buffer("head", head)
        self.weights = nn.Parameter(torch.zeros(n_clauses))

    def forward(self, valuation: Tensor) -> Tensor:
        """Infer new atom valuations with soft weighted clauses.

        Parameters
        ----------
        valuation
            Initial truth values of shape ``(batch, n_atoms)``.

        Returns
        -------
        Tensor
            Refined truth values.
        """
        truth = valuation.clamp(0.0, 1.0)
        clause_weights = torch.softmax(self.weights, dim=0)
        for _ in range(3):
            body = truth[:, self.lhs] * truth[:, self.rhs]
            updates = torch.zeros_like(truth).scatter_add(
                1, self.head.unsqueeze(0).expand(truth.shape[0], -1), body * clause_weights
            )
            truth = torch.maximum(truth, updates.clamp(0.0, 1.0))
        return truth


MENAGERIE_ENTRIES = [
    ("Differentiable ILP (dILP / delta-ILP)", "build", "example_input", "2018", "CD")
]


def build() -> nn.Module:
    """Build a simplified dILP layer.

    Returns
    -------
    nn.Module
        Configured dILP module.
    """
    return DifferentiableILP()


def example_input() -> Tensor:
    """Create atom valuation examples.

    Returns
    -------
    Tensor
        Example valuations with shape ``(2, 8)``.
    """
    return torch.rand(2, 8)
