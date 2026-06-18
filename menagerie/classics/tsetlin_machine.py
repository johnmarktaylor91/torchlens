"""Tsetlin Machine, 2018, Granmo.

Paper: "The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition..."
Clause evaluation is implemented with smooth include/exclude surrogates over
binarized literals. Learning automata updates are omitted in this forward-only module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TsetlinMachine(nn.Module):
    """Differentiable surrogate for Tsetlin clause voting."""

    def __init__(self, n_inputs: int = 12, n_clauses: int = 10) -> None:
        """Initialize clause automata states and polarities.

        Parameters
        ----------
        n_inputs:
            Number of Boolean input variables.
        n_clauses:
            Number of conjunctive clauses.
        """
        super().__init__()
        states = torch.randn(n_clauses, 2 * n_inputs) * 0.7
        polarity = torch.where(torch.arange(n_clauses) % 2 == 0, 1.0, -1.0)
        self.automata_logits = nn.Parameter(states)
        self.register_buffer("polarity", polarity)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate smooth clauses and polarized vote score.

        Parameters
        ----------
        x:
            Integer or float binary input tensor with shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Clause-vote score with shape ``(batch, 1)``.
        """
        bits = x.to(self.automata_logits.dtype).clamp(0.0, 1.0)
        literals = torch.cat((bits, 1.0 - bits), dim=-1)
        include = torch.sigmoid(self.automata_logits)
        literal_support = include * literals[:, None, :] + (1.0 - include)
        clauses = torch.exp(torch.log(literal_support.clamp_min(1.0e-5)).sum(dim=-1))
        score = clauses @ self.polarity[:, None]
        return score / self.polarity.numel() ** 0.5


def build() -> nn.Module:
    """Build a small differentiable Tsetlin Machine surrogate.

    Returns
    -------
    nn.Module
        Configured ``TsetlinMachine`` instance.
    """
    return TsetlinMachine()


def example_input() -> Tensor:
    """Create a binary Tsetlin Machine input.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 12)`` and dtype ``int64``.
    """
    return torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.int64)


MENAGERIE_ENTRIES = [("Tsetlin Machine", "build", "example_input", "2018", "MB1")]
