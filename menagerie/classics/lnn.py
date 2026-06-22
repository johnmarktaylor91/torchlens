"""Logical Neural Networks.

Paper: "Logical Neural Networks", Riegel et al., 2020.

The compact reconstruction keeps LNN's one-neuron-per-formula structure with
weighted real-valued logic over lower/upper truth bounds for AND, OR, and
IMPLIES, producing contradiction-aware formula bounds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightedLogic(nn.Module):
    """Weighted Łukasiewicz logical connective."""

    def __init__(self, arity: int, connective: str) -> None:
        """Initialize connective weights and bias."""

        super().__init__()
        self.connective = connective
        self.raw_weight = nn.Parameter(torch.zeros(arity))
        self.raw_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, bounds: torch.Tensor) -> torch.Tensor:
        """Evaluate weighted truth bounds."""

        weights = torch.nn.functional.softplus(self.raw_weight)
        beta = torch.nn.functional.softplus(self.raw_beta)
        lower, upper = bounds[..., 0], bounds[..., 1]
        if self.connective == "and":
            out_l = torch.clamp(beta - ((1.0 - lower) * weights).sum(dim=-1), 0.0, 1.0)
            out_u = torch.clamp(beta - ((1.0 - upper) * weights).sum(dim=-1), 0.0, 1.0)
        elif self.connective == "or":
            out_l = torch.clamp(1.0 - beta + (lower * weights).sum(dim=-1), 0.0, 1.0)
            out_u = torch.clamp(1.0 - beta + (upper * weights).sum(dim=-1), 0.0, 1.0)
        else:
            antecedent = 1.0 - upper[..., :1]
            consequent = lower[..., 1:2]
            val = torch.cat([antecedent, consequent], dim=-1)
            out_l = torch.clamp(1.0 - beta + (val * weights).sum(dim=-1), 0.0, 1.0)
            out_u = torch.ones_like(out_l)
        return torch.stack([out_l, out_u], dim=-1)


class LNNCompact(nn.Module):
    """Compact formula graph for ``(A AND B) IMPLIES C``."""

    def __init__(self) -> None:
        """Initialize formula neurons."""

        super().__init__()
        self.and_ab = WeightedLogic(2, "and")
        self.or_ab = WeightedLogic(2, "or")
        self.implies = WeightedLogic(2, "implies")

    def forward(self, atom_bounds: torch.Tensor) -> torch.Tensor:
        """Run upward logical inference over truth bounds."""

        a_b = atom_bounds[:, :2]
        c = atom_bounds[:, 2:3]
        conj = self.and_ab(a_b)
        disj = self.or_ab(a_b)
        rule = self.implies(torch.cat([conj.unsqueeze(1), c], dim=1))
        contradiction = torch.relu(rule[..., 0] - rule[..., 1]).unsqueeze(-1)
        return torch.cat([conj, disj, rule, contradiction.expand(-1, 2)], dim=-1)


def build() -> nn.Module:
    """Build compact LNN."""

    return LNNCompact()


def example_input() -> torch.Tensor:
    """Return atom lower/upper truth bounds."""

    return torch.tensor([[[0.8, 1.0], [0.4, 0.9], [0.2, 0.7]]], dtype=torch.float32)


MENAGERIE_ENTRIES = [("LNN", "build", "example_input", "2020", "E7")]
