"""Neural Production Systems for rule-governed visual dynamics.

Paper: Goyal et al. 2021, "Neural Production Systems: Learning Rule-Governed
Visual Dynamics" (NeurIPS), arXiv:2103.01937.

The compact reconstruction keeps the core production-system inductive bias:
object slots bind to rule templates, rule/entity matches are scored sparsely,
and selected rule updates are applied to entity states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralProductionSystem(nn.Module):
    """Slot-rule production system with soft top-rule updates."""

    def __init__(self, entities: int = 4, dim: int = 24, rules: int = 5) -> None:
        """Initialize entity encoders, rule keys, and rule-specific updates.

        Parameters
        ----------
        entities:
            Number of object slots.
        dim:
            Entity state width.
        rules:
            Number of learned rule templates.
        """

        super().__init__()
        self.entities = entities
        self.encoder = nn.Linear(dim, dim)
        self.rule_keys = nn.Parameter(torch.randn(rules, dim))
        self.rule_updates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(rules)
            ]
        )
        self.query = nn.Linear(dim, dim)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """Apply rule-template updates to object slots.

        Parameters
        ----------
        slots:
            Entity slot tensor with shape ``(batch, entities, dim)``.

        Returns
        -------
        torch.Tensor
            Updated entity slots.
        """

        state = torch.tanh(self.encoder(slots))
        match = torch.matmul(self.query(state), self.rule_keys.t())
        rule_weight = torch.softmax(match, dim=-1)
        context = state.mean(dim=1, keepdim=True).expand_as(state)
        updates = []
        for update in self.rule_updates:
            updates.append(update(torch.cat([state, context], dim=-1)))
        stacked = torch.stack(updates, dim=-2)
        sparse = F.one_hot(rule_weight.argmax(dim=-1), num_classes=stacked.shape[-2]).to(
            state.dtype
        )
        gated = (stacked * sparse.unsqueeze(-1)).sum(dim=-2)
        return state + gated


def build() -> nn.Module:
    """Build the compact neural production system.

    Returns
    -------
    nn.Module
        Random-initialized production-system module.
    """

    return NeuralProductionSystem()


def example_input() -> torch.Tensor:
    """Create a small set of object slots.

    Returns
    -------
    torch.Tensor
        Slot tensor with shape ``(1, 4, 24)``.
    """

    return torch.randn(1, 4, 24)


MENAGERIE_ENTRIES = [
    ("Neural Production Systems", "build", "example_input", "2021", "E5"),
]
