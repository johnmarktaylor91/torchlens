"""TensorLog, 2016, Cohen, "TensorLog: A Differentiable Deductive Database".

Paper: Cohen 2016, "TensorLog: A Differentiable Deductive Database."
This simplified module stores small dense relation matrices and answers a query by
softly weighting relation-matrix chains, mirroring TensorLog's compiled matmul view.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TensorLogQuery(nn.Module):
    """Differentiable relation-chain query over entity ids."""

    def __init__(self, n_entities: int = 7, n_relations: int = 3) -> None:
        """Initialize relation matrices and predicate weights.

        Parameters
        ----------
        n_entities
            Number of entities.
        n_relations
            Number of binary predicates.
        """
        super().__init__()
        self.n_entities = n_entities
        self.relations = nn.Parameter(torch.randn(n_relations, n_entities, n_entities) * 0.1)
        self.rule_weights = nn.Parameter(torch.zeros(n_relations))

    def forward(self, heads: Tensor) -> Tensor:
        """Query reachable tail distributions from head entity ids.

        Parameters
        ----------
        heads
            Head entity ids of shape ``(batch,)``.

        Returns
        -------
        Tensor
            Tail entity probabilities.
        """
        one_hot = torch.nn.functional.one_hot(heads, num_classes=self.n_entities).to(
            self.relations.dtype
        )
        weights = torch.softmax(self.rule_weights, dim=0)
        relation = torch.sum(torch.softmax(self.relations, dim=-1) * weights[:, None, None], dim=0)
        tails = one_hot @ relation @ relation
        return tails / tails.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


MENAGERIE_ENTRIES = [("TensorLog", "build", "example_input", "2016", "CD")]


def build() -> nn.Module:
    """Build a simplified TensorLog query module.

    Returns
    -------
    nn.Module
        Configured TensorLog module.
    """
    return TensorLogQuery()


def example_input() -> Tensor:
    """Create head entity id examples.

    Returns
    -------
    Tensor
        Example head ids with shape ``(2,)``.
    """
    return torch.tensor([0, 3], dtype=torch.long)
