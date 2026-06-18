"""Logic Tensor Networks, 2016, Serafini and Garcez, "Logic Tensor Networks".

Paper: Serafini 2016, "Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge."
The module maps object vectors to fuzzy predicate truth values and evaluates a
small Real-Logic formula with product t-norm conjunction and existential pooling.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LogicTensorNetwork(nn.Module):
    """Fuzzy Real-Logic satisfiability module."""

    def __init__(self, object_dim: int = 6) -> None:
        """Initialize neural predicates.

        Parameters
        ----------
        object_dim
            Object vector width.
        """
        super().__init__()
        self.pred_p = nn.Sequential(nn.Linear(object_dim, 8), nn.Tanh(), nn.Linear(8, 1))
        self.pred_q = nn.Sequential(nn.Linear(object_dim, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, objects: Tensor) -> Tensor:
        """Evaluate a fuzzy formula over object vectors.

        Parameters
        ----------
        objects
            Object embeddings of shape ``(batch, objects, object_dim)``.

        Returns
        -------
        Tensor
            Formula satisfiability score per batch item.
        """
        p = torch.sigmoid(self.pred_p(objects))
        q = torch.sigmoid(self.pred_q(objects))
        implication = torch.clamp(1.0 - p + q, 0.0, 1.0)
        forall = implication.mean(dim=1)
        exists = q.amax(dim=1)
        return forall * exists


MENAGERIE_ENTRIES = [("Logic Tensor Networks (LTN)", "build", "example_input", "2016", "CD")]


def build() -> nn.Module:
    """Build a compact Logic Tensor Network.

    Returns
    -------
    nn.Module
        Configured LTN module.
    """
    return LogicTensorNetwork()


def example_input() -> Tensor:
    """Create object vector examples.

    Returns
    -------
    Tensor
        Example objects with shape ``(2, 5, 6)``.
    """
    return torch.randn(2, 5, 6)
