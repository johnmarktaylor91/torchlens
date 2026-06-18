"""Neuro-Symbolic Concept Learner, 2019, Mao et al., "The Neuro-Symbolic Concept Learner".

Paper: Mao 2019, "The Neuro-Symbolic Concept Learner."
This small concept executor applies differentiable concept filters, relation scores,
and soft logical composition over object embeddings for a fixed program fragment.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConceptExecutor(nn.Module):
    """Differentiable fixed-program concept executor."""

    def __init__(self, object_dim: int = 6, n_concepts: int = 4) -> None:
        """Initialize concept and relation embeddings.

        Parameters
        ----------
        object_dim
            Object embedding width.
        n_concepts
            Number of learnable concept filters.
        """
        super().__init__()
        self.concepts = nn.Parameter(torch.randn(n_concepts, object_dim) * 0.2)
        self.relation = nn.Linear(object_dim * 2, 1)

    def forward(self, objects: Tensor) -> Tensor:
        """Execute a soft filter-relate-exist-count program.

        Parameters
        ----------
        objects
            Object embeddings of shape ``(batch, objects, object_dim)``.

        Returns
        -------
        Tensor
            Concatenated existence and count-style answers.
        """
        obj_norm = torch.nn.functional.normalize(objects, dim=-1)
        concept_norm = torch.nn.functional.normalize(self.concepts, dim=-1)
        concept_scores = torch.sigmoid(obj_norm @ concept_norm.T)
        filtered = concept_scores[..., 0] * (1.0 - concept_scores[..., 1])
        left = objects[:, :, None, :].expand(-1, -1, objects.shape[1], -1)
        right = objects[:, None, :, :].expand(-1, objects.shape[1], -1, -1)
        rel = torch.sigmoid(self.relation(torch.cat((left, right), dim=-1))).squeeze(-1)
        related = torch.sum(filtered[:, :, None] * rel, dim=1)
        exists = related.amax(dim=-1, keepdim=True)
        count = related.sum(dim=-1, keepdim=True)
        return torch.cat((exists, count), dim=-1)


MENAGERIE_ENTRIES = [
    ("Neuro-Symbolic Concept Learner (NS-CL)", "build", "example_input", "2019", "CD")
]


def build() -> nn.Module:
    """Build a compact NS-CL-style executor.

    Returns
    -------
    nn.Module
        Configured concept executor.
    """
    return ConceptExecutor()


def example_input() -> Tensor:
    """Create object embedding examples.

    Returns
    -------
    Tensor
        Example objects with shape ``(2, 5, 6)``.
    """
    return torch.randn(2, 5, 6)
