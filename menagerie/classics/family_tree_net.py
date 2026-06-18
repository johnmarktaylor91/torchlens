"""Hinton family-trees distributed-representation net, 1986, as a shared-embedding MLP.

Paper: Hinton 1986, "Learning Distributed Representations of Concepts."
Person and relation embeddings are bound through a bottleneck hidden layer to
produce another person, illustrating emergent distributed role representations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    (
        "Hinton family-trees distributed-representation net",
        "build",
        "example_input",
        "1986",
        "CB",
    )
]


class FamilyTreeNet(nn.Module):
    """Shared-embedding family-relation predictor."""

    def __init__(self, n_people: int = 24, n_relations: int = 12, embed_dim: int = 8) -> None:
        """Initialize person and relation embeddings.

        Parameters
        ----------
        n_people
            Number of person symbols.
        n_relations
            Number of relation symbols.
        embed_dim
            Embedding and bottleneck width.
        """
        super().__init__()
        self.person = nn.Embedding(n_people, embed_dim)
        self.relation = nn.Embedding(n_relations, embed_dim)
        self.hidden = nn.Linear(embed_dim * 3, embed_dim)
        self.readout = nn.Linear(embed_dim, n_people)

    def forward(self, pair: Tensor) -> Tensor:
        """Predict a target person from a person-relation pair.

        Parameters
        ----------
        pair
            Integer tensor of shape ``(batch, 2)`` containing person and relation indices.

        Returns
        -------
        Tensor
            Target-person logits.
        """
        person_vec = self.person(pair[:, 0])
        relation_vec = self.relation(pair[:, 1])
        bound = torch.cat((person_vec, relation_vec, person_vec * relation_vec), dim=-1)
        hidden = torch.sigmoid(self.hidden(bound))
        return self.readout(hidden)


def build() -> nn.Module:
    """Build a small family-tree network.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return FamilyTreeNet()


def example_input() -> Tensor:
    """Create person-relation index examples.

    Returns
    -------
    Tensor
        Example index tensor with shape ``(3, 2)``.
    """
    return torch.tensor([[0, 1], [4, 3], [11, 7]], dtype=torch.long)
