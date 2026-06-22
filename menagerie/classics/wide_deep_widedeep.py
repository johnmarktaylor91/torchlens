"""Wide & Deep Learning for Recommender Systems.

Cheng et al., 2016.
Paper: https://arxiv.org/abs/1606.07792

The architecture jointly trains a memorization-oriented wide linear tower over
cross-product features and a generalization-oriented deep tower over dense
embeddings.  This compact reconstruction keeps both paths and fuses their logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactWideDeep(nn.Module):
    """Compact Wide & Deep recommender classifier."""

    def __init__(self, vocab: int = 32, fields: int = 4, embed_dim: int = 8) -> None:
        """Initialize wide and deep towers.

        Parameters
        ----------
        vocab:
            Cardinality per categorical field.
        fields:
            Number of categorical fields.
        embed_dim:
            Embedding dimension for the deep tower.
        """

        super().__init__()
        self.fields = fields
        self.wide = nn.Linear(fields + fields * (fields - 1) // 2, 1)
        self.embed = nn.Embedding(vocab, embed_dim)
        self.deep = nn.Sequential(
            nn.Linear(fields * embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def _cross_products(self, dense_ids: torch.Tensor) -> torch.Tensor:
        """Build compact cross-product features for the wide tower.

        Parameters
        ----------
        dense_ids:
            Normalized categorical ids with shape ``(batch, fields)``.

        Returns
        -------
        torch.Tensor
            Concatenated raw and pairwise-cross features.
        """

        crosses = []
        for left in range(self.fields):
            for right in range(left + 1, self.fields):
                crosses.append((dense_ids[:, left] * dense_ids[:, right]).unsqueeze(-1))
        return torch.cat([dense_ids, *crosses], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score categorical examples.

        Parameters
        ----------
        x:
            Categorical ids with shape ``(batch, fields)``.

        Returns
        -------
        torch.Tensor
            Fused wide-plus-deep logits.
        """

        dense_ids = x.float() / 31.0
        wide_logit = self.wide(self._cross_products(dense_ids))
        deep_logit = self.deep(self.embed(x).flatten(1))
        return wide_logit + deep_logit


def build() -> nn.Module:
    """Build a compact Wide & Deep model.

    Returns
    -------
    nn.Module
        Random-init Wide & Deep model.
    """

    return CompactWideDeep()


def example_input() -> torch.Tensor:
    """Create categorical recommender input.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape ``(2, 4)``.
    """

    return torch.randint(0, 32, (2, 4))


MENAGERIE_ENTRIES = [
    ("WideDeep-WideDeep", "build", "example_input", "2016", "DC"),
]
