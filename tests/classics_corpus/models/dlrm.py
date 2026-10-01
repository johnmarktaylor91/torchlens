"""DLRM: Deep Learning Recommendation Model.

Naumov et al., 2019, arXiv:1906.00091.  DLRM processes dense features through a
bottom MLP, sparse categorical ids through embedding tables, forms explicit
pairwise dot-product feature interactions, and feeds those interactions to a top
MLP.  This compact version keeps that dense/sparse/interact/top-MLP pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactDLRM(nn.Module):
    """Compact DLRM recommender."""

    def __init__(self, dense_dim: int = 6, embed_dim: int = 8, tables: int = 3) -> None:
        """Initialize DLRM modules.

        Parameters
        ----------
        dense_dim:
            Number of dense input features.
        embed_dim:
            Shared embedding width.
        tables:
            Number of sparse embedding tables.
        """

        super().__init__()
        self.bottom = nn.Sequential(
            nn.Linear(dense_dim, 16), nn.ReLU(), nn.Linear(16, embed_dim), nn.ReLU()
        )
        self.embeddings = nn.ModuleList([nn.Embedding(32, embed_dim) for _ in range(tables)])
        features = tables + 1
        interactions = features * (features - 1) // 2
        self.top = nn.Sequential(
            nn.Linear(embed_dim + interactions, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict recommendation score.

        Parameters
        ----------
        inputs:
            Tuple ``(dense_features, sparse_ids)``.

        Returns
        -------
        torch.Tensor
            Logit score.
        """

        dense, sparse_ids = inputs
        dense_vec = self.bottom(dense)
        sparse_vecs = [emb(sparse_ids[:, idx]) for idx, emb in enumerate(self.embeddings)]
        feats = torch.stack([dense_vec, *sparse_vecs], dim=1)
        gram = torch.matmul(feats, feats.transpose(1, 2))
        tri = torch.tril_indices(feats.shape[1], feats.shape[1], offset=-1, device=feats.device)
        pairwise = gram[:, tri[0], tri[1]]
        return self.top(torch.cat([dense_vec, pairwise], dim=-1))


def build() -> nn.Module:
    """Build compact DLRM.

    Returns
    -------
    nn.Module
        Random-init DLRM reconstruction.
    """

    return CompactDLRM()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create dense and sparse recommendation features.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Dense features and sparse ids.
    """

    return torch.randn(2, 6), torch.randint(0, 32, (2, 3))


MENAGERIE_ENTRIES = [("dlrm", "build", "example_input", "2019", "REC")]
