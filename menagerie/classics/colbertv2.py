"""ColBERTv2: late-interaction retrieval with residual compression.

Santhanam et al., NAACL 2022, "Effective and Efficient Retrieval via Lightweight
Late Interaction."  ColBERTv2 independently encodes query and document tokens,
normalizes projected token vectors, reconstructs compressed document token
vectors from centroids plus residuals, and scores with MaxSim late interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactColBERTv2(nn.Module):
    """Compact ColBERTv2 scorer with residual-compressed document tokens."""

    def __init__(self, vocab: int = 512, dim: int = 48, codebook: int = 8) -> None:
        """Initialize the compact retrieval model.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        dim:
            Encoder and projection width.
        codebook:
            Number of residual-compression centroids.
        """

        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, 4, 4 * dim, batch_first=True, norm_first=True)
        self.embed = nn.Embedding(vocab, dim)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.centroids = nn.Parameter(torch.randn(codebook, dim) * 0.02)
        self.residual = nn.Linear(dim, dim, bias=False)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Encode ids into normalized token vectors.

        Parameters
        ----------
        ids:
            Token ids of shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Normalized token embeddings.
        """

        return nn.functional.normalize(self.proj(self.encoder(self.embed(ids))), dim=-1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Score query/document pairs with MaxSim late interaction.

        Parameters
        ----------
        inputs:
            Tuple ``(query_ids, doc_ids, centroid_ids)``.

        Returns
        -------
        torch.Tensor
            Retrieval score per batch item.
        """

        query_ids, doc_ids, centroid_ids = inputs
        query = self.encode(query_ids)
        doc_raw = self.encode(doc_ids)
        doc = nn.functional.normalize(
            self.centroids[centroid_ids] + 0.25 * self.residual(doc_raw), dim=-1
        )
        maxsim = torch.matmul(query, doc.transpose(-1, -2)).max(dim=-1).values
        return maxsim.sum(dim=-1, keepdim=True)


def build() -> nn.Module:
    """Build a compact ColBERTv2 scorer.

    Returns
    -------
    nn.Module
        Random-init ColBERTv2 reconstruction.
    """

    return CompactColBERTv2()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create query/document/centroid ids.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Query ids, document ids, and centroid assignment ids.
    """

    return (
        torch.randint(0, 512, (1, 6)),
        torch.randint(0, 512, (1, 16)),
        torch.randint(0, 8, (1, 16)),
    )


MENAGERIE_ENTRIES = [("ColBERTv2", "build", "example_input", "2022", "LM")]
