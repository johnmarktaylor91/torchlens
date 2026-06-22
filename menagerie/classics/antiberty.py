"""AntiBERTy: antibody-specific BERT-style masked language model.

AntiBERTy is described by its public repository as an antibody-specific
transformer language model trained on hundreds of millions of natural antibody
sequences. Public summaries note that it follows the BERT encoder architecture.
This compact reconstruction keeps residue embeddings, positional embeddings,
stacked bidirectional transformer encoder blocks, and a masked-LM head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AntiBERTyMini(nn.Module):
    """Compact antibody BERT encoder."""

    def __init__(self, vocab: int = 32, dim: int = 64, layers: int = 2) -> None:
        """Initialize antibody token embeddings and encoder.

        Parameters
        ----------
        vocab:
            Amino-acid/token vocabulary size.
        dim:
            Transformer hidden size.
        layers:
            Number of encoder layers.
        """
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.position = nn.Parameter(torch.randn(1, 32, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=4 * dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.mlm_head = nn.Linear(dim, vocab)
        self.binding_head = nn.Linear(dim, 1)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Encode antibody sequence tokens.

        Parameters
        ----------
        ids:
            Antibody residue token ids with shape ``(batch, length)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Masked-language logits and per-residue binding logits.
        """
        x = self.token(ids) + self.position[:, : ids.shape[1]]
        encoded = self.norm(self.encoder(x))
        return self.mlm_head(encoded), self.binding_head(encoded).squeeze(-1)


def build() -> nn.Module:
    """Build a compact AntiBERTy reconstruction.

    Returns
    -------
    nn.Module
        Random-initialized antibody language model.
    """
    return AntiBERTyMini()


def example_input() -> Tensor:
    """Return antibody residue token ids.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 24)``.
    """
    return torch.randint(0, 32, (1, 24))


MENAGERIE_ENTRIES = [
    ("AntiBERTy", "build", "example_input", "2021", "DE"),
]
