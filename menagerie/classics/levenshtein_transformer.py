"""Levenshtein Transformer insertion-deletion decoder.

Paper: "Levenshtein Transformer", Gu et al., NeurIPS 2019.

LevT replaces strictly left-to-right decoding with alternating deletion and
insertion policies. This compact model keeps a Transformer encoder, a draft
decoder state, a deletion head over existing tokens, and insertion heads for
placeholder counts and token choices.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TinyAttention(nn.Module):
    """Small multi-head attention layer."""

    def __init__(self, d_model: int = 32, n_heads: int = 4) -> None:
        """Initialize query, key, value, and output projections.

        Parameters
        ----------
        d_model:
            Hidden width.
        n_heads:
            Number of attention heads.
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """Apply attention from query states to key/value states.

        Parameters
        ----------
        query:
            Query tensor with shape ``(batch, query_time, d_model)``.
        key_value:
            Key/value tensor with shape ``(batch, kv_time, d_model)``.

        Returns
        -------
        Tensor
            Attention output with the same leading shape as ``query``.
        """
        batch, query_time, _ = query.shape
        kv_time = key_value.shape[1]
        q = self.q(query).view(batch, query_time, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(key_value).view(batch, kv_time, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(key_value).view(batch, kv_time, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).reshape(batch, query_time, -1)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Transformer block with optional cross-attention."""

    def __init__(self, d_model: int = 32) -> None:
        """Initialize attention, cross-attention, and feed-forward layers.

        Parameters
        ----------
        d_model:
            Hidden width.
        """
        super().__init__()
        self.self_attn = TinyAttention(d_model)
        self.cross_attn = TinyAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: Tensor, memory: Tensor | None = None) -> Tensor:
        """Apply a Transformer block.

        Parameters
        ----------
        x:
            Input sequence states.
        memory:
            Optional encoder memory for cross-attention.

        Returns
        -------
        Tensor
            Updated sequence states.
        """
        x = self.norm1(x + self.self_attn(x, x))
        if memory is not None:
            x = self.norm2(x + self.cross_attn(x, memory))
        x = self.norm3(x + self.ff(x))
        return x


class LevenshteinTransformer(nn.Module):
    """Compact Levenshtein Transformer with deletion and insertion policies."""

    def __init__(self, vocab_size: int = 96, d_model: int = 32) -> None:
        """Initialize embeddings, encoder, decoder, and policy heads.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        d_model:
            Hidden width.
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(24, d_model)
        self.encoder = TransformerBlock(d_model)
        self.decoder = TransformerBlock(d_model)
        self.delete_head = nn.Linear(d_model, 2)
        self.placeholder_head = nn.Linear(d_model, 4)
        self.token_head = nn.Linear(d_model, vocab_size)

    def _embed(self, ids: Tensor) -> Tensor:
        """Embed tokens plus learned positions.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Embedded sequence states.
        """
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.token(ids) + self.position(positions)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute compact LevT policy logits.

        Parameters
        ----------
        ids:
            Long source/draft token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Concatenated mean logits from deletion, placeholder, and token heads.
        """
        memory = self.encoder(self._embed(ids))
        draft = self.decoder(self._embed(ids), memory)
        delete_logits = self.delete_head(draft).mean(dim=1)
        placeholder_logits = self.placeholder_head(draft).mean(dim=1)
        token_logits = self.token_head(draft).mean(dim=1)
        return torch.cat([delete_logits, placeholder_logits, token_logits], dim=-1)


def build() -> nn.Module:
    """Build a compact Levenshtein Transformer.

    Returns
    -------
    nn.Module
        Random-initialized Levenshtein Transformer.
    """
    return LevenshteinTransformer()


def example_input() -> Tensor:
    """Return example source/draft token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 12)``.
    """
    return torch.randint(0, 96, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [("Levenshtein Transformer", "build", "example_input", "2019", "DC")]
