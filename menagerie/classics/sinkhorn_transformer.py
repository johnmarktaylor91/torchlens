"""Sinkhorn Transformer compact faithful reconstruction.

Tay et al. 2020, "Sparse Sinkhorn Attention".

The model learns a latent permutation over sequence buckets with Sinkhorn
normalization, sorts keys/values by that soft permutation, applies local
bucketed attention, and unsorts the result. This compact version keeps that
sorting-attention loop with small random-init dimensions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def sinkhorn(logits: Tensor, iterations: int = 4) -> Tensor:
    """Normalize logits into a doubly stochastic matrix.

    Parameters
    ----------
    logits:
        Square matrix logits.
    iterations:
        Number of row/column normalization rounds.

    Returns
    -------
    Tensor
        Doubly stochastic soft permutation.
    """
    scores = logits
    for _ in range(iterations):
        scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        scores = scores - torch.logsumexp(scores, dim=-2, keepdim=True)
    return scores.exp()


class SinkhornBlock(nn.Module):
    """Transformer block with learned bucket sorting attention."""

    def __init__(self, dim: int = 48, heads: int = 4, bucket_size: int = 4) -> None:
        """Initialize attention and feed-forward layers.

        Parameters
        ----------
        dim:
            Token dimension.
        heads:
            Number of attention heads.
        bucket_size:
            Number of tokens per sorted bucket.
        """
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.head_dim = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.sort_net = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply sorted sparse attention.

        Parameters
        ----------
        x:
            Token tensor with shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Updated token tensor.
        """
        batch, time, dim = x.shape
        buckets = time // self.bucket_size
        y = self.norm(x)
        q, k, v = self.to_qkv(y).chunk(3, dim=-1)
        bucket_repr = y.reshape(batch, buckets, self.bucket_size, dim).mean(dim=2)
        sort_logits = self.sort_net(bucket_repr).squeeze(-1)
        logits = sort_logits.unsqueeze(1) - sort_logits.unsqueeze(2)
        perm = sinkhorn(logits)
        k_b = k.reshape(batch, buckets, self.bucket_size, dim)
        v_b = v.reshape(batch, buckets, self.bucket_size, dim)
        k_sorted = torch.einsum("bij,bjtd->bitd", perm, k_b).reshape(batch, time, dim)
        v_sorted = torch.einsum("bij,bjtd->bitd", perm, v_b).reshape(batch, time, dim)
        qh = q.reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)
        kh = k_sorted.reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)
        vh = v_sorted.reshape(batch, time, self.heads, self.head_dim).transpose(1, 2)
        qh = qh.reshape(batch, self.heads, buckets, self.bucket_size, self.head_dim)
        kh = kh.reshape(batch, self.heads, buckets, self.bucket_size, self.head_dim)
        vh = vh.reshape(batch, self.heads, buckets, self.bucket_size, self.head_dim)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        local = torch.matmul(attn, vh).reshape(batch, self.heads, time, self.head_dim)
        local = local.transpose(1, 2).reshape(batch, time, dim)
        x = x + self.out(local)
        return x + self.ff(x)


class SinkhornTransformerCompact(nn.Module):
    """Small language-model-style Sinkhorn Transformer."""

    def __init__(self, vocab: int = 64, dim: int = 48, layers: int = 2) -> None:
        """Initialize embeddings, Sinkhorn blocks, and output head.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        layers:
            Number of blocks.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 16, dim))
        self.blocks = nn.ModuleList([SinkhornBlock(dim=dim) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run compact Sinkhorn Transformer.

        Parameters
        ----------
        tokens:
            Token ids.

        Returns
        -------
        Tensor
            Vocabulary logits.
        """
        x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build() -> nn.Module:
    """Build a compact random-init Sinkhorn Transformer.

    Returns
    -------
    nn.Module
        Compact Sinkhorn Transformer.
    """
    return SinkhornTransformerCompact()


def example_input() -> Tensor:
    """Return token ids.

    Returns
    -------
    Tensor
        Token tensor.
    """
    return torch.randint(0, 64, (1, 16))


MENAGERIE_ENTRIES = [("Sinkhorn-Transformer", "build", "example_input", "2020", "E7")]
