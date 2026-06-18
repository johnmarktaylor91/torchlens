"""MELT memory-efficient looped transformer, 2026, Vendrell et al.

Paper: 2026, "Memory-Efficient Looped Transformer."
A single shared transformer block is applied repeatedly while a gated cache summarizes
keys and values; this compact version omits specialized constant-memory kernels.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MELT(nn.Module):
    """Shared-block transformer with gated recurrent key-value cache."""

    def __init__(self, vocab_size: int = 256, dim: int = 32, loops: int = 3) -> None:
        """Initialize token embeddings, shared block, cache gate, and readout.

        Parameters
        ----------
        vocab_size
            Number of token ids.
        dim
            Embedding width.
        loops
            Number of block reuse iterations.
        """
        super().__init__()
        self.loops = loops
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.cache_gate = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply the shared transformer block with a gated cache.

        Parameters
        ----------
        tokens
            Token ids with shape ``(batch, 128)``.

        Returns
        -------
        Tensor
            Token logits.
        """
        x = self.embedding(tokens)
        cache = torch.zeros_like(x)
        for _ in range(self.loops):
            mixed, _ = self.attn(self.norm1(x), self.norm1(x + cache), self.norm1(x + cache))
            x = x + mixed
            new_cache = self.norm2(x)
            gate = torch.sigmoid(self.cache_gate(torch.cat((cache, new_cache), dim=-1)))
            cache = gate * new_cache + (1.0 - gate) * cache
            x = x + self.ff(cache)
        return self.out(x)


MENAGERIE_ENTRIES = [
    ("MELT (Memory-Efficient Looped Transformer)", "build", "example_input", "2026", "DA")
]


def build() -> nn.Module:
    """Build a MELT module.

    Returns
    -------
    nn.Module
        Configured MELT module.
    """
    return MELT()


def example_input() -> Tensor:
    """Create token ids.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 128)``.
    """
    return torch.randint(0, 256, (1, 128), dtype=torch.long)
