"""Shared helpers for TorchLens Total Audit notebooks."""

from __future__ import annotations

import torch
from torch import nn


class TinyTransformer(nn.Module):
    """Small deterministic transformer-like model for recipe notebooks."""

    def __init__(
        self,
        *,
        vocab_size: int = 11,
        d_model: int = 8,
        n_heads: int = 2,
        max_len: int = 6,
    ) -> None:
        """Initialize embeddings, attention, MLP, and language-model head.

        Parameters
        ----------
        vocab_size:
            Number of token IDs accepted by the embedding table.
        d_model:
            Hidden width used by the block.
        n_heads:
            Attention head count.
        max_len:
            Maximum sequence length supported by positional embeddings.
        """

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_len, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run one transformer-style block over token IDs.

        Parameters
        ----------
        tokens:
            Integer token IDs with shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Per-position logits with shape ``(batch, seq_len, vocab_size)``.
        """

        positions = self.pos_embed[: tokens.shape[1]].unsqueeze(0)
        hidden = self.token_embed(tokens) + positions
        attended, _weights = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = self.ln_1(hidden + attended)
        hidden = self.ln_2(hidden + self.mlp(hidden))
        return self.head(hidden)


def tiny_model(seed: int = 0) -> nn.Module:
    """Return a deterministic three-layer MLP for audit notebooks.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    nn.Module
        Three-layer MLP with deterministic initial weights.
    """

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def tiny_transformer(seed: int = 0) -> TinyTransformer:
    """Return a deterministic tiny transformer-like model.

    Parameters
    ----------
    seed:
        Torch RNG seed used before constructing the model.

    Returns
    -------
    TinyTransformer
        Evaluation-mode transformer-like module with deterministic weights.
    """

    torch.manual_seed(seed)
    return TinyTransformer().eval()
