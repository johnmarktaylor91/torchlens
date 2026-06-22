"""Linformer language model with low-rank causal self-attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalLinformerAttention(nn.Module):
    """Low-rank Linformer attention with an autoregressive causal mask."""

    def __init__(self, dim: int = 32, heads: int = 4, seq_len: int = 16, rank: int = 8) -> None:
        """Initialize projections and Linformer E/F sequence projections.

        Parameters
        ----------
        dim:
            Hidden dimension.
        heads:
            Number of attention heads.
        seq_len:
            Maximum sequence length.
        rank:
            Low-rank projected key/value length.
        """
        super().__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.rank = rank
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.e_proj = nn.Parameter(torch.randn(seq_len, rank) * 0.02)
        self.f_proj = nn.Parameter(torch.randn(seq_len, rank) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Apply low-rank causal attention.

        Parameters
        ----------
        x:
            Token embeddings of shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Contextualized embeddings.
        """
        batch, time, _ = x.shape
        q = self.q(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        e = self.e_proj[:time]
        f = self.f_proj[:time]
        causal_support = torch.tril(torch.ones(time, time, dtype=torch.bool, device=x.device))
        causal_e = causal_support.float().unsqueeze(-1) * e.unsqueeze(0)
        causal_f = causal_support.float().unsqueeze(-1) * f.unsqueeze(0)
        low_k = torch.einsum("qtr,bhtd->bhqrd", causal_e, k)
        low_v = torch.einsum("qtr,bhtd->bhqrd", causal_f, v)
        scores = torch.einsum("bhtd,bhtrd->bhtr", q, low_k) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        y = torch.einsum("bhtr,bhtrd->bhtd", attn, low_v).transpose(1, 2).reshape(batch, time, -1)
        return self.out(y)


class LinformerLM(nn.Module):
    """Autoregressive Linformer language model."""

    def __init__(self, vocab: int = 64, dim: int = 32, seq_len: int = 16) -> None:
        """Initialize token embeddings, causal Linformer block, and LM head.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Hidden dimension.
        seq_len:
            Maximum sequence length.
        """
        super().__init__()
        self.token = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        self.attn = CausalLinformerAttention(dim=dim, seq_len=seq_len)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict next-token logits with causal low-rank attention.

        Parameters
        ----------
        tokens:
            Integer token ids ``(batch, time)``.

        Returns
        -------
        Tensor
            Vocabulary logits ``(batch, time, vocab)``.
        """
        x = self.token(tokens) + self.pos[:, : tokens.shape[1]]
        x = self.norm(x + self.attn(x))
        x = self.norm(x + self.ffn(x))
        return self.lm_head(x)


def build() -> nn.Module:
    """Build a compact Linformer LM.

    Returns
    -------
    nn.Module
        Causal Linformer language model.
    """
    return LinformerLM().eval()


def example_input() -> Tensor:
    """Return token ids for Linformer LM tracing.

    Returns
    -------
    Tensor
        Token ids.
    """
    return torch.randint(0, 64, (1, 16))


MENAGERIE_ENTRIES = [
    ("linformer_lm", "build", "example_input", "2020", "NLP/LLM/text"),
]
