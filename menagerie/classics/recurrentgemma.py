"""RecurrentGemma / Griffin hybrid sequence model.

Paper: Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models,
De et al. 2024; RecurrentGemma is Google's Gemma-family Griffin implementation.

The compact model alternates RG-LRU recurrent temporal mixing, local sliding
window attention, RMSNorm, and gated MLP blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize scale parameters."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension."""

        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RGLRU(nn.Module):
    """Real-gated linear recurrent unit with diagonal recurrence."""

    def __init__(self, dim: int) -> None:
        """Initialize RG-LRU gates and recurrence."""

        super().__init__()
        self.in_proj = nn.Linear(dim, dim)
        self.input_gate = nn.Linear(dim, dim)
        self.recurrence_gate = nn.Linear(dim, dim)
        self.a_param = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan the sequence with input-dependent gates."""

        a = torch.sigmoid(self.a_param)
        state = torch.zeros(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        outs = []
        values = self.in_proj(x)
        i_gate = torch.sigmoid(self.input_gate(x))
        r_gate = torch.sigmoid(self.recurrence_gate(x))
        for step in range(x.shape[1]):
            state = a * r_gate[:, step] * state + i_gate[:, step] * values[:, step]
            outs.append(state)
        return torch.stack(outs, dim=1)


class LocalMQA(nn.Module):
    """Local sliding-window multi-query attention."""

    def __init__(self, dim: int = 32, heads: int = 4, window: int = 4) -> None:
        """Initialize local attention projections."""

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, self.head_dim)
        self.v = nn.Linear(dim, self.head_dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal local MQA."""

        batch, tokens, _ = x.shape
        q = self.q(x).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        idx = torch.arange(tokens, device=x.device)
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1) < self.window) & (
            idx.unsqueeze(0) >= idx.unsqueeze(1)
        )
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e4)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.o(out.transpose(1, 2).reshape(batch, tokens, -1))


class GriffinBlock(nn.Module):
    """Hybrid temporal mixing and gated MLP block."""

    def __init__(self, dim: int = 32, use_attention: bool = False) -> None:
        """Initialize the block."""

        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = LocalMQA(dim) if use_attention else RGLRU(dim)
        self.norm2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, dim * 2)
        self.up = nn.Linear(dim, dim * 2)
        self.down = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal mixer and gated feed-forward network."""

        x = x + self.mixer(self.norm1(x))
        y = self.norm2(x)
        x = x + self.down(F.gelu(self.gate(y)) * self.up(y))
        return x


class RecurrentGemma(nn.Module):
    """Compact RecurrentGemma language model."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize embeddings, Griffin blocks, and language head."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [GriffinBlock(dim, False), GriffinBlock(dim, True), GriffinBlock(dim, False)]
        )
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict token logits."""

        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build() -> nn.Module:
    """Build compact RecurrentGemma."""

    return RecurrentGemma()


def example_input() -> torch.Tensor:
    """Return token ids."""

    return torch.randint(0, 128, (1, 8))


MENAGERIE_ENTRIES = [("recurrentgemma", "build", "example_input", "2024", "sequence/recurrent")]
