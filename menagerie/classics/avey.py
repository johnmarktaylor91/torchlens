"""Avey: sequence-length-invariant ranker plus neural processor architecture.

Paper: "Don't Pay Attention" / Avey1, 2025.

Avey slices sequences into chunks, ranks relevant chunks for each local context,
and applies a neural processor rather than full quadratic self-attention or a
left-to-right recurrence.  This compact reconstruction keeps the ranker,
top-k retrieval, enricher, contextualizer, and fuser components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AveyProcessor(nn.Module):
    """Neural processor with enricher, contextualizer, and fuser units."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize processor projections."""

        super().__init__()
        self.enricher = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.contextualizer = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.fuser = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))

    def forward(self, local: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        """Fuse local and retrieved chunk representations."""

        enriched = self.enricher(local)
        context = self.contextualizer(torch.cat((local, retrieved), dim=-1))
        return self.fuser(torch.cat((enriched, context), dim=-1))


class Avey(nn.Module):
    """Compact Avey language model."""

    def __init__(self, vocab: int = 256, dim: int = 48, chunk: int = 4, top_k: int = 2) -> None:
        """Initialize embedding, ranker, processor, and language head."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.ranker = nn.Linear(dim, dim, bias=False)
        self.processor = AveyProcessor(dim)
        self.head = nn.Linear(dim, vocab)
        self.chunk = chunk
        self.top_k = top_k

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Score tokens using chunk ranking and neural processing."""

        x = self.embed(tokens.clamp(0, self.embed.num_embeddings - 1))
        bsz, length, dim = x.shape
        pad = (self.chunk - length % self.chunk) % self.chunk
        x_pad = F.pad(x, (0, 0, 0, pad))
        chunks = x_pad.reshape(bsz, -1, self.chunk, dim).mean(dim=2)
        scores = torch.matmul(self.ranker(chunks), chunks.transpose(1, 2)) / (dim**0.5)
        weights, index = torch.topk(scores, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        bank = chunks.unsqueeze(1).expand(-1, chunks.shape[1], -1, -1)
        gathered = torch.gather(bank, 2, index.unsqueeze(-1).expand(-1, -1, -1, dim))
        retrieved = (gathered * weights.unsqueeze(-1)).sum(dim=2)
        processed = self.processor(chunks, retrieved)
        token_ctx = processed.repeat_interleave(self.chunk, dim=1)[:, :length]
        return self.head(x + token_ctx)


def build() -> nn.Module:
    """Build compact Avey."""

    return Avey()


def example_input() -> torch.Tensor:
    """Return token IDs for Avey."""

    return torch.randint(0, 256, (1, 16))


MENAGERIE_ENTRIES = [
    ("Avey", "build", "example_input", "2025", "sequence"),
]
