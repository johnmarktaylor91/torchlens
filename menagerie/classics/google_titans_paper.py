"""Titans: Learning to Memorize at Test Time.

Behrouz et al., 2025.
Paper: https://arxiv.org/abs/2501.00663

Titans augments short-term attention with a neural long-term memory module.  The
paper studies memory placement variants: Memory as Context (MAC), Memory as Gate
(MAG), and Memory as Layer (MAL).  This compact reconstruction keeps a
differentiable surprise-gated associative memory and exposes all three placement
patterns under separate menagerie entries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    """Surprise-gated associative long-term memory module."""

    def __init__(self, dim: int) -> None:
        """Initialize key, value, query, and surprise projections.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        """

        super().__init__()
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.query = nn.Linear(dim, dim, bias=False)
        self.surprise = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read from a differentiable test-time memory built from the sequence.

        Parameters
        ----------
        x:
            Token states with shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Retrieved long-term memory states.
        """

        keys = F.normalize(self.key(x), dim=-1)
        values = self.value(x)
        novelty = torch.sigmoid(self.surprise(x))
        stored = values * novelty
        scores = torch.matmul(self.query(x), keys.transpose(-1, -2)) / (x.shape[-1] ** 0.5)
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device).tril()
        scores = scores.masked_fill(causal.unsqueeze(0) == 0, -1.0e4)
        read = torch.softmax(scores, dim=-1) @ stored
        return self.out(read)


class TitansBlock(nn.Module):
    """Titans block with selectable memory placement."""

    def __init__(self, dim: int, heads: int, mode: str) -> None:
        """Initialize a Titans block.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        heads:
            Number of short-term attention heads.
        mode:
            One of ``"context"``, ``"gate"``, or ``"layer"``.
        """

        super().__init__()
        self.mode = mode
        self.memory = NeuralMemory(dim)
        self.memory_token = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def _short_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal short-term self-attention.

        Parameters
        ----------
        x:
            Token states.

        Returns
        -------
        torch.Tensor
            Attention output.
        """

        mask = torch.ones(x.shape[1], x.shape[1], device=x.device).triu(1).bool()
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a Titans memory-placement block.

        Parameters
        ----------
        x:
            Token states with shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Updated token states.
        """

        mem = self.memory(self.norm(x))
        if self.mode == "context":
            context = self.memory_token(mem.mean(dim=1, keepdim=True))
            attended = self._short_attention(torch.cat([context, x], dim=1))[:, 1:]
            x = x + attended
        elif self.mode == "gate":
            attended = self._short_attention(self.norm(x))
            gate = torch.sigmoid(self.gate(torch.cat([attended, mem], dim=-1)))
            x = x + gate * mem + (1.0 - gate) * attended
        else:
            x = x + mem
            x = x + self._short_attention(self.norm(x))
        return x + self.ffn(x)


class CompactTitansLM(nn.Module):
    """Compact language model for one Titans memory-placement variant."""

    def __init__(self, mode: str, vocab: int = 64, dim: int = 32) -> None:
        """Initialize embeddings, Titans blocks, and LM head.

        Parameters
        ----------
        mode:
            Memory placement variant.
        vocab:
            Vocabulary size.
        dim:
            Hidden dimension.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([TitansBlock(dim, 4, mode), TitansBlock(dim, 4, mode)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict token logits.

        Parameters
        ----------
        ids:
            Token ids with shape ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build_memory_as_context() -> nn.Module:
    """Build Titans Memory-as-Context model.

    Returns
    -------
    nn.Module
        Random-init MAC model.
    """

    return CompactTitansLM("context")


def build_memory_as_gate() -> nn.Module:
    """Build Titans Memory-as-Gate model.

    Returns
    -------
    nn.Module
        Random-init MAG model.
    """

    return CompactTitansLM("gate")


def build_memory_as_layer() -> nn.Module:
    """Build Titans Memory-as-Layer model.

    Returns
    -------
    nn.Module
        Random-init MAL model.
    """

    return CompactTitansLM("layer")


def example_input() -> torch.Tensor:
    """Create a compact token sequence.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape ``(1, 10)``.
    """

    return torch.randint(0, 64, (1, 10))


MENAGERIE_ENTRIES = [
    (
        "google:Titans-paper-memory-as-context",
        "build_memory_as_context",
        "example_input",
        "2025",
        "DC",
    ),
    ("google:Titans-paper-memory-as-gate", "build_memory_as_gate", "example_input", "2025", "DC"),
    ("google:Titans-paper-memory-as-layer", "build_memory_as_layer", "example_input", "2025", "DC"),
]
