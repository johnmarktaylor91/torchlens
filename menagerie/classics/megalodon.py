"""Megalodon: CEMA and normalized attention for long-context language models.

Paper: "Megalodon: Efficient LLM Pretraining and Inference with Unlimited
Context Length", Ma et al., 2024.

The compact reconstruction keeps Megalodon's load-bearing block ingredients:
complex exponential moving average (CEMA), timestep normalization, normalized
local attention, and a gated two-hop residual feed-forward path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEMA(nn.Module):
    """Complex exponential moving average token mixer."""

    def __init__(self, dim: int) -> None:
        """Initialize complex decay and input projection."""

        super().__init__()
        self.log_decay = nn.Parameter(torch.full((dim,), -1.0))
        self.freq = nn.Parameter(torch.linspace(0.0, 1.0, dim))
        self.in_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply real part of a complex EMA recurrence."""

        u = self.in_proj(x)
        real = torch.zeros_like(u[:, 0])
        imag = torch.zeros_like(real)
        decay = torch.sigmoid(self.log_decay)
        cos = torch.cos(self.freq)
        sin = torch.sin(self.freq)
        outs = []
        for t in range(x.shape[1]):
            new_real = decay * (real * cos - imag * sin) + u[:, t]
            new_imag = decay * (real * sin + imag * cos)
            real, imag = new_real, new_imag
            outs.append(real)
        return torch.stack(outs, dim=1)


class TimestepNorm(nn.Module):
    """Normalize over the autoregressive timestep axis."""

    def __init__(self, eps: float = 1e-5) -> None:
        """Initialize timestep norm."""

        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize sequence positions for each channel."""

        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)


class MegalodonBlock(nn.Module):
    """Compact Megalodon block."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize CEMA, attention, and gated residual path."""

        super().__init__()
        self.norm = TimestepNorm()
        self.cema = CEMA(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Megalodon token mixing."""

        y = self.norm(x)
        y = y + self.cema(y)
        attn, _ = self.attn(F.normalize(y, dim=-1), F.normalize(y, dim=-1), y)
        x = x + attn
        return x + torch.sigmoid(self.gate(x)) * self.ff(self.norm(x))


class MegalodonCompact(nn.Module):
    """Compact language model with one Megalodon block."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize embeddings and LM head."""

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.block = MegalodonBlock(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict token logits."""

        return self.head(self.block(self.embed(ids)))


def build() -> nn.Module:
    """Build compact Megalodon."""

    return MegalodonCompact()


def example_input() -> torch.Tensor:
    """Return token IDs."""

    return torch.randint(0, 128, (1, 16))


MENAGERIE_ENTRIES = [("Megalodon", "build", "example_input", "2024", "E7")]
