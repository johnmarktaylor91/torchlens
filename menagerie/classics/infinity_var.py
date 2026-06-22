"""Infinity-VAR compact bitwise visual autoregressive model.

Paper: Han et al., 2025, "Infinity: Scaling Bitwise AutoRegressive Modeling for
High-Resolution Image Synthesis".

Infinity redefines visual autoregression as bitwise prediction over an
effectively infinite visual vocabulary and adds bitwise self-correction.  This
compact reconstruction keeps text conditioning, causal VAR tokens, bit logits,
and an iterative correction head over predicted bit planes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class BitwiseVARBlock(nn.Module):
    """Causal transformer block for bitwise visual tokens."""

    def __init__(self, dim: int = 48, heads: int = 4) -> None:
        """Initialize attention and feed-forward layers."""

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked self-attention to visual bit tokens."""

        length = x.shape[1]
        mask = torch.full((length, length), float("-inf"), device=x.device).triu(1)
        x = self.norm1(x + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0])
        return self.norm2(x + self.ffn(x))


class InfinityVAR(nn.Module):
    """Compact Infinity-style bitwise autoregressive image generator."""

    def __init__(self, vocab: int = 128, dim: int = 48, tokens: int = 16, bits: int = 8) -> None:
        """Initialize token, text, bit, and self-correction heads."""

        super().__init__()
        self.tokens = tokens
        self.bits = bits
        self.text = nn.Embedding(vocab, dim)
        self.bit_in = nn.Linear(bits, dim)
        self.pos = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([BitwiseVARBlock(dim) for _ in range(2)])
        self.bit_logits = nn.Linear(dim, bits)
        self.correction = nn.Sequential(nn.Linear(bits + dim, dim), nn.GELU(), nn.Linear(dim, bits))
        self.patch = nn.Linear(bits, 3 * 4 * 4)

    def forward(self, text_ids: Tensor, seed_bits: Tensor) -> Tensor:
        """Predict corrected bit planes and decode them into image patches."""

        text_context = self.text(text_ids).mean(dim=1, keepdim=True)
        x = self.bit_in(seed_bits) + self.pos + text_context
        for block in self.blocks:
            x = block(x)
        logits = self.bit_logits(x)
        soft_bits = torch.sigmoid(logits)
        corrected = torch.sigmoid(self.correction(torch.cat([soft_bits, x], dim=-1)) + logits)
        patches = self.patch(corrected).view(text_ids.shape[0], self.tokens, 3, 4, 4)
        grid = patches.view(text_ids.shape[0], 4, 4, 3, 4, 4)
        return grid.permute(0, 3, 1, 4, 2, 5).reshape(text_ids.shape[0], 3, 16, 16)


def build() -> nn.Module:
    """Build the compact Infinity-VAR model."""

    return InfinityVAR().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return text ids and teacher-forced seed bit tokens."""

    return torch.randint(0, 128, (1, 6)), torch.rand(1, 16, 8)


MENAGERIE_ENTRIES = [
    ("Infinity-VAR", "build", "example_input", "2025", "GEN"),
]
