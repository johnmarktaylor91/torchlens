"""CosyVoice 2 combined LLM and chunk-aware flow compact reconstruction.

Paper: CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language
Models (2024).

CosyVoice 2 keeps the LLM-to-speech-token front end and adds chunk-aware causal
flow matching for streaming acoustic generation.  This compact model exposes
finite-scalar quantized speech-token conditioning, a causal LM context, and a
block-causal flow transformer over acoustic chunks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FSQ(nn.Module):
    """Finite-scalar quantization surrogate for CosyVoice 2 speech tokens."""

    def __init__(self, dim: int = 48, levels: int = 5) -> None:
        """Initialize projection and quantization level count."""

        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.levels = levels

    def forward(self, x: Tensor) -> Tensor:
        """Quantize continuous token features to finite scalar levels."""

        y = torch.tanh(self.proj(x))
        return torch.round((self.levels - 1) * y) / (self.levels - 1)


class CosyVoice2LLMFlow(nn.Module):
    """Compact combined CosyVoice 2 LLM and streaming flow model."""

    def __init__(self, vocab: int = 256, mel_bins: int = 32, dim: int = 48) -> None:
        """Initialize token LM, FSQ conditioning, chunk mask, and flow head."""

        super().__init__()
        self.text = nn.Embedding(vocab, dim)
        self.semantic = nn.Linear(dim, dim)
        self.fsq = FSQ(dim)
        self.mel = nn.Linear(mel_bins, dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.lm = nn.TransformerEncoder(layer, 1)
        self.flow = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(dim, mel_bins)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Predict chunk-aware acoustic flow from text ids and noisy mel chunks."""

        text_ids, noisy_mel = inputs
        lm_context = self.lm(self.text(text_ids))
        semantic = self.fsq(self.semantic(lm_context)).mean(dim=1, keepdim=True)
        hidden = self.mel(noisy_mel) + semantic
        length = hidden.shape[1]
        mask = torch.full((length, length), float("-inf"), device=hidden.device).triu(3)
        return self.head(self.flow(hidden, mask=mask))


def build() -> nn.Module:
    """Build a compact random-init CosyVoice 2 LLM-flow model."""

    return CosyVoice2LLMFlow().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return text token ids and noisy acoustic chunks."""

    return (torch.randint(0, 256, (1, 8)), torch.randn(1, 10, 32))


MENAGERIE_ENTRIES = [
    ("cosyvoice2_llm_flow", "build", "example_input", "2024", "DC"),
]
