"""CosyVoice Qwen2 language-model stage compact reconstruction.

Paper: CosyVoice / CosyVoice 2 scalable zero-shot TTS reports (Du et al.,
2024; CosyVoice 2, 2024).

CosyVoice uses an autoregressive LLM to convert text plus prompt speech context
into supervised semantic speech-token sequences.  This compact version keeps
Qwen2-style RMSNorm, rotary causal self-attention, and a speech-token head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by Qwen-style decoders."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize scale and epsilon."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize activations by root mean square."""

        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class Qwen2Block(nn.Module):
    """Small causal decoder block with rotary phase injection."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        """Initialize attention and SwiGLU feed-forward layers."""

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, dim * 2, bias=False)
        self.up = nn.Linear(dim, dim * 2, bias=False)
        self.down = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal rotary attention and SwiGLU MLP."""

        bsz, length, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)
        pos = torch.arange(length, device=x.device).float()
        phase = torch.sin(pos)[None, None, :, None]
        q = q + phase
        k = k + phase
        score = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        mask = torch.full((length, length), float("-inf"), device=x.device).triu(1)
        attn = torch.matmul(torch.softmax(score + mask, dim=-1), v)
        x = x + self.out(attn.transpose(1, 2).reshape(bsz, length, dim))
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class CosyVoiceQwen2LM(nn.Module):
    """Compact CosyVoice Qwen2 semantic speech-token LM."""

    def __init__(self, text_vocab: int = 256, speech_vocab: int = 128, dim: int = 64) -> None:
        """Initialize text/speech embeddings, decoder blocks, and token head."""

        super().__init__()
        self.text = nn.Embedding(text_vocab, dim)
        self.prompt_speech = nn.Embedding(speech_vocab, dim)
        self.blocks = nn.ModuleList([Qwen2Block(dim) for _ in range(2)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, speech_vocab, bias=False)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Predict next semantic speech-token logits from text and prompt tokens."""

        text_ids, prompt_ids = inputs
        x = torch.cat([self.text(text_ids), self.prompt_speech(prompt_ids)], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build() -> nn.Module:
    """Build a compact random-init CosyVoice Qwen2 LM."""

    return CosyVoiceQwen2LM().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return text token ids and prompt speech token ids."""

    return (torch.randint(0, 256, (1, 8)), torch.randint(0, 128, (1, 6)))


MENAGERIE_ENTRIES = [
    ("CosyVoice_Qwen2LM", "build", "example_input", "2024", "DC"),
]
