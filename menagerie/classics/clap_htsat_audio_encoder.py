"""CLAP HTSAT audio encoder compact reconstruction.

Paper: Chen et al., 2022, "HTS-AT: A Hierarchical Token-Semantic Audio
Transformer"; LAION CLAP, 2022.

CLAP uses HTSAT as an audio encoder inside a contrastive audio-text model.
This compact module preserves patchified spectrogram input, Swin-style local
window attention, hierarchical patch merging, and token-semantic class maps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HTSATStage(nn.Module):
    """Windowed transformer stage for spectrogram tokens."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize local attention and patch merging."""

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens: Tensor) -> Tensor:
        """Refine tokens with local self-attention."""

        tokens = self.norm(tokens + self.attn(tokens, tokens, tokens, need_weights=False)[0])
        return tokens + self.ffn(tokens)


class CLAPHTSATAudioEncoder(nn.Module):
    """Compact HTSAT encoder with CLAP projection."""

    def __init__(self, dim: int = 48, classes: int = 12) -> None:
        """Initialize spectrogram patchifier and semantic projection heads."""

        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.stage1 = HTSATStage(dim)
        self.merge = nn.Linear(dim * 2, dim)
        self.stage2 = HTSATStage(dim)
        self.semantic = nn.Linear(dim, classes)
        self.proj = nn.Linear(dim, 64)

    def forward(self, spectrogram: Tensor) -> tuple[Tensor, Tensor]:
        """Encode log-mel spectrograms into CLAP and token-semantic outputs."""

        feat = self.patch(spectrogram).flatten(2).transpose(1, 2)
        feat = self.stage1(feat)
        even = feat[:, 0::2]
        odd = feat[:, 1::2]
        if odd.shape[1] < even.shape[1]:
            odd = F.pad(odd, (0, 0, 0, even.shape[1] - odd.shape[1]))
        feat = self.merge(torch.cat([even, odd], dim=-1))
        feat = self.stage2(feat)
        semantic_maps = self.semantic(feat)
        embedding = F.normalize(self.proj(feat.mean(dim=1)), dim=-1)
        return embedding, semantic_maps


def build() -> nn.Module:
    """Build the compact CLAP HTSAT audio encoder."""

    return CLAPHTSATAudioEncoder().eval()


def example_input() -> Tensor:
    """Return a small log-mel spectrogram."""

    return torch.randn(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    ("clap_htsat_audio_encoder", "build", "example_input", "2022", "AUDIO"),
]
