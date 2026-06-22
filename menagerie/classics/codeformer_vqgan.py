"""CodeFormer: transformer codebook lookup for blind face restoration.

Paper: "Towards Robust Blind Face Restoration with Codebook Lookup
Transformer", Zhou et al., NeurIPS 2022.

This compact model keeps CodeFormer's distinctive path: a low-quality encoder
produces spatial tokens, a transformer predicts codebook logits, quantized
code vectors are looked up from a learned VQ codebook, and a decoder reconstructs
the restored face.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Small convolutional encoder for degraded face images."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize the encoder."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an RGB face image into spatial features."""

        return self.net(x)


class CodeFormerCompact(nn.Module):
    """Compact CodeFormer/VQGAN face-restoration model."""

    def __init__(self, channels: int = 32, codes: int = 64) -> None:
        """Initialize the compact CodeFormer."""

        super().__init__()
        self.encoder = ConvEncoder(channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dim_feedforward=channels * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.to_logits = nn.Linear(channels, codes)
        self.codebook = nn.Embedding(codes, channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restore a degraded aligned face crop."""

        feat = self.encoder(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        contextual = self.transformer(tokens)
        probs = torch.softmax(self.to_logits(contextual), dim=-1)
        quant = torch.matmul(probs, self.codebook.weight).transpose(1, 2).view(b, c, h, w)
        restored = self.decoder(torch.cat([feat, quant], dim=1))
        return torch.tanh(restored + F.interpolate(x, size=restored.shape[-2:], mode="bilinear"))


def build_codeformer_vqgan() -> nn.Module:
    """Build compact CodeFormer."""

    return CodeFormerCompact()


def example_input() -> torch.Tensor:
    """Return a small aligned RGB face crop."""

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "CodeFormer VQGAN (transformer codebook lookup face restoration)",
        "build_codeformer_vqgan",
        "example_input",
        "2022",
        "E7",
    )
]
