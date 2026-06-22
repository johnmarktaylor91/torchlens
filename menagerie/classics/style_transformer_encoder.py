"""Style Transformer Encoder for GAN Inversion.

Hu et al., CVPR 2022.
Paper: https://arxiv.org/abs/2203.07543
Source: https://github.com/sapphire497/style-transformer

The Style Transformer encoder maps a real image to the W+ latent space of
StyleGAN, enabling high-fidelity GAN inversion.

Architecture:
  1. CNN Feature Extractor -- a small convolutional backbone (inspired by
     feature pyramid / ResNet-style) that produces multi-scale spatial
     feature maps from the input image.

  2. Transformer Decoder -- a stack of transformer decoder layers where:
       * The QUERIES are num_styles learned style tokens (one per W+ layer).
       * The KEYS/VALUES come from the CNN feature maps (cross-attention).
     This is the distinctive primitive: per-layer style queries that each
     learn to attend to the image features and produce the corresponding
     latent code.

  3. Linear projection -- maps each style token from dim to latent_dim.

  Output: (B, num_styles, latent_dim) -- the W+ latent code sequence.

Compact config: CNN produces 16x16 features, 4 style queries, 2 transformer
decoder layers, latent_dim=64.  Input: (1, 3, 64, 64).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------


class CNNExtractor(nn.Module):
    """Small CNN backbone that produces flattened spatial tokens for cross-attention."""

    def __init__(self, in_ch: int = 3, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_ch, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32 -> 16
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 16 -> 16 (refine)
            nn.Conv2d(64, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)  # (B, out_dim, H/4, W/4)
        B, C, H, W = feat.shape
        # Flatten to sequence: (B, H*W, C)
        return feat.flatten(2).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Transformer Decoder
# ---------------------------------------------------------------------------


class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block: self-attn on queries + cross-attn to features."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Self-attention over style queries
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q2)
        # Cross-attention: queries attend to CNN features
        q3, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm2(queries + q3)
        # Feed-forward
        queries = self.norm3(queries + self.ff(queries))
        return queries


class StyleTransformerDecoder(nn.Module):
    """Stack of transformer decoder blocks."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(dim, num_heads, ff_dim) for _ in range(num_layers)]
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            queries = layer(queries, memory)
        return queries


# ---------------------------------------------------------------------------
# Full Style Transformer Encoder
# ---------------------------------------------------------------------------


class StyleTransformerEncoder(nn.Module):
    """Style Transformer: CNN features + learned style queries + transformer decoder.

    Input:  (B, 3, H, W) image.
    Output: (B, num_styles, latent_dim) W+ latent codes.
    """

    def __init__(
        self,
        num_styles: int = 4,
        latent_dim: int = 64,
        feat_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim

        # CNN backbone
        self.cnn = CNNExtractor(in_ch=3, out_dim=feat_dim)

        # Learned style queries: one per W+ layer
        self.style_queries = nn.Parameter(torch.randn(1, num_styles, feat_dim))

        # Transformer decoder
        self.transformer = StyleTransformerDecoder(
            dim=feat_dim, num_heads=num_heads, ff_dim=feat_dim * 4, num_layers=num_layers
        )

        # Project each style token to latent_dim
        self.proj = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Extract CNN features: (B, spatial_tokens, feat_dim)
        memory = self.cnn(x)

        # Expand learned style queries to batch
        queries = self.style_queries.expand(B, -1, -1)  # (B, num_styles, feat_dim)

        # Transformer: style queries cross-attend to CNN features
        out = self.transformer(queries, memory)  # (B, num_styles, feat_dim)

        # Project to latent space
        latent = self.proj(out)  # (B, num_styles, latent_dim)
        return latent


# ---------------------------------------------------------------------------
# Builder + example
# ---------------------------------------------------------------------------


def build_style_transformer_encoder() -> nn.Module:
    """Build compact Style Transformer encoder."""
    return StyleTransformerEncoder(
        num_styles=4,
        latent_dim=64,
        feat_dim=64,
        num_heads=4,
        num_layers=2,
    )


def example_input_ste() -> torch.Tensor:
    """(1, 3, 64, 64) image for GAN inversion."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Style Transformer Encoder (GAN inversion: CNN features + style queries + cross-attention -> W+ latent)",
        "build_style_transformer_encoder",
        "example_input_ste",
        "2022",
        "DC",
    ),
]
