"""SSTN: Spectral-Spatial Transformer Network for Hyperspectral Image Classification.

Zhong et al., IEEE Transactions on Geoscience and Remote Sensing, 2021.
Paper: https://arxiv.org/abs/2104.00776
Source: https://github.com/zgr6010/HSI_SSTN (and related reimplementations)

SSTN addresses hyperspectral image (HSI) classification, where each pixel has
hundreds of spectral bands (analogous to an ultra-deep-channel 2D image). The key
challenge is capturing both:
  (a) Spectral relationships: correlations across bands for the same spatial pixel.
  (b) Spatial relationships: local neighbourhood texture/context around each pixel.

SSTN uses two separate Transformer modules:
  1. Spectral Transformer: processes the spectral sequence (band tokens) to learn
     inter-band correlations. Input: the bands are treated as tokens (B, bands, 1)
     with a learned projection per spatial query.
  2. Spatial Transformer: processes the spatial neighbourhood tokens (patch pixels)
     to learn spatial context. Input: a local patch around the query pixel, where
     each spatial location is a token with its full spectral vector.

The two attention outputs are combined (concatenated + linear fusion) and fed to
a classification head.

Compact faithfulness:
  - Input HSI patch: (1, bands, patch_h, patch_w) where bands=30, patch=11x11.
  - Spectral attention: treats band axis as sequence (tokens = bands).
  - Spatial attention: treats spatial positions as sequence (tokens = patch pixels).
  - Both use standard multi-head self-attention with LN + residual.
  - Classification head: MLP -> num_classes.
  - Output: (1, num_classes) class logits for the centre pixel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral Transformer Module
# ---------------------------------------------------------------------------


class SpectralTransformer(nn.Module):
    """Attention over the spectral (band) dimension.

    Treats each band as a token (position = band index, feature = spatial summary).
    For compact classification: we use the mean over spatial dims as the band feature.

    Input: (B, bands, H, W) -> per-band embedding -> (B, bands, embed_dim).
    Output: (B, embed_dim) spectral context via CLS token or mean pooling.
    """

    def __init__(self, bands: int, embed_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.band_embed = nn.Linear(1, embed_dim)  # each band's mean -> embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W)
        B, bands, H, W = x.shape

        # Summarise each band with its spatial mean: (B, bands, 1)
        band_feats = x.mean(dim=[-2, -1], keepdim=True).squeeze(-1).transpose(1, 2)
        # Actually: (B, H*W, bands) — attend per-band across all spatial points
        # Simpler: use band mean -> embed -> attend
        band_feats = x.mean(dim=[-2, -1]).unsqueeze(-1)  # (B, bands, 1)
        tokens = self.band_embed(band_feats)  # (B, bands, embed_dim)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+bands, embed_dim)

        tokens = self.transformer(tokens)
        # Use CLS token as spectral context
        return self.norm(tokens[:, 0])  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Spatial Transformer Module
# ---------------------------------------------------------------------------


class SpatialTransformer(nn.Module):
    """Attention over spatial positions in a local patch.

    Each spatial pixel is a token; features are the full spectral vector at that
    pixel. Captures neighbourhood context via self-attention.

    Input: (B, bands, H, W) -> flattened spatial tokens -> (B, H*W, embed_dim).
    Output: (B, embed_dim) spatial context via CLS token.
    """

    def __init__(self, bands: int, embed_dim: int, num_heads: int = 2) -> None:
        super().__init__()
        self.pixel_embed = nn.Linear(bands, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W)
        B, bands, H, W = x.shape

        # Rearrange to spatial tokens: each pixel = (bands,) -> embed_dim
        pixels = x.flatten(2).permute(0, 2, 1)  # (B, H*W, bands)
        tokens = self.pixel_embed(pixels)  # (B, H*W, embed_dim)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+H*W, embed_dim)

        tokens = self.transformer(tokens)
        return self.norm(tokens[:, 0])  # (B, embed_dim)


# ---------------------------------------------------------------------------
# Full SSTN
# ---------------------------------------------------------------------------


class SSTN(nn.Module):
    """SSTN: Spectral-Spatial Transformer for HSI classification (compact reimpl).

    Input: (B, bands, patch_h, patch_w) — HSI patch.
    Output: (B, num_classes) class logits for the centre pixel.
    """

    def __init__(
        self,
        bands: int = 30,
        embed_dim: int = 32,
        num_heads: int = 2,
        num_classes: int = 16,
    ) -> None:
        super().__init__()
        self.spectral_transformer = SpectralTransformer(bands, embed_dim, num_heads)
        self.spatial_transformer = SpatialTransformer(bands, embed_dim, num_heads)

        # Fusion and classification
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, bands, H, W)
        spec_feat = self.spectral_transformer(x)  # (B, embed_dim)
        spat_feat = self.spatial_transformer(x)  # (B, embed_dim)
        fused = torch.cat([spec_feat, spat_feat], dim=-1)  # (B, embed_dim*2)
        fused = self.fusion(fused)  # (B, embed_dim)
        return self.classifier(fused)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_sstn() -> nn.Module:
    """Build SSTN (spectral + spatial Transformer, 30 bands, 11x11 patch, 16 classes)."""
    return SSTN(bands=30, embed_dim=32, num_heads=2, num_classes=16)


def example_input() -> torch.Tensor:
    """HSI patch: (1, 30, 11, 11) — 30 bands, 11x11 spatial window."""
    return torch.randn(1, 30, 11, 11)


MENAGERIE_ENTRIES = [
    (
        "SSTN (Spectral-Spatial Transformer for HSI classification, dual spectral+spatial attention)",
        "build_sstn",
        "example_input",
        "2021",
        "DC",
    ),
]
