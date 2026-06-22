"""TransBTS: Transformer-based Brain Tumor Segmentation.

Wang et al., MICCAI 2021.
Paper: https://arxiv.org/abs/2103.04430
Source: https://github.com/Wenxuan-1119/TransBTS

TransBTS combines 3D CNN encoder-decoder (U-Net-like) with a Transformer bottleneck
for multi-modal brain tumor segmentation from MRI (BraTS dataset):

Architecture:
  1. 3D CNN Encoder: successive 3D conv blocks with max-pooling downsample the
     4-modality MRI volume through 4 encoding stages, extracting hierarchical
     3D features (skip connections are stored at each stage).
  2. Transformer Bottleneck: the coarsest encoder feature map is reshaped into
     a 1D token sequence and processed by a stack of standard transformer layers
     (multi-head self-attention + FFN). This captures long-range dependencies
     across the 3D volume that CNNs struggle with due to limited receptive field.
  3. 3D CNN Decoder: skip connections from the encoder are concatenated with
     upsampled decoder features, and 3D conv blocks refine the representations
     at each resolution back to the original volume size.
  4. Segmentation Head: 1x1x1 conv -> num_classes.

The key architectural choice (the THING that makes TransBTS unique) is the
CNN-Transformer hybrid: the CNN handles local 3D feature extraction while the
Transformer handles global long-range dependencies at the bottleneck.

Compact faithfulness:
  - Input: (1, 4, 32, 32, 32) — batch=1, 4 MRI modalities, 32^3 volume.
  - 2 encoder stages (not 4), channels [8, 16], for a quick compact trace.
  - Transformer bottleneck: 1 layer, 4 heads.
  - 2 decoder stages with skip connections.
  - Output: (1, num_classes, 32, 32, 32) segmentation logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Conv Block
# ---------------------------------------------------------------------------


class ConvBlock3D(nn.Module):
    """3D conv block: Conv3d -> BN -> ReLU (x2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer bottleneck
# ---------------------------------------------------------------------------


class TransformerBottleneck(nn.Module):
    """Standard transformer encoder used as bottleneck after CNN encoder."""

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C) — flattened 3D tokens
        return self.transformer(x)


# ---------------------------------------------------------------------------
# Full TransBTS
# ---------------------------------------------------------------------------


class TransBTS(nn.Module):
    """TransBTS brain tumor segmentation (compact random-init reimpl).

    Input: (B, 4, D, H, W) — 4 MRI modalities.
    Output: (B, num_classes, D, H, W) segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 8,
        num_classes: int = 4,
        transformer_dim: int = 16,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
    ) -> None:
        super().__init__()
        c = base_channels
        self.transformer_dim = transformer_dim

        # ---- Encoder ----
        self.enc1 = ConvBlock3D(in_channels, c)  # -> (B, c, D, H, W)
        self.pool1 = nn.MaxPool3d(2)  # -> (B, c, D/2, H/2, W/2)
        self.enc2 = ConvBlock3D(c, c * 2)  # -> (B, 2c, D/2, H/2, W/2)
        self.pool2 = nn.MaxPool3d(2)  # -> (B, 2c, D/4, H/4, W/4)

        # Projection to transformer dim
        self.to_trans = nn.Conv3d(c * 2, transformer_dim, 1)

        # ---- Transformer bottleneck ----
        self.transformer = TransformerBottleneck(transformer_dim, num_heads, num_transformer_layers)

        # Projection back
        self.from_trans = nn.Conv3d(transformer_dim, c * 2, 1)

        # ---- Decoder ----
        self.up2 = nn.ConvTranspose3d(c * 2, c * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(c * 2 + c * 2, c * 2)  # + skip from enc2

        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock3D(c + c, c)  # + skip from enc1

        # ---- Segmentation head ----
        self.seg_head = nn.Conv3d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D, H, W = x.shape

        # Encoder
        e1 = self.enc1(x)  # (B, c, D, H, W)
        e2 = self.enc2(self.pool1(e1))  # (B, 2c, D/2, H/2, W/2)
        bottleneck = self.pool2(e2)  # (B, 2c, D/4, H/4, W/4)

        # Project to transformer dim
        btl = self.to_trans(bottleneck)  # (B, trans_dim, D/4, H/4, W/4)
        _, C, D4, H4, W4 = btl.shape

        # Flatten for transformer: (B, N, C)
        tokens = btl.flatten(2).permute(0, 2, 1)  # (B, D4*H4*W4, trans_dim)
        tokens = self.transformer(tokens)
        # Reshape back
        btl = tokens.permute(0, 2, 1).reshape(B, C, D4, H4, W4)

        # Project back to CNN channels
        btl = self.from_trans(btl)  # (B, 2c, D/4, H/4, W/4)

        # Decoder
        d2 = self.up2(btl)  # (B, 2c, D/2, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 2c, D/2, H/2, W/2)

        d1 = self.up1(d2)  # (B, c, D, H, W)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, c, D, H, W)

        out = self.seg_head(d1)  # (B, num_classes, D, H, W)
        return out


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_transbts() -> nn.Module:
    """Build TransBTS brain tumor segmentation (2-stage encoder, 1 transformer layer)."""
    return TransBTS(
        in_channels=4,
        base_channels=8,
        num_classes=4,
        transformer_dim=16,
        num_heads=4,
        num_transformer_layers=1,
    )


def example_input() -> torch.Tensor:
    """4-modality MRI volume: (1, 4, 32, 32, 32)."""
    return torch.randn(1, 4, 32, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "TransBTS (3D CNN encoder + Transformer bottleneck + 3D CNN decoder for brain tumor segmentation)",
        "build_transbts",
        "example_input",
        "2021",
        "DC",
    ),
]
