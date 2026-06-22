"""LSTR: Lane Shape Transformer -- DETR-like end-to-end lane detection.

Liu et al., WACV 2021.
Paper: https://arxiv.org/abs/2011.04233
Source: https://github.com/liuruijin17/LSTR
Also covered by target name: PytorchAutoDrive-LSTR

LSTR applies a DETR-style transformer encoder-decoder to lane detection,
directly regressing polynomial lane shape parameters from a set of
fixed lane queries (learned positional embeddings).

Distinctive architecture:
  1. CNN backbone (ResNet-style) producing image features
  2. Positional encoding added to flattened backbone features
  3. Transformer encoder processes the sequence of patch features
  4. Transformer decoder: fixed-size set of learned lane queries attend
     to encoder memory and output lane shape parameters
  5. Prediction head: per-query MLP predicting:
     - Existence score (binary classification)
     - Polynomial curve parameters (coefficients of a degree-3 polynomial
       mapping vertical position y -> horizontal position x)
     - Vertical extent: y_start and y_end of the lane

The "lane shape" parameterization is LSTR's key contribution vs anchor-based methods.

Architecture notes / faithful-core simplifications:
  - Compact backbone: 3-stage CNN (ResNet stub) at reduced widths
  - embed_dim=64, encoder/decoder depth=2, n_heads=4 (published: 256d, 2+2 layers)
  - n_queries=8 lane queries (published: 7)
  - Polynomial degree=3 (as published)
  - Input: (1, 3, 64, 128) -- small for fast tracing
  - trace+draw verified 2026-06-21
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Compact CNN backbone
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + idt)


class CNNBackbone(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(base, base), BasicBlock(base, base))
        self.layer2 = nn.Sequential(
            BasicBlock(base, base * 2, stride=2), BasicBlock(base * 2, base * 2)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(base * 2, base * 4, stride=2), BasicBlock(base * 4, base * 4)
        )
        self.out_ch = base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, out_ch, H/16, W/16)


# ============================================================
# Positional encoding (2D sine/cosine)
# ============================================================


class PositionEncoding2D(nn.Module):
    """2D sinusoidal position encoding added to backbone feature tokens."""

    def __init__(self, embed_dim: int, max_hw: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Create a fixed 2D positional encoding table
        pe = torch.zeros(max_hw, max_hw, embed_dim)
        pos_h = torch.arange(max_hw).unsqueeze(1).float()
        pos_w = torch.arange(max_hw).unsqueeze(1).float()
        dim = embed_dim // 4
        div = torch.exp(torch.arange(0, dim, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe_h = torch.zeros(max_hw, embed_dim // 2)
        pe_h[:, 0::2] = torch.sin(pos_h * div)
        pe_h[:, 1::2] = torch.cos(pos_h * div)
        pe_w = torch.zeros(max_hw, embed_dim // 2)
        pe_w[:, 0::2] = torch.sin(pos_w * div)
        pe_w[:, 1::2] = torch.cos(pos_w * div)
        # Combine H and W encodings
        pe_2d = torch.zeros(max_hw, max_hw, embed_dim)
        pe_2d[:, :, : embed_dim // 2] = pe_h.unsqueeze(1).expand(max_hw, max_hw, embed_dim // 2)
        pe_2d[:, :, embed_dim // 2 :] = pe_w.unsqueeze(0).expand(max_hw, max_hw, embed_dim // 2)
        self.register_buffer("pe_2d", pe_2d)

    def forward(self, x_tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Add positional encoding to (B, H*W, D) tokens."""
        pe = self.pe_2d[:h, :w, :].reshape(h * w, self.embed_dim)
        return x_tokens + pe.unsqueeze(0)


# ============================================================
# LSTR Transformer (encoder-decoder)
# ============================================================


class LSTRTransformer(nn.Module):
    """LSTR transformer: encoder on image tokens + decoder with lane queries."""

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        enc_depth: int = 2,
        dec_depth: int = 2,
        mlp_ratio: float = 4.0,
        n_queries: int = 8,
    ) -> None:
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_depth)

        # Learned lane queries (fixed set, one per potential lane)
        self.query_embed = nn.Embedding(n_queries, embed_dim)
        self.n_queries = n_queries

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory: (B, N, D) -- image tokens after projection
        Returns lane query features: (B, n_queries, D)
        """
        enc_out = self.encoder(memory)  # (B, N, D)
        B = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, n_q, D)
        dec_out = self.decoder(queries, enc_out)  # (B, n_queries, D)
        return dec_out


# ============================================================
# Lane shape prediction head (polynomial regression)
# ============================================================


class LaneShapeHead(nn.Module):
    """Per-query MLP predicting polynomial lane curve parameters.

    Each lane query outputs:
      - exist: 1 float (logit for lane existence)
      - poly_coeffs: (poly_degree+1) polynomial coefficients (x = sum c_k * y^k)
      - y_start, y_end: 2 floats (vertical extent of the lane, normalized [0,1])

    Total per query: 1 + (poly_degree + 1) + 2
    """

    def __init__(self, embed_dim: int, poly_degree: int = 3) -> None:
        super().__init__()
        self.poly_degree = poly_degree
        out_dim = 1 + (poly_degree + 1) + 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, queries: torch.Tensor):
        """queries: (B, n_queries, D)"""
        out = self.mlp(queries)  # (B, n_queries, out_dim)
        exist = out[..., 0:1]
        poly = out[..., 1 : 1 + self.poly_degree + 1]
        extent = out[..., -2:]
        return exist, poly, extent


# ============================================================
# Full LSTR
# ============================================================


class LSTR(nn.Module):
    """LSTR: CNN backbone + transformer encoder-decoder + polynomial lane head."""

    def __init__(
        self,
        base: int = 32,
        embed_dim: int = 64,
        n_heads: int = 4,
        enc_depth: int = 2,
        dec_depth: int = 2,
        n_queries: int = 8,
        poly_degree: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = CNNBackbone(base)
        # Project backbone features to embed_dim
        self.input_proj = nn.Conv2d(self.backbone.out_ch, embed_dim, 1)
        self.pos_enc = PositionEncoding2D(embed_dim, max_hw=64)
        self.transformer = LSTRTransformer(
            embed_dim=embed_dim,
            n_heads=n_heads,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            n_queries=n_queries,
        )
        self.lane_head = LaneShapeHead(embed_dim, poly_degree)

    def forward(self, x: torch.Tensor):
        # Backbone
        feat = self.backbone(x)  # (B, out_ch, h, w)
        feat = self.input_proj(feat)  # (B, embed_dim, h, w)
        B, D, h, w = feat.shape
        # Flatten to tokens + add 2D positional encoding
        tokens = feat.flatten(2).transpose(1, 2)  # (B, h*w, D)
        tokens = self.pos_enc(tokens, h, w)
        # Transformer encoder-decoder
        lane_feats = self.transformer(tokens)  # (B, n_queries, D)
        # Predict polynomial lane shapes
        exist, poly, extent = self.lane_head(lane_feats)
        return exist, poly, extent


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_lstr() -> nn.Module:
    return LSTR(
        base=32,
        embed_dim=64,
        n_heads=4,
        enc_depth=2,
        dec_depth=2,
        n_queries=8,
        poly_degree=3,
    )


def example_input() -> torch.Tensor:
    """RGB image (1, 3, 64, 128) for fast tracing."""
    return torch.randn(1, 3, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "LSTR (Lane Shape Transformer: DETR-style polynomial lane curve regression)",
        "build_lstr",
        "example_input",
        "2021",
        "DC",
    ),
]
