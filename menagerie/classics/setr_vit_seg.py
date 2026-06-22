"""SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers.

Zheng et al., CVPR 2021.
Paper: https://arxiv.org/abs/2012.15840
Source: https://github.com/fudan-zvg/SETR

SETR uses a plain ViT (Vision Transformer) encoder -- the image is split into
fixed-size patches, flattened to 1D tokens, processed by pure transformer encoder
layers -- combined with three different decoder heads:

  - SETR-Naive: simple linear projection + bilinear upsample to output resolution
  - SETR-PUP: progressive upsampling decoder -- 4 stages each with a 3x3 conv
    and 2x bilinear upsample to recover spatial resolution step by step
  - SETR-MLA: multi-level aggregation decoder -- takes feature maps from multiple
    (evenly spaced) transformer layers, projects each, sums them via progressive
    conv blocks, final 1x1 to output classes

Architecture notes / faithful-core simplifications:
  - ViT encoder: patch_size=8, image=64x64 -> 8x8=64 tokens (full paper uses 16x16
    patches on 480x480 images -> 900 tokens + cls_token; we replicate the
    patching, pos-embed, cls-token, and transformer block structure faithfully)
  - Compact dims: embed_dim=64, depth=4 (paper: 1024d, 24 layers)
  - All three decoder variants share the same ViT encoder
  - Input: (1, 3, 64, 64)
  - trace+draw verified 2026-06-21
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ViT Encoder (shared)
# ============================================================


class PatchEmbed(nn.Module):
    """Image -> patch tokens (no CLS in grid; we add a CLS token separately)."""

    def __init__(
        self, img_size: int = 64, patch_size: int = 8, in_ch: int = 3, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """Return (B, N, D) patch tokens + (B, D, h, w) spatial grid for MLA."""
        feat = self.proj(x)  # (B, D, h, w)
        B, D, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens, feat, h, w


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Pure ViT encoder for SETR.

    Exposes intermediate layer outputs for MLA decoder.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_ch: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size
        self.depth = depth

    def forward(self, x: torch.Tensor):
        """
        Returns:
          final_tokens: (B, N, D)  -- tokens after all layers (excluding CLS)
          intermediates: list of (B, N, D) feature tensors at equally spaced layers
          h, w: patch-grid spatial dims
        """
        tokens, _, h, w = self.patch_embed(x)
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)
        tokens = tokens + self.pos_embed
        intermediates = []
        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i % max(1, self.depth // 4) == (self.depth // 4 - 1):
                intermediates.append(tokens[:, 1:])  # exclude CLS
        tokens = self.norm(tokens)
        final_tokens = tokens[:, 1:]  # (B, N, D)
        return final_tokens, intermediates, h, w


# ============================================================
# SETR-Naive decoder
# ============================================================


class SETRNaive(nn.Module):
    """SETR-Naive: ViT encoder + simple linear project + bilinear upsample."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_ch: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        num_classes: int = 19,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_ch, embed_dim, depth)
        self.head = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        tokens, _, h, w = self.encoder(x)  # (B, N, D)
        B, N, D = tokens.shape
        feat = tokens.transpose(1, 2).view(B, D, h, w)  # (B, D, h, w)
        out = self.head(feat)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ============================================================
# SETR-PUP decoder
# ============================================================


class SETRPup(nn.Module):
    """SETR-PUP: ViT encoder + progressive upsampling decoder (4 stages x 2x)."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_ch: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        num_classes: int = 19,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_ch, embed_dim, depth)
        # 4 stages: each 3x3 conv + BN + ReLU then 2x bilinear upsample
        # patch_size=8 -> 4 stages of 2x to recover full resolution
        n_up = int(math.log2(patch_size))  # 3 for patch_size=8
        self.up_stages = nn.ModuleList()
        ch = embed_dim
        for i in range(n_up):
            out_ch = ch // 2 if ch > num_classes * 2 else max(ch, num_classes)
            self.up_stages.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            ch = out_ch
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        tokens, _, h, w = self.encoder(x)
        B, N, D = tokens.shape
        feat = tokens.transpose(1, 2).view(B, D, h, w)
        for stage in self.up_stages:
            feat = stage(feat)
            feat = F.interpolate(feat, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.head(feat)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ============================================================
# SETR-MLA decoder
# ============================================================


class SETRMla(nn.Module):
    """SETR-MLA: ViT encoder + multi-level aggregation decoder.

    Takes feature maps from evenly-spaced transformer layers,
    projects each to a smaller channel dim, progressively fuses
    via addition and conv refinement, then outputs class logits.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_ch: int = 3,
        embed_dim: int = 64,
        depth: int = 4,
        num_classes: int = 19,
        mla_ch: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_ch, embed_dim, depth)
        # Project each intermediate layer to mla_ch
        n_int = depth // max(1, depth // 4)
        self.n_intermediates = min(depth, 4)  # collect at most 4 intermediate outputs
        self.proj = nn.ModuleList(
            [nn.Conv2d(embed_dim, mla_ch, 1, bias=False) for _ in range(self.n_intermediates)]
        )
        # Refinement conv for each level
        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(mla_ch, mla_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(mla_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in range(self.n_intermediates)
            ]
        )
        self.head = nn.Conv2d(mla_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        tokens, intermediates, h, w = self.encoder(x)

        # Use as many intermediates as we have projectors
        feats = []
        for i, interm in enumerate(intermediates[-self.n_intermediates :]):
            B, N, D = interm.shape
            f = interm.transpose(1, 2).view(B, D, h, w)
            feats.append(self.proj[i](f))

        # Aggregate from deepest to shallowest (progressive add)
        agg = None
        for i in range(len(feats) - 1, -1, -1):
            rf = self.refine[i](feats[i])
            if agg is not None:
                if agg.shape[2:] != rf.shape[2:]:
                    agg = F.interpolate(
                        agg, size=rf.shape[2:], mode="bilinear", align_corners=False
                    )
                rf = rf + agg
            agg = rf

        out = self.head(agg)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


# ============================================================
# Builders + example inputs + entries
# ============================================================


def build_setr_naive() -> nn.Module:
    return SETRNaive(img_size=64, patch_size=8, embed_dim=64, depth=4)


def build_setr_pup() -> nn.Module:
    return SETRPup(img_size=64, patch_size=8, embed_dim=64, depth=4)


def build_setr_mla() -> nn.Module:
    return SETRMla(img_size=64, patch_size=8, embed_dim=64, depth=4, mla_ch=32)


def example_input() -> torch.Tensor:
    """Small RGB image (1, 3, 64, 64) for fast tracing."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SETR-Naive (ViT encoder + linear upsample decoder)",
        "build_setr_naive",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "SETR-PUP (ViT encoder + progressive upsampling decoder)",
        "build_setr_pup",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "SETR-MLA (ViT encoder + multi-level aggregation decoder)",
        "build_setr_mla",
        "example_input",
        "2021",
        "DC",
    ),
]
