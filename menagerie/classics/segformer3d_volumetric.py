"""SegFormer3D: Efficient 3D Volumetric SegFormer for Medical Segmentation.

Inspired by: Xie et al. SegFormer (NeurIPS 2021, arXiv 2105.15203) extended to 3D.
3D extension papers include:
  - "3D Medical Image Segmentation with Transformer" (various 2022 works)
  - SegFormer3D variants applied to BraTS, liver, etc.
Source inspiration: https://github.com/huggingface/transformers (SegFormer 2D)
  adapted to 3D by adding 3D convolutions and 3D spatial reduction attention.

SegFormer3D is a hierarchical 3D vision transformer for volumetric segmentation:
  1. Overlapping 3D Patch Embedding: stride-2 3D conv with overlap creates tokens
     from each stage of a multi-scale feature hierarchy.
  2. Efficient 3D Self-Attention with Spatial Reduction: instead of attending over
     all N^3 voxel tokens (expensive), the key/value tokens are spatially reduced
     (via a 3D conv with stride=R) before attention, reducing complexity from O(N^6)
     to O(N^3 * N^3/R^3). This is the key efficiency primitive.
  3. Mix-FFN: 3D depth-wise conv inside the MLP to inject local 3D context.
  4. All-MLP Decoder: upsamples and fuses multi-scale features into a segmentation
     map, avoiding the heavy FPN decoder common in 3D CNNs.

Compact faithfulness:
  - Input: (1, 4, 16, 16, 16) — batch=1, 4 MRI modalities, 16^3 volume.
  - 2 hierarchical stages with spatial-reduction attention.
  - Spatial reduction ratio R=2 (SR-Attn).
  - All-MLP decoder.
  - Output: (1, num_classes, 16, 16, 16) segmentation logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D Overlapping Patch Embedding
# ---------------------------------------------------------------------------


class OverlapPatchEmbed3D(nn.Module):
    """3D overlapping patch embedding: strided conv with overlap.

    (B, C_in, D, H, W) -> (B, C_out, D', H', W')
    D' = D / stride, etc.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', embed_dim)
        x = self.norm(x)
        return x, (D, H, W)


# ---------------------------------------------------------------------------
# Spatial-Reduction Attention (3D SR-Attn)
# ---------------------------------------------------------------------------


class SRAttention3D(nn.Module):
    """3D Spatial-Reduction Multi-Head Self-Attention.

    Keys and values are computed on a spatially-reduced version of the input
    (via a 3D conv with stride=sr_ratio), reducing the attention sequence length
    and thus computational cost.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        sr_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction: 3D conv to reduce key/value sequence
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x: torch.Tensor, shape3d: tuple) -> torch.Tensor:
        # x: (B, N, C), N = D*H*W
        B, N, C = x.shape
        D, H, W = shape3d

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Spatial reduction: reshape to 3D, apply stride-sr_ratio conv, flatten back
            x_3d = x.transpose(1, 2).reshape(B, C, D, H, W)
            x_sr = self.sr(x_3d)  # (B, C, D/sr, H/sr, W/sr)
            x_sr = x_sr.flatten(2).transpose(1, 2)  # (B, N_sr, C)
            x_sr = self.sr_norm(x_sr)
            kv = self.kv(x_sr)
            N_sr = x_sr.shape[1]
        else:
            kv = self.kv(x)
            N_sr = N

        kv = kv.reshape(B, N_sr, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# Mix-FFN (3D depthwise conv inside MLP)
# ---------------------------------------------------------------------------


class MixFFN3D(nn.Module):
    """Mix-FFN: Linear -> 3D depthwise conv -> Linear with GELU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dw_conv = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor, shape3d: tuple) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        D, H, W = shape3d
        x = self.fc1(x)
        # Reshape to 3D for depthwise conv
        x = x.transpose(1, 2).reshape(B, -1, D, H, W)
        x = self.dw_conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# SegFormer3D Transformer Block
# ---------------------------------------------------------------------------


class SegFormer3DBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 2, sr_ratio: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SRAttention3D(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN3D(dim, dim * 2)

    def forward(self, x: torch.Tensor, shape3d: tuple) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), shape3d)
        x = x + self.ffn(self.norm2(x), shape3d)
        return x


# ---------------------------------------------------------------------------
# All-MLP Decoder
# ---------------------------------------------------------------------------


class AllMLPDecoder3D(nn.Module):
    """All-MLP Decoder: linear project each stage, upsample, fuse, predict."""

    def __init__(self, in_channels_list: list, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.projs = nn.ModuleList([nn.Linear(c, embed_dim) for c in in_channels_list])
        self.fuse = nn.Linear(embed_dim * len(in_channels_list), embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, features: list, target_shape: tuple) -> torch.Tensor:
        # features: list of (B, N_i, C_i); target_shape: (D, H, W)
        D, H, W = target_shape
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.projs)):
            B, N, C = feat.shape
            # Determine spatial dims: assume cube root (approximate)
            side = round(N ** (1 / 3))
            feat_p = proj(feat)  # (B, N, embed_dim)
            feat_p = feat_p.transpose(1, 2).reshape(B, -1, side, side, side)
            feat_p = F.interpolate(feat_p, size=(D, H, W), mode="trilinear", align_corners=False)
            feat_p = feat_p.flatten(2).transpose(1, 2)  # (B, D*H*W, embed_dim)
            projected.append(feat_p)

        x = torch.cat(projected, dim=-1)  # (B, D*H*W, embed_dim * n_stages)
        x = self.fuse(x)  # (B, D*H*W, embed_dim)
        x = self.head(x)  # (B, D*H*W, num_classes)
        B, N, nc = x.shape
        x = x.transpose(1, 2).reshape(B, nc, D, H, W)
        return x


# ---------------------------------------------------------------------------
# Full SegFormer3D
# ---------------------------------------------------------------------------


class SegFormer3D(nn.Module):
    """SegFormer3D volumetric segmentation network (compact random-init reimpl).

    Input: (B, in_channels, D, H, W) — e.g. (1, 4, 16, 16, 16) for BraTS.
    Output: (B, num_classes, D, H, W) segmentation logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dims: list = None,
        num_heads: list = None,
        sr_ratios: list = None,
        depths: list = None,
        num_classes: int = 4,
        decoder_dim: int = 32,
    ) -> None:
        super().__init__()
        if embed_dims is None:
            embed_dims = [32, 64]
        if num_heads is None:
            num_heads = [2, 4]
        if sr_ratios is None:
            sr_ratios = [2, 1]
        if depths is None:
            depths = [1, 1]

        self.num_stages = len(embed_dims)

        # Patch embeddings for each stage
        self.patch_embeds = nn.ModuleList()
        self.patch_embeds.append(
            OverlapPatchEmbed3D(in_channels, embed_dims[0], patch_size=3, stride=2)
        )
        for i in range(1, self.num_stages):
            self.patch_embeds.append(
                OverlapPatchEmbed3D(embed_dims[i - 1], embed_dims[i], patch_size=3, stride=2)
            )

        # Transformer blocks per stage
        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            stage_blocks = nn.ModuleList(
                [
                    SegFormer3DBlock(embed_dims[i], num_heads[i], sr_ratios[i])
                    for _ in range(depths[i])
                ]
            )
            self.blocks.append(stage_blocks)

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in embed_dims])

        # All-MLP decoder
        self.decoder = AllMLPDecoder3D(embed_dims, decoder_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, D_in, H_in, W_in = x.shape
        features = []
        prev_feat = prev_shape = None  # carried across stages (set at each iter's end)

        for i in range(self.num_stages):
            # Patch embedding
            if i == 0:
                x_stage = x
            else:
                # Rebuild spatial from previous stage tokens
                x_stage = prev_feat.transpose(1, 2).reshape(B, -1, *prev_shape)

            tokens, shape3d = self.patch_embeds[i](x_stage)
            # Transformer blocks
            for blk in self.blocks[i]:
                tokens = blk(tokens, shape3d)
            tokens = self.norms[i](tokens)
            features.append(tokens)
            # Carry this stage's tokens/spatial shape so the next stage can rebuild
            # its volumetric input (hierarchical multi-stage encoder).
            prev_feat = tokens
            prev_shape = shape3d
            prev_feat = tokens
            prev_shape = shape3d

        # All-MLP decode to original resolution
        seg = self.decoder(features, (D_in, H_in, W_in))
        return seg


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_segformer3d() -> nn.Module:
    """Build SegFormer3D (2 stages, embed_dims=[32,64], BraTS-style 4-modal input)."""
    return SegFormer3D(
        in_channels=4,
        embed_dims=[32, 64],
        num_heads=[2, 4],
        sr_ratios=[2, 1],
        depths=[1, 1],
        num_classes=4,
        decoder_dim=32,
    )


def example_input() -> torch.Tensor:
    """4-modality MRI volume: (1, 4, 16, 16, 16)."""
    return torch.randn(1, 4, 16, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "SegFormer3D (3D volumetric SegFormer, SR-Attention + Mix-FFN + All-MLP decoder)",
        "build_segformer3d",
        "example_input",
        "2022",
        "DC",
    ),
]
