"""PointBERT and Point-MAE: Masked pre-training for point cloud Transformers.

PointBERT: Yu et al., CVPR 2022. arXiv:2111.14819
  Source: https://github.com/lulutang0608/Point-BERT

Point-MAE: Pang et al., ECCV 2022. arXiv:2203.06604
  Source: https://github.com/Pang-Yatian/Point-MAE

Both share the same two-stage tokenization/masking + Transformer architecture:

  Stage 1 - Point cloud tokenizer (Point Embedding):
    * Divide the input point cloud into G local groups (mini-patches) via FPS
      to select G centers, then kNN to gather k neighbors per center.
    * Each group is encoded by a shared "mini-PointNet" (DGCNN-style EdgeConv
      or simple shared MLP on relative coordinates) -> a d-dim token per group.

  Stage 2 - Masked Transformer:
    * Randomly mask a fraction (75%) of the G tokens.
    * PointBERT: use a pre-trained dVAE's discrete tokens as BERT targets;
      standard [MASK] token replaces masked positions.
    * Point-MAE: only the *visible* tokens (unmasked) are processed by the
      Transformer; a lightweight decoder reconstructs the masked tokens.

Three MENAGERIE_ENTRIES:
  a) PointBERT_PointTransformer: the tokenizer + full Transformer backbone
     (all tokens visible, standard ViT-style -- the fine-tuning / downstream
     inference graph, not the BERT masking).
  b) PointMAE_PointTransformer: the same tokenizer + full Transformer backbone
     (fine-tuning mode, used after pre-training; architecture identical to PointBERT).
  c) PointMAE_Point_MAE_pretrain: the full pre-training graph -- tokenizer +
     mask + visible-only encoder + mask decoder that reconstructs masked patches.

Faithful compact reimplementation.  Simplifications:
  - G=32 groups (paper uses 64-128), k=16 neighbors.
  - Transformer depth 2 (paper uses 12), d_model=128 (paper 384/768).
  - dVAE target (PointBERT) replaced by a random linear projection for tracing.
  - Point-MAE decoder: 1 Transformer layer + linear head reconstructing k*3 coords.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[batch, farthest].unsqueeze(1)
        dist = ((xyz - c) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(dim=-1)[1]
    return centroids


def knn_group(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """Return (B, G, k, 3) relative coords of k-NN for each center."""
    B, N, _ = xyz.shape
    G = centers.shape[1]
    dists = torch.cdist(centers, xyz)  # (B, G, N)
    idx = dists.topk(k, dim=-1, largest=False)[1]  # (B, G, k)
    bi = torch.arange(B, device=xyz.device).view(B, 1, 1)
    grouped = xyz[bi, idx]  # (B, G, k, 3)
    rel = grouped - centers.unsqueeze(2)  # relative to center
    return rel  # (B, G, k, 3)


# -----------------------------------------------------------------------
# Point Cloud Tokenizer (shared by PointBERT and Point-MAE)
# -----------------------------------------------------------------------


class PointPatchEmbed(nn.Module):
    """Tokenize a point cloud into G patch tokens.

    For each of G FPS-selected centers, gather k-NN relative coords,
    apply a shared mini-PointNet MLP, and max-pool to get a d_model token.
    """

    def __init__(self, G: int = 32, k: int = 16, d_model: int = 128) -> None:
        super().__init__()
        self.G = G
        self.k = k
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """xyz: (B, N, 3) -> tokens (B, G, d_model), centers (B, G, 3)."""
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.G)  # (B, G)
        bi = torch.arange(B, device=xyz.device).unsqueeze(1)
        centers = xyz[bi, fps_idx]  # (B, G, 3)

        rel = knn_group(xyz, centers, self.k)  # (B, G, k, 3)
        # Apply shared MLP to each neighbor: flatten to (B*G*k, 3)
        rel_flat = rel.reshape(B * self.G * self.k, 3)
        feat = self.mlp[0](rel_flat)  # Linear
        feat = self.mlp[1](feat)  # BN
        feat = self.mlp[2](feat)  # ReLU
        feat = self.mlp[3](feat)  # Linear -> (B*G*k, d_model)
        feat = feat.reshape(B, self.G, self.k, self.d_model)
        tokens = feat.max(dim=2)[0]  # (B, G, d_model)
        return tokens, centers


# -----------------------------------------------------------------------
# Positional encoding (3D -> d_model via MLP)
# -----------------------------------------------------------------------


class Pos3dEncoding(nn.Module):
    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )

    def forward(self, centers: torch.Tensor) -> torch.Tensor:
        """centers: (B, G, 3) -> (B, G, d_model)."""
        return self.net(centers)


# -----------------------------------------------------------------------
# Transformer block (standard)
# -----------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, ff_dim: int = 256) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


# -----------------------------------------------------------------------
# PointBERT Transformer backbone (fine-tuning / inference mode)
# Full set of G tokens processed by Transformer -> CLS token -> class logit
# -----------------------------------------------------------------------


class PointBERTTransformer(nn.Module):
    """PointBERT backbone: tokenizer + Transformer, all G tokens visible."""

    def __init__(
        self,
        G: int = 32,
        k: int = 16,
        d_model: int = 128,
        depth: int = 2,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.patch_embed = PointPatchEmbed(G=G, k=k, d_model=d_model)
        self.pos_enc = Pos3dEncoding(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) -> (B, num_classes) class logits."""
        tokens, centers = self.patch_embed(xyz)  # (B, G, d_model)
        pos = self.pos_enc(centers)  # (B, G, d_model)
        tokens = tokens + pos

        B = xyz.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, G+1, d_model)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])  # CLS -> logit


# -----------------------------------------------------------------------
# Point-MAE pre-training model
# Encoder: only visible (unmasked) tokens
# Decoder: mask tokens + positional encoding -> reconstruct masked patches
# -----------------------------------------------------------------------


class PointMAEPretraining(nn.Module):
    """Point-MAE pre-training: masked patch prediction.

    The encoder only processes visible tokens; the decoder inputs the
    encoder output + learned [MASK] tokens (for masked positions) + position
    embeddings, and predicts the raw XYZ coordinates of the masked patches.
    """

    def __init__(
        self,
        G: int = 32,
        k: int = 16,
        d_model: int = 128,
        enc_depth: int = 2,
        dec_depth: int = 1,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.G = G
        self.k = k
        self.mask_ratio = mask_ratio
        self.patch_embed = PointPatchEmbed(G=G, k=k, d_model=d_model)
        self.pos_enc = Pos3dEncoding(d_model)

        # Encoder Transformer (visible tokens only)
        self.enc_blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(enc_depth)])
        self.enc_norm = nn.LayerNorm(d_model)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Decoder Transformer (all tokens: visible + masked)
        self.dec_blocks = nn.ModuleList([TransformerBlock(d_model) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(d_model)

        # Prediction head: reconstruct k*3 coords of the masked patch
        self.pred_head = nn.Linear(d_model, k * 3)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) -> (B, num_masked, k, 3) predicted patch coords."""
        B, N, _ = xyz.shape
        tokens, centers = self.patch_embed(xyz)  # (B, G, d_model)
        pos = self.pos_enc(centers)  # (B, G, d_model)
        tokens = tokens + pos

        # Random masking: split G tokens into visible/masked
        G = self.G
        num_mask = int(G * self.mask_ratio)
        # Shuffle
        noise = torch.rand(B, G, device=xyz.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, : G - num_mask]  # (B, num_visible)

        # Gather visible tokens
        visible = tokens.gather(
            1, ids_keep.unsqueeze(-1).expand(B, G - num_mask, tokens.shape[-1])
        )  # (B, num_visible, d_model)

        # Encoder on visible tokens only
        for blk in self.enc_blocks:
            visible = blk(visible)
        visible = self.enc_norm(visible)

        # Decoder: reconstruct full token sequence
        # Insert mask tokens at masked positions
        mask_tokens = self.mask_token.expand(B, num_mask, -1)  # (B, num_mask, d)
        # Restore order: interleave visible + mask tokens
        full = torch.cat([visible, mask_tokens], dim=1)  # (B, G, d_model)
        # Reorder to original positions
        full = full.gather(1, ids_restore.unsqueeze(-1).expand(B, G, full.shape[-1]))
        # Add full positional encoding
        full = full + pos
        for blk in self.dec_blocks:
            full = blk(full)
        full = self.dec_norm(full)

        # Predict only masked positions
        ids_masked = ids_shuffle[:, G - num_mask :]  # (B, num_mask)
        masked_pred = full.gather(
            1, ids_masked.unsqueeze(-1).expand(B, num_mask, full.shape[-1])
        )  # (B, num_mask, d_model)
        pred = self.pred_head(masked_pred)  # (B, num_mask, k*3)
        return pred.reshape(B, num_mask, self.k, 3)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_pointbert() -> nn.Module:
    """PointBERT: point-patch tokenizer + Transformer backbone (fine-tuning mode)."""
    return PointBERTTransformer(G=8, k=4, d_model=64, depth=2, num_classes=40)


def build_pointmae_transformer() -> nn.Module:
    """Point-MAE Transformer backbone (same architecture as PointBERT, fine-tuning mode)."""
    return PointBERTTransformer(G=8, k=4, d_model=64, depth=2, num_classes=40)


def build_pointmae_pretrain() -> nn.Module:
    """Point-MAE pre-training model (masked encoder + decoder reconstruction)."""
    return PointMAEPretraining(G=8, k=4, d_model=64, enc_depth=2, dec_depth=1, mask_ratio=0.75)


def example_input() -> torch.Tensor:
    """Compact point cloud (1, 32, 3) for fast trace+draw."""
    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "PointBERT (FPS-kNN tokenizer + Transformer, BERT-style masked pre-training backbone)",
        "build_pointbert",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "Point-MAE Transformer (FPS-kNN tokenizer + Transformer, fine-tuning mode)",
        "build_pointmae_transformer",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "Point-MAE pre-training (masked patch prediction: visible-only encoder + decoder)",
        "build_pointmae_pretrain",
        "example_input",
        "2022",
        "DC",
    ),
]
