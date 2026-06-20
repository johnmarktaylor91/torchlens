"""Point Transformer (Hengshuang Zhao et al.).

Zhao, Jiang, Jia, Torr & Koltun, ICCV 2021.
Paper: https://arxiv.org/abs/2012.09164

The Point Transformer layer is a *vector* self-attention operating on local kNN
neighborhoods of a point cloud. For each center point i and neighbor j it forms
the attention vector

    gamma( phi(x_i) - psi(x_j) + delta_ij )  (subtraction relation)

where delta_ij = theta(p_i - p_j) is a learned positional encoding MLP over the
relative 3D coordinates. The same delta is added to the value branch. Attention
is softmax-normalized over the neighborhood, then aggregated as a weighted sum of
(alpha(x_j) + delta_ij). gamma is a small MLP producing per-channel (vector)
attention weights -- this is the defining departure from scalar dot-product
attention.

This is a faithful random-init reimplementation with pure-torch kNN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbors of each point (B, N, k)."""
    dist = torch.cdist(xyz, xyz)
    return dist.topk(k, dim=-1, largest=False)[1]


class PointTransformerLayer(nn.Module):
    """Vector self-attention layer with subtraction relation + positional encoding."""

    def __init__(self, dim: int, k: int = 16) -> None:
        super().__init__()
        self.k = k
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        # Positional encoding MLP theta: relative xyz (3) -> dim.
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        # Attention weight MLP gamma: dim -> dim (vector attention).
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        b, n, _ = xyz.shape
        idx = knn_indices(xyz, self.k)  # (B, N, k)
        batch_idx = torch.arange(b, device=xyz.device).view(b, 1, 1)

        q = self.to_q(feats)  # (B, N, dim)
        k = self.to_k(feats)
        v = self.to_v(feats)

        k_neigh = k[batch_idx, idx]  # (B, N, k, dim)
        v_neigh = v[batch_idx, idx]  # (B, N, k, dim)
        xyz_neigh = xyz[batch_idx, idx]  # (B, N, k, 3)
        rel_pos = xyz.unsqueeze(2) - xyz_neigh  # (B, N, k, 3)
        delta = self.pos_mlp(rel_pos)  # (B, N, k, dim)

        # Subtraction relation: phi(x_i) - psi(x_j) + delta.
        rel = q.unsqueeze(2) - k_neigh + delta  # (B, N, k, dim)
        attn = self.attn_mlp(rel)  # (B, N, k, dim) vector attention logits
        attn = F.softmax(attn, dim=2)  # normalize over neighborhood

        agg = attn * (v_neigh + delta)  # (B, N, k, dim)
        out = agg.sum(dim=2)  # (B, N, dim)
        return out


class PointTransformerBlock(nn.Module):
    """Residual Point-Transformer block: linear -> PT attention -> linear, with skip."""

    def __init__(self, dim: int, k: int = 16) -> None:
        super().__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.attn = PointTransformerLayer(dim, k=k)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        identity = feats
        x = F.relu(self.linear_in(feats))
        x = self.attn(xyz, x)
        x = self.linear_out(x)
        return F.relu(x + identity)


class PointTransformerClassifier(nn.Module):
    """Point cloud classifier from stacked Point-Transformer blocks."""

    def __init__(self, dim: int = 32, num_classes: int = 40, k: int = 16) -> None:
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.block1 = PointTransformerBlock(dim, k=k)
        self.block2 = PointTransformerBlock(dim, k=k)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feats = self.embed(xyz)
        feats = self.block1(xyz, feats)
        feats = self.block2(xyz, feats)
        pooled = feats.max(dim=1)[0]
        return self.classifier(pooled)


def build() -> nn.Module:
    """Build the Point Transformer point-cloud classifier."""
    return PointTransformerClassifier(dim=32, num_classes=40, k=16)


def example_input() -> torch.Tensor:
    """Small point cloud ``(1, 64, 3)`` (few points keeps the kNN-attention trace compact)."""
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "Point Transformer (vector self-attention on point clouds, Zhao ICCV'21)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
