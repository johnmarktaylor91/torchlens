"""PointMLP: Rethinking Network Design and Local Geometry in Point Cloud Learning.

Ma, Zhang, Xu & Liu (Monash / Tsinghua), ICLR 2022, arXiv:2202.07123.
Source: https://github.com/ma-xu/pointMLP-pytorch

PointMLP removes all sophisticated local geometry extractors and shows that a
pure residual-MLP with a simple GEOMETRIC AFFINE MODULE (GAM) can achieve SOTA.
Distinctive primitives:
  1. Geometric Affine Module (GAM): normalize a local neighborhood by subtracting
     the centroid and dividing by a learned scale alpha + shift beta (per-channel
     affine), so the local geometry is in a canonical form before the MLP.
  2. Residual-MLP blocks: plain MLP with skip connections (no attention, no
     convolutions with kernel tricks, no adaptive sampling).
  3. The full architecture is: FPS -> GAM(normalize neighborhood) ->
     residual-MLP-stack -> max-pool -> repeat hierarchically.

NOTE: FPS replaced by random sampling to avoid loop-unrolling.
Compact: 64 points -> 16 -> 4; 2 residual blocks per stage; small dims.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Utilities                                                                   #
# --------------------------------------------------------------------------- #


def random_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    return torch.stack([torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)])


def knn_query(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    dists = torch.cdist(new_xyz, xyz)
    return torch.topk(dists, k, dim=-1, largest=False)[1]


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    batch = torch.arange(B, device=points.device).view(B, *([1] * (idx.dim() - 1))).expand_as(idx)
    return points[batch, idx]


# --------------------------------------------------------------------------- #
#  Geometric Affine Module (GAM) -- the key PointMLP primitive                #
# --------------------------------------------------------------------------- #


class GeometricAffine(nn.Module):
    """Normalize local neighborhoods with learned affine (alpha, beta).

    Given grouped points (B, S, K, C), subtract centroid and apply
    per-channel learned scale alpha and shift beta.  This puts each local
    neighborhood into a canonical coordinate frame with no distance bias.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        # grouped: (B, S, K, C)
        mean = grouped.mean(dim=2, keepdim=True)  # (B, S, 1, C)
        std = grouped.std(dim=2, keepdim=True) + 1e-5  # (B, S, 1, C)
        norm = (grouped - mean) / std  # (B, S, K, C)
        return self.alpha * norm + self.beta


# --------------------------------------------------------------------------- #
#  Residual MLP block                                                          #
# --------------------------------------------------------------------------- #


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*S, dim) or (B, dim)
        return self.act(x + self.net(x))


# --------------------------------------------------------------------------- #
#  PointMLP stage                                                              #
# --------------------------------------------------------------------------- #


class PointMLPStage(nn.Module):
    """One hierarchical PointMLP stage: sample -> GAM -> residual MLP -> max-pool."""

    def __init__(self, npoint: int, k: int, in_dim: int, out_dim: int, n_blocks: int = 2) -> None:
        super().__init__()
        self.npoint = npoint
        self.k = k
        self.proj = nn.Linear(in_dim * k, out_dim)
        self.gam = GeometricAffine(in_dim)
        self.blocks = nn.ModuleList([ResMLPBlock(out_dim) for _ in range(n_blocks)])
        self.skip_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple:
        # xyz: (B, N, 3), feat: (B, N, C)
        B, N, C = feat.shape
        sample_idx = random_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, sample_idx)  # (B, S, 3)

        knn_idx = knn_query(xyz, new_xyz, self.k)  # (B, S, k)
        grouped = index_points(feat, knn_idx)  # (B, S, k, C)

        # Geometric Affine normalization (the key PointMLP step)
        grouped = self.gam(grouped)  # (B, S, k, C)

        # Max-pool then project
        S = self.npoint
        pooled = grouped.max(dim=2)[0]  # (B, S, C) - max over k neighbors
        out = self.proj(grouped.reshape(B, S, -1))  # (B, S, out_dim)

        # Residual MLP blocks
        out_flat = out.reshape(B * S, -1)
        for blk in self.blocks:
            out_flat = blk(out_flat)
        out = out_flat.reshape(B, S, -1)  # (B, S, out_dim)
        return new_xyz, out


# --------------------------------------------------------------------------- #
#  Full PointMLP network                                                       #
# --------------------------------------------------------------------------- #


class PointMLP(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Input projection: xyz (3) -> 32
        self.input_proj = nn.Linear(3, 32)
        # Stage 1: 64 -> 16 points
        self.stage1 = PointMLPStage(npoint=16, k=8, in_dim=32, out_dim=64, n_blocks=2)
        # Stage 2: 16 -> 4 points
        self.stage2 = PointMLPStage(npoint=4, k=4, in_dim=64, out_dim=128, n_blocks=2)
        # Global: max pool all
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        feat = self.input_proj(xyz)  # (B, N, 32)
        xyz1, feat1 = self.stage1(xyz, feat)
        xyz2, feat2 = self.stage2(xyz1, feat1)
        global_feat = feat2.max(dim=1)[0]  # (B, 128)
        return self.head(global_feat)


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointMLP(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_pointmlp() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointMLP (geometric affine module + residual-MLP hierarchy)",
        "build_pointmlp",
        "example_input",
        "2022",
        "DC",
    ),
]
