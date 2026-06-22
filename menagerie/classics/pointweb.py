"""PointWeb: Enhancing Local Neighborhood Features for Point Cloud Processing.

Zhao, Hu, Wang & Huang (Zhejiang University), CVPR 2019, arXiv:1901.08396.
Source: https://github.com/hszhao/PointWeb

PointWeb's distinctive primitive is the Adaptive Feature Adjustment (AFA) module:
  - For each local neighborhood of K points, densely connect ALL K*(K-1) ordered
    pairs (i, j) via a pairwise feature-difference map.
  - An MLP on each pair's feature difference learns an "impact" (how much point j
    should adjust point i's features).
  - These impacts are aggregated per-point, producing a context-aware adjustment
    that is added back to the original features.
  - This dense pairwise connectivity captures the full local topology -- unlike
    PointNet's max-pool which ignores inter-point relationships.

Compact config: K=4 neighbors, 64 points -> 16 -> 4.
NOTE: FPS replaced by random sampling to avoid loop-unrolling.
"""

from __future__ import annotations

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
#  Adaptive Feature Adjustment (AFA) -- PointWeb's key primitive              #
# --------------------------------------------------------------------------- #


class AFA(nn.Module):
    """Adaptive Feature Adjustment: dense pairwise impact between ALL points in
    a local region.

    For each pair (i, j) in the K-neighborhood:
      - Compute feature difference f_i - f_j
      - MLP maps this difference to an adjustment delta_f_i (impact of j on i)
    Sum impacts over j for each i -> adjustment added to original features.
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        # MLP on per-pair feature difference: feat_dim -> feat_dim
        self.impact_mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, grouped: torch.Tensor) -> torch.Tensor:
        """grouped: (B, S, K, C) -- local neighborhood features.
        Returns adjusted features (B, S, K, C).
        """
        B, S, K, C = grouped.shape

        # Pairwise feature differences: f_i - f_j for all (i,j)
        # grouped[:, :, :, None, :] - grouped[:, :, None, :, :] -> (B, S, K, K, C)
        fi = grouped.unsqueeze(3)  # (B, S, K, 1, C)
        fj = grouped.unsqueeze(2)  # (B, S, 1, K, C)
        diff = fi - fj  # (B, S, K, K, C) all pairs

        # Compute impact for each pair
        impact = self.impact_mlp(diff)  # (B, S, K, K, C)

        # Aggregate: for each point i, sum impacts from all j
        adjustment = impact.sum(dim=3) / K  # (B, S, K, C)

        return grouped + adjustment


# --------------------------------------------------------------------------- #
#  PointWeb set-abstraction layer                                              #
# --------------------------------------------------------------------------- #


class PointWebLayer(nn.Module):
    """PointWeb SA layer: gather neighborhood, apply AFA, then max-pool."""

    def __init__(self, npoint: int, k: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.npoint = npoint
        self.k = k
        # Initial projection + xyz concat
        self.proj = nn.Sequential(
            nn.Linear(in_dim + 3, out_dim),
            nn.ReLU(inplace=True),
        )
        self.afa = AFA(out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple:
        B, N, C = feat.shape
        sample_idx = random_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, sample_idx)  # (B, S, 3)

        knn_idx = knn_query(xyz, new_xyz, self.k)  # (B, S, k)
        grouped_xyz = index_points(xyz, knn_idx)  # (B, S, k, 3)
        grouped_feat = index_points(feat, knn_idx)  # (B, S, k, C)

        rel_xyz = grouped_xyz - new_xyz.unsqueeze(2)  # relative
        x = torch.cat([grouped_feat, rel_xyz], dim=-1)  # (B, S, k, C+3)

        S = self.npoint
        x = self.proj(x)  # (B, S, k, out_dim)

        # AFA: dense pairwise feature adjustment
        x = self.afa(x)  # (B, S, k, out_dim)

        # Max-pool over k neighbors
        out = x.max(dim=2)[0]  # (B, S, out_dim)
        out = self.bn(out.reshape(B * S, -1)).reshape(B, S, -1)
        return new_xyz, F.relu(out, inplace=True)


# --------------------------------------------------------------------------- #
#  Full PointWeb network                                                       #
# --------------------------------------------------------------------------- #


class PointWebNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Stage 1: 64 -> 16, k=4
        self.layer1 = PointWebLayer(npoint=16, k=4, in_dim=3, out_dim=32)
        # Stage 2: 16 -> 4, k=4
        self.layer2 = PointWebLayer(npoint=4, k=4, in_dim=32, out_dim=64)
        # Global
        self.layer3 = PointWebLayer(npoint=1, k=4, in_dim=64, out_dim=128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz1, f1 = self.layer1(xyz, xyz)
        xyz2, f2 = self.layer2(xyz1, f1)
        _, f3 = self.layer3(xyz2, f2)
        return self.head(f3.squeeze(1))


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointWebNet(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_pointweb_model() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointWeb (adaptive feature adjustment via dense pairwise impact map)",
        "build_pointweb_model",
        "example_input",
        "2019",
        "DC",
    ),
]
