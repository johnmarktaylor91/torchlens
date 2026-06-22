"""PointASNL: Robust Point Clouds Processing using Nonlocal Spatial Correlations.

Yan, Zheng, Li, Wang, Cui, Qi & Tian (Meituan / Stanford), CVPR 2020, arXiv:2003.00492.
Source: https://github.com/yanx27/PointASNL

PointASNL has two distinctive primitives:
  1. Adaptive Sampling (AS) module: instead of fixed FPS centroids, learn to
     RE-WEIGHT and SHIFT sampled points via self-attention over their neighbors.
     A small attention network predicts per-point offsets + importance weights,
     so the sampled centroid positions adapt to the data.
  2. Local-NonLocal (L-NL) module: each SA layer combines a LOCAL branch
     (standard grouped PointNet on neighbors) with a NONLOCAL branch (global
     self-attention across all centroids to capture long-range correlations).
     Their outputs are added.

Compact config: 64 points -> 16 AS-centroids -> 4 AS-centroids.
NOTE: FPS replaced by random sampling for tractable tracing.
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
#  Adaptive Sampling (AS) module                                               #
# --------------------------------------------------------------------------- #


class AdaptiveSampling(nn.Module):
    """AS module: re-weight/shift sampled point positions via neighbor attention.

    For each initial (random-sampled) centroid:
    - Gather k neighbors.
    - Compute attention scores (small MLP on relative xyz).
    - Weighted combination of neighbor positions gives the adapted centroid.
    - Attention-weighted feature sum gives adapted features.
    """

    def __init__(self, in_channel: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        # Attention score: relative xyz (3) -> scalar weight
        self.attn_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )
        # Feature transform for neighbor features
        self.feat_proj = nn.Linear(in_channel, in_channel)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple:
        """xyz (B,N,3), feat (B,N,C) -> adapted_xyz (B,S,3), adapted_feat (B,S,C)."""
        B, N, C = feat.shape
        S = N // 4  # sample 1/4 of points

        # Initial random centroids
        sample_idx = random_sample(xyz, S)
        init_xyz = index_points(xyz, sample_idx)  # (B, S, 3)
        init_feat = index_points(feat, sample_idx)  # (B, S, C)

        # Adaptive: for each centroid, attend over k neighbors to shift position
        knn_idx = knn_query(xyz, init_xyz, self.k)  # (B, S, k)
        grouped_xyz = index_points(xyz, knn_idx)  # (B, S, k, 3)
        grouped_feat = index_points(feat, knn_idx)  # (B, S, k, C)

        delta_xyz = grouped_xyz - init_xyz.unsqueeze(2)  # relative (B, S, k, 3)

        # Attention weights over neighbors
        attn_logits = self.attn_net(delta_xyz).squeeze(-1)  # (B, S, k)
        attn = F.softmax(attn_logits, dim=-1)  # (B, S, k)

        # Adapted centroid = weighted sum of neighbor positions
        adapted_xyz = (attn.unsqueeze(-1) * grouped_xyz).sum(dim=2)  # (B, S, 3)

        # Adapted feature = weighted sum of neighbor features + projection
        adapted_feat = (attn.unsqueeze(-1) * self.feat_proj(grouped_feat)).sum(dim=2)  # (B, S, C)
        adapted_feat = adapted_feat + init_feat  # residual

        return adapted_xyz, adapted_feat


# --------------------------------------------------------------------------- #
#  Local-NonLocal (L-NL) module                                                #
# --------------------------------------------------------------------------- #


class LocalNonLocalModule(nn.Module):
    """L-NL: combines local PointNet with global self-attention.

    local branch: grouped MLP + max-pool over k neighbors
    nonlocal branch: scaled dot-product self-attention across all S centroids
    output: local + nonlocal (element-wise sum)
    """

    def __init__(self, in_channel: int, out_channel: int, k: int = 8) -> None:
        super().__init__()
        self.k = k

        # Local branch: shared MLP on grouped neighbors
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel + 3, out_channel),
            nn.ReLU(inplace=True),
        )

        # Nonlocal branch: self-attention
        self.q_proj = nn.Linear(in_channel, out_channel)
        self.k_proj = nn.Linear(in_channel, out_channel)
        self.v_proj = nn.Linear(in_channel, out_channel)
        self.out_proj = nn.Linear(out_channel, out_channel)

        self.norm = nn.LayerNorm(out_channel)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """xyz (B,S,3), feat (B,S,C) -> (B,S,out_channel)."""
        B, S, C = feat.shape

        # ----- Local branch -----
        knn_idx = knn_query(xyz, xyz, self.k)  # (B, S, k) -- self-neighbors
        grouped_xyz = index_points(xyz, knn_idx)  # (B, S, k, 3)
        grouped_feat = index_points(feat, knn_idx)  # (B, S, k, C)
        delta = grouped_xyz - xyz.unsqueeze(2)  # relative (B, S, k, 3)
        local_in = torch.cat([grouped_feat, delta], dim=-1)  # (B, S, k, C+3)
        local_out = self.local_mlp(local_in)  # (B, S, k, out_channel)
        local_out = local_out.max(dim=2)[0]  # (B, S, out_channel)

        # ----- Nonlocal branch (global self-attention over S centroids) -----
        q = self.q_proj(feat)  # (B, S, D)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        scale = q.shape[-1] ** -0.5
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * scale, dim=-1)  # (B, S, S)
        nonlocal_out = self.out_proj(torch.bmm(attn, v))  # (B, S, out_channel)

        # Combine
        out = local_out + nonlocal_out
        return self.norm(out)


# --------------------------------------------------------------------------- #
#  Full PointASNL network                                                      #
# --------------------------------------------------------------------------- #


class PointASNLNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Input embedding
        self.input_proj = nn.Linear(3, 32)

        # Stage 1: AS module + L-NL
        self.as1 = AdaptiveSampling(in_channel=32, k=8)
        self.lnl1 = LocalNonLocalModule(in_channel=32, out_channel=64, k=8)

        # Stage 2: AS module + L-NL
        self.as2 = AdaptiveSampling(in_channel=64, k=4)
        self.lnl2 = LocalNonLocalModule(in_channel=64, out_channel=128, k=4)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        feat = self.input_proj(xyz)  # (B, N, 32)

        # Stage 1
        xyz1, feat1 = self.as1(xyz, feat)  # (B, N//4, 32) AS-adapted
        feat1 = self.lnl1(xyz1, feat1)  # (B, N//4, 64)

        # Stage 2
        xyz2, feat2 = self.as2(xyz1, feat1)  # (B, N//16, 64)
        feat2 = self.lnl2(xyz2, feat2)  # (B, N//16, 128)

        global_feat = feat2.max(dim=1)[0]  # (B, 128)
        return self.head(global_feat)


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = PointASNLNet(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_pointasnl_model() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointASNL (adaptive sampling + local-nonlocal point cloud learning)",
        "build_pointasnl_model",
        "example_input",
        "2020",
        "DC",
    ),
]
