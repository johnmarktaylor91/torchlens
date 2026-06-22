"""PointNet++ with Multi-Scale Grouping (MSG).

Qi, Yi, Su & Guibas (Stanford), NeurIPS 2017, arXiv:1706.02413.
Source: https://github.com/charlesq34/pointnet2

PointNet++ MSG is a variant of PointNet++ where each Set Abstraction (SA)
layer groups neighbors at MULTIPLE radii simultaneously and concatenates the
per-scale PointNet features. This is the distinctive MSG primitive:
  - For each centroid, run ball-query at radius r1, r2 (multiple scales).
  - Apply a separate shared MLP (PointNet) to each scale group.
  - Max-pool each scale -> concatenate per-scale features.
  - This multi-resolution local feature is propagated hierarchically.

NOTE: True FPS (farthest-point sampling) uses a Python for-loop that unrolls
into a large graph under TorchLens tracing. We use random-index sampling
(torch.randperm) which is algebraically equivalent for the atlas visualization
goal and avoids loop-unrolling. The rest of the MSG architecture is faithful:
multi-radius ball-query, per-scale shared MLPs, max-pool + concat.

Compact config: 64 points -> 8 centroids (2 scales) -> 4 centroids (2 scales)
-> global; tiny MLP widths. CPU, random init, forward-only.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Geometry utilities                                                          #
# --------------------------------------------------------------------------- #


def random_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Random (rather than farthest-point) sampling to avoid loop unrolling.
    xyz: (B, N, 3) -> indices (B, npoint).
    """
    B, N, _ = xyz.shape
    idx = torch.stack([torch.randperm(N, device=xyz.device)[:npoint] for _ in range(B)])
    return idx


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Ball-query grouping. Returns indices (B, S, nsample)."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz)  # (B, S, N)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[dists > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    group_idx = torch.where(group_idx == N, first, group_idx)
    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points (B, N, C); idx (B, ...) -> (B, ..., C)."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx]


# --------------------------------------------------------------------------- #
#  Building blocks                                                             #
# --------------------------------------------------------------------------- #


class SharedMLP(nn.Module):
    """Per-point shared MLP (1x1 conv stack) used inside the PointNet layer."""

    def __init__(self, channels: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAbstractionMSG(nn.Module):
    """Multi-Scale Grouping (MSG) Set Abstraction layer.

    Runs ball-query at multiple radii, applies a separate shared MLP to each
    scale, max-pools neighbors per scale, then concatenates all scale features.
    This is the core distinctive primitive of PointNet++ MSG vs SSG.
    """

    def __init__(
        self,
        npoint: int,
        radii: List[float],
        nsamples: List[int],
        in_channel: int,
        mlp_specs: List[List[int]],
    ) -> None:
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlp_specs)
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.mlps = nn.ModuleList()
        for spec in mlp_specs:
            self.mlps.append(SharedMLP([in_channel + 3] + spec))

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """xyz (B,N,3), points (B,N,C) -> new_xyz (B,S,3), new_feat (B,S,sum_mlp[-1])."""
        sample_idx = random_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, sample_idx)  # (B, S, 3)

        scale_feats = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples, self.mlps):
            group_idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, S, K)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)  # relative
            if points is not None:
                grouped_pts = index_points(points, group_idx)
                grouped = torch.cat([grouped_xyz, grouped_pts], dim=-1)  # (B, S, K, 3+C)
            else:
                grouped = grouped_xyz  # (B, S, K, 3)
            grouped = grouped.permute(0, 3, 2, 1)  # (B, 3+C, K, S)
            feat = mlp(grouped)  # (B, mlp[-1], K, S)
            feat = torch.max(feat, dim=2)[0].transpose(1, 2)  # (B, S, mlp[-1])
            scale_feats.append(feat)

        # Concatenate all scale features along the channel dim -- the MSG concat.
        new_feat = torch.cat(scale_feats, dim=-1)  # (B, S, sum_mlp[-1])
        return new_xyz, new_feat


# --------------------------------------------------------------------------- #
#  Full MSG classification network                                             #
# --------------------------------------------------------------------------- #


class PointNet2MSG(nn.Module):
    """PointNet++ Multi-Scale Grouping (MSG) classification network."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # SA1: 2 scales
        self.sa1 = SetAbstractionMSG(
            npoint=8,
            radii=[0.2, 0.4],
            nsamples=[4, 8],
            in_channel=0,
            mlp_specs=[[16, 32], [16, 32]],
        )
        # SA2: 2 scales; input channels = 32+32=64
        self.sa2 = SetAbstractionMSG(
            npoint=4,
            radii=[0.4, 0.8],
            nsamples=[4, 8],
            in_channel=64,
            mlp_specs=[[32, 64], [32, 64]],
        )
        # Global SA: group all remaining points; input channels = 64+64=128
        self.global_mlp = SharedMLP([128 + 3, 64, 128])
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)

        # Global SA: group all remaining points
        B, S, _ = l2_xyz.shape
        grouped_xyz = l2_xyz.unsqueeze(1)  # (B, 1, S, 3)
        grouped = torch.cat([grouped_xyz, l2_pts.unsqueeze(1)], dim=-1)  # (B, 1, S, 3+128)
        grouped = grouped.permute(0, 3, 2, 1)  # (B, 3+C, S, 1)
        feat = self.global_mlp(grouped)  # (B, 128, S, 1)
        feat = torch.max(feat, dim=2)[0].squeeze(-1)  # (B, 128)
        return self.classifier(feat)


# --------------------------------------------------------------------------- #
#  Wrapper & menagerie interface                                               #
# --------------------------------------------------------------------------- #


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = PointNet2MSG(num_classes=10)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.model(xyz)


def build_pointnet2_msg() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    """Small point cloud (1, 64, 3)."""
    torch.manual_seed(0)
    return torch.randn(1, 64, 3)


MENAGERIE_ENTRIES = [
    (
        "PointNet++ MSG (multi-scale grouping set abstraction)",
        "build_pointnet2_msg",
        "example_input",
        "2017",
        "DC",
    ),
]
