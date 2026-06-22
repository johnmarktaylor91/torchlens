"""PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.

Qi, Yi, Su & Guibas (Stanford), NeurIPS 2017, arXiv:1706.02413.
Source: https://github.com/charlesq34/pointnet2  (and the CUDA ``pointnet2-ops`` /
``pointnet2_ops_lib`` used by most PyTorch ports).

PointNet++'s defining module is the **Set Abstraction (SA) layer**, stacked
hierarchically:
  1. Sampling: farthest-point sampling (FPS) selects ``npoint`` centroids.
  2. Grouping: ball query gathers up to ``nsample`` neighbors within ``radius`` of
     each centroid (single-scale grouping, SSG).
  3. PointNet layer: a shared MLP on each grouped neighbor's [feature, relative-xyz],
     followed by a max-pool over neighbors -> one feature per centroid.
Two SA layers + a global SA (group-all) feed a small classifier.

The CEILING in the menagerie is ``pointnet2-ops``: FPS, ball-query, and grouping are
shipped as custom CUDA kernels. They are OPTIMIZATIONS of operations expressible in
plain torch -- FPS is an iterative argmax over running min-distances, ball-query is a
radius threshold on a pairwise distance matrix, grouping is a gather. This module
reimplements the full SSG architecture (two SA layers + group-all + classifier) with
pure-torch FPS / ball-query (via ``torch.cdist``) so it traces and renders.

Small cloud: 256 points -> 64 -> 16 centroids.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative farthest-point sampling. xyz: (B, N, 3) -> indices (B, npoint)."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)  # deterministic start
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)  # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Group neighbors within ``radius`` of each centroid. Returns indices (B, S, nsample)."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz)  # (B, S, N)
    # Default to the nearest indices, then mask out-of-radius to the closest in-radius.
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[dists > radius] = N  # sentinel
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # (B, S, nsample)
    # Replace sentinels (no neighbor) with the first (nearest) valid neighbor.
    first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    group_idx = torch.where(group_idx == N, first, group_idx)
    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index. points: (B, N, C); idx: (B, ...) -> (B, ..., C)."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx]


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


class SetAbstraction(nn.Module):
    """Single-scale-grouping (SSG) set-abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: List[int],
        group_all: bool = False,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp = SharedMLP([in_channel + 3] + mlp)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):
        # xyz: (B, N, 3); points: (B, N, C) feature (or None).
        B, N, _ = xyz.shape
        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            grouped = (
                torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
                if points is not None
                else grouped_xyz
            )
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)
            group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B, S, K)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)  # relative coords
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped = grouped_xyz
        # grouped: (B, S, K, C+3) -> conv expects (B, C, K, S).
        grouped = grouped.permute(0, 3, 2, 1)
        feat = self.mlp(grouped)  # (B, mlp[-1], K, S)
        feat = torch.max(feat, dim=2)[0].transpose(1, 2)  # max over neighbors -> (B, S, mlp[-1])
        return new_xyz, feat


class PointNet2SSG(nn.Module):
    """PointNet++ (SSG) classification network."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(64, 0.2, 16, in_channel=0, mlp=[32, 32, 64])
        self.sa2 = SetAbstraction(16, 0.4, 16, in_channel=64, mlp=[64, 64, 128])
        self.sa3 = SetAbstraction(
            None, None, None, in_channel=128, mlp=[128, 256, 256], group_all=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 1, 256)
        global_feat = l3_points.squeeze(1)  # (B, 256)
        return self.classifier(global_feat)


def build_pointnet2() -> nn.Module:
    """Build PointNet++ (SSG) classifier."""
    return PointNet2SSG(num_classes=10)


def example_input() -> torch.Tensor:
    """A small point cloud ``(1, 256, 3)``."""
    return torch.randn(1, 256, 3)


MENAGERIE_ENTRIES = [
    (
        "PointNet++ (hierarchical set abstraction, SSG)",
        "build_pointnet2",
        "example_input",
        "2017",
        "DC",
    ),
]
