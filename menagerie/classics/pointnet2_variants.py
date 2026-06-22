"""PointNet++ Multi-Scale Grouping (MSG) variants.

Qi, Yi, Su & Guibas (Stanford), NeurIPS 2017, arXiv:1706.02413.
Source: https://github.com/charlesq34/pointnet2

This module covers the TWO architectural variants NOT in pointnet2.py (SSG):
  1. PointNet2_cls_msg -- classification with multi-scale grouping (MSG).
  2. PointNet2_part_seg_msg -- part segmentation with MSG set abstraction +
     feature propagation (FP) upsampling decoder.

MSG vs SSG:
  In SSG each set abstraction uses a single (radius, nsample) grouping.
  In MSG it uses MULTIPLE radii at each SA layer and concatenates the per-scale
  features -- capturing multi-scale context at each hierarchical level.

Part segmentation adds a decoder of Feature Propagation (FP) layers that
upsample back to the original point resolution using inverse-distance-weighted
interpolation of features from SA layers.

Both use the same pure-torch FPS / ball-query helpers from pointnet2.py logic,
reimplemented here so this file is self-contained.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Shared primitives (duplicated from pointnet2.py for self-containedness)
# -----------------------------------------------------------------------


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    batch = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def ball_query(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).expand(B, S, N).clone()
    group_idx[dists > radius] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    group_idx = torch.where(group_idx == N, first, group_idx)
    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx]


class SharedMLP(nn.Module):
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


# -----------------------------------------------------------------------
# Multi-Scale Grouping (MSG) Set Abstraction
# -----------------------------------------------------------------------


class SetAbstractionMSG(nn.Module):
    """MSG set-abstraction: multiple (radius, nsample, mlp) scales, features concatenated."""

    def __init__(
        self,
        npoint: int,
        radius_list: List[float],
        nsample_list: List[int],
        in_channel: int,
        mlp_list: List[List[int]],
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlps = nn.ModuleList()
        for mlp_channels in mlp_list:
            self.mlps.append(SharedMLP([in_channel + 3] + mlp_channels))

    def forward(
        self,
        xyz: torch.Tensor,
        points: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = xyz.shape[0]
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)

        new_points_list = []
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlps):
            group_idx = ball_query(radius, nsample, xyz, new_xyz)  # (B, S, K)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_pts = index_points(points, group_idx)
                grouped = torch.cat([grouped_xyz, grouped_pts], dim=-1)
            else:
                grouped = grouped_xyz
            grouped = grouped.permute(0, 3, 2, 1)  # (B, C+3, K, S)
            feat = mlp(grouped)
            feat = torch.max(feat, dim=2)[0].transpose(1, 2)  # (B, S, mlp[-1])
            new_points_list.append(feat)

        new_points = torch.cat(new_points_list, dim=-1)  # (B, S, sum_mlp[-1])
        return new_xyz, new_points


# -----------------------------------------------------------------------
# MSG Classification network
# -----------------------------------------------------------------------


class PointNet2MsgClassifier(nn.Module):
    """PointNet++ classification with multi-scale grouping (MSG)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # SA1: 2 scales -- compact for fast trace
        self.sa1 = SetAbstractionMSG(
            npoint=16,
            radius_list=[0.2, 0.4],
            nsample_list=[4, 8],
            in_channel=0,
            mlp_list=[[16, 32], [16, 32]],
        )
        sa1_out = 32 + 32  # = 64
        # SA2: 2 scales
        self.sa2 = SetAbstractionMSG(
            npoint=4,
            radius_list=[0.4, 0.8],
            nsample_list=[4, 8],
            in_channel=sa1_out,
            mlp_list=[[32, 64], [32, 64]],
        )
        sa2_out = 64 + 64  # = 128
        # Global SA (group-all, large radius)
        self.sa3 = SetAbstractionMSG(
            npoint=1,
            radius_list=[1000.0],
            nsample_list=[4],
            in_channel=sa2_out,
            mlp_list=[[128, 128]],
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        _, l3_pts = self.sa3(l2_xyz, l2_pts)
        global_feat = l3_pts.squeeze(1)
        return self.classifier(global_feat)


# -----------------------------------------------------------------------
# Feature Propagation (FP) for part segmentation
# -----------------------------------------------------------------------


class FeaturePropagation(nn.Module):
    """FP layer: inverse-distance interpolation + MLP for upsampling decoder."""

    def __init__(self, in_channel: int, mlp: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        cin = in_channel
        for cout in mlp:
            layers += [nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout), nn.ReLU(inplace=True)]
            cin = cout
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,  # upsampled positions  (B, N1, 3)
        xyz2: torch.Tensor,  # downsampled positions (B, N2, 3)
        points1: Optional[torch.Tensor],  # (B, N1, C1) skip features
        points2: torch.Tensor,  # (B, N2, C2) low-res features
    ) -> torch.Tensor:
        B, N1, _ = xyz1.shape
        N2 = xyz2.shape[1]

        if N2 == 1:
            # Trivial: broadcast the single global feature
            interp = points2.expand(B, N1, -1)
        else:
            dists = torch.cdist(xyz1, xyz2)  # (B, N1, N2)
            k = min(3, N2)
            d, idx = dists.topk(k, dim=-1, largest=False)  # (B, N1, k)
            d = d.clamp(min=1e-10)
            w = 1.0 / d  # (B, N1, k)
            w = w / w.sum(dim=-1, keepdim=True)

            # Gather features at idx: (B, N1, k, C2)
            bi = torch.arange(B, device=xyz2.device).view(B, 1, 1)
            gathered = points2[bi, idx]  # (B, N1, k, C2)
            interp = (w.unsqueeze(-1) * gathered).sum(dim=2)  # (B, N1, C2)

        if points1 is not None:
            new_points = torch.cat([points1, interp], dim=-1)  # (B, N1, C1+C2)
        else:
            new_points = interp

        new_points = new_points.transpose(1, 2)  # (B, C, N1) for Conv1d
        return self.mlp(new_points).transpose(1, 2)  # (B, N1, mlp[-1])


# -----------------------------------------------------------------------
# Part Segmentation network
# -----------------------------------------------------------------------


class PointNet2PartSegMSG(nn.Module):
    """PointNet++ part segmentation with MSG encoder + FP decoder."""

    def __init__(self, num_part_classes: int = 50) -> None:
        super().__init__()
        # Encoder (MSG SA layers) -- compact for fast trace
        self.sa1 = SetAbstractionMSG(
            npoint=8,
            radius_list=[0.2, 0.4],
            nsample_list=[4, 8],
            in_channel=0,
            mlp_list=[[16, 32], [16, 32]],
        )
        sa1_out = 32 + 32  # 64
        self.sa2 = SetAbstractionMSG(
            npoint=2,
            radius_list=[0.4, 0.8],
            nsample_list=[4, 8],
            in_channel=sa1_out,
            mlp_list=[[32, 64], [32, 64]],
        )
        sa2_out = 128
        self.sa3 = SetAbstractionMSG(
            npoint=1,
            radius_list=[1000.0],
            nsample_list=[2],
            in_channel=sa2_out,
            mlp_list=[[128, 128]],
        )

        # Decoder (FP layers, upsampling back to original resolution)
        self.fp3 = FeaturePropagation(128 + sa2_out, [128, 64])
        self.fp2 = FeaturePropagation(64 + sa1_out, [64, 32])
        # FP1: upsample back to original N points
        self.fp1 = FeaturePropagation(32, [32, 32])

        # Per-point classifier
        self.head = nn.Sequential(
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_part_classes, 1),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B, N, 3)
        l0_xyz = xyz  # (B, N, 3)

        l1_xyz, l1_pts = self.sa1(xyz, None)  # (B, 64, 3), (B, 64, 96)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)  # (B, 16, 3), (B, 16, 256)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)  # (B, 1, 3), (B, 1, 256)

        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)  # (B, sa2_npts, 64)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)  # (B, sa1_npts, 32)
        l0_pts = self.fp1(l0_xyz, l1_xyz, None, l1_pts)  # (B, N, 32)

        out = self.head(l0_pts.transpose(1, 2))  # (B, num_classes, N)
        return out.transpose(1, 2)  # (B, N, num_classes)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_cls_msg() -> nn.Module:
    """PointNet++ MSG classification network."""
    return PointNet2MsgClassifier(num_classes=10)


def build_part_seg_msg() -> nn.Module:
    """PointNet++ MSG part-segmentation network (encoder + FP decoder)."""
    return PointNet2PartSegMSG(num_part_classes=50)


def example_input() -> torch.Tensor:
    """Compact point cloud (1, 32, 3) for fast trace+draw."""
    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "PointNet++ MSG classifier (multi-scale-grouping set abstraction)",
        "build_cls_msg",
        "example_input",
        "2017",
        "DC",
    ),
    (
        "PointNet++ part segmentation MSG (MSG encoder + FP upsampling decoder)",
        "build_part_seg_msg",
        "example_input",
        "2017",
        "DC",
    ),
]
