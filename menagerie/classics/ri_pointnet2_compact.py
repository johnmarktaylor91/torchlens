"""Compact PointNet++ SSG classifier for RI atlas rows.

Qi, Yi, Su & Guibas, 2017, "PointNet++: Deep Hierarchical Feature Learning
on Point Sets in a Metric Space."  Source: https://github.com/charlesq34/pointnet2

This is a smaller single-scale-grouping (SSG) PointNet++ reimplementation for
draw-constrained catalog rows.  It preserves the defining architecture:
farthest-point sampling, radius/ball grouping, shared pointwise MLP over
relative coordinates, max pooling over local neighborhoods, stacked set
abstraction, and a classifier.  The cloud is reduced to 32 -> 8 -> 2 centroids.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Sample centroids with deterministic farthest-point sampling.

    Parameters
    ----------
    xyz:
        Point coordinates of shape ``(B, N, 3)``.
    npoint:
        Number of centroids to sample.

    Returns
    -------
    torch.Tensor
        Sampled centroid indices of shape ``(B, npoint)``.
    """

    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, num_points), 1e10, device=xyz.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    batch = torch.arange(batch_size, device=xyz.device)
    for idx in range(npoint):
        centroids[:, idx] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather batched point tensors by batched indices.

    Parameters
    ----------
    points:
        Point or feature tensor of shape ``(B, N, C)``.
    idx:
        Long indices of shape ``(B, ...)``.

    Returns
    -------
    torch.Tensor
        Gathered tensor of shape ``(B, ..., C)``.
    """

    batch_size = points.shape[0]
    view_shape = [batch_size] + [1] * (idx.dim() - 1)
    batch = torch.arange(batch_size, device=points.device).view(view_shape).expand_as(idx)
    return points[batch, idx]


def ball_query(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return fixed-size local neighborhoods around centroids.

    Parameters
    ----------
    radius:
        Ball-query radius.
    nsample:
        Maximum number of neighbors per centroid.
    xyz:
        Full point coordinates of shape ``(B, N, 3)``.
    new_xyz:
        Centroid coordinates of shape ``(B, S, 3)``.

    Returns
    -------
    torch.Tensor
        Neighbor indices of shape ``(B, S, nsample)``.
    """

    batch_size, num_points, _ = xyz.shape
    num_centroids = new_xyz.shape[1]
    dists = torch.cdist(new_xyz, xyz)
    group_idx = (
        torch.arange(num_points, device=xyz.device)
        .view(1, 1, num_points)
        .expand(batch_size, num_centroids, num_points)
        .clone()
    )
    group_idx[dists > radius] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    first = group_idx[:, :, 0:1].expand(-1, -1, nsample)
    return torch.where(group_idx == num_points, first, group_idx)


class SharedMLP(nn.Module):
    """Shared pointwise MLP implemented with ``1x1`` convolutions."""

    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv2d(channels[idx], channels[idx + 1], 1),
                    nn.BatchNorm2d(channels[idx + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared MLP.

        Parameters
        ----------
        x:
            Grouped point features of shape ``(B, C, K, S)``.

        Returns
        -------
        torch.Tensor
            Transformed features of shape ``(B, C_out, K, S)``.
        """

        return self.net(x)


class SetAbstraction(nn.Module):
    """PointNet++ single-scale set-abstraction layer."""

    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list[int],
        group_all: bool = False,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp = SharedMLP([in_channel + 3, *mlp])

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply sampling, grouping, shared MLP, and local max pooling.

        Parameters
        ----------
        xyz:
            Point coordinates of shape ``(B, N, 3)``.
        points:
            Optional point features of shape ``(B, N, C)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            New coordinates and centroid features.
        """

        batch_size = xyz.shape[0]
        if self.group_all:
            new_xyz = torch.zeros(batch_size, 1, 3, device=xyz.device, dtype=xyz.dtype)
            grouped_xyz = xyz.unsqueeze(1)
            grouped = (
                torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
                if points is not None
                else grouped_xyz
            )
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
            group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)
            grouped = (
                torch.cat([grouped_xyz, index_points(points, group_idx)], dim=-1)
                if points is not None
                else grouped_xyz
            )
        grouped = grouped.permute(0, 3, 2, 1)
        features = self.mlp(grouped)
        features = torch.max(features, dim=2)[0].transpose(1, 2)
        return new_xyz, features


class CompactPointNet2SSG(nn.Module):
    """Compact PointNet++ SSG classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.sa1 = SetAbstraction(8, 0.35, 8, in_channel=0, mlp=[16, 24])
        self.sa2 = SetAbstraction(2, 0.70, 8, in_channel=24, mlp=[32, 48])
        self.sa3 = SetAbstraction(1, 1.0, 2, in_channel=48, mlp=[64], group_all=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        xyz:
            Point coordinates of shape ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(B, num_classes)``.
        """

        xyz1, points1 = self.sa1(xyz, None)
        xyz2, points2 = self.sa2(xyz1, points1)
        _, points3 = self.sa3(xyz2, points2)
        return self.classifier(points3.squeeze(1))


def build_pointnet2_cls_ssg_compact() -> nn.Module:
    """Build the compact PointNet++ SSG classifier.

    Returns
    -------
    nn.Module
        Compact PointNet++ SSG classifier.
    """

    return CompactPointNet2SSG()


def example_input() -> torch.Tensor:
    """Return a compact point-cloud example.

    Returns
    -------
    torch.Tensor
        Point cloud of shape ``(1, 32, 3)``.
    """

    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "PointNet++ compact SSG classifier (FPS + ball-query set abstraction)",
        "build_pointnet2_cls_ssg_compact",
        "example_input",
        "2017",
        "DC",
    ),
]
