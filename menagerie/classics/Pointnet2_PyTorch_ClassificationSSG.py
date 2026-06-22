"""PointNet++ SSG classifier with hierarchical set abstraction.

Paper: PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.

Qi et al. (NeurIPS 2017) replace PointNet's single global pooling with
hierarchical set-abstraction layers: farthest-point sampling, local ball-query
grouping, shared PointNet MLPs inside each neighborhood, and symmetric max
pooling.  This compact random-init reconstruction keeps the single-scale
grouping (SSG) classification path used by the Pointnet2 PyTorch SSG model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _square_distance(src: Tensor, dst: Tensor) -> Tensor:
    """Compute pairwise squared distances.

    Parameters
    ----------
    src:
        Source points of shape ``(B, N, C)``.
    dst:
        Destination points of shape ``(B, M, C)``.

    Returns
    -------
    Tensor
        Squared distances of shape ``(B, N, M)``.
    """

    return torch.cdist(src, dst).pow(2)


def _farthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """Select centroids with deterministic farthest-point sampling.

    Parameters
    ----------
    xyz:
        Point coordinates ``(B, N, 3)``.
    npoint:
        Number of centroids.

    Returns
    -------
    Tensor
        Long indices ``(B, npoint)``.
    """

    batch, num_points, _ = xyz.shape
    centroids = torch.zeros(batch, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, num_points), 1e10, device=xyz.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch, device=xyz.device)
    for idx in range(npoint):
        centroids[:, idx] = farthest
        centroid = xyz[batch_indices, farthest].view(batch, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1)[1]
    return centroids


def _index_points(points: Tensor, idx: Tensor) -> Tensor:
    """Gather points by batched indices.

    Parameters
    ----------
    points:
        Tensor ``(B, N, C)``.
    idx:
        Long indices ``(B, ...)``.

    Returns
    -------
    Tensor
        Gathered tensor ``(B, ..., C)``.
    """

    batch = points.shape[0]
    view_shape = [batch] + [1] * (idx.dim() - 1)
    repeat_shape = [1] + list(idx.shape[1:])
    batch_indices = torch.arange(batch, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx]


def _query_ball(radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor) -> Tensor:
    """Find local neighborhood indices for each centroid.

    Parameters
    ----------
    radius:
        Ball-query radius.
    nsample:
        Maximum neighbors.
    xyz:
        All points ``(B, N, 3)``.
    new_xyz:
        Centroids ``(B, S, 3)``.

    Returns
    -------
    Tensor
        Neighborhood indices ``(B, S, nsample)``.
    """

    dist = _square_distance(new_xyz, xyz)
    nearest = dist.argsort(dim=-1)[:, :, :nsample]
    nearest_dist = torch.gather(dist, -1, nearest)
    fallback = nearest[:, :, :1].expand_as(nearest)
    return torch.where(nearest_dist <= radius * radius, nearest, fallback)


class SharedPointMLP(nn.Module):
    """Shared PointNet MLP for grouped local neighborhoods."""

    def __init__(self, channels: list[int]) -> None:
        """Initialize 1x1 Conv-BN-ReLU layers.

        Parameters
        ----------
        channels:
            Channel sizes from input to output.
        """

        super().__init__()
        layers: list[nn.Module] = []
        for in_ch, out_ch in zip(channels, channels[1:], strict=False):
            layers.extend(
                [nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU()]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the shared MLP.

        Parameters
        ----------
        x:
            Grouped features ``(B, C, nsample, npoint)``.

        Returns
        -------
        Tensor
            Transformed features.
        """

        return self.net(x)


class SetAbstractionSSG(nn.Module):
    """PointNet++ single-scale set-abstraction layer."""

    def __init__(
        self, npoint: int | None, radius: float, nsample: int, in_channels: int, mlp: list[int]
    ) -> None:
        """Initialize sampling, grouping, and local PointNet.

        Parameters
        ----------
        npoint:
            Number of sampled centroids, or ``None`` for global abstraction.
        radius:
            Ball-query radius.
        nsample:
            Number of neighbors per centroid.
        in_channels:
            Input point-feature channels, excluding xyz.
        mlp:
            Local PointNet channel sizes.
        """

        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = SharedPointMLP([in_channels + 3, *mlp])

    def forward(self, xyz: Tensor, points: Tensor | None) -> tuple[Tensor, Tensor]:
        """Apply SSG set abstraction.

        Parameters
        ----------
        xyz:
            Coordinates ``(B, N, 3)``.
        points:
            Optional point features ``(B, C, N)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            New coordinates ``(B, S, 3)`` and features ``(B, C_out, S)``.
        """

        if self.npoint is None:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1)
            grouped_points = points.transpose(1, 2).unsqueeze(1) if points is not None else None
        else:
            fps_idx = _farthest_point_sample(xyz, self.npoint)
            new_xyz = _index_points(xyz, fps_idx)
            group_idx = _query_ball(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = _index_points(xyz, group_idx)
            grouped_points = (
                _index_points(points.transpose(1, 2), group_idx) if points is not None else None
            )
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        grouped = (
            grouped_xyz
            if grouped_points is None
            else torch.cat([grouped_xyz, grouped_points], dim=-1)
        )
        grouped = grouped.permute(0, 3, 2, 1)
        features = self.mlp(grouped).max(dim=2)[0]
        return new_xyz, features


class PointNet2SSGClassifier(nn.Module):
    """Compact PointNet++ SSG classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize hierarchical abstraction layers and classifier.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.sa1 = SetAbstractionSSG(16, 0.25, 12, 0, [32, 32, 64])
        self.sa2 = SetAbstractionSSG(4, 0.55, 8, 64, [64, 64, 128])
        self.sa3 = SetAbstractionSSG(None, 1.0, 4, 128, [128, 256])
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, points: Tensor) -> Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        points:
            Coordinates ``(B, N, 3)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        xyz1, feat1 = self.sa1(points, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)
        return self.fc2(F.relu(self.fc1(feat3.squeeze(-1))))


def build() -> nn.Module:
    """Build a compact random-init PointNet++ SSG classifier.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return PointNet2SSGClassifier().eval()


def example_input() -> Tensor:
    """Return a small point cloud.

    Returns
    -------
    Tensor
        Point tensor ``(1, 48, 3)``.
    """

    return torch.randn(1, 48, 3)


MENAGERIE_ENTRIES = [
    ("Pointnet2_PyTorch_ClassificationSSG", "build", "example_input", "2017", "PC"),
]
