"""PointNet compact classification and dense-segmentation classics.

Qi et al., CVPR 2017, arXiv:1612.00593.
Paper: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.

PointNet applies shared MLPs to every point, predicts input and feature transform
matrices with T-Net modules, aggregates a global feature by symmetric max pooling,
and uses either a global classifier or a per-point segmentation head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP1d(nn.Module):
    """Shared pointwise MLP implemented as 1x1 convolutions."""

    def __init__(self, channels: list[int]) -> None:
        """Initialize a stack of 1x1 Conv-BN-ReLU layers.

        Parameters
        ----------
        channels:
            Channel sizes from input through output.
        """

        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(len(channels) - 1):
            layers.extend(
                [
                    nn.Conv1d(channels[idx], channels[idx + 1], 1, bias=False),
                    nn.BatchNorm1d(channels[idx + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared MLP.

        Parameters
        ----------
        x:
            Point features ``(B, C, N)``.

        Returns
        -------
        torch.Tensor
            Transformed point features.
        """

        return self.net(x)


class TNet(nn.Module):
    """PointNet transformation network for input or feature alignment."""

    def __init__(self, dim: int) -> None:
        """Initialize the transform predictor.

        Parameters
        ----------
        dim:
            Size of the square transform matrix.
        """

        super().__init__()
        self.dim = dim
        self.mlp = SharedMLP1d([dim, 32, 64, 128])
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, dim * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a near-identity transform matrix.

        Parameters
        ----------
        x:
            Point features ``(B, dim, N)``.

        Returns
        -------
        torch.Tensor
            Transform matrices ``(B, dim, dim)``.
        """

        batch = x.shape[0]
        pooled = torch.max(self.mlp(x), dim=-1)[0]
        mat = self.fc2(F.relu_(self.fc1(pooled))).view(batch, self.dim, self.dim)
        eye = torch.eye(self.dim, device=x.device).unsqueeze(0)
        return mat + eye


class PointNetFeature(nn.Module):
    """PointNet feature extractor with input and feature transforms."""

    def __init__(self, global_feature: bool = True) -> None:
        """Initialize PointNet feature stages.

        Parameters
        ----------
        global_feature:
            Whether to return only the global pooled feature.
        """

        super().__init__()
        self.global_feature = global_feature
        self.input_tnet = TNet(3)
        self.mlp1 = SharedMLP1d([3, 32, 64])
        self.feature_tnet = TNet(64)
        self.mlp2 = SharedMLP1d([64, 128, 256])

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Extract global or dense PointNet features.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Global feature ``(B, 256)`` or dense feature ``(B, 320, N)``.
        """

        x = points.transpose(1, 2)
        x = torch.bmm(self.input_tnet(x), x)
        local = self.mlp1(x)
        local = torch.bmm(self.feature_tnet(local), local)
        global_map = self.mlp2(local)
        pooled = torch.max(global_map, dim=-1)[0]
        if self.global_feature:
            return pooled
        repeated = pooled.unsqueeze(-1).expand(-1, -1, points.shape[1])
        return torch.cat([local, repeated], dim=1)


class PointNetCls(nn.Module):
    """PointNet classification head."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize the classifier.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.features = PointNetFeature(global_feature=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, classes)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.features(points))


class PointNetDenseCls(nn.Module):
    """PointNet dense per-point segmentation head."""

    def __init__(self, classes: int = 4) -> None:
        """Initialize the dense classifier.

        Parameters
        ----------
        classes:
            Number of per-point labels.
        """

        super().__init__()
        self.features = PointNetFeature(global_feature=False)
        self.head = nn.Sequential(
            nn.Conv1d(320, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, classes, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Predict per-point labels.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Per-point logits ``(B, N, classes)``.
        """

        return self.head(self.features(points)).transpose(1, 2)


def build_pointnet_cls() -> nn.Module:
    """Build a compact PointNet classifier.

    Returns
    -------
    nn.Module
        Random-init PointNet classifier.
    """

    return PointNetCls()


def build_pointnet_dense_cls() -> nn.Module:
    """Build a compact PointNet dense classifier.

    Returns
    -------
    nn.Module
        Random-init PointNet segmentation model.
    """

    return PointNetDenseCls()


def build_pointnet_feat() -> nn.Module:
    """Build a compact PointNet feature extractor.

    Returns
    -------
    nn.Module
        Random-init PointNet feature extractor.
    """

    return PointNetFeature(global_feature=True)


def example_points() -> torch.Tensor:
    """Create a compact point cloud.

    Returns
    -------
    torch.Tensor
        Point tensor ``(1, 32, 3)``.
    """

    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    ("pointnet", "build_pointnet_cls", "example_points", "2017", "GEO"),
    ("PointNet_PointNetCls", "build_pointnet_cls", "example_points", "2017", "GEO"),
    ("PointNet_PointNetDenseCls", "build_pointnet_dense_cls", "example_points", "2017", "GEO"),
    ("PointNet_PointNetfeat", "build_pointnet_feat", "example_points", "2017", "GEO"),
]
