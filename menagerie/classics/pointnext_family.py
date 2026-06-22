"""PointNeXt / OpenPoints compact point-cloud models.

PointNeXt revisits PointNet++ with improved training and architectural scaling:
set-abstraction style local aggregation, residual inverted bottleneck MLP blocks,
and S/B/L/XL width-depth scaling in OpenPoints.  This reconstruction keeps the
architecture compact and CPU-friendly while preserving local neighborhood
pooling and residual pointwise feature refinement.

Sources: "PointNeXt: Revisiting PointNet++ with Improved Training and Scaling
Strategies" (NeurIPS 2022) and the OpenPoints model zoo.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ScaleName = Literal["s", "b", "l", "xl"]


class LocalAggregation(nn.Module):
    """PointNeXt local neighborhood aggregation.

    Parameters
    ----------
    channels:
        Feature channel count.
    k:
        Number of nearest neighbors.
    """

    def __init__(self, channels: int, k: int = 6) -> None:
        super().__init__()
        self.k = k
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.mix = nn.Linear(channels, channels)

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Aggregate nearest-neighbor features.

        Parameters
        ----------
        xyz:
            Point coordinates of shape ``(batch, points, 3)``.
        feats:
            Point features of shape ``(batch, points, channels)``.

        Returns
        -------
        torch.Tensor
            Aggregated features.
        """

        dist = torch.cdist(xyz, xyz)
        idx = dist.topk(self.k, largest=False).indices
        batch, points, channels = feats.shape
        gather_idx = idx.unsqueeze(-1).expand(batch, points, self.k, channels)
        neigh_feats = torch.gather(
            feats.unsqueeze(1).expand(batch, points, points, channels), 2, gather_idx
        )
        pos_idx = idx.unsqueeze(-1).expand(batch, points, self.k, 3)
        rel_pos = torch.gather(xyz.unsqueeze(1).expand(batch, points, points, 3), 2, pos_idx)
        rel_pos = rel_pos - xyz.unsqueeze(2)
        encoded = neigh_feats + self.pos_mlp(rel_pos)
        return self.mix(encoded.max(dim=2).values)


class InvResMLP(nn.Module):
    """PointNeXt inverted residual MLP block.

    Parameters
    ----------
    channels:
        Feature channel count.
    expansion:
        Hidden expansion ratio.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual pointwise MLP refinement.

        Parameters
        ----------
        x:
            Point features.

        Returns
        -------
        torch.Tensor
            Refined point features.
        """

        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        return residual + self.fc2(x)


class PointNeXtTiny(nn.Module):
    """Compact scaled PointNeXt classifier.

    Parameters
    ----------
    scale:
        OpenPoints scale key.
    num_classes:
        Output class count.
    """

    def __init__(self, scale: ScaleName, num_classes: int = 13) -> None:
        super().__init__()
        widths = {"s": 24, "b": 32, "l": 40, "xl": 48}
        depths = {"s": 1, "b": 2, "l": 3, "xl": 4}
        width = widths[scale]
        depth = depths[scale]
        self.stem = nn.Linear(3, width)
        self.aggregation = LocalAggregation(width)
        self.blocks = nn.ModuleList([InvResMLP(width) for _ in range(depth)])
        self.down = nn.Linear(width, width)
        self.head = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, num_classes))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        xyz:
            Point coordinates of shape ``(batch, points, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        feats = F.relu(self.stem(xyz))
        feats = F.relu(self.aggregation(xyz, feats))
        for block in self.blocks:
            feats = block(feats)
        sampled = feats[:, ::2]
        sampled = F.relu(self.down(sampled))
        pooled = torch.cat((sampled.mean(dim=1), sampled.max(dim=1).values), dim=-1)
        return self.head(pooled)


def _build(scale: ScaleName) -> PointNeXtTiny:
    """Build a scaled PointNeXt model.

    Parameters
    ----------
    scale:
        OpenPoints scale key.

    Returns
    -------
    PointNeXtTiny
        Random-initialized PointNeXt variant.
    """

    return PointNeXtTiny(scale)


def example_input() -> torch.Tensor:
    """Create a compact point cloud.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(2, 32, 3)``.
    """

    return torch.rand(2, 32, 3)


def build_pointnext_s() -> PointNeXtTiny:
    """Build PointNeXt-S.

    Returns
    -------
    PointNeXtTiny
        S-scale model.
    """

    return _build("s")


def build_pointnext_b() -> PointNeXtTiny:
    """Build PointNeXt-B.

    Returns
    -------
    PointNeXtTiny
        B-scale model.
    """

    return _build("b")


def build_pointnext_l() -> PointNeXtTiny:
    """Build PointNeXt-L.

    Returns
    -------
    PointNeXtTiny
        L-scale model.
    """

    return _build("l")


def build_pointnext_xl() -> PointNeXtTiny:
    """Build PointNeXt-XL.

    Returns
    -------
    PointNeXtTiny
        XL-scale model.
    """

    return _build("xl")


MENAGERIE_ENTRIES = [
    ("PointNeXt-S", "build_pointnext_s", "example_input", "2022", "pointcloud"),
    ("pointnext_b_openpoints", "build_pointnext_b", "example_input", "2022", "pointcloud"),
    ("pointnext_l_openpoints", "build_pointnext_l", "example_input", "2022", "pointcloud"),
    ("pointnext_s_openpoints", "build_pointnext_s", "example_input", "2022", "pointcloud"),
    ("pointnext_xl_openpoints", "build_pointnext_xl", "example_input", "2022", "pointcloud"),
]
