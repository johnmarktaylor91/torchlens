"""Point Transformer V2.

Paper: "Point Transformer V2: Grouped Vector Attention and Partition-based
Pooling", Wu et al., NeurIPS 2022.

The compact reconstruction keeps grouped vector attention with learned grouped
weight encoding, multiplicative positional encoding, and partition-based pooling
over voxel-like point partitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupedVectorAttention(nn.Module):
    """Point Transformer V2 grouped vector attention."""

    def __init__(self, channels: int = 24, groups: int = 4) -> None:
        """Initialize grouped attention projections."""

        super().__init__()
        self.groups = groups
        self.group_dim = channels // groups
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.pos = nn.Sequential(nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.weight_encoding = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, groups)
        )
        self.out = nn.Linear(channels, channels)

    def forward(self, feat: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        """Apply all-pairs compact grouped vector attention."""

        rel = coord[:, :, None, :] - coord[:, None, :, :]
        pos = self.pos(rel)
        q = self.q(feat)[:, :, None, :]
        k = self.k(feat)[:, None, :, :]
        v = self.v(feat)[:, None, :, :] + pos
        logits = self.weight_encoding(q - k + pos).permute(0, 1, 3, 2)
        weights = torch.softmax(logits, dim=-1)
        grouped_v = v.view(v.shape[0], v.shape[1], v.shape[2], self.groups, self.group_dim)
        out = torch.einsum("bngm,bnmgd->bngd", weights, grouped_v).reshape(feat.shape)
        multiplier = torch.sigmoid(pos.mean(dim=2))
        return self.out(out * multiplier)


class PointTransformerV2Compact(nn.Module):
    """Compact point-cloud classifier with partition pooling."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize stem, attention, pooling, and head."""

        super().__init__()
        self.stem = nn.Linear(6, channels)
        self.attn = GroupedVectorAttention(channels)
        self.pool_proj = nn.Linear(channels, channels)
        self.head = nn.Linear(channels, 6)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud."""

        coord = points[..., :3]
        feat = torch.relu(self.stem(points))
        feat = feat + self.attn(feat, coord)
        first_partition = self.pool_proj(feat[:, : feat.shape[1] // 2]).mean(dim=1)
        second_partition = self.pool_proj(feat[:, feat.shape[1] // 2 :]).mean(dim=1)
        return self.head(torch.stack([first_partition, second_partition], dim=1).mean(dim=1))


def build() -> nn.Module:
    """Build compact Point Transformer V2."""

    return PointTransformerV2Compact()


def example_input() -> torch.Tensor:
    """Return point coordinates plus features."""

    return torch.randn(1, 12, 6)


MENAGERIE_ENTRIES = [("PointTransformerV2", "build", "example_input", "2022", "E7")]
