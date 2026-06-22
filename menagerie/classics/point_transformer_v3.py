"""Point Transformer V3: Simpler, Faster, Stronger.

Wu et al., CVPR 2024.
Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html

PTv3 replaces expensive point-cloud neighborhood search with locality-preserving
serialization: points are sorted along space-filling keys, grouped into patches,
and processed by grouped vector attention inside those serialized patches.  This
compact reconstruction keeps the load-bearing primitives: coordinate
serialization, patch grouping, grouped-vector attention, a feed-forward block,
and a point-wise classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _serialization_key(coords: torch.Tensor) -> torch.Tensor:
    """Compute a compact Morton-like key for each normalized 3-D point.

    Parameters
    ----------
    coords:
        Point coordinates with shape ``(batch, points, 3)``.

    Returns
    -------
    torch.Tensor
        Integer-ish sortable keys with shape ``(batch, points)``.
    """

    grid = torch.clamp((coords + 1.0) * 15.5, min=0.0, max=31.0).long()
    return grid[..., 0] * 1024 + grid[..., 1] * 32 + grid[..., 2]


class GroupedVectorAttention(nn.Module):
    """PTv3-style grouped vector attention inside serialized point patches."""

    def __init__(self, dim: int, groups: int = 4) -> None:
        """Initialize grouped vector attention projections.

        Parameters
        ----------
        dim:
            Embedding dimension.
        groups:
            Number of channel groups for vector attention.
        """

        super().__init__()
        self.groups = groups
        self.group_dim = dim // groups
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.weight = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, groups))
        self.out = nn.Linear(dim, dim)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Apply grouped vector attention over serialized patches.

        Parameters
        ----------
        feats:
            Patch features with shape ``(batch, patches, patch_size, dim)``.
        coords:
            Patch coordinates with shape ``(batch, patches, patch_size, 3)``.

        Returns
        -------
        torch.Tensor
            Updated patch features with shape ``(batch, patches, patch_size, dim)``.
        """

        q = self.q(feats)
        k = self.k(feats)
        v = self.v(feats)
        rel = coords.unsqueeze(3) - coords.unsqueeze(2)
        pos = self.pos(rel)
        logits = self.weight(q.unsqueeze(3) - k.unsqueeze(2) + pos)
        attn = torch.softmax(logits, dim=3)
        vg = (v.unsqueeze(2) + pos).view(*v.shape[:3], feats.shape[2], self.groups, self.group_dim)
        out = (attn.unsqueeze(-1) * vg).sum(dim=3).reshape_as(feats)
        return self.out(out)


class PTv3Block(nn.Module):
    """Serialized patch attention block used by compact PTv3."""

    def __init__(self, dim: int, groups: int = 4) -> None:
        """Initialize normalization, attention, and MLP layers.

        Parameters
        ----------
        dim:
            Embedding dimension.
        groups:
            Number of vector-attention groups.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GroupedVectorAttention(dim, groups)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Run one residual PTv3 patch block.

        Parameters
        ----------
        feats:
            Patch features with shape ``(batch, patches, patch_size, dim)``.
        coords:
            Patch coordinates with shape ``(batch, patches, patch_size, 3)``.

        Returns
        -------
        torch.Tensor
            Updated patch features.
        """

        feats = feats + self.attn(self.norm1(feats), coords)
        return feats + self.mlp(self.norm2(feats))


class CompactPointTransformerV3(nn.Module):
    """Compact PTv3 with serialization, patching, and grouped vector attention."""

    def __init__(self, in_dim: int = 6, dim: int = 32, patch_size: int = 8) -> None:
        """Initialize a small PTv3 point classifier.

        Parameters
        ----------
        in_dim:
            Input feature dimension including coordinates.
        dim:
            Hidden embedding dimension.
        patch_size:
            Number of serialized points per patch.
        """

        super().__init__()
        self.patch_size = patch_size
        self.stem = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([PTv3Block(dim), PTv3Block(dim)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify each point after serialized patch attention.

        Parameters
        ----------
        x:
            Point tensor with shape ``(batch, points, 6)``; first three channels
            are coordinates.

        Returns
        -------
        torch.Tensor
            Per-point logits in original point order with shape ``(batch, points, 5)``.
        """

        coords = x[..., :3]
        order = torch.argsort(_serialization_key(coords), dim=1)
        gather_feat = order.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        gather_xyz = order.unsqueeze(-1).expand(-1, -1, 3)
        xs = torch.gather(x, 1, gather_feat)
        cs = torch.gather(coords, 1, gather_xyz)
        bsz, n_points, _ = xs.shape
        patches = n_points // self.patch_size
        feats = self.stem(xs).view(bsz, patches, self.patch_size, -1)
        cpatch = cs.view(bsz, patches, self.patch_size, 3)
        for block in self.blocks:
            feats = block(feats, cpatch)
        flat = feats.view(bsz, n_points, -1)
        inv = torch.argsort(order, dim=1).unsqueeze(-1).expand_as(flat)
        return self.head(torch.gather(flat, 1, inv))


def build() -> nn.Module:
    """Build a compact random-init Point Transformer V3.

    Returns
    -------
    nn.Module
        Compact PTv3 model.
    """

    return CompactPointTransformerV3()


def example_input() -> torch.Tensor:
    """Create a small point cloud input.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 32, 6)``.
    """

    return torch.randn(1, 32, 6)


MENAGERIE_ENTRIES = [
    ("point_transformer_v3", "build", "example_input", "2024", "DC"),
]
