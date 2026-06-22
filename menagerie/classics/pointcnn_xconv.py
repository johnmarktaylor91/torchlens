"""PointCNN X-Conv point-cloud operator.

Li et al., NeurIPS 2018.
Paper: https://proceedings.neurips.cc/paper/2018/hash/f5f8590cd58a54e94377e6ae2eded4d9-Abstract.html

PointCNN learns an X-transformation from local point coordinates, uses it to
permute/weight neighboring point features, and then applies a convolution-like
linear operator.  This compact reconstruction keeps the distinctive learned
X-Conv primitive with deterministic local neighborhoods.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class XConv(nn.Module):
    """Compact PointCNN X-Conv layer."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 8) -> None:
        """Initialize local coordinate lifting, X-transform, and projection.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        out_dim:
            Output feature dimension.
        k:
            Neighborhood size.
        """

        super().__init__()
        self.k = k
        self.lift = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, in_dim))
        self.x_trans = nn.Sequential(nn.Linear(k * 3, k * k), nn.ReLU(), nn.Linear(k * k, k * k))
        self.proj = nn.Linear(k * in_dim, out_dim)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Apply X-Conv over local nearest-coordinate neighborhoods.

        Parameters
        ----------
        coords:
            Point coordinates with shape ``(batch, points, 3)``.
        feats:
            Point features with shape ``(batch, points, in_dim)``.

        Returns
        -------
        torch.Tensor
            Output point features with shape ``(batch, points, out_dim)``.
        """

        dists = torch.cdist(coords, coords)
        idx = torch.topk(dists, self.k, dim=-1, largest=False).indices
        gather_xyz = idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        gather_feat = idx.unsqueeze(-1).expand(-1, -1, -1, feats.shape[-1])
        nbr_xyz = torch.gather(
            coords.unsqueeze(1).expand(-1, coords.shape[1], -1, -1), 2, gather_xyz
        )
        nbr_feat = torch.gather(
            feats.unsqueeze(1).expand(-1, feats.shape[1], -1, -1), 2, gather_feat
        )
        rel = nbr_xyz - coords.unsqueeze(2)
        lifted = nbr_feat + self.lift(rel)
        xmat = self.x_trans(rel.flatten(2)).view(coords.shape[0], coords.shape[1], self.k, self.k)
        mixed = torch.matmul(xmat, lifted)
        return self.proj(mixed.flatten(2))


class CompactPointCNN(nn.Module):
    """Small PointCNN classifier built from stacked X-Conv layers."""

    def __init__(self, in_dim: int = 6, hidden: int = 24) -> None:
        """Initialize stem, X-Conv stack, and classifier.

        Parameters
        ----------
        in_dim:
            Input point feature dimension including coordinates.
        hidden:
            Hidden feature width.
        """

        super().__init__()
        self.stem = nn.Linear(in_dim, hidden)
        self.xconv1 = XConv(hidden, hidden)
        self.xconv2 = XConv(hidden, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        x:
            Point tensor with shape ``(batch, points, 6)``.

        Returns
        -------
        torch.Tensor
            Cloud-level logits.
        """

        coords = x[..., :3]
        feats = torch.relu(self.stem(x))
        feats = torch.relu(self.xconv1(coords, feats))
        feats = torch.relu(self.xconv2(coords, feats))
        return self.head(feats.mean(dim=1))


def build() -> nn.Module:
    """Build a compact PointCNN.

    Returns
    -------
    nn.Module
        Random-init PointCNN.
    """

    return CompactPointCNN()


def example_input() -> torch.Tensor:
    """Create a small point-cloud input.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 24, 6)``.
    """

    return torch.randn(1, 24, 6)


MENAGERIE_ENTRIES = [
    ("pointcnn_xconv", "build", "example_input", "2018", "DC"),
]
