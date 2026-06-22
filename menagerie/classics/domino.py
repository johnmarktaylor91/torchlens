"""DoMINO decomposable multi-scale iterative neural operator.

Ranade et al. (2025), "DoMINO: A Decomposable Multi-scale Iterative Neural
Operator for Modeling Large Scale Engineering Simulations."  DoMINO is described
as a local, multi-scale point-cloud neural operator for external aerodynamics:
geometry and query points are encoded with local geometric neighborhoods, refined
iteratively across scales, and decoded into surface or volume flow quantities.
This compact reconstruction keeps point-cloud geometry encoding, radius-free
k-neighborhood message passing, multi-scale coarse-to-fine refinement, and field
decoding on query points.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalPointOperator(nn.Module):
    """Local point-cloud operator using learned relative-coordinate messages."""

    def __init__(self, channels: int, k: int = 4) -> None:
        """Initialize the local operator.

        Parameters
        ----------
        channels:
            Feature channel count.
        k:
            Number of nearest neighbors.
        """

        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2 + 3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU()
        )

    def forward(self, points: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Apply local k-neighborhood message passing.

        Parameters
        ----------
        points:
            Point coordinates of shape ``(B, N, 3)``.
        feats:
            Point features of shape ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Updated point features.
        """

        dist = torch.cdist(points, points)
        knn = dist.topk(self.k + 1, largest=False).indices[:, :, 1:]
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
        nbr_points = points[batch, knn]
        nbr_feats = feats[batch, knn]
        center_points = points.unsqueeze(2).expand_as(nbr_points)
        center_feats = feats.unsqueeze(2).expand_as(nbr_feats)
        edge = torch.cat([center_feats, nbr_feats, nbr_points - center_points], dim=-1)
        msg = self.edge_mlp(edge).mean(dim=2)
        return feats + self.update(msg)


class CompactDoMINO(nn.Module):
    """Compact multi-scale iterative point-cloud neural operator."""

    def __init__(self, channels: int = 32, out_channels: int = 4) -> None:
        """Initialize the compact DoMINO model.

        Parameters
        ----------
        channels:
            Hidden point feature width.
        out_channels:
            Number of predicted flow quantities.
        """

        super().__init__()
        self.geom_embed = nn.Sequential(
            nn.Linear(6, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.query_embed = nn.Sequential(
            nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        self.coarse = LocalPointOperator(channels, k=4)
        self.fine = LocalPointOperator(channels, k=4)
        self.cross = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.decode = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, out_channels),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Predict flow quantities at query points.

        Parameters
        ----------
        data:
            Tensor of shape ``(B, N, 9)`` containing geometry coordinates,
            normals, and query coordinates.

        Returns
        -------
        torch.Tensor
            Query-point flow predictions of shape ``(B, N, out_channels)``.
        """

        geom = data[..., :3]
        normals = data[..., 3:6]
        query = data[..., 6:9]
        feats = self.geom_embed(torch.cat([geom, normals], dim=-1))
        coarse_points = geom[:, ::2]
        coarse_feats = self.coarse(coarse_points, feats[:, ::2])
        coarse_up = F.interpolate(
            coarse_feats.transpose(1, 2), size=geom.shape[1], mode="linear"
        ).transpose(1, 2)
        feats = feats + coarse_up
        feats = self.fine(geom, feats)
        q = self.query_embed(query)
        attended, _ = self.cross(q, feats, feats, need_weights=False)
        return self.decode(q + attended)


def build() -> nn.Module:
    """Build the compact DoMINO model.

    Returns
    -------
    nn.Module
        Random-init point-cloud neural operator in evaluation mode.
    """

    return CompactDoMINO().eval()


def example_input() -> torch.Tensor:
    """Return compact geometry, normal, and query point data.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 16, 9)``.
    """

    geom = torch.randn(1, 16, 3)
    normals = F.normalize(torch.randn(1, 16, 3), dim=-1)
    query = geom + 0.05 * torch.randn(1, 16, 3)
    return torch.cat([geom, normals, query], dim=-1)


MENAGERIE_ENTRIES = [
    ("DoMINO", "build", "example_input", "2025", "E5"),
]
