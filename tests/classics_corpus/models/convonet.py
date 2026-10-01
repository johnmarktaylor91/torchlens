"""Convolutional Occupancy Networks (ConvONet).

Peng, Niemeyer, Mescheder, Pollefeys & Geiger, ECCV 2020.
Paper: https://arxiv.org/abs/2003.04618
Source: https://github.com/autonomousvision/convolutional_occupancy_networks

ConvONet extends occupancy networks with *spatially structured* latent
representations.  The key contribution over ONet is that instead of a single
global latent code, it uses a *local* feature encoding:

1. **LocalPoolPointnet (encoder)**: a PointNet-style encoder that:
   - Processes input points with a shared MLP.
   - Projects features onto a structured grid (plane or voxel).
   - Applies 2D/3D convolutions over the grid for context aggregation.
   - At query points, bilinearly interpolates features from the grid.

2. **LocalDecoder (decoder)**: given the interpolated local features at
   query positions, an MLP maps to an occupancy logit.

Architecture in this reimplementation:
  - Encoder: LocalPoolPointnet with an XY-plane feature grid (32x32 resolution).
    Input points (B, N, 3) -> shared MLP -> scatter max onto 32x32 grid -> 2 Conv2d
    aggregation layers -> feature plane c_plane (B, 32, 32, 32).
  - Decoder: LocalDecoder -- interpolates features from c_plane at query
    points, then MLP(concat(features, xyz)) -> occupancy logit.
  - Two MENAGERIE_ENTRIES: one for the encoder, one for the full model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Encoder: LocalPoolPointnet
# -----------------------------------------------------------------------


class LocalPoolPointnet(nn.Module):
    """PointNet-based encoder that scatters features onto a 2D feature plane.

    Args:
        in_ch: input feature channels (3 for raw xyz).
        c_dim: feature channels in the latent plane.
        plane_res: resolution of the feature plane (H=W=plane_res).
        hidden_dim: hidden MLP channels.
    """

    def __init__(
        self,
        in_ch: int = 3,
        c_dim: int = 32,
        plane_res: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.plane_res = plane_res
        self.c_dim = c_dim

        # Shared PointNet MLP (no global pooling yet)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, c_dim),
        )

        # 2D convolutional aggregator over the plane
        self.plane_conv = nn.Sequential(
            nn.Conv2d(c_dim, c_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_dim, c_dim, 3, padding=1),
        )

    def _scatter_to_plane(
        self,
        xyz: torch.Tensor,
        feats: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter-max point features onto the XY plane grid.

        xyz: (B, N, 3) in [-0.5, 0.5]; feats: (B, N, c_dim)
        -> plane: (B, c_dim, plane_res, plane_res)
        """
        B, N, C = feats.shape
        R = self.plane_res
        # Normalise XY coords to [0, R-1]
        xy = (xyz[:, :, :2] + 0.5).clamp(0, 1 - 1e-6) * R  # (B, N, 2)
        # Integer voxel index
        xi = xy[:, :, 0].long().clamp(0, R - 1)  # (B, N)
        yi = xy[:, :, 1].long().clamp(0, R - 1)

        # Flat index into (R, R) grid
        flat_idx = yi * R + xi  # (B, N)

        # Scatter max onto plane: (B, R*R, c_dim)
        plane_flat = torch.full((B, R * R, C), -1e8, device=feats.device, dtype=feats.dtype)
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(B, N, C)
        plane_flat.scatter_reduce_(1, flat_idx_exp, feats, reduce="amax", include_self=True)
        # Replace -1e8 (empty cells) with zeros
        plane_flat = plane_flat.clamp(min=0)
        plane = plane_flat.reshape(B, R, R, C).permute(0, 3, 1, 2)  # (B, C, R, R)
        return plane

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (B, N, 3) -> plane feature (B, c_dim, plane_res, plane_res)."""
        feats = self.point_mlp(xyz)  # (B, N, c_dim)
        plane = self._scatter_to_plane(xyz, feats)  # (B, c_dim, R, R)
        plane = self.plane_conv(plane)  # (B, c_dim, R, R)
        return plane


# -----------------------------------------------------------------------
# Decoder: LocalDecoder
# -----------------------------------------------------------------------


class LocalDecoder(nn.Module):
    """MLP decoder that interpolates plane features at query positions.

    Args:
        c_dim: feature channels in the latent plane.
        hidden_dim: hidden MLP width.
    """

    def __init__(self, c_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _interpolate_plane(
        self,
        plane: torch.Tensor,
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinear interpolation from plane at xy positions of query points.

        plane: (B, c_dim, R, R); xyz: (B, M, 3) in [-0.5, 0.5]
        -> (B, M, c_dim)
        """
        # Normalise to [-1, 1] for grid_sample
        xy = xyz[:, :, :2] * 2  # (B, M, 2)
        # grid_sample expects (B, 1, M, 2) grid
        grid = xy.unsqueeze(1)  # (B, 1, M, 2)
        # plane: (B, c_dim, R, R) -> grid_sample -> (B, c_dim, 1, M)
        interp = F.grid_sample(
            plane, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B, c_dim, 1, M)
        return interp.squeeze(2).permute(0, 2, 1)  # (B, M, c_dim)

    def forward(self, plane: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """plane: (B, c_dim, R, R); xyz: (B, M, 3) -> (B, M) occupancy logits."""
        c = self._interpolate_plane(plane, xyz)  # (B, M, c_dim)
        x = torch.cat([c, xyz], dim=-1)  # (B, M, c_dim+3)
        B, M, _ = x.shape
        out = self.mlp(x.reshape(B * M, -1))  # (B*M, 1)
        return out.reshape(B, M)


# -----------------------------------------------------------------------
# Full ConvONet model
# -----------------------------------------------------------------------


class ConvONet(nn.Module):
    """Full Convolutional Occupancy Network: encoder + decoder."""

    def __init__(self, c_dim: int = 32, plane_res: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = LocalPoolPointnet(
            in_ch=3, c_dim=c_dim, plane_res=plane_res, hidden_dim=hidden_dim
        )
        self.decoder = LocalDecoder(c_dim=c_dim, hidden_dim=hidden_dim)

    def forward(self, input_pts: torch.Tensor, query_pts: torch.Tensor) -> torch.Tensor:
        """
        input_pts: (B, N, 3) input point cloud.
        query_pts: (B, M, 3) query positions.
        Returns: (B, M) occupancy logits.
        """
        plane = self.encoder(input_pts)  # (B, c_dim, R, R)
        return self.decoder(plane, query_pts)  # (B, M)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_encoder() -> nn.Module:
    """Build ConvONet LocalPoolPointnet encoder only."""
    return LocalPoolPointnet(in_ch=3, c_dim=32, plane_res=16, hidden_dim=64)


def build_full() -> nn.Module:
    """Build full ConvONet (LocalPoolPointnet encoder + LocalDecoder)."""
    return ConvONet(c_dim=32, plane_res=16, hidden_dim=64)


def example_encoder_input() -> torch.Tensor:
    """Input point cloud (1, 64, 3) for LocalPoolPointnet encoder."""
    return torch.randn(1, 64, 3)


def example_full_input() -> list:
    """Input point cloud + query points for full ConvONet."""
    input_pts = torch.randn(1, 64, 3) * 0.4  # scale to ~[-0.5, 0.5]
    query_pts = torch.randn(1, 32, 3) * 0.4
    return [input_pts, query_pts]


MENAGERIE_ENTRIES = [
    (
        "ConvONet LocalPoolPointnet (scatter-max plane encoder + 2D conv aggregation)",
        "build_encoder",
        "example_encoder_input",
        "2020",
        "DC",
    ),
    (
        "ConvONet full (LocalPoolPointnet encoder + bilinear-interpolate LocalDecoder)",
        "build_full",
        "example_full_input",
        "2020",
        "DC",
    ),
]
