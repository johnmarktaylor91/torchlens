"""DirectVoxGO (DVGO): Direct Voxel Grid Optimisation.

Sun, Sun & Chen, CVPR 2022.
Paper: https://arxiv.org/abs/2111.11215
Source: https://github.com/sunset1995/DirectVoxGO

DirectVoxGO is an explicit neural radiance field that replaces MLP-based implicit
scene representations with directly optimisable 3D voxel grids.  Its distinctive
primitives are:

  1. **Density grid**: a 3D array of scalar log-density values; trilinear interpolation
     samples the density at any continuous 3D point.
  2. **Feature grid**: a 3D array of feature vectors; also trilinear-interpolated.
  3. **Post-activation** of the interpolated density via ``softplus``.
  4. **Shallow RGB MLP**: a tiny 2-layer MLP that maps the trilinearly-interpolated
     feature vector + view direction (spherical harmonics or direct) -> RGB.

In DirectVoxGO coarse-to-fine: a coarse density-only pass finds occupied voxels;
a fine pass adds the feature grid + RGB head.

Simplifications in this reimplementation:
  - Grid size 16^3 (paper uses e.g. 160^3 fine; 16^3 faithfully shows the topology).
  - Feature channels = 12 instead of the paper's 12 (same, actually).
  - RGB head: 2-layer MLP matching the paper's architecture (feature_dim+view_dim -> 128 -> 3).
  - View direction encoding: direct concatenation of 3-D direction vector.
  - Batch of sample points is a tiny (N_pts, 3) tensor on the unit cube.
  - Only the fine model is reproduced here; a separate coarse-only build is also provided.

This faithfully shows the voxel-grid-trilinear-interp + shallow-MLP-rgb-head topology
that makes DVGO architecturally distinctive from implicit NeRFs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: trilinear voxel grid lookup
# ---------------------------------------------------------------------------


def _trilinear_grid_sample(grid: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    """Sample a voxel grid at continuous xyz coordinates via trilinear interpolation.

    Args:
        grid: (C, D, H, W) float tensor (channel-first voxel grid).
        xyz:  (N, 3) float tensor of query points in [-1, 1]^3 (the standard
              ``grid_sample`` normalised coordinate space).
    Returns:
        (N, C) sampled features.
    """
    # grid_sample expects (N, C, D, H, W) input and (N, 1, 1, N_pts, 3) coords.
    N = xyz.shape[0]
    grid = grid.unsqueeze(0)  # (1, C, D, H, W)
    query = xyz.view(1, 1, 1, N, 3)  # (1, 1, 1, N, 3) grid coords
    # grid_sample convention: coords in (W, H, D) order (x=W, y=H, z=D)
    out = F.grid_sample(
        grid,
        query,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, C, 1, 1, N)
    return out.squeeze(0).squeeze(1).squeeze(1).T  # (N, C)


# ---------------------------------------------------------------------------
# DVGO Coarse model (density grid only)
# ---------------------------------------------------------------------------


class DVGOCoarse(nn.Module):
    """DVGO coarse model: density voxel grid with trilinear interpolation.

    Corresponds to the coarse-pass model in DirectVoxGO which optimises only
    a density grid to quickly find occupied space.

    Forward: sample a batch of 3D points from the voxel grid, return log-densities.
    """

    def __init__(self, grid_size: int = 16) -> None:
        super().__init__()
        # Density grid: (1, D, H, W) -- one channel (log-density before softplus)
        self.density_grid = nn.Parameter(torch.zeros(1, grid_size, grid_size, grid_size))

    def density(self, xyz_normalised: torch.Tensor) -> torch.Tensor:
        """Return post-activation density at (N, 3) normalised points in [-1, 1]^3."""
        log_density = _trilinear_grid_sample(self.density_grid, xyz_normalised)  # (N, 1)
        return F.softplus(log_density + 1.0)  # post-activation (bias=1 as in paper)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (N, 3) points in [-1, 1]^3.  Returns (N, 1) density values."""
        return self.density(xyz)


# ---------------------------------------------------------------------------
# DVGO Fine model (density grid + feature grid + RGB MLP head)
# ---------------------------------------------------------------------------


class DVGOFine(nn.Module):
    """DVGO fine model: density grid + feature grid + shallow RGB MLP.

    This is the architecturally-distinctive part of DirectVoxGO:
      - density_grid: trilinearly-interpolated log-density -> softplus density
      - feature_grid: trilinearly-interpolated feature vector per sample point
      - rgb_head: tiny 2-layer MLP (feature + view_dir -> RGB)

    The feature_grid and rgb_head together replace the deep NeRF MLP.
    """

    def __init__(
        self,
        grid_size: int = 16,
        feature_channels: int = 12,
        view_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.view_dim = view_dim

        # Explicit voxel grids (the defining contribution)
        self.density_grid = nn.Parameter(torch.zeros(1, grid_size, grid_size, grid_size))
        self.feature_grid = nn.Parameter(
            torch.randn(feature_channels, grid_size, grid_size, grid_size) * 0.1
        )

        # Shallow RGB MLP head: (feature_channels + view_dim) -> hidden -> 3
        self.rgb_head = nn.Sequential(
            nn.Linear(feature_channels + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),  # RGB in [0, 1]
        )

    def density(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return post-activation density at (N, 3) normalised coords."""
        log_density = _trilinear_grid_sample(self.density_grid, xyz)  # (N, 1)
        return F.softplus(log_density + 1.0)

    def forward(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> dict:
        """Forward pass over a batch of sample points.

        Args:
            xyz: (N, 3) query points in [-1, 1]^3 (normalised scene cube).
            view_dirs: (N, 3) unit viewing direction vectors.

        Returns:
            dict with keys:
              'density': (N, 1) -- post-activation voxel density.
              'rgb':     (N, 3) -- predicted colour for each sample.
        """
        # 1. Trilinear density lookup + post-activation
        density = self.density(xyz)  # (N, 1)

        # 2. Trilinear feature lookup
        features = _trilinear_grid_sample(self.feature_grid, xyz)  # (N, C)

        # 3. RGB head: concat feature + view direction, predict colour
        rgb_input = torch.cat([features, view_dirs], dim=-1)  # (N, C+3)
        rgb = self.rgb_head(rgb_input)  # (N, 3)

        return {"density": density, "rgb": rgb}


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_dvgo_coarse() -> nn.Module:
    """Build a DVGO coarse model (density grid only, 16^3 grid)."""
    return DVGOCoarse(grid_size=16)


def build_dvgo_fine() -> nn.Module:
    """Build a DVGO fine model (density grid + feature grid + RGB MLP, 16^3 grid)."""
    return DVGOFine(grid_size=16, feature_channels=12, hidden_dim=64)


def example_input_coarse() -> torch.Tensor:
    """Example batch of 3D query points (32, 3) in [-1, 1]^3 for DVGO coarse."""
    return torch.rand(32, 3) * 2.0 - 1.0


def example_input_fine() -> list:
    """Example (xyz, view_dirs) for DVGO fine model -- returns list for tl.trace."""
    xyz = torch.rand(32, 3) * 2.0 - 1.0
    view_dirs = F.normalize(torch.randn(32, 3), dim=-1)
    return [xyz, view_dirs]


MENAGERIE_ENTRIES = [
    (
        "DirectVoxGO DVGO-Coarse (explicit density voxel grid + trilinear interp)",
        "build_dvgo_coarse",
        "example_input_coarse",
        "2022",
        "DC",
    ),
    (
        "DirectVoxGO DVGO-Fine (explicit density+feature grids + shallow RGB MLP)",
        "build_dvgo_fine",
        "example_input_fine",
        "2022",
        "DC",
    ),
]
