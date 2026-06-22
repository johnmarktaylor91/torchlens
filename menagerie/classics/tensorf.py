"""TensoRF: Tensorial Radiance Fields.

Chen, Xu, Geiger, Yu & Su, ECCV 2022.
Paper: https://arxiv.org/abs/2203.09517
Source: https://github.com/apchenstu/TensoRF

TensoRF represents a radiance field as a factored tensor decomposition of a 4D
grid (XYZ + feature channels).  Two factorisation variants:

  1. **TensorCP** (CP / CANDECOMP-PARAFAC decomposition):
     Represents the 3D spatial grid as a sum of rank-1 trilinear terms.
     For K components, the density/feature at xyz is:
       F(x,y,z) = sum_{k=1}^{K} v_x^k(x) * v_y^k(y) * v_z^k(z)
     where v_axis^k is a 1D vector of length N_grid sampled via linear interpolation.
     Concatenate feature components -> shallow MLP -> RGB.

  2. **TensorVMSplit** (Vector-Matrix / VM decomposition):
     Represents the 3D grid as a sum of vector-matrix outer products across 3 planes.
     Three plane pairs (XY, XZ, YZ): each pair has a 2D matrix M_plane and a 1D vector
     v_axis perpendicular to that plane.  For 3 axes:
       F(x,y,z) = sum_mode [ M_XY(x,y) * v_Z(z)
                            + M_XZ(x,z) * v_Y(y)
                            + M_YZ(y,z) * v_X(x) ]
     Separates density components (n_comp_density per mode) and
     appearance components (n_comp_app per mode).
     Feature vector from appearance components -> shallow MLP -> RGB.

Simplifications in this reimplementation:
  - Grid resolution: 32 along each axis (paper uses 300+; 32 faithfully shows topology).
  - n_comp_density = 4, n_comp_app = 8 per mode (paper uses 16/48; same structure).
  - RGB MLP: 2-layer (feature -> 128 -> 3) with sigmoid; paper uses same width.
  - View direction: direct 3-vec (paper also encodes with PE or SH basis, optionally).
  - Batch: 64 sample points.
  - Density and appearance are separate components (as in the TensoRF source).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: 1D grid interpolation (nearest-aligned, clipped to valid range)
# ---------------------------------------------------------------------------


def _sample_1d(vec: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample a (C, N_grid) vector at continuous coords in [-1, 1] using linear interp.

    Args:
        vec:    (C, N_grid) learnable 1D feature vectors.
        coords: (N_pts,) float in [-1, 1].

    Returns:
        (N_pts, C) sampled features.
    """
    # Normalise to [0, N_grid-1]
    N_grid = vec.shape[1]
    idx = (coords + 1.0) * 0.5 * (N_grid - 1)  # (N_pts,)
    idx = idx.clamp(0.0, N_grid - 1.0)
    idx_lo = idx.long().clamp(0, N_grid - 2)
    idx_hi = (idx_lo + 1).clamp(0, N_grid - 1)
    w_hi = (idx - idx_lo.float()).unsqueeze(0)  # (1, N_pts)
    w_lo = 1.0 - w_hi
    # vec: (C, N_grid)  ->  (C, N_pts) via index
    f_lo = vec[:, idx_lo]  # (C, N_pts)
    f_hi = vec[:, idx_hi]  # (C, N_pts)
    return (f_lo * w_lo + f_hi * w_hi).T  # (N_pts, C)


def _sample_2d(mat: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sample a (C, N, N) 2D matrix at continuous (u, v) in [-1, 1] using bilinear interp.

    Args:
        mat: (C, N_grid, N_grid) learnable 2D feature map.
        u:   (N_pts,) float in [-1, 1] -- first spatial coordinate.
        v:   (N_pts,) float in [-1, 1] -- second spatial coordinate.

    Returns:
        (N_pts, C) sampled features.
    """
    N_pts = u.shape[0]
    # Use grid_sample: expects (1, C, H, W) input and (1, 1, N_pts, 2) coords
    grid = mat.unsqueeze(0)  # (1, C, N, N)
    coords = torch.stack([v, u], dim=-1)  # (N_pts, 2) -- grid_sample is (x=col, y=row)
    coords = coords.view(1, 1, N_pts, 2)  # (1, 1, N_pts, 2)
    out = F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=True)
    # out: (1, C, 1, N_pts) -> (N_pts, C)
    return out.squeeze(0).squeeze(1).T  # (N_pts, C)


# ---------------------------------------------------------------------------
# TensorCP: CP decomposition radiance field
# ---------------------------------------------------------------------------


class TensorCP(nn.Module):
    """TensoRF with CP (CANDECOMP-PARAFAC) decomposition.

    Radiance grid = sum of K rank-1 terms, each = v_X(x) * v_Y(y) * v_Z(z).
    Separate density and appearance components.
    """

    def __init__(
        self,
        grid_res: int = 32,
        n_comp_density: int = 4,
        n_comp_app: int = 8,
        app_dim: int = 27,
        hidden_dim: int = 64,
        view_dim: int = 3,
    ) -> None:
        super().__init__()
        self.n_comp_density = n_comp_density
        self.n_comp_app = n_comp_app

        # --- Density vectors (3 axes x n_comp_density) ---
        # Each is (n_comp_density, grid_res): one 1D vector per component per axis
        self.density_vec_x = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)
        self.density_vec_y = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)
        self.density_vec_z = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)

        # --- Appearance vectors (3 axes x n_comp_app x app_dim) ---
        # Encode appearance_dim features per component
        self.app_vec_x = nn.Parameter(torch.randn(n_comp_app * app_dim, grid_res) * 0.1)
        self.app_vec_y = nn.Parameter(torch.randn(n_comp_app * app_dim, grid_res) * 0.1)
        self.app_vec_z = nn.Parameter(torch.randn(n_comp_app * app_dim, grid_res) * 0.1)
        self.app_dim = app_dim

        # RGB MLP head
        self.rgb_head = nn.Sequential(
            nn.Linear(app_dim + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def compute_density_feature(self, xyz: torch.Tensor) -> torch.Tensor:
        """CP density feature at (N, 3) query points (normalised to [-1,1])."""
        # Sample each axis vector for each component
        fx = _sample_1d(self.density_vec_x, xyz[:, 0])  # (N, n_comp)
        fy = _sample_1d(self.density_vec_y, xyz[:, 1])  # (N, n_comp)
        fz = _sample_1d(self.density_vec_z, xyz[:, 2])  # (N, n_comp)
        # CP: element-wise product across axes, then sum over components -> scalar
        density_feat = (fx * fy * fz).sum(dim=-1, keepdim=True)  # (N, 1)
        return density_feat

    def compute_app_feature(self, xyz: torch.Tensor) -> torch.Tensor:
        """CP appearance feature at (N, 3) query points."""
        n_comp = self.n_comp_app
        dim = self.app_dim
        fx = _sample_1d(self.app_vec_x, xyz[:, 0])  # (N, n_comp*app_dim)
        fy = _sample_1d(self.app_vec_y, xyz[:, 1])
        fz = _sample_1d(self.app_vec_z, xyz[:, 2])
        # Reshape to (N, n_comp, app_dim), multiply, sum over components
        fx = fx.view(-1, n_comp, dim)
        fy = fy.view(-1, n_comp, dim)
        fz = fz.view(-1, n_comp, dim)
        app_feat = (fx * fy * fz).sum(dim=1)  # (N, app_dim)
        return app_feat

    def forward(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            xyz:       (N, 3) query points in [-1, 1]^3.
            view_dirs: (N, 3) unit viewing directions.

        Returns:
            dict with 'density' (N,1), 'rgb' (N,3).
        """
        density_feat = self.compute_density_feature(xyz)
        density = F.relu(density_feat)  # (N, 1)

        app_feat = self.compute_app_feature(xyz)  # (N, app_dim)
        rgb_in = torch.cat([app_feat, view_dirs], dim=-1)  # (N, app_dim+3)
        rgb = self.rgb_head(rgb_in)  # (N, 3)

        return {"density": density, "rgb": rgb}


# ---------------------------------------------------------------------------
# TensorVMSplit: VM (vector-matrix) decomposition radiance field
# ---------------------------------------------------------------------------


class TensorVMSplit(nn.Module):
    """TensoRF with VM (Vector-Matrix) decomposition.

    The 3D field is decomposed as a sum of vector-matrix outer products:
      F(x,y,z) = sum over 3 modes of [ M_plane(u,v) * v_perp(w) ]
    where (u,v,w) is one of the 3 axis-aligned plane/axis pairings:
      mode XY: M_XY(x,y) * v_Z(z)
      mode XZ: M_XZ(x,z) * v_Y(y)
      mode YZ: M_YZ(y,z) * v_X(x)

    Separate density (n_comp_density per mode) and appearance (n_comp_app per mode).
    """

    def __init__(
        self,
        grid_res: int = 32,
        n_comp_density: int = 4,
        n_comp_app: int = 8,
        app_dim: int = 27,
        hidden_dim: int = 64,
        view_dim: int = 3,
    ) -> None:
        super().__init__()
        self.n_comp_density = n_comp_density
        self.n_comp_app = n_comp_app
        self.app_dim = app_dim

        # --- Density VM decomposition ---
        # 3 planes: XY, XZ, YZ each (n_comp, grid, grid)
        self.density_mat_xy = nn.Parameter(torch.randn(n_comp_density, grid_res, grid_res) * 0.1)
        self.density_mat_xz = nn.Parameter(torch.randn(n_comp_density, grid_res, grid_res) * 0.1)
        self.density_mat_yz = nn.Parameter(torch.randn(n_comp_density, grid_res, grid_res) * 0.1)
        # 3 perpendicular vectors: Z, Y, X
        self.density_vec_z = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)
        self.density_vec_y = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)
        self.density_vec_x = nn.Parameter(torch.randn(n_comp_density, grid_res) * 0.1)

        # --- Appearance VM decomposition ---
        # Planes encode (n_comp * app_dim) channels
        nc_app_dim = n_comp_app * app_dim
        self.app_mat_xy = nn.Parameter(torch.randn(nc_app_dim, grid_res, grid_res) * 0.1)
        self.app_mat_xz = nn.Parameter(torch.randn(nc_app_dim, grid_res, grid_res) * 0.1)
        self.app_mat_yz = nn.Parameter(torch.randn(nc_app_dim, grid_res, grid_res) * 0.1)
        self.app_vec_z = nn.Parameter(torch.randn(nc_app_dim, grid_res) * 0.1)
        self.app_vec_y = nn.Parameter(torch.randn(nc_app_dim, grid_res) * 0.1)
        self.app_vec_x = nn.Parameter(torch.randn(nc_app_dim, grid_res) * 0.1)

        # RGB MLP head
        self.rgb_head = nn.Sequential(
            nn.Linear(app_dim + view_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def _density_feature(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute VM density feature at (N, 3) coords in [-1, 1]^3."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        # mode XY: M_XY(x,y) hadamard v_Z(z) -> (N, n_comp) -> sum
        f_xy = _sample_2d(self.density_mat_xy, x, y)  # (N, n_comp)
        f_z = _sample_1d(self.density_vec_z, z)  # (N, n_comp)
        # mode XZ
        f_xz = _sample_2d(self.density_mat_xz, x, z)
        f_y = _sample_1d(self.density_vec_y, y)
        # mode YZ
        f_yz = _sample_2d(self.density_mat_yz, y, z)
        f_x = _sample_1d(self.density_vec_x, x)

        # Sum across modes (each hadamard product then sum across components)
        density_feat = (
            (f_xy * f_z).sum(-1, keepdim=True)
            + (f_xz * f_y).sum(-1, keepdim=True)
            + (f_yz * f_x).sum(-1, keepdim=True)
        )  # (N, 1)
        return density_feat

    def _app_feature(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute VM appearance feature at (N, 3) coords in [-1, 1]^3."""
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        n_comp = self.n_comp_app
        dim = self.app_dim

        def _vm_mode(mat_feat, vec_feat):
            # mat_feat: (N, n_comp*dim), vec_feat: (N, n_comp*dim)
            mf = mat_feat.view(-1, n_comp, dim)
            vf = vec_feat.view(-1, n_comp, dim)
            return (mf * vf).sum(dim=1)  # (N, dim)

        # mode XY plane x Z vec
        f_xy = _sample_2d(self.app_mat_xy, x, y)
        f_z = _sample_1d(self.app_vec_z, z)
        app_xy = _vm_mode(f_xy, f_z)

        # mode XZ plane x Y vec
        f_xz = _sample_2d(self.app_mat_xz, x, z)
        f_y = _sample_1d(self.app_vec_y, y)
        app_xz = _vm_mode(f_xz, f_y)

        # mode YZ plane x X vec
        f_yz = _sample_2d(self.app_mat_yz, y, z)
        f_x = _sample_1d(self.app_vec_x, x)
        app_yz = _vm_mode(f_yz, f_x)

        return app_xy + app_xz + app_yz  # (N, app_dim)

    def forward(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            xyz:       (N, 3) query points in [-1, 1]^3.
            view_dirs: (N, 3) unit viewing directions.

        Returns:
            dict with 'density' (N,1), 'rgb' (N,3).
        """
        density_feat = self._density_feature(xyz)
        density = F.relu(density_feat)  # (N, 1)

        app_feat = self._app_feature(xyz)  # (N, app_dim)
        rgb_in = torch.cat([app_feat, view_dirs], dim=-1)
        rgb = self.rgb_head(rgb_in)

        return {"density": density, "rgb": rgb}


# ---------------------------------------------------------------------------
# Build functions and example inputs
# ---------------------------------------------------------------------------


def build_tensorf_cp() -> nn.Module:
    """Build TensoRF TensorCP (CP decomposition, 32^3 grid)."""
    return TensorCP(
        grid_res=32,
        n_comp_density=4,
        n_comp_app=8,
        app_dim=27,
        hidden_dim=64,
    )


def build_tensorf_vm() -> nn.Module:
    """Build TensoRF TensorVMSplit (VM decomposition, 32^3 grid)."""
    return TensorVMSplit(
        grid_res=32,
        n_comp_density=4,
        n_comp_app=8,
        app_dim=27,
        hidden_dim=64,
    )


def example_input() -> list:
    """Example (xyz, view_dirs) for TensoRF models -- returns list for tl.trace."""
    xyz = torch.rand(64, 3) * 2.0 - 1.0
    view_dirs = F.normalize(torch.randn(64, 3), dim=-1)
    return [xyz, view_dirs]


MENAGERIE_ENTRIES = [
    (
        "TensoRF TensorCP (CP tensor decomposition radiance field)",
        "build_tensorf_cp",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "TensoRF TensorVMSplit (VM vector-matrix tensor decomposition radiance field)",
        "build_tensorf_vm",
        "example_input",
        "2022",
        "DC",
    ),
]
