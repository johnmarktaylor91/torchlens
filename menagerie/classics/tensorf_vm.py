"""TensoRF: Tensorial Radiance Fields with Vector-Matrix Decomposition.

Yu et al., ECCV 2022.
Paper: https://arxiv.org/abs/2203.09517
Source: https://github.com/apchenstu/TensoRF

TensoRF factorizes the 4D radiance field tensor T(x, y, z, feature) into a
low-rank sum of Vector-Matrix (VM) outer products:

  T ≈ sum_r  v_r(axis1) ⊗ M_r(axis2, axis3)

where for each axis combination (XY, XZ, YZ):
  - v_r is a 1D vector of length equal to the resolution along one axis.
  - M_r is a 2D matrix (feature map) over the plane formed by the other two axes.

This VM decomposition separates the 3D spatial indexing into lightweight
1D + 2D components, achieving much more compact storage than a full 3D grid
while still representing arbitrary continuous radiance fields.

At query time for a 3D point (x, y, z):
  1. Trilinearly sample each factorized component using F.grid_sample.
  2. Compute the outer-product feature by multiplying each sampled v_r(axis) with
     the sampled M_r(other axes).
  3. Sum contributions -> density sigma and appearance features.
  4. A tiny MLP decodes appearance features + view direction -> RGB color.

Compact faithfulness:
  - Small grid resolution (16^3), 4 components per plane (R=4), feature_dim=16.
  - Input: a batch of 3D sample points (1, N, 3) with N=256 rays/samples.
  - Output: (1, N, 4) = (sigma, R, G, B) per-sample.
  - All three axis combinations (XY, XZ, YZ) are present in the graph.
  - The VM outer-product sampling and the MLP decoder are the key primitives shown.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VM Decomposition components
# ---------------------------------------------------------------------------


class VMDecomposition(nn.Module):
    """TensoRF Vector-Matrix (VM) decomposition for one feature type.

    Factorizes a 3D feature grid into R sums of (1D vector) x (2D matrix):
      T(x,y,z) = sum_r  v_r^Z(z) * M_r^XY(x, y)    (XY plane)
               + sum_r  v_r^Y(y) * M_r^XZ(x, z)    (XZ plane)
               + sum_r  v_r^X(x) * M_r^YZ(y, z)    (YZ plane)

    Args:
        resolution: grid resolution along each axis.
        num_components: rank R (number of components per plane).
        feature_dim: feature dimension per component.
    """

    def __init__(
        self,
        resolution: int = 16,
        num_components: int = 4,
        feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.num_components = num_components
        self.feature_dim = feature_dim

        # XY plane: matrix M^XY (R, feature_dim, resX, resY)
        # Z vector: v^Z (R, feature_dim, 1, resZ) — using 1D grid_sample via 2D with H=1
        self.mat_xy = nn.Parameter(
            torch.randn(1, num_components * feature_dim, resolution, resolution) * 0.1
        )
        self.vec_z = nn.Parameter(torch.randn(1, num_components * feature_dim, resolution, 1) * 0.1)

        # XZ plane: matrix M^XZ (R, feature_dim, resX, resZ)
        self.mat_xz = nn.Parameter(
            torch.randn(1, num_components * feature_dim, resolution, resolution) * 0.1
        )
        self.vec_y = nn.Parameter(torch.randn(1, num_components * feature_dim, resolution, 1) * 0.1)

        # YZ plane: matrix M^YZ (R, feature_dim, resY, resZ)
        self.mat_yz = nn.Parameter(
            torch.randn(1, num_components * feature_dim, resolution, resolution) * 0.1
        )
        self.vec_x = nn.Parameter(torch.randn(1, num_components * feature_dim, resolution, 1) * 0.1)

    def _sample_plane(self, plane: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
        """Sample a 2D plane at given coordinates.

        Args:
            plane: (1, C, H, W) feature plane.
            coords_2d: (B, N, 1, 2) normalized coordinates in [-1, 1].

        Returns:
            (B, C, N)
        """
        # grid_sample expects (B, C, H_out, W_out) output with grid (B, H_out, W_out, 2)
        sampled = F.grid_sample(
            plane, coords_2d, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, C, N, 1)
        return sampled.squeeze(-1)  # (1, C, N)

    def _sample_line(self, line: torch.Tensor, coord_1d: torch.Tensor) -> torch.Tensor:
        """Sample a 1D line (stored as 2D with H=res, W=1).

        Args:
            line: (1, C, res, 1)
            coord_1d: (B, N, 1, 1) normalized coordinates in [-1, 1] for the H axis.

        Returns:
            (B, C, N)
        """
        sampled = F.grid_sample(
            line, coord_1d, mode="bilinear", padding_mode="border", align_corners=True
        )  # (1, C, N, 1)
        return sampled.squeeze(-1)  # (1, C, N)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """Query VM decomposition at 3D points.

        Args:
            pts: (B, N, 3) normalized coordinates in [-1, 1]^3.

        Returns:
            features: (B, N, feature_dim * num_components) summed feature vector.
        """
        B, N, _ = pts.shape
        x = pts[..., 0:1]  # (B, N, 1)
        y = pts[..., 1:2]
        z = pts[..., 2:3]

        # Build 2D sampling grids: (B, N, 1, 2)
        grid_xy = torch.cat([x, y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
        grid_xz = torch.cat([x, z], dim=-1).unsqueeze(2)
        grid_yz = torch.cat([y, z], dim=-1).unsqueeze(2)

        # Build 1D sampling grids: (B, N, 1, 2) with W coord = 0 (center of the 1-wide grid)
        zeros = torch.zeros_like(x)
        line_z = torch.cat([zeros, z], dim=-1).unsqueeze(2)  # (B, N, 1, 2): (w=0, h=z)
        line_y = torch.cat([zeros, y], dim=-1).unsqueeze(2)
        line_x = torch.cat([zeros, x], dim=-1).unsqueeze(2)

        # Expand params for batch (they are (1, C, H, W))
        mat_xy = self.mat_xy.expand(B, -1, -1, -1)
        mat_xz = self.mat_xz.expand(B, -1, -1, -1)
        mat_yz = self.mat_yz.expand(B, -1, -1, -1)
        vec_z = self.vec_z.expand(B, -1, -1, -1)
        vec_y = self.vec_y.expand(B, -1, -1, -1)
        vec_x = self.vec_x.expand(B, -1, -1, -1)

        # Sample matrix planes: (B, C, N)
        f_mat_xy = self._sample_plane(mat_xy, grid_xy)  # (B, C, N)
        f_mat_xz = self._sample_plane(mat_xz, grid_xz)
        f_mat_yz = self._sample_plane(mat_yz, grid_yz)

        # Sample vector lines: (B, C, N)
        f_vec_z = self._sample_line(vec_z, line_z)
        f_vec_y = self._sample_line(vec_y, line_y)
        f_vec_x = self._sample_line(vec_x, line_x)

        # VM outer products: element-wise multiply matrix * vector per component
        # Each contributes (B, R*feat_dim, N)
        feat_xy = f_mat_xy * f_vec_z  # XY plane * Z vector
        feat_xz = f_mat_xz * f_vec_y  # XZ plane * Y vector
        feat_yz = f_mat_yz * f_vec_x  # YZ plane * X vector

        # Sum all contributions: (B, C, N)
        feat = feat_xy + feat_xz + feat_yz  # (B, R*feat_dim, N)
        return feat.permute(0, 2, 1)  # (B, N, R*feat_dim)


# ---------------------------------------------------------------------------
# Density VM decomposition (simpler: no feature_dim, just scalar per component)
# ---------------------------------------------------------------------------


class DensityVM(nn.Module):
    """VM decomposition for density (scalar output per point)."""

    def __init__(self, resolution: int = 16, num_components: int = 4) -> None:
        super().__init__()
        self.vm = VMDecomposition(resolution, num_components, feature_dim=1)
        self.feature_dim = num_components  # R scalar features summed

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, 3)
        feat = self.vm(pts)  # (B, N, R)
        sigma = feat.sum(-1, keepdim=True).relu()  # (B, N, 1)
        return sigma


# ---------------------------------------------------------------------------
# Tiny MLP for color prediction
# ---------------------------------------------------------------------------


class ColorMLP(nn.Module):
    """Tiny MLP: appearance features + view direction -> RGB."""

    def __init__(self, in_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + 3, hidden),  # + 3 for view direction
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        # feat: (B, N, in_features); dirs: (B, N, 3)
        x = torch.cat([feat, dirs], dim=-1)
        return self.net(x)  # (B, N, 3)


# ---------------------------------------------------------------------------
# Full TensoRF model
# ---------------------------------------------------------------------------


class TensoRF(nn.Module):
    """TensoRF VM-decomposed radiance field (compact random-init reimpl).

    Input: (B, N, 3) — batch of 3D sample points in [-1, 1]^3.
    Output: (B, N, 4) — sigma, R, G, B per point.

    Note: view direction is synthesized as zeros for compact tracing (the
    ColorMLP still receives it as input, preserving the architecture shape).
    """

    def __init__(
        self,
        resolution: int = 16,
        num_components: int = 4,
        feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.density_vm = DensityVM(resolution, num_components)
        self.appearance_vm = VMDecomposition(resolution, num_components, feature_dim)
        app_feat_size = num_components * feature_dim
        self.color_mlp = ColorMLP(in_features=app_feat_size)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        # pts: (B, N, 3)
        B, N, _ = pts.shape

        # VM-factorized density
        sigma = self.density_vm(pts)  # (B, N, 1)

        # VM-factorized appearance features
        app_feat = self.appearance_vm(pts)  # (B, N, R*feat_dim)

        # View direction (zero for compact trace; architecture shape preserved)
        dirs = torch.zeros(B, N, 3, device=pts.device, dtype=pts.dtype)

        # Color MLP
        rgb = self.color_mlp(app_feat, dirs)  # (B, N, 3)

        return torch.cat([sigma, rgb], dim=-1)  # (B, N, 4)


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_tensorf_vm() -> nn.Module:
    """Build TensoRF VM (resolution=16, 4 components, feature_dim=16)."""
    return TensoRF(resolution=16, num_components=4, feature_dim=16)


def example_input() -> torch.Tensor:
    """Batch of 3D sample points: (1, 256, 3) in [-1, 1]^3."""
    return torch.rand(1, 256, 3) * 2 - 1


MENAGERIE_ENTRIES = [
    (
        "TensoRF VM (Vector-Matrix factorized radiance field, VM outer-product grid sampling + color MLP)",
        "build_tensorf_vm",
        "example_input",
        "2022",
        "DC",
    ),
]
