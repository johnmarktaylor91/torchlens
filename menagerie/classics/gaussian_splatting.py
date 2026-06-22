"""3D Gaussian Splatting: a learnable explicit Gaussian scene representation.

Kerbl, Kopanas, Leimkuehler & Drettakis (INRIA), SIGGRAPH 2023, arXiv:2308.04079.
Source: https://github.com/graphdeco-inria/gaussian-splatting  (the
``diff_gaussian_rasterization`` CUDA submodule).

3D Gaussian Splatting represents a scene as a set of ``N`` anisotropic 3D Gaussians,
each carrying LEARNED per-Gaussian attributes that are optimized directly:
  - position (mean) xyz           (3,)
  - scale (log-space)             (3,)
  - rotation quaternion           (4,)
  - opacity (logit)               (1,)
  - view-dependent color via spherical-harmonics coefficients (SH degree 3 = 16
    coeffs x 3 channels = 48,) evaluated for the current view direction.

The CEILING in the menagerie is ``diff_gaussian_rasterization``: a custom CUDA
TILE-based rasterizer that alpha-composites the projected Gaussians into an image.
That rasterizer is a FIXED, non-learned differentiable renderer -- a fast
implementation of an alpha-blend over splats, not part of the model's learned
graph. The LEARNED part is the per-Gaussian parameter set + the SH-to-RGB color
evaluation. Following the menagerie precedent (cross_view_splatter), we faithfully
reimplement the learned scene module and STOP at the per-Gaussian parameter map:
activations, scales, rotations (normalized quaternions), opacities, and the
view-evaluated RGB colors. No rasterization.

This is the canonical OPTIMIZATION-BASED 3DGS (a free Gaussian point set), distinct
from feed-forward image-conditioned splat predictors. Small scene: N=256 Gaussians.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _eval_sh_degree3(sh_coeffs: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate degree-3 RGB spherical harmonics at view directions.

    sh_coeffs: (N, 16, 3) per-Gaussian SH coefficients.
    dirs:      (N, 3) unit view directions (Gaussian-to-camera).
    Returns:   (N, 3) RGB.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    basis = torch.stack(
        [
            0.28209479 * torch.ones_like(x),
            -0.48860251 * y,
            0.48860251 * z,
            -0.48860251 * x,
            1.09254843 * x * y,
            -1.09254843 * y * z,
            0.31539157 * (3 * zz - 1),
            -1.09254843 * x * z,
            0.54627421 * (xx - yy),
            -0.59004358 * y * (3 * xx - yy),
            2.89061144 * x * y * z,
            -0.45704579 * y * (5 * zz - 1),
            0.37317633 * z * (5 * zz - 3),
            -0.45704579 * x * (5 * zz - 1),
            1.44530572 * z * (xx - yy),
            -0.59004358 * x * (xx - 3 * yy),
        ],
        dim=-1,
    )  # (N, 16)
    rgb = torch.einsum("nc,ncd->nd", basis, sh_coeffs)  # (N, 3)
    return rgb + 0.5  # SH-to-RGB offset (paper convention)


class GaussianScene(nn.Module):
    """Learnable 3D Gaussian scene: per-Gaussian attributes -> activated param map."""

    def __init__(self, n_gaussians: int = 256, sh_degree: int = 3) -> None:
        super().__init__()
        self.n = n_gaussians
        n_sh = (sh_degree + 1) ** 2  # 16 for degree 3
        # Directly-optimized per-Gaussian parameters (random init, as at scene start).
        self.means = nn.Parameter(torch.randn(n_gaussians, 3) * 0.5)
        self.scales_raw = nn.Parameter(torch.randn(n_gaussians, 3) * 0.1 - 2.0)  # log-scale
        self.quats_raw = nn.Parameter(torch.randn(n_gaussians, 4))
        self.opacity_raw = nn.Parameter(torch.randn(n_gaussians, 1))
        self.sh = nn.Parameter(torch.randn(n_gaussians, n_sh, 3) * 0.1)

    def forward(self, view_dir: torch.Tensor) -> torch.Tensor:
        # view_dir: (1, 3) global view direction; broadcast per Gaussian.
        means = self.means
        scales = torch.exp(self.scales_raw)  # activation: scale > 0
        quats = F.normalize(self.quats_raw, dim=-1)  # unit rotation quaternion
        opacity = torch.sigmoid(self.opacity_raw)  # in (0, 1)
        dirs = F.normalize(view_dir.expand(self.n, -1), dim=-1)
        colors = torch.sigmoid(_eval_sh_degree3(self.sh, dirs))  # (N, 3) RGB
        # Concatenated per-Gaussian parameter map (the learned scene; pre-rasterization).
        return torch.cat([means, scales, quats, opacity, colors], dim=-1)  # (N, 14)


def build_gaussian_splatting() -> nn.Module:
    """Build a 3D Gaussian Splatting scene (learned per-Gaussian param map; no raster)."""
    return GaussianScene(n_gaussians=256, sh_degree=3)


def example_input() -> torch.Tensor:
    """A single global view direction ``(1, 3)``."""
    return F.normalize(torch.randn(1, 3), dim=-1)


MENAGERIE_ENTRIES = [
    (
        "3D Gaussian Splatting (learnable per-Gaussian scene params)",
        "build_gaussian_splatting",
        "example_input",
        "2023",
        "DC",
    ),
]
