"""Plenoxels radiance-field renderer in torch tensor form.

Fridovich-Keil et al. (CVPR 2022), "Plenoxels: Radiance Fields without Neural
Networks."  Plenoxels represent a scene as an explicit sparse voxel grid of
density and spherical-harmonic color coefficients, then render rays by
trilinear interpolation and differentiable volume compositing.  This compact
Torch module is not a learned neural net; it exposes the faithful differentiable
rendering computation so TorchLens can render the base-environment graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlenoxelRenderer(nn.Module):
    """Compact explicit voxel-grid radiance-field renderer."""

    def __init__(self, grid_size: int = 8, sh_basis: int = 4) -> None:
        """Initialize voxel density and color coefficients.

        Parameters
        ----------
        grid_size:
            Resolution of the dense compact grid.
        sh_basis:
            Number of low-order spherical-harmonic basis terms.
        """

        super().__init__()
        self.density = nn.Parameter(torch.randn(1, 1, grid_size, grid_size, grid_size) * 0.1)
        self.color = nn.Parameter(
            torch.randn(1, 3 * sh_basis, grid_size, grid_size, grid_size) * 0.1
        )
        self.sh_basis = sh_basis

    def _basis(self, dirs: torch.Tensor) -> torch.Tensor:
        """Evaluate low-order spherical-harmonic-like basis terms.

        Parameters
        ----------
        dirs:
            Unit ray directions of shape ``(B, R, 3)``.

        Returns
        -------
        torch.Tensor
            Basis tensor of shape ``(B, R, K)``.
        """

        return torch.stack(
            [torch.ones_like(dirs[..., 0]), dirs[..., 0], dirs[..., 1], dirs[..., 2]],
            dim=-1,
        )[..., : self.sh_basis]

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        """Render RGB values along compact rays.

        Parameters
        ----------
        rays:
            Tensor ``(B, R, 6)`` containing ray origins and directions in
            normalized coordinates.

        Returns
        -------
        torch.Tensor
            RGB ray colors of shape ``(B, R, 3)``.
        """

        origins = rays[..., :3]
        dirs = F.normalize(rays[..., 3:6], dim=-1)
        samples = torch.linspace(0.1, 1.0, 8, device=rays.device, dtype=rays.dtype)
        pts = origins.unsqueeze(2) + samples.view(1, 1, -1, 1) * dirs.unsqueeze(2)
        grid = (pts.clamp(-1.0, 1.0)).view(rays.shape[0], -1, 1, 1, 3)
        sigma = F.softplus(
            F.grid_sample(
                self.density.expand(rays.shape[0], -1, -1, -1, -1), grid, align_corners=True
            )
        )
        coeff = F.grid_sample(
            self.color.expand(rays.shape[0], -1, -1, -1, -1), grid, align_corners=True
        )
        coeff = coeff.view(rays.shape[0], 3, self.sh_basis, rays.shape[1], samples.numel()).permute(
            0, 3, 4, 1, 2
        )
        basis = self._basis(dirs).unsqueeze(2).unsqueeze(3)
        rgb = torch.sigmoid((coeff * basis).sum(dim=-1))
        alpha = 1.0 - torch.exp(-sigma.view(rays.shape[0], rays.shape[1], samples.numel()) * 0.15)
        trans = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-6], dim=-1), dim=-1
        )[..., :-1]
        weights = alpha * trans
        return (weights.unsqueeze(-1) * rgb).sum(dim=2)


def build() -> nn.Module:
    """Build the compact Plenoxels renderer.

    Returns
    -------
    nn.Module
        Random-init explicit renderer in evaluation mode.
    """

    return PlenoxelRenderer().eval()


def example_input() -> torch.Tensor:
    """Return compact ray origins and directions.

    Returns
    -------
    torch.Tensor
        Ray tensor of shape ``(1, 12, 6)``.
    """

    origins = torch.zeros(1, 12, 3)
    dirs = torch.randn(1, 12, 3)
    return torch.cat([origins, dirs], dim=-1)


MENAGERIE_ENTRIES = [
    ("plenoxels_torch", "build", "example_input", "2022", "E5"),
]
