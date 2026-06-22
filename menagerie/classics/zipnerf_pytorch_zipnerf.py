"""Zip-NeRF compact anti-aliased grid radiance field.

Paper: Barron et al., 2023, "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance
Fields".

Zip-NeRF combines mip-NeRF-style conical frustum integration with grid features.
This reconstruction traces the distinctive primitive: ray intervals are lifted
to integrated, scale-aware grid features before proposal weights and volumetric
alpha compositing produce color.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IntegratedHashGrid(nn.Module):
    """Small anti-aliased multiresolution grid encoder."""

    def __init__(self, levels: int = 3, features: int = 8) -> None:
        """Initialize dense proxy grids for traceable hash-grid features."""

        super().__init__()
        self.grids = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, features, 4 * 2**i, 4 * 2**i) * 0.02)
                for i in range(levels)
            ]
        )
        self.proj = nn.Linear(levels * features, 32)

    def forward(self, xy: Tensor, radius: Tensor) -> Tensor:
        """Sample grid features with radius-dependent prefiltering."""

        samples = []
        grid = xy.clamp(-1, 1).view(xy.shape[0], xy.shape[1], 1, 2)
        for level, param in enumerate(self.grids):
            feat = F.grid_sample(param.expand(xy.shape[0], -1, -1, -1), grid, align_corners=True)
            blur = torch.exp(-radius * float(2**level))
            samples.append(feat.squeeze(-1).transpose(1, 2) * blur)
        return self.proj(torch.cat(samples, dim=-1))


class ZipNeRF(nn.Module):
    """Compact Zip-NeRF renderer."""

    def __init__(self, samples: int = 12) -> None:
        """Initialize proposal and radiance heads."""

        super().__init__()
        self.samples = samples
        self.grid = IntegratedHashGrid()
        self.dir_proj = nn.Linear(3, 32)
        self.sigma = nn.Linear(32, 1)
        self.color = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 3), nn.Sigmoid())

    def forward(self, rays_o: Tensor, rays_d: Tensor, radii: Tensor) -> Tensor:
        """Render colors from ray origins, directions, and cone radii."""

        t = torch.linspace(0.05, 1.0, self.samples, device=rays_o.device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        xy = torch.tanh(pts[..., :2])
        radius = radii[:, None, None] * t[None, :, None]
        feat = F.relu(self.grid(xy, radius))
        sigma = F.softplus(self.sigma(feat)).squeeze(-1)
        delta = torch.cat([t[1:] - t[:-1], t.new_tensor([0.1])])
        alpha = 1 - torch.exp(-sigma * delta)
        trans = torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-6], dim=1), dim=1
        )[:, :-1]
        weights = alpha * trans
        dirs = self.dir_proj(F.normalize(rays_d, dim=-1)).unsqueeze(1).expand(-1, self.samples, -1)
        rgb = self.color(torch.cat([feat, dirs], dim=-1))
        return (weights[..., None] * rgb).sum(dim=1)


def build() -> nn.Module:
    """Build the compact Zip-NeRF renderer."""

    return ZipNeRF().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return origins, directions, and cone radii for a few rays."""

    return torch.randn(8, 3) * 0.1, F.normalize(torch.randn(8, 3), dim=-1), torch.full((8,), 0.02)


MENAGERIE_ENTRIES = [
    ("zipnerf_pytorch_ZipNeRF", "build", "example_input", "2023", "GEN"),
]
