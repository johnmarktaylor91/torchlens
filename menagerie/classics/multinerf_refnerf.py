"""Ref-NeRF compact structured view-dependent radiance field.

Paper: Verbin et al. 2022, "Ref-NeRF: Structured View-Dependent Appearance for
Neural Radiance Fields." This renderer predicts normals, roughness, diffuse
color, and specular color from an integrated directional encoding of reflected
view directions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics.multinerf_mipnerf360 import MLP, integrated_posenc


class RefNeRF(nn.Module):
    """Compact Ref-NeRF renderer."""

    def __init__(self, samples: int = 8) -> None:
        """Initialize geometric and directional fields.

        Parameters
        ----------
        samples:
            Samples per ray.
        """

        super().__init__()
        self.samples = samples
        enc_dim = 3 * (1 + 2 * 4)
        self.geo = MLP(enc_dim, 48, 8)
        self.spec = MLP(3 * (1 + 2 * 3) + 2, 48, 3)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        """Render RGB with reflection-direction conditioning.

        Parameters
        ----------
        rays_o:
            Ray origins.
        rays_d:
            Ray directions.

        Returns
        -------
        torch.Tensor
            RGB values.
        """

        t = torch.linspace(0.1, 1.2, self.samples, device=rays_o.device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        enc = integrated_posenc(pts, torch.full_like(pts, 0.01))
        geo = self.geo(enc)
        normal = F.normalize(geo[..., :3], dim=-1)
        diffuse = torch.sigmoid(geo[..., 3:6])
        rough = torch.sigmoid(geo[..., 6:7])
        sigma = F.softplus(geo[..., 7])
        view = -F.normalize(rays_d[:, None, :].expand_as(normal), dim=-1)
        refl = 2 * (normal * view).sum(dim=-1, keepdim=True) * normal - view
        ide = integrated_posenc(refl, rough.expand_as(refl), levels=3)
        spec = torch.sigmoid(
            self.spec(torch.cat([ide, rough, (normal * view).sum(dim=-1, keepdim=True)], dim=-1))
        )
        weights = torch.softmax(sigma, dim=1)
        return (weights[..., None] * (diffuse + spec * (1 - rough))).sum(dim=1)


def build() -> nn.Module:
    """Build compact Ref-NeRF.

    Returns
    -------
    nn.Module
        Ref-NeRF renderer.
    """

    return RefNeRF()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create rays.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Renderer inputs.
    """

    return torch.randn(4, 3), F.normalize(torch.randn(4, 3), dim=-1)


MENAGERIE_ENTRIES = [("multinerf_refnerf", "build", "example_input", "2022", "E7")]
