"""RawNeRF compact linear-HDR radiance field.

Paper: Mildenhall et al. 2022, "NeRF in the Dark: High Dynamic Range View
Synthesis from Noisy Raw Images." This variant preserves linear raw RGB output
with exposure control before a differentiable camera tonemapping path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics.multinerf_mipnerf360 import MLP, integrated_posenc


class RawNeRF(nn.Module):
    """Compact RawNeRF renderer with exposure and tonemapping."""

    def __init__(self, samples: int = 8) -> None:
        """Initialize raw radiance field.

        Parameters
        ----------
        samples:
            Samples per ray.
        """

        super().__init__()
        self.samples = samples
        self.field = MLP(3 * (1 + 2 * 4) + 4, 48, 5)

    def forward(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, exposure: torch.Tensor
    ) -> torch.Tensor:
        """Render tonemapped raw RGB rays.

        Parameters
        ----------
        rays_o:
            Ray origins.
        rays_d:
            Ray directions.
        exposure:
            Exposure scalar per ray.

        Returns
        -------
        torch.Tensor
            Tonemapped RGB.
        """

        t = torch.linspace(0.1, 1.2, self.samples, device=rays_o.device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        enc = integrated_posenc(pts, torch.full_like(pts, 0.01))
        exp_feat = exposure[:, None, None].expand(-1, self.samples, 1)
        raw = self.field(torch.cat([enc, rays_d[:, None, :].expand_as(pts), exp_feat], dim=-1))
        linear_rgb = F.softplus(raw[..., :3])
        sigma = F.softplus(raw[..., 3])
        weights = torch.softmax(sigma, dim=1)
        hdr = (weights[..., None] * linear_rgb).sum(dim=1) * exposure[:, None]
        return hdr / (1.0 + hdr)


def build() -> nn.Module:
    """Build compact RawNeRF.

    Returns
    -------
    nn.Module
        RawNeRF renderer.
    """

    return RawNeRF()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create rays and exposure values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Renderer inputs.
    """

    return torch.randn(4, 3), F.normalize(torch.randn(4, 3), dim=-1), torch.ones(4)


MENAGERIE_ENTRIES = [("multinerf_rawnerf", "build", "example_input", "2022", "E7")]
