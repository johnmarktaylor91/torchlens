"""Mip-NeRF 360 compact proposal-and-NeRF renderer.

Paper: Barron et al. 2022, "Mip-NeRF 360: Unbounded Anti-Aliased Neural
Radiance Fields." The model keeps conical-frustum-style integrated position
features, proposal-density resampling, scene contraction, and final NeRF color
and density prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def integrated_posenc(x: torch.Tensor, cov: torch.Tensor, levels: int = 4) -> torch.Tensor:
    """Encode positions with Gaussian-integrated sinusoidal features.

    Parameters
    ----------
    x:
        Sample means.
    cov:
        Sample covariance diagonal.
    levels:
        Number of frequency bands.

    Returns
    -------
    torch.Tensor
        Encoded samples.
    """

    feats = [x]
    for level in range(levels):
        scale = float(2**level)
        atten = torch.exp(-0.5 * cov * scale * scale)
        feats.extend([atten * torch.sin(scale * x), atten * torch.cos(scale * x)])
    return torch.cat(feats, dim=-1)


class MLP(nn.Module):
    """Small residual MLP used by MultiNeRF."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        """Initialize MLP.

        Parameters
        ----------
        in_dim:
            Input width.
        hidden:
            Hidden width.
        out_dim:
            Output width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.net(x)


class MipNeRF360(nn.Module):
    """Compact Mip-NeRF 360 renderer."""

    def __init__(self, samples: int = 8) -> None:
        """Initialize proposal and radiance fields.

        Parameters
        ----------
        samples:
            Samples per ray.
        """

        super().__init__()
        self.samples = samples
        enc_dim = 3 * (1 + 2 * 4)
        self.proposal = MLP(enc_dim, 32, 1)
        self.field = MLP(enc_dim + 3, 48, 4)

    def forward(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, radii: torch.Tensor
    ) -> torch.Tensor:
        """Render RGB rays through proposal-weighted samples.

        Parameters
        ----------
        rays_o:
            Ray origins.
        rays_d:
            Ray directions.
        radii:
            Cone radii per ray.

        Returns
        -------
        torch.Tensor
            Rendered RGB values.
        """

        t = torch.linspace(0.1, 1.5, self.samples, device=rays_o.device)
        pts = rays_o[:, None, :] + rays_d[:, None, :] * t[None, :, None]
        pts = pts / (1.0 + pts.norm(dim=-1, keepdim=True))
        cov = radii[:, None, :] * t[None, :, None].square()
        enc = integrated_posenc(pts, cov)
        prop_sigma = F.softplus(self.proposal(enc)).squeeze(-1)
        weights = torch.softmax(prop_sigma, dim=-1)
        dirs = rays_d[:, None, :].expand_as(pts)
        raw = self.field(torch.cat([enc, dirs], dim=-1))
        rgb = torch.sigmoid(raw[..., :3])
        sigma = F.softplus(raw[..., 3])
        alpha = 1.0 - torch.exp(-sigma * weights)
        return (alpha[..., None] * rgb).sum(dim=1)


def build() -> nn.Module:
    """Build compact Mip-NeRF 360.

    Returns
    -------
    nn.Module
        Renderer.
    """

    return MipNeRF360()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create ray origins, directions, and radii.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Renderer inputs.
    """

    return torch.randn(4, 3), F.normalize(torch.randn(4, 3), dim=-1), torch.full((4, 3), 0.02)


MENAGERIE_ENTRIES = [("multinerf_mipnerf360", "build", "example_input", "2022", "E7")]
