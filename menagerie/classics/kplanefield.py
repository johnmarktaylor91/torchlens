"""K-Planes explicit radiance field.

Fridovich-Keil et al. (CVPR 2023), "K-Planes: Explicit Radiance Fields in
Space, Time, and Appearance."  K-Planes factorizes a d-dimensional radiance
field into learned 2-D feature planes for each coordinate pair, multiplies or
sums sampled plane features, and decodes density/color.  This compact version
uses the 4-D dynamic case with six coordinate-pair planes.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class KPlaneField(nn.Module):
    """Compact 4-D K-Planes field."""

    def __init__(self, resolution: int = 16, features: int = 8) -> None:
        """Initialize learned coordinate-pair feature planes.

        Parameters
        ----------
        resolution:
            Plane resolution.
        features:
            Feature channels per plane.
        """

        super().__init__()
        self.pairs = tuple(itertools.combinations(range(4), 2))
        self.planes = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, features, resolution, resolution) * 0.05)
                for _ in self.pairs
            ]
        )
        self.decoder = nn.Sequential(nn.Linear(features, 24), nn.GELU(), nn.Linear(24, 4))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate density and RGB at 4-D coordinates.

        Parameters
        ----------
        coords:
            Coordinates of shape ``(B, N, 4)`` in ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Density and RGB tensor of shape ``(B, N, 4)``.
        """

        features = None
        for pair, plane in zip(self.pairs, self.planes, strict=True):
            grid = coords[..., pair].view(coords.shape[0], coords.shape[1], 1, 2)
            sampled = F.grid_sample(
                plane.expand(coords.shape[0], -1, -1, -1), grid, align_corners=True
            )
            sampled = sampled.squeeze(-1).transpose(1, 2)
            features = sampled if features is None else features * sampled
        assert features is not None
        out = self.decoder(features)
        return torch.cat([F.softplus(out[..., :1]), torch.sigmoid(out[..., 1:])], dim=-1)


def build() -> nn.Module:
    """Build the compact K-Planes field.

    Returns
    -------
    nn.Module
        Random-init K-Planes field in evaluation mode.
    """

    return KPlaneField().eval()


def example_input() -> torch.Tensor:
    """Return compact 4-D coordinates.

    Returns
    -------
    torch.Tensor
        Coordinate tensor of shape ``(1, 20, 4)``.
    """

    return torch.rand(1, 20, 4) * 2.0 - 1.0


MENAGERIE_ENTRIES = [
    ("KPlaneField", "build", "example_input", "2023", "E5"),
]
