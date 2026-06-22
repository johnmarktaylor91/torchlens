"""IM-NET: Learning Implicit Fields for Generative Shape Modeling.

Chen & Zhang, CVPR 2019.
Paper: https://arxiv.org/abs/1812.02822
Source: https://github.com/czq142857/IM-NET (and IM-NET-pytorch)

IM-NET is an implicit-field decoder (the "IM" generator/decoder). It maps a
latent shape code concatenated with a 3D query point coordinate to a single
inside/outside occupancy value, through a stack of fully-connected layers whose
hidden widths halve at each stage. The original generator (``generator`` in the
IM-NET decoder) uses widths gf_dim*16, *8, *4, *2, *1 with leaky-ReLU and a
final sigmoid; the latent code is re-concatenated to the running feature at
every layer (a skip-style conditioning).

This is a faithful random-init reimplementation of the IM-NET decoder/generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMNetGenerator(nn.Module):
    """IM-NET implicit-field generator (decoder).

    Concatenates the per-shape latent code ``z`` to each 3D query point and
    regresses an occupancy value in [0, 1]. Hidden widths follow the original
    ``gf_dim`` schedule (16/8/4/2/1 x base) with the latent code re-injected at
    every fully-connected layer.
    """

    def __init__(self, z_dim: int = 256, point_dim: int = 3, gf_dim: int = 128) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        d_in = z_dim + point_dim
        self.linear_1 = nn.Linear(d_in, gf_dim * 16)
        self.linear_2 = nn.Linear(gf_dim * 16 + d_in, gf_dim * 8)
        self.linear_3 = nn.Linear(gf_dim * 8 + d_in, gf_dim * 4)
        self.linear_4 = nn.Linear(gf_dim * 4 + d_in, gf_dim * 2)
        self.linear_5 = nn.Linear(gf_dim * 2 + d_in, gf_dim * 1)
        self.linear_6 = nn.Linear(gf_dim * 1, 1)

    def forward(self, points: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3); z: (B, z_dim)
        zs = z.unsqueeze(1).expand(-1, points.shape[1], -1)
        pointz = torch.cat([points, zs], dim=2)
        h1 = F.leaky_relu(self.linear_1(pointz), negative_slope=0.02)
        h2 = F.leaky_relu(self.linear_2(torch.cat([h1, pointz], dim=2)), negative_slope=0.02)
        h3 = F.leaky_relu(self.linear_3(torch.cat([h2, pointz], dim=2)), negative_slope=0.02)
        h4 = F.leaky_relu(self.linear_4(torch.cat([h3, pointz], dim=2)), negative_slope=0.02)
        h5 = F.leaky_relu(self.linear_5(torch.cat([h4, pointz], dim=2)), negative_slope=0.02)
        h6 = torch.sigmoid(self.linear_6(h5))
        return h6


class _IMNetWrapper(nn.Module):
    """Single-input wrapper: splits a packed tensor into points + latent code.

    The packed input has shape ``(B, N + 1, z_dim + 3)`` where the first ``N``
    rows are query points (their last 3 columns are coordinates) and the final
    row carries the latent code. This keeps the menagerie example_input a single
    tensor while exercising the true two-argument IM-NET forward.
    """

    def __init__(self, model: IMNetGenerator) -> None:
        super().__init__()
        self.model = model

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        z_dim = self.model.z_dim
        points = packed[:, :-1, -3:]
        z = packed[:, -1, :z_dim]
        return self.model(points, z)


def build() -> nn.Module:
    """Build the IM-NET implicit-field generator wrapped for single-tensor input."""
    return _IMNetWrapper(IMNetGenerator(z_dim=256, point_dim=3, gf_dim=128))


def example_input() -> torch.Tensor:
    """Packed tensor ``(1, N+1, z_dim+3)`` carrying query points + a latent code."""
    n_points = 64
    z_dim = 256
    return torch.randn(1, n_points + 1, z_dim + 3)


MENAGERIE_ENTRIES = [
    (
        "IM-NET (implicit-field shape generator/decoder)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
