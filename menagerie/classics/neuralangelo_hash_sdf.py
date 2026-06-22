"""Neuralangelo hash-grid SDF renderer with numerical gradients.

Paper: Neuralangelo: High-Fidelity Neural Surface Reconstruction.

Li et al. (CVPR 2023) combine multi-resolution hash-grid positional encoding
with neural signed-distance fields and volume rendering.  Its load-bearing
primitive is the use of finite-difference numerical gradients at hash-grid
resolution to regularize SDF normals beyond local cells.  This compact module
keeps hash-grid encoding, SDF/color MLPs, ray samples, NeuS-style alpha, and
finite-difference normals.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class HashGridEncoding(nn.Module):
    """Tiny multi-resolution hash-grid encoder."""

    def __init__(self, levels: int = 4, features: int = 2, table_size: int = 64) -> None:
        """Initialize trainable hash tables.

        Parameters
        ----------
        levels:
            Number of grid resolutions.
        features:
            Features per level.
        table_size:
            Entries per hash table.
        """

        super().__init__()
        self.levels = levels
        self.features = features
        self.table_size = table_size
        self.tables = nn.Parameter(torch.randn(levels, table_size, features) * 0.01)

    def _hash(self, corners: Tensor) -> Tensor:
        """Hash integer 3D grid corners.

        Parameters
        ----------
        corners:
            Integer grid coordinates.

        Returns
        -------
        Tensor
            Hash table indices.
        """

        primes = torch.tensor([1, 2654435761, 805459861], device=corners.device, dtype=torch.long)
        return (corners.long() * primes).sum(dim=-1).remainder(self.table_size)

    def forward(self, x: Tensor) -> Tensor:
        """Encode normalized coordinates.

        Parameters
        ----------
        x:
            Coordinates in roughly ``[-1, 1]``.

        Returns
        -------
        Tensor
            Concatenated hash features.
        """

        x01 = (x + 1.0) * 0.5
        feats = []
        for level in range(self.levels):
            res = 4 * (2**level)
            grid = x01 * (res - 1)
            base = torch.floor(grid)
            frac = grid - base
            acc = x.new_zeros(*x.shape[:-1], self.features)
            for ox in (0, 1):
                for oy in (0, 1):
                    for oz in (0, 1):
                        corner = base + torch.tensor([ox, oy, oz], device=x.device)
                        weight = torch.prod(
                            torch.where(
                                torch.tensor([ox, oy, oz], device=x.device) == 1, frac, 1 - frac
                            ),
                            dim=-1,
                            keepdim=True,
                        )
                        acc = acc + weight * self.tables[level, self._hash(corner)]
            feats.append(acc)
        return torch.cat(feats, dim=-1)


class NeuralangeloCompact(nn.Module):
    """Compact hash-grid SDF volume renderer."""

    def __init__(self, hidden: int = 48) -> None:
        """Initialize encoder and neural fields.

        Parameters
        ----------
        hidden:
            Hidden width.
        """

        super().__init__()
        self.encoding = HashGridEncoding()
        enc_dim = self.encoding.levels * self.encoding.features
        self.sdf_mlp = nn.Sequential(
            nn.Linear(enc_dim, hidden), nn.Softplus(beta=100), nn.Linear(hidden, 1)
        )
        self.color_mlp = nn.Sequential(
            nn.Linear(enc_dim + 3 + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),
        )
        self.inv_s = nn.Parameter(torch.tensor(10.0))

    def _sdf(self, points: Tensor) -> Tensor:
        """Evaluate signed distance.

        Parameters
        ----------
        points:
            3D sample points.

        Returns
        -------
        Tensor
            SDF values.
        """

        return self.sdf_mlp(self.encoding(points)).squeeze(-1)

    def _numerical_normal(self, points: Tensor, eps: float = 0.05) -> Tensor:
        """Compute finite-difference SDF normals.

        Parameters
        ----------
        points:
            3D sample points.
        eps:
            Grid-scale offset.

        Returns
        -------
        Tensor
            Unit normal estimates.
        """

        basis = torch.eye(3, device=points.device) * eps
        grads = []
        for axis in range(3):
            grads.append(
                (self._sdf(points + basis[axis]) - self._sdf(points - basis[axis])) / (2 * eps)
            )
        return F.normalize(torch.stack(grads, dim=-1), dim=-1)

    def forward(self, rays_o: Tensor, rays_d: Tensor) -> tuple[Tensor, Tensor]:
        """Render RGB and expected depth for short rays.

        Parameters
        ----------
        rays_o:
            Ray origins ``(B, R, 3)``.
        rays_d:
            Unit ray directions ``(B, R, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            RGB colors and expected depths.
        """

        steps = torch.linspace(0.2, 1.2, 12, device=rays_o.device)
        points = rays_o.unsqueeze(2) + steps.view(1, 1, -1, 1) * rays_d.unsqueeze(2)
        sdf = self._sdf(points)
        normals = self._numerical_normal(points)
        enc = self.encoding(points)
        color = self.color_mlp(
            torch.cat([enc, normals, rays_d.unsqueeze(2).expand_as(points)], dim=-1)
        )
        density = torch.sigmoid(-sdf * self.inv_s)
        alpha = 1.0 - torch.exp(-density * 0.1)
        trans = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-5], dim=-1), dim=-1
        )[..., :-1]
        weights = alpha * trans
        rgb = (weights.unsqueeze(-1) * color).sum(dim=2)
        depth = (weights * steps.view(1, 1, -1)).sum(dim=2)
        return rgb, depth


def build() -> nn.Module:
    """Build compact random-init Neuralangelo renderer.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return NeuralangeloCompact().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a tiny ray bundle.

    Returns
    -------
    tuple[Tensor, Tensor]
        Ray origins and directions.
    """

    origins = torch.zeros(1, 8, 3)
    dirs = F.normalize(torch.randn(1, 8, 3), dim=-1)
    return origins, dirs


MENAGERIE_ENTRIES = [("neuralangelo_hash_sdf", "build", "example_input", "2023", "3D")]
