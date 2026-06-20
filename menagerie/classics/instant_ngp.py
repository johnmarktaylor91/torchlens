"""Instant-NGP: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.

Mueller, Evans, Schied & Keller (NVIDIA), SIGGRAPH 2022, arXiv:2201.05989.
Source: https://github.com/NVlabs/instant-ngp  (the ``tiny-cuda-nn`` hash encoder).

Instant-NGP's distinctive contribution is the **multiresolution hash-grid encoding**:
a 3D query point is encoded by, at each of ``L`` geometrically-spaced resolution
levels, finding its enclosing voxel, hashing the 8 voxel corners into a fixed-size
hash table of trainable feature vectors, and TRILINEARLY interpolating the looked-up
features. The per-level features are concatenated and fed to a small "tiny" MLP that
predicts density + a geometry feature; a second MLP combines the geometry feature with
a directional encoding to predict RGB color.

The CEILING in the menagerie is the ``tinycudann`` package: the hash lookup +
fused-MLP are a custom CUDA kernel that must be compiled with nvcc (no pip wheel).
The kernel is purely an OPTIMIZATION of an operation expressible in plain torch:
the hash function (spatial hash of integer voxel coords), the table gather, and the
trilinear blend are all standard tensor ops. This module reimplements the full
architecture -- hash-grid encoding (faithful spatial hash + trilinear interp) plus the
density/color MLPs -- in pure torch at small width so it traces and renders.

Config here: L=8 levels, T=2^14 table size, F=2 features/level, base_res=16,
per_level_scale~1.4 (the published default progression); density MLP = 1 hidden
layer, color MLP = 2 hidden layers (matching the paper's "1 hidden layer" geometry
net + "2 hidden layers" color net). Input is a batch of (xyz, view-dir) samples.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large primes used by the instant-ngp spatial hash (pi_1, pi_2, pi_3 in the paper;
# the first coordinate uses 1).
_PRIMES = (1, 2654435761, 805459861)


class HashGridEncoder(nn.Module):
    """Multiresolution hash-grid positional encoding (faithful, pure torch).

    For each level, voxel corners are spatially hashed into a trainable feature
    table and the 8 corner features are trilinearly interpolated. Per-level
    encodings are concatenated.
    """

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 14,
        base_resolution: int = 16,
        per_level_scale: float = 1.4,
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.table_size = 2**log2_hashmap_size
        # Geometric resolution progression per level (paper eq.: N_l = floor(N_min * b^l)).
        resolutions = [int(base_resolution * (per_level_scale**level)) for level in range(n_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))
        # One trainable feature table per level.
        self.tables = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.table_size, n_features_per_level) * 1e-4)
                for _ in range(n_levels)
            ]
        )
        self.output_dim = n_levels * n_features_per_level
        # 8 unit-cube corner offsets for trilinear interpolation.
        corners = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=torch.long
        )
        self.register_buffer("corners", corners)  # (8, 3)
        prime = torch.tensor(_PRIMES, dtype=torch.long)
        self.register_buffer("primes", prime)  # (3,)

    def _spatial_hash(self, voxel_coords: torch.Tensor) -> torch.Tensor:
        """Instant-NGP spatial hash of integer voxel coords -> table index (mod T)."""
        # voxel_coords: (..., 3) integer. XOR of (coord * prime) per axis, mod table_size.
        h = voxel_coords[..., 0] * self.primes[0]
        h = h ^ (voxel_coords[..., 1] * self.primes[1])
        h = h ^ (voxel_coords[..., 2] * self.primes[2])
        return h % self.table_size

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (N, 3) in [0, 1].
        n = xyz.shape[0]
        per_level = []
        for level in range(self.n_levels):
            res = int(self.resolutions[level].item())
            scaled = xyz * res  # (N, 3) continuous grid coords
            base = torch.floor(scaled).long()  # (N, 3) lower corner
            frac = scaled - base.float()  # (N, 3) in [0, 1)
            # Gather the 8 corner features and trilinearly blend.
            corner_voxels = base.unsqueeze(1) + self.corners.unsqueeze(0)  # (N, 8, 3)
            idx = self._spatial_hash(corner_voxels)  # (N, 8)
            feats = self.tables[level][idx.reshape(-1)].reshape(
                n, 8, self.n_features_per_level
            )  # (N, 8, F)
            # Trilinear weights: prod over axes of (corner ? frac : 1 - frac).
            cw = self.corners.unsqueeze(0).float()  # (1, 8, 3)
            w = cw * frac.unsqueeze(1) + (1 - cw) * (1 - frac).unsqueeze(1)  # (N, 8, 3)
            w = w.prod(dim=-1, keepdim=True)  # (N, 8, 1)
            per_level.append((feats * w).sum(dim=1))  # (N, F)
        return torch.cat(per_level, dim=-1)  # (N, L*F)


class SphericalHarmonicsDirEncoding(nn.Module):
    """Degree-3 spherical-harmonics directional encoding (16 coeffs), as in instant-ngp."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 16

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        # d: (N, 3) unit view directions. Real SH up to l=3 (16 basis functions).
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        xx, yy, zz = x * x, y * y, z * z
        out = torch.stack(
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
        )
        return out  # (N, 16)


class InstantNGP(nn.Module):
    """Instant-NGP NeRF field: hash-grid encoding -> density MLP -> color MLP."""

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 14,
        geo_feat_dim: int = 15,
        hidden_dim: int = 64,
        color_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = HashGridEncoder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.dir_encoder = SphericalHarmonicsDirEncoding()
        # Density (geometry) network: 1 hidden layer (paper).
        self.density_net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1 + geo_feat_dim),  # sigma + geometry feature
        )
        # Color network: 2 hidden layers (paper).
        self.color_net = nn.Sequential(
            nn.Linear(geo_feat_dim + self.dir_encoder.output_dim, color_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(color_hidden_dim, color_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(color_hidden_dim, 3),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: (N, 6) = [xyz (in [0,1]), view-dir (unit)].
        xyz, view = samples[:, :3], samples[:, 3:]
        enc = self.encoder(xyz)
        h = self.density_net(enc)
        sigma = F.softplus(h[:, :1])  # density >= 0
        geo_feat = h[:, 1:]
        dir_enc = self.dir_encoder(F.normalize(view, dim=-1))
        rgb = torch.sigmoid(self.color_net(torch.cat([geo_feat, dir_enc], dim=-1)))
        return torch.cat([rgb, sigma], dim=-1)  # (N, 4) = RGBsigma


def build_instant_ngp() -> nn.Module:
    """Build Instant-NGP NeRF field (hash-grid encoding + density/color MLPs)."""
    return InstantNGP()


def example_input() -> torch.Tensor:
    """A small batch ``(64, 6)`` of [xyz in [0,1], view-dir] ray samples."""
    xyz = torch.rand(64, 3)
    d = F.normalize(torch.randn(64, 3), dim=-1)
    return torch.cat([xyz, d], dim=-1)


MENAGERIE_ENTRIES = [
    (
        "Instant-NGP (multiresolution hash-grid NeRF encoder)",
        "build_instant_ngp",
        "example_input",
        "2022",
        "DC",
    ),
]
