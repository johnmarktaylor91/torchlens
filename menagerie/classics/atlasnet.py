"""AtlasNet: A Papier-Mache Approach to Learning 3D Surface Generation.

Groueix et al., CVPR 2018.
Paper: https://arxiv.org/abs/1802.05384
Source: https://github.com/ThibaultGroueix/AtlasNet

AtlasNet represents a 3D shape as the union of ``N`` parametric surface patches.
Each patch has its own small MLP decoder that maps a 2D sample drawn from the unit
square, concatenated with a global shape latent, to a 3D point on the surface.
The defining structure is this bank of per-patch MLP decoders (the "papier-mache"
of learned 2D->3D charts) that together tile the object surface.

This is a faithful random-init reimplementation of the AtlasNet decoder
(``model.py`` ``PointGenCon`` x ``nb_primitives``):
  - bottleneck_size = 1024 (global shape latent)
  - nb_primitives = 4 patches, each a 4-layer 1D-conv MLP
    (in -> in -> in/2 -> in/4 -> 3) with BN + ReLU and a final tanh
  - per patch the 2D grid sample (2 dims) is concatenated to the latent
Input is the global shape latent ``(B, 1024)``; the decoder samples a fixed grid
internally and emits ``(B, 3, N*num_points)`` surface points.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointGenCon(nn.Module):
    """Per-patch 1D-conv MLP decoder mapping (latent + 2D sample) -> 3D point."""

    def __init__(self, bottleneck_size: int = 1024 + 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = nn.Conv1d(bottleneck_size, bottleneck_size // 2, 1)
        self.conv3 = nn.Conv1d(bottleneck_size // 2, bottleneck_size // 4, 1)
        self.conv4 = nn.Conv1d(bottleneck_size // 4, 3, 1)
        self.bn1 = nn.BatchNorm1d(bottleneck_size)
        self.bn2 = nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = nn.BatchNorm1d(bottleneck_size // 4)
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.th(self.conv4(x))


class AtlasNetDecoder(nn.Module):
    """Bank of ``nb_primitives`` per-patch MLP decoders tiling the surface."""

    def __init__(
        self,
        bottleneck_size: int = 1024,
        nb_primitives: int = 4,
        num_points_per_primitive: int = 64,
    ) -> None:
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.num_points_per_primitive = num_points_per_primitive
        self.decoders = nn.ModuleList(
            [PointGenCon(bottleneck_size + 2) for _ in range(nb_primitives)]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.size(0)
        n = self.num_points_per_primitive
        outputs = []
        for decoder in self.decoders:
            # Sample a fixed 2D grid in the unit square for this patch.
            grid = torch.rand(batch_size, 2, n, device=latent.device, dtype=latent.dtype)
            latent_tiled = latent.unsqueeze(2).expand(batch_size, self.bottleneck_size, n)
            y = torch.cat((grid, latent_tiled), dim=1)  # (B, 2 + bottleneck, n)
            outputs.append(decoder(y))  # (B, 3, n)
        return torch.cat(outputs, dim=2)  # (B, 3, nb_primitives * n)


def build() -> nn.Module:
    """Build the AtlasNet decoder (4 patches, bottleneck 1024)."""
    return AtlasNetDecoder(bottleneck_size=1024, nb_primitives=4, num_points_per_primitive=64)


def example_input() -> torch.Tensor:
    """Example global shape latent ``(2, 1024)`` for the AtlasNet decoder."""
    return torch.randn(2, 1024)


MENAGERIE_ENTRIES = [
    (
        "AtlasNet decoder (per-patch MLP surface charts)",
        "build",
        "example_input",
        "2018",
        "DC",
    ),
]
