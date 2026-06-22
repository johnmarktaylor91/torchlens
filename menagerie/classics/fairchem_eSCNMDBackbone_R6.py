"""FairChem eSCNMDBackbone R6 compact reconstruction.

The eSCN/EquiformerV2 line replaces expensive SO(3) convolutions with efficient
spherical-channel updates and maintains scalar/vector features for atomistic dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from menagerie.classics.fairchem_EScAIPBackbone_R6 import RadialBasis, example_graph


class CompactESCNMD(nn.Module):
    """Compact scalar/vector eSCN molecular-dynamics backbone."""

    def __init__(self, dim: int = 48, layers: int = 2) -> None:
        """Initialize compact eSCNMD.

        Parameters
        ----------
        dim:
            Scalar feature dimension.
        layers:
            Number of message-passing layers.
        """

        super().__init__()
        self.embed = nn.Embedding(16, dim)
        self.radial = RadialBasis()
        self.spherical = nn.Linear(9, dim)
        self.scalar_msg = nn.ModuleList([nn.Linear(dim + 8, dim) for _ in range(layers)])
        self.channel_mix = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.vector_gate = nn.ModuleList([nn.Linear(dim, 1) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.energy = nn.Linear(dim, 1)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict energy and equivariant vector force proxy.

        Parameters
        ----------
        z:
            Atomic IDs.
        pos:
            Cartesian positions.
        edge_index:
            Directed edges.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Energy and force proxy.
        """

        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-6)
        direction = rel / dist[:, None]
        sph = torch.cat(
            [
                direction,
                direction[:, :1] * direction,
                direction[:, 1:2] * direction,
            ],
            dim=-1,
        )
        radial = self.radial(dist)
        h = self.embed(z)
        v = torch.zeros_like(pos)
        spherical_channels = self.spherical(sph)
        for scalar_msg, channel_mix, vector_gate, norm in zip(
            self.scalar_msg,
            self.channel_mix,
            self.vector_gate,
            self.norms,
            strict=True,
        ):
            edge_h = torch.cat([h[src] + channel_mix(spherical_channels), radial], dim=-1)
            msg = F.silu(scalar_msg(edge_h))
            h = norm(h + torch.zeros_like(h).index_add(0, dst, msg))
            gate = vector_gate(msg)
            v = v + torch.zeros_like(v).index_add(0, dst, gate * direction)
        return self.energy(h).sum(), v


def build_fairchem_eSCNMDBackbone_R6() -> nn.Module:
    """Build compact FairChem eSCNMDBackbone R6.

    Returns
    -------
    nn.Module
        Random-init compact eSCNMD.
    """

    return CompactESCNMD(layers=2)


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact eSCNMD inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic graph inputs.
    """

    return example_graph()


build = build_fairchem_eSCNMDBackbone_R6

MENAGERIE_ENTRIES = [
    (
        "fairchem_eSCNMDBackbone_R6",
        "build_fairchem_eSCNMDBackbone_R6",
        "example_input",
        "2024",
        "E6",
    ),
]
