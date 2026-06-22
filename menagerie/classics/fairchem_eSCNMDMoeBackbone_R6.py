"""FairChem eSCNMDMoeBackbone R6 compact reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.fairchem_eSCNMDBackbone_R6 import CompactESCNMD
from menagerie.classics.fairchem_EScAIPBackbone_R6 import example_graph


class CompactESCNMDMoE(nn.Module):
    """Compact mixture-of-experts wrapper around eSCNMD backbones."""

    def __init__(self, experts: int = 3) -> None:
        """Initialize eSCNMD MoE.

        Parameters
        ----------
        experts:
            Number of compact experts.
        """

        super().__init__()
        self.experts = nn.ModuleList([CompactESCNMD(layers=1) for _ in range(experts)])
        self.router = nn.Linear(3, experts)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route atomistic graph inputs through eSCNMD experts.

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
            Mixture energy and force proxy.
        """

        pooled_pos = pos.mean(dim=0, keepdim=True)
        weights = torch.softmax(self.router(pooled_pos), dim=-1).squeeze(0)
        energy = pos.new_zeros(())
        forces = torch.zeros_like(pos)
        for idx, expert in enumerate(self.experts):
            e, f = expert(z, pos, edge_index)
            energy = energy + weights[idx] * e
            forces = forces + weights[idx] * f
        return energy, forces


def build_fairchem_eSCNMDMoeBackbone_R6() -> nn.Module:
    """Build compact FairChem eSCNMD MoE backbone.

    Returns
    -------
    nn.Module
        Random-init compact eSCNMD MoE.
    """

    return CompactESCNMDMoE()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact eSCNMD MoE inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic graph inputs.
    """

    return example_graph()


build = build_fairchem_eSCNMDMoeBackbone_R6

MENAGERIE_ENTRIES = [
    (
        "fairchem_eSCNMDMoeBackbone_R6",
        "build_fairchem_eSCNMDMoeBackbone_R6",
        "example_input",
        "2024",
        "E6",
    ),
]
