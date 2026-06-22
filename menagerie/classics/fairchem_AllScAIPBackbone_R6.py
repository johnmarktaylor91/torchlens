"""FairChem AllScAIPBackbone R6 compact reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.fairchem_EScAIPBackbone_R6 import CompactEScAIPBackbone, example_graph


class CompactAllScAIPBackbone(CompactEScAIPBackbone):
    """AllScAIP with local neighbor attention followed by global all-to-all attention."""

    def __init__(self) -> None:
        """Initialize local EScAIP blocks and global atom attention."""

        super().__init__(layers=2)
        self.global_attn = nn.MultiheadAttention(48, 4, batch_first=True)
        self.global_norm = nn.LayerNorm(48)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict energy and forces using local then all-to-all node attention."""

        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = torch.linalg.vector_norm(rel, dim=-1).clamp_min(1e-6)
        directions = rel / dist[:, None]
        radial = self.radial(dist)
        atoms = self.embed(z)
        vector_accum = torch.zeros_like(pos)
        for block in self.blocks:
            atoms, edge_vec = block(atoms, radial, edge_index, directions)
            vector_accum = vector_accum.index_add(0, dst, edge_vec)
        global_atoms = self.global_attn(
            atoms.unsqueeze(0), atoms.unsqueeze(0), atoms.unsqueeze(0), need_weights=False
        )[0].squeeze(0)
        atoms = self.global_norm(atoms + global_atoms)
        energy = self.energy(atoms).sum()
        forces = self.force(atoms) + vector_accum
        return energy, forces


def build_fairchem_AllScAIPBackbone_R6() -> nn.Module:
    """Build compact all-scale FairChem ScAIP backbone.

    Returns
    -------
    nn.Module
        Random-init compact all-scale ScAIP.
    """

    return CompactAllScAIPBackbone()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact atomistic graph inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic graph inputs.
    """

    return example_graph()


build = build_fairchem_AllScAIPBackbone_R6

MENAGERIE_ENTRIES = [
    (
        "fairchem_AllScAIPBackbone_R6",
        "build_fairchem_AllScAIPBackbone_R6",
        "example_input",
        "2024",
        "E6",
    ),
]
