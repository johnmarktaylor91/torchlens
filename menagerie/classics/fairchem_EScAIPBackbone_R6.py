"""FairChem EScAIPBackbone R6 compact reconstruction.

Paper: EScAIP: Efficiently Scaled Attention Interatomic Potential, 2024.  EScAIP
uses graph neural-network blocks with attention over neighbor/edge representations for
atomistic energies and forces.  This compact model keeps radial edge features,
neighbor-level multi-head attention, residual atom updates, and energy/force heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasis(nn.Module):
    """Gaussian radial basis expansion for interatomic distances."""

    def __init__(self, count: int = 8, cutoff: float = 5.0) -> None:
        """Initialize radial basis centers.

        Parameters
        ----------
        count:
            Number of radial basis functions.
        cutoff:
            Maximum compact distance.
        """

        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, count))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances into Gaussian features.

        Parameters
        ----------
        dist:
            Pairwise distances.

        Returns
        -------
        torch.Tensor
            Radial features.
        """

        return torch.exp(-self.gamma.abs() * (dist[..., None] - self.centers) ** 2)


class NeighborAttentionBlock(nn.Module):
    """Neighbor-level attention block for atomistic graphs."""

    def __init__(self, dim: int, radial_dim: int = 8, heads: int = 4) -> None:
        """Initialize neighbor attention.

        Parameters
        ----------
        dim:
            Atom feature dimension.
        radial_dim:
            Radial feature dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.edge = nn.Linear(dim * 2 + radial_dim, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )

    def forward(
        self,
        atoms: torch.Tensor,
        radial: torch.Tensor,
        edge_index: torch.Tensor,
        directions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update atoms by attending over incoming neighbor representations.

        Parameters
        ----------
        atoms:
            Atom features of shape ``(nodes, dim)``.
        radial:
            Edge radial features.
        edge_index:
            Directed edges with shape ``(2, edges)``.
        directions:
            Unit edge directions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated atom features and edge vector messages.
        """

        src, dst = edge_index
        edge_feat = self.edge(torch.cat([atoms[src], atoms[dst], radial], dim=-1))
        q = self.q(atoms[dst]).view(-1, self.heads, self.head_dim)
        k = self.k(edge_feat).view(-1, self.heads, self.head_dim)
        v = self.v(edge_feat).view(-1, self.heads, self.head_dim)
        logits = (q * k).sum(dim=-1) / (self.head_dim**0.5)
        attn = torch.zeros_like(logits)
        for node in range(atoms.shape[0]):
            mask = dst == node
            if bool(mask.any()):
                attn[mask] = torch.softmax(logits[mask], dim=0)
        msg = (attn[..., None] * v).reshape(edge_feat.shape[0], -1)
        agg = torch.zeros_like(atoms).index_add(0, dst, msg)
        atoms = atoms + self.out(agg)
        atoms = atoms + self.ff(atoms)
        edge_vectors = directions * msg[:, :3].mean(dim=-1, keepdim=True)
        return atoms, edge_vectors


class CompactEScAIPBackbone(nn.Module):
    """Compact EScAIP atomistic potential backbone."""

    def __init__(self, atoms: int = 5, dim: int = 48, layers: int = 2, vocab: int = 16) -> None:
        """Initialize compact EScAIP.

        Parameters
        ----------
        atoms:
            Number of compact atoms.
        dim:
            Atom feature dimension.
        layers:
            Number of attention blocks.
        vocab:
            Atomic-number vocabulary size.
        """

        super().__init__()
        self.atoms = atoms
        self.embed = nn.Embedding(vocab, dim)
        self.radial = RadialBasis()
        self.blocks = nn.ModuleList([NeighborAttentionBlock(dim) for _ in range(layers)])
        self.energy = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        self.force = nn.Linear(dim, 3)

    def forward(
        self, z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict compact atomistic energy and force proxy.

        Parameters
        ----------
        z:
            Atomic IDs of shape ``(nodes,)``.
        pos:
            Cartesian positions of shape ``(nodes, 3)``.
        edge_index:
            Directed edge indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Scalar energy and per-atom force proxy.
        """

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
        energy = self.energy(atoms).sum()
        forces = self.force(atoms) + vector_accum
        return energy, forces


def example_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a compact directed molecular graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic IDs, positions, and directed edge index.
    """

    z = torch.tensor([6, 1, 1, 8, 7])
    pos = torch.randn(5, 3)
    edges = [(i, j) for i in range(5) for j in range(5) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return z, pos, edge_index


def build_fairchem_EScAIPBackbone_R6() -> nn.Module:
    """Build compact FairChem EScAIPBackbone R6.

    Returns
    -------
    nn.Module
        Random-init compact EScAIP backbone.
    """

    return CompactEScAIPBackbone(layers=2)


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact EScAIP inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atomic graph inputs.
    """

    return example_graph()


build = build_fairchem_EScAIPBackbone_R6

MENAGERIE_ENTRIES = [
    (
        "fairchem_EScAIPBackbone_R6",
        "build_fairchem_EScAIPBackbone_R6",
        "example_input",
        "2024",
        "E6",
    ),
]
