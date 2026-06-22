"""SevenNet compact faithful reconstruction.

SevenNet is a Scalable EquiVariance-Enabled Neural Network interatomic
potential package whose core model is based on NequIP. The compact model here
uses atom embeddings, radial edge bases, scalar/vector message passing, gated
equivariant updates, and per-atom energy readout for random-init tracing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SevenNetLayer(nn.Module):
    """Scalar-vector equivariant message-passing layer."""

    def __init__(self, hidden: int, radial: int) -> None:
        """Initialize message and update projections.

        Parameters
        ----------
        hidden:
            Hidden scalar/vector channel count.
        radial:
            Radial basis dimension.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(radial, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 3),
        )
        self.scalar_update = nn.Linear(hidden * 2, hidden)
        self.vector_gate = nn.Linear(hidden, hidden)

    def forward(
        self, scalar: Tensor, vector: Tensor, pos: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply one equivariant message-passing update.

        Parameters
        ----------
        scalar:
            Scalar atom features with shape ``(atoms, hidden)``.
        vector:
            Vector atom features with shape ``(atoms, hidden, 3)``.
        pos:
            Atomic positions with shape ``(atoms, 3)``.
        edge_index:
            Directed edge indices with shape ``(2, edges)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated scalar and vector features.
        """
        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = rel / dist
        centers = torch.linspace(0.0, 2.5, 8, device=pos.device)
        radial = torch.exp(-((dist - centers) ** 2) * 2.0)
        gate_s, gate_v, gate_mix = self.edge_mlp(radial).chunk(3, dim=-1)
        scalar_msg = scalar[src] * gate_s
        vector_msg = vector[src] * gate_v.unsqueeze(-1) + gate_mix.unsqueeze(-1) * unit.unsqueeze(1)
        agg_s = torch.zeros_like(scalar).index_add(0, dst, scalar_msg)
        agg_v = torch.zeros_like(vector).index_add(0, dst, vector_msg)
        scalar = scalar + torch.tanh(self.scalar_update(torch.cat((scalar, agg_s), dim=-1)))
        vector = vector + torch.tanh(self.vector_gate(scalar)).unsqueeze(-1) * agg_v
        return scalar, vector


class SevenNetCompact(nn.Module):
    """Compact SevenNet/NequIP-style interatomic potential."""

    def __init__(self, num_species: int = 8, hidden: int = 24, layers: int = 2) -> None:
        """Initialize compact SevenNet layers.

        Parameters
        ----------
        num_species:
            Number of atomic species.
        hidden:
            Hidden scalar/vector channels.
        layers:
            Number of message-passing layers.
        """
        super().__init__()
        self.embed = nn.Embedding(num_species, hidden)
        self.layers = nn.ModuleList([SevenNetLayer(hidden, radial=8) for _ in range(layers)])
        self.energy = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, atomic_numbers: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        """Predict total energy from atoms and neighbor edges.

        Parameters
        ----------
        atomic_numbers:
            Atomic species ids.
        pos:
            Atomic positions.
        edge_index:
            Directed neighbor graph.

        Returns
        -------
        Tensor
            Single total-energy prediction.
        """
        scalar = self.embed(atomic_numbers)
        vector = pos.new_zeros(pos.shape[0], scalar.shape[-1], 3)
        for layer in self.layers:
            scalar, vector = layer(scalar, vector, pos, edge_index)
        invariant = vector.pow(2).sum(dim=-1).sqrt()
        per_atom = self.energy(torch.cat((scalar, invariant), dim=-1))
        return per_atom.sum(dim=0)


def build() -> nn.Module:
    """Build a compact random-init SevenNet model.

    Returns
    -------
    nn.Module
        Compact SevenNet reconstruction.
    """
    return SevenNetCompact()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small molecular graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Atomic species, positions, and directed edge indices.
    """
    atomic_numbers = torch.tensor([0, 1, 2, 0, 0], dtype=torch.long)
    pos = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 0], [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]],
        dtype=torch.long,
    )
    return atomic_numbers, pos, edge_index


MENAGERIE_ENTRIES = [
    ("SevenNet / SevenNet-Omni", "build", "example_input", "2024", "E7"),
    ("sevennet_0", "build", "example_input", "2024", "E7"),
]
