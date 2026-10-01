"""M3GNet compact materials graph network.

Chen and Ong, 2022, "A Universal Graph Deep Learning Interatomic Potential for
the Periodic Table".  M3GNet augments atomistic graph message passing with
three-body interactions from coordinates/lattice-aware geometry.  This compact
classic traces atom embeddings, radial edge messages, angular triplet messages,
and per-atom energy readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class M3GNet(nn.Module):
    """Compact invariant M3GNet-style potential."""

    def __init__(self, species: int = 16, hidden: int = 32) -> None:
        """Initialize embeddings and message functions.

        Parameters
        ----------
        species:
            Number of atomic species ids.
        hidden:
            Hidden width.
        """
        super().__init__()
        self.embed = nn.Embedding(species, hidden)
        self.radial = nn.Linear(6, hidden)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.energy = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, atom_types: Tensor, positions: Tensor, edge_index: Tensor) -> Tensor:
        """Predict total energy from atoms and coordinates.

        Parameters
        ----------
        atom_types:
            Atomic numbers/species ids with shape ``(atoms,)``.
        positions:
            Cartesian positions with shape ``(atoms, 3)``.
        edge_index:
            Directed edges with shape ``(2, edges)``.

        Returns
        -------
        Tensor
            Scalar total energy.
        """
        src, dst = edge_index
        h = self.embed(atom_types)
        vec = positions[dst] - positions[src]
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        basis = torch.cat(
            (dist, torch.sin(dist), torch.cos(dist), dist.square(), 1.0 / dist, vec[:, :1]), dim=-1
        )
        edge = self.radial(basis)
        msg = self.edge_mlp(torch.cat((h[src], h[dst], edge), dim=-1))
        agg = torch.zeros_like(h).index_add(0, dst, msg)
        h = h + agg
        triplet = torch.zeros_like(h)
        for center in range(atom_types.shape[0]):
            mask = dst == center
            incoming = torch.nonzero(mask, as_tuple=False).flatten()
            if incoming.numel() > 1:
                v = F.normalize(vec[incoming], dim=-1)
                cosang = torch.matmul(v, v.t()).mean().view(1, 1)
                pooled = edge[incoming].mean(dim=0, keepdim=True)
                triplet[center : center + 1] = self.triplet_mlp(torch.cat((pooled, cosang), dim=-1))
        h = h + triplet
        return self.energy(h).sum(dim=0)


def build() -> nn.Module:
    """Build compact M3GNet.

    Returns
    -------
    nn.Module
        Random-initialized M3GNet reconstruction.
    """
    return M3GNet().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small directed atom graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Atomic species, positions, and edge indices.
    """
    atom_types = torch.tensor([6, 8, 1, 1, 14, 3], dtype=torch.long)
    positions = torch.randn(6, 3)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 1, 2, 4, 0], [1, 0, 0, 1, 5, 4, 4, 5, 2, 3]],
        dtype=torch.long,
    )
    return atom_types, positions, edge_index


MENAGERIE_ENTRIES = [
    ("M3GNet", "build", "example_input", "2022", "DC"),
]
