"""LEFTNet compact equivariant graph neural network.

Paper: Du et al., 2023, "A New Perspective on Building Efficient and
Expressive 3D Equivariant Graph Neural Networks".

LEFTNet combines local substructure encoding with frame transition encoding.
This compact reconstruction builds edge-wise local frames from relative
coordinates, transports vector features through frame transitions, and updates
node scalars/vectors equivariantly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LEFTLayer(nn.Module):
    """Local-frame equivariant message passing layer."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize scalar and vector message functions."""

        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scalar = nn.Linear(dim, dim)
        self.vector_gate = nn.Linear(dim, 1)

    def forward(
        self, h: Tensor, v: Tensor, pos: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update scalar and vector node features using frame transitions."""

        src, dst = edge_index
        rel = pos[dst] - pos[src]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        e1 = rel / dist
        seed = torch.roll(e1, shifts=1, dims=0)
        e2 = F.normalize(seed - (seed * e1).sum(-1, keepdim=True) * e1, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        frame = torch.stack([e1, e2, e3], dim=-1)
        local_vec = torch.einsum("edk,ed->ek", frame, v[src])
        msg = self.edge_mlp(torch.cat([h[src], h[dst], dist], dim=-1))
        transported = torch.einsum("edk,ek->ed", frame, local_vec * self.vector_gate(msg))
        agg_h = torch.zeros_like(h).index_add(0, dst, msg)
        agg_v = torch.zeros_like(v).index_add(0, dst, transported)
        return h + self.scalar(agg_h), v + agg_v


class LEFTNet(nn.Module):
    """Compact LEFTNet molecular graph model."""

    def __init__(self, atom_types: int = 16, dim: int = 32) -> None:
        """Initialize atom embedding and local-frame layers."""

        super().__init__()
        self.embed = nn.Embedding(atom_types, dim)
        self.layers = nn.ModuleList([LEFTLayer(dim), LEFTLayer(dim)])
        self.readout = nn.Sequential(nn.Linear(dim + 1, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, atoms: Tensor, pos: Tensor, edge_index: Tensor) -> Tensor:
        """Predict a graph property from atoms and 3D positions."""

        h = self.embed(atoms)
        v = torch.zeros(pos.shape[0], 3, device=pos.device)
        for layer in self.layers:
            h, v = layer(h, v, pos, edge_index)
        graph = torch.cat([h.mean(dim=0), v.norm(dim=-1).mean().unsqueeze(0)], dim=0)
        return self.readout(graph)


def build() -> nn.Module:
    """Build compact LEFTNet."""

    return LEFTNet().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return atom ids, coordinates, and directed edges."""

    atoms = torch.tensor([6, 1, 1, 8, 7], dtype=torch.long)
    pos = torch.randn(5, 3)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3, 4, 1, 3], [1, 2, 0, 0, 4, 3, 3, 1]],
        dtype=torch.long,
    )
    return atoms, pos, edge_index


MENAGERIE_ENTRIES = [
    ("leftnet", "build", "example_input", "2023", "GRAPH"),
]
