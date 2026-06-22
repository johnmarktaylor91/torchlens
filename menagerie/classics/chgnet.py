"""CHGNet compact Crystal Hamiltonian Graph Neural Network.

Deng et al., "CHGNet as a pretrained universal neural network potential for
charge-informed atomistic modelling", Nature Machine Intelligence 2023.
CHGNet uses atom, bond, and angle graphs with radial/angular basis expansions,
message passing, weighted aggregation, and auxiliary magnetic-moment/charge-aware
regularization.  This compact version keeps the atom-bond-angle update and
multi-head energy/magmoms outputs without the dependency-heavy potential package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAngleBlock(nn.Module):
    """CHGNet-style atom-bond update with angle-conditioned bond messages."""

    def __init__(self, atom_dim: int, bond_dim: int, angle_dim: int) -> None:
        """Initialize angle, bond, and atom update projections.

        Parameters
        ----------
        atom_dim:
            Atom embedding width.
        bond_dim:
            Bond embedding width.
        angle_dim:
            Angle basis width.
        """

        super().__init__()
        self.angle_mlp = nn.Sequential(nn.Linear(bond_dim * 2 + angle_dim, bond_dim), nn.SiLU())
        self.bond_mlp = nn.Sequential(nn.Linear(atom_dim * 2 + bond_dim, bond_dim), nn.SiLU())
        self.atom_mlp = nn.Sequential(nn.Linear(atom_dim + bond_dim, atom_dim), nn.SiLU())
        self.atom_norm = nn.LayerNorm(atom_dim)
        self.bond_norm = nn.LayerNorm(bond_dim)

    def forward(
        self,
        atom: torch.Tensor,
        bond: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        angle_src: torch.Tensor,
        angle_dst: torch.Tensor,
        angle_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one atom-bond-angle interaction update.

        Parameters
        ----------
        atom:
            Atom features ``(B, N, A)``.
        bond:
            Directed bond features ``(B, E, Bdim)``.
        src:
            Source atom index per directed bond.
        dst:
            Destination atom index per directed bond.
        angle_src:
            First bond index in an angle triplet.
        angle_dst:
            Second bond index in an angle triplet.
        angle_basis:
            Angle basis features ``(B, M, G)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated atom and bond features.
        """

        angle_msg = self.angle_mlp(
            torch.cat([bond[:, angle_src], bond[:, angle_dst], angle_basis], dim=-1)
        )
        bond_delta = torch.zeros_like(bond).index_add(1, angle_dst, angle_msg)
        src_atom = atom[:, src]
        dst_atom = atom[:, dst]
        bond_msg = self.bond_mlp(torch.cat([src_atom, dst_atom, bond + bond_delta], dim=-1))
        bond = self.bond_norm(bond + bond_msg)
        atom_aggr = torch.zeros(atom.shape[0], atom.shape[1], bond.shape[-1], device=atom.device)
        atom_aggr = atom_aggr.index_add(1, dst, bond)
        atom = self.atom_norm(atom + self.atom_mlp(torch.cat([atom, atom_aggr], dim=-1)))
        return atom, bond


class CHGNetCompact(nn.Module):
    """Compact CHGNet potential with atom, bond, and angle graphs."""

    def __init__(self, atom_in: int = 8, bond_in: int = 6, angle_in: int = 4) -> None:
        """Initialize CHGNet embeddings and prediction heads.

        Parameters
        ----------
        atom_in:
            Atom feature width.
        bond_in:
            Bond radial feature width.
        angle_in:
            Angle basis width.
        """

        super().__init__()
        self.atom_embed = nn.Linear(atom_in, 24)
        self.bond_embed = nn.Linear(bond_in, 16)
        self.blocks = nn.ModuleList([EdgeAngleBlock(24, 16, angle_in) for _ in range(2)])
        self.energy_head = nn.Sequential(nn.Linear(24, 24), nn.SiLU(), nn.Linear(24, 1))
        self.magmom_head = nn.Linear(24, 1)

    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        angle_src: torch.Tensor,
        angle_dst: torch.Tensor,
        angle_basis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict compact CHGNet energy and site magnetic moments.

        Parameters
        ----------
        atom_features:
            Atom features ``(B, N, F)``.
        bond_features:
            Directed bond features ``(B, E, R)``.
        src:
            Source atom indices.
        dst:
            Destination atom indices.
        angle_src:
            Incoming bond indices for angle graph.
        angle_dst:
            Outgoing bond indices for angle graph.
        angle_basis:
            Angle basis features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Total energy and per-site magnetic moment proxy.
        """

        atom = F.silu(self.atom_embed(atom_features))
        bond = F.silu(self.bond_embed(bond_features))
        for block in self.blocks:
            atom, bond = block(atom, bond, src, dst, angle_src, angle_dst, angle_basis)
        site_energy = self.energy_head(atom)
        magmom = self.magmom_head(atom)
        return site_energy.sum(dim=1), magmom


class CHGNetTensorInput(nn.Module):
    """Tensor-fronted compact CHGNet with fixed crystal graph indices."""

    def __init__(self) -> None:
        """Initialize the wrapped CHGNet and fixed example crystal buffers."""

        super().__init__()
        self.model = CHGNetCompact()
        inputs = example_graph_input()
        self.register_buffer("atom_features", inputs[0])
        self.register_buffer("bond_features", inputs[1])
        self.register_buffer("src", inputs[2])
        self.register_buffer("dst", inputs[3])
        self.register_buffer("angle_src", inputs[4])
        self.register_buffer("angle_dst", inputs[5])
        self.register_buffer("angle_basis", inputs[6])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict energy and magnetic moments from a simple tensor input.

        Parameters
        ----------
        x:
            Small conditioning tensor. Its mean shifts atom, bond, and angle
            features while the fixed crystal graph indices remain buffers.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Total energy and per-site magnetic moment proxy.
        """

        scale = x.mean().to(dtype=self.atom_features.dtype)
        return self.model(
            self.atom_features + scale,
            self.bond_features + scale,
            self.src,
            self.dst,
            self.angle_src,
            self.angle_dst,
            self.angle_basis + scale,
        )


def build() -> nn.Module:
    """Build a compact random-init CHGNet model.

    Returns
    -------
    nn.Module
        CHGNet-style potential.
    """

    return CHGNetTensorInput()


def build_graph_tuple() -> nn.Module:
    """Build the raw graph-tuple compact CHGNet model.

    Returns
    -------
    nn.Module
        Tuple-input CHGNetCompact.
    """

    return CHGNetCompact()


def example_input() -> torch.Tensor:
    """Create a simple tensor input.

    Returns
    -------
    torch.Tensor
        Conditioning tensor.
    """

    return torch.randn(1, 4)


def example_graph_input() -> tuple[torch.Tensor, ...]:
    """Create a small directed crystal graph with angle triplets.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Atom features, bond features, graph indices, and angle bases.
    """

    src = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 0, 3])
    angle_src = torch.tensor([0, 2, 4, 6])
    angle_dst = torch.tensor([2, 4, 6, 0])
    return (
        torch.randn(1, 4, 8),
        torch.randn(1, 8, 6),
        src,
        dst,
        angle_src,
        angle_dst,
        torch.randn(1, 4, 4),
    )


MENAGERIE_ENTRIES = [
    (
        "CHGNet (atom-bond-angle charge-informed crystal Hamiltonian GNN)",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "chgnet (atom-bond-angle charge-informed crystal Hamiltonian GNN)",
        "build",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "chgnet graph-tuple reference",
        "build_graph_tuple",
        "example_graph_input",
        "2023",
        "DC",
    ),
]
