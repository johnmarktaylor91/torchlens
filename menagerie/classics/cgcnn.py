"""Crystal Graph Convolutional Neural Network (CGCNN) compact classic.

Xie and Grossman, "Crystal Graph Convolutional Neural Networks for an Accurate
and Interpretable Prediction of Material Properties", Physical Review Letters
2018.  CGCNN represents a crystal as an atom graph with neighbor bond features;
each convolution computes a gated message from central atom, neighbor atom, and
bond features, then pools atom embeddings to a crystal-level property.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CGCNNConv(nn.Module):
    """CGCNN gated crystal graph convolution."""

    def __init__(self, atom_dim: int, bond_dim: int) -> None:
        """Initialize the filter/core message projection.

        Parameters
        ----------
        atom_dim:
            Atom embedding width.
        bond_dim:
            Bond feature width.
        """

        super().__init__()
        self.message = nn.Linear(atom_dim * 2 + bond_dim, atom_dim * 2)
        self.norm = nn.LayerNorm(atom_dim)

    def forward(
        self, atom: torch.Tensor, nbr_idx: torch.Tensor, bond: torch.Tensor
    ) -> torch.Tensor:
        """Update atoms from neighbor atoms and bond features.

        Parameters
        ----------
        atom:
            Atom features ``(B, N, D)``.
        nbr_idx:
            Neighbor indices ``(B, N, K)``.
        bond:
            Bond features ``(B, N, K, E)``.

        Returns
        -------
        torch.Tensor
            Updated atom features.
        """

        bsz, atoms, degree = nbr_idx.shape
        batch = torch.arange(bsz, device=atom.device).view(bsz, 1, 1).expand(-1, atoms, degree)
        nbr = atom[batch, nbr_idx]
        center = atom.unsqueeze(2).expand_as(nbr)
        gate_raw, core_raw = self.message(torch.cat([center, nbr, bond], dim=-1)).chunk(2, dim=-1)
        msg = torch.sigmoid(gate_raw) * F.softplus(core_raw)
        return self.norm(atom + msg.sum(dim=2))


class CGCNNCompact(nn.Module):
    """Small CGCNN for random-init crystal property prediction."""

    def __init__(self, atom_in: int = 8, bond_in: int = 6, hidden: int = 24) -> None:
        """Initialize atom embedding, gated convolutions, and pooling head.

        Parameters
        ----------
        atom_in:
            Raw atom feature width.
        bond_in:
            Raw bond feature width.
        hidden:
            Hidden atom embedding width.
        """

        super().__init__()
        self.atom_embed = nn.Linear(atom_in, hidden)
        self.bond_embed = nn.Linear(bond_in, hidden // 2)
        self.convs = nn.ModuleList([CGCNNConv(hidden, hidden // 2) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, 1))

    def forward(
        self, atom_features: torch.Tensor, nbr_idx: torch.Tensor, bond_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict a crystal-level scalar from atom and bond graph features.

        Parameters
        ----------
        atom_features:
            Atom features ``(B, N, F)``.
        nbr_idx:
            Neighbor index tensor ``(B, N, K)``.
        bond_features:
            Bond features ``(B, N, K, E)``.

        Returns
        -------
        torch.Tensor
            Crystal property prediction ``(B, 1)``.
        """

        atom = F.softplus(self.atom_embed(atom_features))
        bond = F.softplus(self.bond_embed(bond_features))
        for conv in self.convs:
            atom = conv(atom, nbr_idx, bond)
        return self.head(atom.mean(dim=1))


def build() -> nn.Module:
    """Build a compact CGCNN model.

    Returns
    -------
    nn.Module
        CGCNN property predictor.
    """

    return CGCNNCompact()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a toy periodic-neighbor crystal graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Atom features, neighbor indices, and bond features.
    """

    nbr_idx = torch.tensor([[[1, 2, 3], [0, 2, 4], [0, 1, 5], [0, 4, 5], [1, 3, 5], [2, 3, 4]]])
    return torch.randn(1, 6, 8), nbr_idx, torch.randn(1, 6, 3, 6)


MENAGERIE_ENTRIES = [
    (
        "CGCNN (gated atom-bond crystal graph convolution)",
        "build",
        "example_input",
        "2018",
        "DC",
    ),
]
