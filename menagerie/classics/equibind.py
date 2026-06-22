"""EquiBind: SE(3)-equivariant drug binding structure prediction.

Stärk et al., ICML 2022.  EquiBind uses ligand/receptor geometric message
passing, cross-molecule attention, and coordinate updates to directly predict a
bound ligand pose and binding location.  This compact version keeps scalar
feature updates driven by pairwise distances and vector coordinate shifts, so
translation/rotation structure is represented without external dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EquivariantLayer(nn.Module):
    """Distance-gated scalar and coordinate message-passing layer."""

    def __init__(self, dim: int) -> None:
        """Initialize message functions.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.edge = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.coord_gate = nn.Linear(dim, 1)
        self.node = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(
        self, h_src: torch.Tensor, x_src: torch.Tensor, h_ctx: torch.Tensor, x_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update source features and coordinates from context atoms.

        Parameters
        ----------
        h_src:
            Source atom features.
        x_src:
            Source coordinates.
        h_ctx:
            Context atom features.
        x_ctx:
            Context coordinates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated features and coordinates.
        """

        rel = x_src[:, :, None, :] - x_ctx[:, None, :, :]
        dist = rel.pow(2).sum(dim=-1, keepdim=True)
        pair = torch.cat(
            [
                h_src[:, :, None, :].expand(-1, -1, h_ctx.shape[1], -1),
                h_ctx[:, None, :, :].expand(-1, h_src.shape[1], -1, -1),
                dist,
            ],
            dim=-1,
        )
        msg = self.edge(pair)
        attn = torch.softmax(-dist.squeeze(-1), dim=-1).unsqueeze(-1)
        pooled = (attn * msg).sum(dim=2)
        shift = (attn * self.coord_gate(msg) * rel).sum(dim=2) / max(1, x_ctx.shape[1])
        return h_src + self.node(torch.cat([h_src, pooled], dim=-1)), x_src - shift


class CompactEquiBind(nn.Module):
    """Compact ligand/receptor EquiBind-style coordinate updater."""

    def __init__(self, atom_types: int = 16, dim: int = 32) -> None:
        """Initialize compact EquiBind.

        Parameters
        ----------
        atom_types:
            Number of atom-type ids.
        dim:
            Hidden width.
        """

        super().__init__()
        self.embed = nn.Embedding(atom_types, dim)
        self.lig_self = EquivariantLayer(dim)
        self.rec_self = EquivariantLayer(dim)
        self.cross = EquivariantLayer(dim)
        self.pose_head = nn.Linear(dim, 1)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Predict compact ligand pose coordinates.

        Parameters
        ----------
        inputs:
            Tuple ``(ligand_types, ligand_xyz, receptor_types, receptor_xyz)``.

        Returns
        -------
        torch.Tensor
            Updated ligand coordinates and binding center flattened together.
        """

        lig_t, lig_x, rec_t, rec_x = inputs
        lig_h = self.embed(lig_t)
        rec_h = self.embed(rec_t)
        lig_h, lig_x = self.lig_self(lig_h, lig_x, lig_h, lig_x)
        rec_h, rec_x = self.rec_self(rec_h, rec_x, rec_h, rec_x)
        lig_h, lig_x = self.cross(lig_h, lig_x, rec_h, rec_x)
        weights = torch.softmax(self.pose_head(lig_h).squeeze(-1), dim=-1).unsqueeze(-1)
        center = (weights * lig_x).sum(dim=1)
        return torch.cat([lig_x.flatten(1), center], dim=-1)


def build() -> nn.Module:
    """Build compact EquiBind.

    Returns
    -------
    nn.Module
        Random-init EquiBind-style model.
    """

    return CompactEquiBind()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create ligand/receptor atom ids and coordinates.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Atom ids and coordinates.
    """

    return (
        torch.randint(0, 16, (1, 5)),
        torch.randn(1, 5, 3),
        torch.randint(0, 16, (1, 9)),
        torch.randn(1, 9, 3),
    )


MENAGERIE_ENTRIES = [("EquiBind", "build", "example_input", "2022", "GEO")]
