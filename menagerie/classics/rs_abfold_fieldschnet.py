# FAITHFUL REIMPLEMENTATION from arXiv:2010.14942 (no public code) -- A/B codex
"""FieldSchNet with SchNet, dipole-field, and dipole-dipole updates."""

from __future__ import annotations

import torch
from torch import nn


class RadialMLP(nn.Module):
    """Distance-conditioned radial filter."""

    def __init__(self, out_dim: int, hidden_dim: int = 16) -> None:
        """Initialize the radial filter.

        Parameters
        ----------
        out_dim:
            Output channel count.
        hidden_dim:
            Hidden channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate radial filter values.

        Parameters
        ----------
        distances:
            Pairwise distances.

        Returns
        -------
        torch.Tensor
            Filter values.
        """
        return self.net(distances.unsqueeze(-1))


class FieldSchNetLayer(nn.Module):
    """One FieldSchNet interaction layer."""

    def __init__(self, features: int) -> None:
        """Initialize the interaction layer.

        Parameters
        ----------
        features:
            Scalar and dipole feature channels.
        """
        super().__init__()
        self.charge_filter = RadialMLP(features)
        self.charge_update = nn.Sequential(
            nn.Linear(features, features), nn.SiLU(), nn.Linear(features, features)
        )
        self.dipole_from_scalar = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )
        self.field_proj = nn.Linear(features, features)
        self.dipole_pair_filter = RadialMLP(features)
        self.dipole_pair_update = nn.Sequential(
            nn.Linear(features, features),
            nn.SiLU(),
            nn.Linear(features, features),
        )

    def forward(
        self,
        scalars: torch.Tensor,
        dipoles: torch.Tensor,
        positions: torch.Tensor,
        external_field: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply SchNet, dipole-field, and dipole-dipole refinements.

        Parameters
        ----------
        scalars:
            Scalar atom features ``(batch, atoms, features)``.
        dipoles:
            Dipole features ``(batch, atoms, features, 3)``.
        positions:
            Atomic coordinates ``(batch, atoms, 3)``.
        external_field:
            Field vectors at atoms ``(batch, atoms, 3)``.
        mask:
            Pair mask excluding self interactions.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated scalar and dipole features.
        """
        rij = positions[:, :, None, :] - positions[:, None, :, :]
        distances = rij.norm(dim=-1).clamp_min(1e-4)
        cutoff = torch.exp(-distances) * mask
        charge_filter = self.charge_filter(distances) * cutoff.unsqueeze(-1)
        charge_messages = (scalars[:, None, :, :] * charge_filter).sum(dim=2)
        w_update = self.charge_update(charge_messages)

        dipole_weights = self.dipole_from_scalar(scalars)
        dipole_messages = (
            dipole_weights[:, None, :, :, None]
            * rij[:, :, :, None, :]
            * cutoff[:, :, :, None, None]
        ).sum(dim=2)
        new_dipoles = dipoles + dipole_messages

        field_term = (
            self.field_proj(new_dipoles.transpose(-1, -2)).transpose(-1, -2)
            * external_field[:, :, None, :]
        )
        u_update = field_term.sum(dim=-1)

        unit = rij / distances.unsqueeze(-1)
        identity = torch.eye(3, device=positions.device, dtype=positions.dtype)
        tensor = (3.0 * unit[..., :, None] * unit[..., None, :] - identity) / distances[
            ..., None, None
        ] ** 3
        pair_filter = self.dipole_pair_filter(distances) * cutoff.unsqueeze(-1)
        left = torch.einsum("bnfc,bnmcq->bnmfq", new_dipoles, tensor)
        pair_energy = (left * new_dipoles[:, None, :, :, :]).sum(dim=-1) * pair_filter
        v_update = self.dipole_pair_update(pair_energy.sum(dim=2))
        return scalars + w_update + u_update + v_update, new_dipoles


class FieldSchNet(nn.Module):
    """FieldSchNet energy model for atoms in an external vector field."""

    def __init__(
        self, max_atomic_number: int = 18, features: int = 16, interactions: int = 3
    ) -> None:
        """Initialize FieldSchNet.

        Parameters
        ----------
        max_atomic_number:
            Maximum supported atomic number.
        features:
            Feature channels.
        interactions:
            Number of iterative interaction layers.
        """
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_number + 1, features)
        self.layers = nn.ModuleList([FieldSchNetLayer(features) for _ in range(interactions)])
        self.energy = nn.Sequential(
            nn.Linear(features, features), nn.SiLU(), nn.Linear(features, 1)
        )

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Predict molecular energy from atoms, coordinates, and local fields.

        Parameters
        ----------
        atoms:
            Tensor ``(batch, atoms, 7)`` with ``Z, x, y, z, field_x, field_y, field_z``.

        Returns
        -------
        torch.Tensor
            Total energy per molecule.
        """
        atomic_numbers = atoms[..., 0].long().clamp(min=0, max=18)
        positions = atoms[..., 1:4]
        fields = atoms[..., 4:7]
        scalars = self.embedding(atomic_numbers)
        dipoles = torch.zeros(*scalars.shape, 3, device=atoms.device, dtype=atoms.dtype)
        atom_count = atoms.shape[1]
        eye = torch.eye(atom_count, device=atoms.device, dtype=atoms.dtype)
        mask = 1.0 - eye
        for layer in self.layers:
            scalars, dipoles = layer(scalars, dipoles, positions, fields, mask)
        return self.energy(scalars).sum(dim=1).squeeze(-1)


def build_fieldschnet() -> FieldSchNet:
    """Build a tiny traceable FieldSchNet.

    Returns
    -------
    FieldSchNet
        Tiny FieldSchNet model.
    """
    return FieldSchNet()


def example_input_fieldschnet() -> torch.Tensor:
    """Create atoms with coordinates and external fields.

    Returns
    -------
    torch.Tensor
        Example atom tensor.
    """
    return torch.tensor(
        [
            [
                [6.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0],
                [8.0, 0.8, 0.2, 0.0, 0.0, 0.1, 0.0],
                [1.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.1],
            ]
        ]
    )


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("FieldSchNet", "build_fieldschnet", "example_input_fieldschnet", 2021, "REIMPL")
]
