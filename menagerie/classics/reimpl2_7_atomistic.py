"""Compact SchNetPack-family atomistic reimplementations for shard 7.

Sources checked: SchNet continuous-filter convolutions, PaiNN equivariant
message passing, SchNetPack SO(3) equivariant layers, and SpookyNet electronic
state plus nonlocal transformer architecture.  These are dependency-free
random-init PyTorch reconstructions for TorchLens rendering.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RadialBasis(nn.Module):
    """Gaussian radial basis expansion for interatomic distances."""

    def __init__(self, n_rbf: int = 8, cutoff: float = 5.0) -> None:
        """Initialize RBF centers.

        Parameters
        ----------
        n_rbf:
            Number of radial basis functions.
        cutoff:
            Maximum represented distance.
        """
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = 10.0 / cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Expand distances into Gaussian radial features.

        Parameters
        ----------
        distances:
            Pairwise distances.

        Returns
        -------
        Tensor
            Radial basis features.
        """
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers).square())


class SchNetInteraction(nn.Module):
    """SchNet continuous-filter convolution interaction block."""

    def __init__(self, channels: int = 32, n_rbf: int = 8) -> None:
        """Initialize continuous-filter convolution.

        Parameters
        ----------
        channels:
            Atom embedding width.
        n_rbf:
            Radial basis count.
        """
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.in_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def forward(self, h: Tensor, rbf: Tensor) -> Tensor:
        """Apply pairwise continuous-filter message passing.

        Parameters
        ----------
        h:
            Atom features.
        rbf:
            Pairwise radial basis features.

        Returns
        -------
        Tensor
            Updated atom features.
        """
        filt = self.filter_net(rbf)
        msg = filt * self.in_proj(h)[:, None, :, :]
        return h + self.out_proj(msg.sum(dim=2))


class SchNetModel(nn.Module):
    """Compact SchNet atomistic network."""

    def __init__(self, channels: int = 32, layers: int = 2) -> None:
        """Initialize compact SchNet.

        Parameters
        ----------
        channels:
            Atom embedding width.
        layers:
            Number of interaction blocks.
        """
        super().__init__()
        self.embedding = nn.Embedding(32, channels)
        self.rbf = RadialBasis()
        self.layers = nn.ModuleList([SchNetInteraction(channels) for _ in range(layers)])
        self.energy = nn.Linear(channels, 1)

    def forward(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Predict atomwise energy from atoms and coordinates.

        Parameters
        ----------
        atomic_numbers:
            Atomic numbers.
        positions:
            Cartesian coordinates.

        Returns
        -------
        Tensor
            Molecular energy.
        """
        h = self.embedding(atomic_numbers)
        dist = torch.cdist(positions, positions)
        rbf = self.rbf(dist)
        for layer in self.layers:
            h = layer(h, rbf)
        return self.energy(h).sum(dim=1)


class FieldSchNetModel(SchNetModel):
    """SchNet variant with explicit molecule-field interaction path."""

    def __init__(self, channels: int = 40, layers: int = 2) -> None:
        """Initialize field-conditioned SchNet.

        Parameters
        ----------
        channels:
            Atom embedding width.
        layers:
            Number of interaction blocks.
        """

        super().__init__(channels=channels, layers=layers)
        self.field_proj = nn.Linear(3, channels)
        self.dipole_gate = nn.Linear(channels, channels)

    def forward(self, atomic_numbers: Tensor, positions: Tensor, field: Tensor) -> Tensor:
        """Predict energy with external-field and dipole coupling.

        Parameters
        ----------
        atomic_numbers:
            Atomic numbers.
        positions:
            Cartesian coordinates.
        field:
            External electric field vector.

        Returns
        -------
        Tensor
            Field-conditioned molecular energy.
        """

        h = self.embedding(atomic_numbers)
        dist = torch.cdist(positions, positions)
        rbf = self.rbf(dist)
        field_feat = self.field_proj(field).unsqueeze(1)
        dipole = (positions * field.unsqueeze(1)).sum(dim=-1, keepdim=True)
        h = h + torch.tanh(self.dipole_gate(field_feat)) * dipole
        for layer in self.layers:
            h = layer(h, rbf) + field_feat
        return self.energy(h).sum(dim=1)


class PaiNNLayer(nn.Module):
    """PaiNN scalar/vector equivariant message-passing layer."""

    def __init__(self, channels: int = 24, n_rbf: int = 8) -> None:
        """Initialize PaiNN update.

        Parameters
        ----------
        channels:
            Scalar/vector channel count.
        n_rbf:
            Radial basis count.
        """
        super().__init__()
        self.filter = nn.Sequential(
            nn.Linear(n_rbf, channels * 3), nn.SiLU(), nn.Linear(channels * 3, channels * 3)
        )
        self.scalar_update = nn.Linear(channels, channels)
        self.vector_gate = nn.Linear(channels, channels)
        self.mix = nn.Linear(channels * 2, channels)

    def forward(self, s: Tensor, v: Tensor, rel: Tensor, rbf: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one PaiNN interaction/update step.

        Parameters
        ----------
        s:
            Scalar atom features.
        v:
            Vector atom features ``(batch, atoms, channels, 3)``.
        rel:
            Pairwise relative vectors.
        rbf:
            Pairwise radial basis features.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated scalar and vector features.
        """
        f_scalar, f_vec, f_gate = self.filter(rbf).chunk(3, dim=-1)
        unit = rel / rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scalar_msg = (f_scalar * self.scalar_update(s)[:, None, :, :]).sum(dim=2)
        vector_msg = torch.einsum("bijc,bijd->bicd", f_vec, unit)
        gate = torch.sigmoid(self.vector_gate(s)).unsqueeze(-1)
        v = v + vector_msg * gate
        v_norm = v.square().sum(dim=-1).sqrt()
        s = s + self.mix(torch.cat((scalar_msg, v_norm), dim=-1))
        return s, v


class PaiNNModel(nn.Module):
    """Compact PaiNN polarizable atom interaction network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize compact PaiNN.

        Parameters
        ----------
        channels:
            Feature channel count.
        """
        super().__init__()
        self.embedding = nn.Embedding(32, channels)
        self.rbf = RadialBasis()
        self.layers = nn.ModuleList([PaiNNLayer(channels), PaiNNLayer(channels)])
        self.readout = nn.Linear(channels, 1)

    def forward(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Predict energy using scalar/vector equivariant features.

        Parameters
        ----------
        atomic_numbers:
            Atomic numbers.
        positions:
            Cartesian coordinates.

        Returns
        -------
        Tensor
            Molecular energy.
        """
        s = self.embedding(atomic_numbers)
        v = torch.zeros(*s.shape, 3, device=s.device, dtype=s.dtype)
        rel = positions[:, :, None, :] - positions[:, None, :, :]
        rbf = self.rbf(rel.norm(dim=-1))
        for layer in self.layers:
            s, v = layer(s, v, rel, rbf)
        return self.readout(s).sum(dim=1)


class SO3Net(nn.Module):
    """SO3net-style spherical/radial equivariant atomistic network."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize compact SO3net.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """
        super().__init__()
        self.embedding = nn.Embedding(32, channels)
        self.rbf = RadialBasis()
        self.radial = nn.Linear(8, channels)
        self.spherical = nn.Linear(9, channels)
        self.gate = nn.Linear(channels, channels)
        self.readout = nn.Linear(channels, 1)

    def forward(self, atomic_numbers: Tensor, positions: Tensor) -> Tensor:
        """Apply simplified SO(3) radial/spherical message passing.

        Parameters
        ----------
        atomic_numbers:
            Atomic numbers.
        positions:
            Cartesian coordinates.

        Returns
        -------
        Tensor
            Molecular energy.
        """
        h = self.embedding(atomic_numbers)
        rel = positions[:, :, None, :] - positions[:, None, :, :]
        dist = rel.norm(dim=-1).clamp_min(1e-6)
        unit = rel / dist.unsqueeze(-1)
        harmonics = torch.cat(
            (
                unit,
                unit.square(),
                unit[..., 0:1] * unit[..., 1:2],
                unit[..., 1:2] * unit[..., 2:3],
                unit[..., 0:1] * unit[..., 2:3],
            ),
            dim=-1,
        )
        filt = self.radial(self.rbf(dist)) * self.spherical(harmonics)
        msg = (filt * h[:, None, :, :]).sum(dim=2)
        h = h + torch.tanh(self.gate(msg))
        return self.readout(h).sum(dim=1)


class SpookyNet(nn.Module):
    """SpookyNet-style electronic-state-aware nonlocal atomistic network."""

    def __init__(self, channels: int = 32, heads: int = 4) -> None:
        """Initialize compact SpookyNet.

        Parameters
        ----------
        channels:
            Hidden channel count.
        heads:
            Transformer attention heads.
        """
        super().__init__()
        self.embedding = nn.Embedding(32, channels)
        self.charge_spin = nn.Linear(2, channels)
        self.local = SchNetInteraction(channels)
        self.rbf = RadialBasis()
        self.nonlocal_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.energy = nn.Linear(channels, 1)
        self.charge_head = nn.Linear(channels, 1)

    def forward(
        self, atomic_numbers: Tensor, positions: Tensor, electronic_state: Tensor
    ) -> Tensor:
        """Predict energy with electronic degrees of freedom and corrections.

        Parameters
        ----------
        atomic_numbers:
            Atomic numbers.
        positions:
            Cartesian coordinates.
        electronic_state:
            Total charge and spin features.

        Returns
        -------
        Tensor
            Energy plus charge-equilibration style correction.
        """
        h = self.embedding(atomic_numbers) + self.charge_spin(electronic_state).unsqueeze(1)
        dist = torch.cdist(positions, positions).clamp_min(1e-6)
        h = self.local(h, self.rbf(dist))
        h = h + self.nonlocal_attn(h, h, h, need_weights=False)[0]
        atom_energy = self.energy(h).sum(dim=1)
        partial_charge = self.charge_head(h).squeeze(-1)
        coulomb = (partial_charge[:, :, None] * partial_charge[:, None, :] / dist).mean(
            dim=(1, 2), keepdim=True
        )
        nuclear = atomic_numbers.float().sum(dim=1, keepdim=True) * 0.01
        return atom_energy + coulomb + nuclear


def build_schnet() -> nn.Module:
    """Build compact SchNet."""
    return SchNetModel()


def build_painn() -> nn.Module:
    """Build compact PaiNN."""
    return PaiNNModel()


def build_field_schnet() -> nn.Module:
    """Build compact field-aware SchNetPack model."""
    return FieldSchNetModel(channels=40, layers=2)


def build_so3net() -> nn.Module:
    """Build compact SO3net."""
    return SO3Net()


def build_spookynet() -> nn.Module:
    """Build compact SpookyNet."""
    return SpookyNet()


def example_molecule() -> tuple[Tensor, Tensor]:
    """Return atomic numbers and positions."""
    z = torch.tensor([[6, 1, 1, 8, 1, 7]], dtype=torch.long)
    r = torch.randn(1, 6, 3)
    return z, r


def example_field_molecule() -> tuple[Tensor, Tensor, Tensor]:
    """Return atomic numbers, positions, and an external field vector."""

    z, r = example_molecule()
    return z, r, torch.tensor([[0.2, -0.1, 0.05]])


def example_spookynet() -> tuple[Tensor, Tensor, Tensor]:
    """Return atoms, positions, and charge/spin state."""
    z, r = example_molecule()
    return z, r, torch.tensor([[0.0, 1.0]])


MENAGERIE_ENTRIES = [
    ("SchNet (SchNetPack)", "build_schnet", "example_molecule", "2017", "DC"),
    ("SchNetPack PaiNN", "build_painn", "example_molecule", "2021", "DC"),
    ("FieldSchNet (SchNetPack)", "build_field_schnet", "example_field_molecule", "2019", "DC"),
    ("SO3net (SchNetPack)", "build_so3net", "example_molecule", "2022", "DC"),
    ("SpookyNet", "build_spookynet", "example_spookynet", "2021", "DC"),
]
