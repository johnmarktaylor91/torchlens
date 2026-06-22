"""Compact SchNetPack-style atomistic neural networks.

SchNet and DTNN model molecules with atom embeddings, pairwise distances, and
continuous-filter/tensor interaction passes. PaiNN adds rotationally equivariant
scalar-vector message passing. FieldSchNet injects an external field into atomic
representations. SO3Net uses directional spherical bases on interatomic vectors.
All variants here are random-init compact PyTorch reconstructions for TorchLens
rendering in the base environment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise displacement vectors and distances.

    Parameters
    ----------
    pos:
        Atomic positions shaped ``(batch, atoms, 3)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Displacements and distances.
    """

    disp = pos[:, :, None, :] - pos[:, None, :, :]
    dist = torch.linalg.vector_norm(disp + 1e-6, dim=-1)
    return disp, dist


class GaussianRBF(nn.Module):
    """Gaussian radial basis expansion."""

    def __init__(self, n_rbf: int = 8, cutoff: float = 4.0) -> None:
        """Initialize fixed centers and width.

        Parameters
        ----------
        n_rbf:
            Number of radial basis functions.
        cutoff:
            Distance cutoff.
        """

        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances and apply smooth cosine cutoff.

        Parameters
        ----------
        dist:
            Pairwise distances.

        Returns
        -------
        torch.Tensor
            Radial basis features.
        """

        rbf = torch.exp(-self.gamma.abs() * (dist[..., None] - self.centers).pow(2))
        cut = 0.5 * (torch.cos(torch.clamp(dist, max=self.cutoff) * torch.pi / self.cutoff) + 1.0)
        return rbf * cut[..., None]


class ContinuousFilterInteraction(nn.Module):
    """SchNet continuous-filter convolution interaction."""

    def __init__(self, hidden: int = 16, n_rbf: int = 8) -> None:
        """Initialize filter-generating network.

        Parameters
        ----------
        hidden:
            Atomic feature width.
        n_rbf:
            Number of radial basis functions.
        """

        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.dense = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor) -> torch.Tensor:
        """Apply continuous-filter message passing.

        Parameters
        ----------
        h:
            Atomic scalar features.
        rbf:
            Pairwise radial basis features.

        Returns
        -------
        torch.Tensor
            Updated scalar features.
        """

        filt = self.filter_net(rbf)
        msg = filt * self.dense(h)[:, None, :, :]
        return h + self.out(msg.sum(dim=2))


class SchNetLike(nn.Module):
    """SchNet/DTNN-style atomistic energy model."""

    def __init__(self, tensor_gate: bool = False, field: bool = False, so3: bool = False) -> None:
        """Initialize atomistic model.

        Parameters
        ----------
        tensor_gate:
            Use DTNN tensor-product distance gate.
        field:
            Inject external field channels.
        so3:
            Add directional spherical edge basis.
        """

        super().__init__()
        self.tensor_gate = tensor_gate
        self.field = field
        self.so3 = so3
        self.embed = nn.Embedding(16, 16)
        self.rbf = GaussianRBF()
        self.interactions = nn.ModuleList([ContinuousFilterInteraction() for _ in range(2)])
        self.tensor = nn.Bilinear(16, 8, 16) if tensor_gate else None
        self.field_proj = nn.Linear(3, 16) if field else None
        self.so3_proj = nn.Linear(11, 16) if so3 else None
        self.readout = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict molecular energy from atomic numbers and positions.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, atoms, 7)`` with atomic number, xyz position,
            and optional external field vector.

        Returns
        -------
        torch.Tensor
            Molecular scalar prediction.
        """

        z = x[..., 0].round().long().clamp(0, 15)
        pos = x[..., 1:4]
        field = x[..., 4:7]
        disp, dist = _pairwise(pos)
        rbf = self.rbf(dist)
        h = self.embed(z)
        if self.field_proj is not None:
            h = h + self.field_proj(field)
        if self.so3_proj is not None:
            direction = disp / dist[..., None].clamp_min(1e-4)
            basis = torch.cat([rbf, direction], dim=-1)
            h = h + self.so3_proj(basis).sum(dim=2)
        for interaction in self.interactions:
            if self.tensor is not None:
                gate = self.tensor(h[:, :, None, :].expand(-1, -1, h.shape[1], -1), rbf).sum(dim=2)
                h = h + torch.tanh(gate)
            h = interaction(h, rbf)
        return self.readout(h).sum(dim=1)


class PaiNNLike(nn.Module):
    """PaiNN scalar-vector equivariant message passing model."""

    def __init__(self) -> None:
        """Initialize scalar and vector update networks."""

        super().__init__()
        self.embed = nn.Embedding(16, 16)
        self.rbf = GaussianRBF()
        self.scalar_msg = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 16))
        self.vector_msg = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
        self.mix = nn.Linear(32, 16)
        self.readout = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar property with PaiNN scalar-vector updates.

        Parameters
        ----------
        x:
            Atom tensor shaped ``(batch, atoms, 7)``.

        Returns
        -------
        torch.Tensor
            Molecular scalar prediction.
        """

        z = x[..., 0].round().long().clamp(0, 15)
        pos = x[..., 1:4]
        disp, dist = _pairwise(pos)
        direction = disp / dist[..., None].clamp_min(1e-4)
        rbf = self.rbf(dist)
        s = self.embed(z)
        v = x.new_zeros(x.shape[0], x.shape[1], 3, 16)
        for _ in range(2):
            smsg = self.scalar_msg(rbf) * s[:, None, :, :]
            vmsg = (
                self.vector_msg(rbf).unsqueeze(-1) * direction[..., None] * s[:, None, :, None, :]
            )
            s = s + smsg.sum(dim=2)
            v = v + vmsg.sum(dim=2)
            s = s + self.mix(torch.cat([s, torch.linalg.vector_norm(v, dim=2)], dim=-1))
        return self.readout(s).sum(dim=1)


class NeuralNetworkPotential(nn.Module):
    """SchNetPack-style wrapper around a representation and atomwise head."""

    def __init__(self, representation: nn.Module) -> None:
        """Initialize the potential wrapper.

        Parameters
        ----------
        representation:
            Atomistic representation model.
        """

        super().__init__()
        self.representation = representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the atomistic potential.

        Parameters
        ----------
        x:
            Atom tensor.

        Returns
        -------
        torch.Tensor
            Potential prediction.
        """

        return self.representation(x)


def example_atoms() -> torch.Tensor:
    """Return a tiny molecule tensor.

    Returns
    -------
    torch.Tensor
        Atomic numbers, positions, and field vector.
    """

    z = torch.tensor([[[6.0], [1.0], [8.0], [1.0]]])
    pos = torch.randn(1, 4, 3) * 0.7
    field = torch.randn(1, 4, 3) * 0.1
    return torch.cat([z, pos, field], dim=-1)


def build_gaussian_model() -> nn.Module:
    """Build radial-basis Gaussian atomistic model.

    Returns
    -------
    nn.Module
        Continuous-filter model.
    """

    return SchNetLike()


def build_dtnn() -> nn.Module:
    """Build DTNN tensor-interaction model.

    Returns
    -------
    nn.Module
        DTNN reconstruction.
    """

    return SchNetLike(tensor_gate=True)


def build_schnet() -> nn.Module:
    """Build SchNet continuous-filter model.

    Returns
    -------
    nn.Module
        SchNet reconstruction.
    """

    return SchNetLike()


def build_fieldschnet() -> nn.Module:
    """Build FieldSchNet with external field coupling.

    Returns
    -------
    nn.Module
        FieldSchNet reconstruction.
    """

    return SchNetLike(field=True)


def build_so3net() -> nn.Module:
    """Build SO3Net with directional spherical basis.

    Returns
    -------
    nn.Module
        SO3Net reconstruction.
    """

    return SchNetLike(so3=True)


def build_painn() -> nn.Module:
    """Build PaiNN scalar-vector message passing model.

    Returns
    -------
    nn.Module
        PaiNN reconstruction.
    """

    return PaiNNLike()


def build_neural_network_potential() -> nn.Module:
    """Build SchNetPack NeuralNetworkPotential wrapper.

    Returns
    -------
    nn.Module
        Wrapped atomistic model.
    """

    return NeuralNetworkPotential(SchNetLike())


MENAGERIE_ENTRIES = [
    ("GaussianModel", "build_gaussian_model", "example_atoms", "2017", "atomistic"),
    (
        "schnetpack_NeuralNetworkPotential",
        "build_neural_network_potential",
        "example_atoms",
        "2019",
        "atomistic",
    ),
    ("PaiNN", "build_painn", "example_atoms", "2021", "atomistic/equivariant"),
    ("DTNN", "build_dtnn", "example_atoms", "2017", "atomistic"),
    ("painn", "build_painn", "example_atoms", "2021", "atomistic/equivariant"),
    ("PaiNN (SchNetPack)", "build_painn", "example_atoms", "2021", "atomistic/equivariant"),
    ("schnetpack_PaiNN", "build_painn", "example_atoms", "2021", "atomistic/equivariant"),
    ("schnetpack_PaiNN_R6", "build_painn", "example_atoms", "2021", "atomistic/equivariant"),
    ("schnetpack_SchNet", "build_schnet", "example_atoms", "2017", "atomistic"),
    ("schnetpack_SchNet_R6", "build_schnet", "example_atoms", "2017", "atomistic"),
    ("schnetpack_FieldSchNet", "build_fieldschnet", "example_atoms", "2023", "atomistic/field"),
    ("schnetpack_FieldSchNet_R6", "build_fieldschnet", "example_atoms", "2023", "atomistic/field"),
    ("schnetpack_SO3net", "build_so3net", "example_atoms", "2023", "atomistic/equivariant"),
    ("schnetpack_SO3net_R6", "build_so3net", "example_atoms", "2023", "atomistic/equivariant"),
]
