# FAITHFUL REIMPLEMENTATION from arXiv:1912.01398 (no public code) -- A/B codex
"""Tiny Tensor-Embedded Atom Network with scalar, vector, and tensor states."""

from __future__ import annotations

import torch
from torch import nn


class SmoothActivation(nn.Module):
    """Smooth nonlinearity standing in for TeaNet's smooth activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a smooth activation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor.
        """
        return torch.nn.functional.silu(x)


class ChannelLinear(nn.Module):
    """Linear channel mixing for scalar/vector/tensor channel axes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize channel mixing.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix only the final channel axis.

        Parameters
        ----------
        x:
            Tensor with channels in the final dimension.

        Returns
        -------
        torch.Tensor
            Tensor with mixed channels.
        """
        return self.linear(x)


class TeaNetBlock(nn.Module):
    """Local interaction block carrying scalar, vector, tensor, and bond states."""

    def __init__(self, scalar_channels: int, vector_channels: int, tensor_channels: int) -> None:
        """Initialize a TeaNet local interaction block.

        Parameters
        ----------
        scalar_channels:
            Atom and bond scalar channels.
        vector_channels:
            Atom and bond vector channels.
        tensor_channels:
            Atom tensor channels.
        """
        super().__init__()
        self.act = SmoothActivation()
        self.atom_pre = ChannelLinear(scalar_channels + vector_channels, scalar_channels)
        self.atom_scalar = ChannelLinear(scalar_channels, scalar_channels)
        self.atom_vector = ChannelLinear(vector_channels, vector_channels)
        self.atom_tensor = ChannelLinear(tensor_channels, tensor_channels)
        self.bond_pre = ChannelLinear(scalar_channels + vector_channels, scalar_channels)
        self.sym = ChannelLinear(scalar_channels * 2 + vector_channels * 3, scalar_channels)
        self.asym = ChannelLinear(scalar_channels + vector_channels * 2, scalar_channels)
        self.to_atom_s = ChannelLinear(scalar_channels, scalar_channels)
        self.to_atom_v = ChannelLinear(scalar_channels, vector_channels)
        self.to_atom_t = ChannelLinear(scalar_channels, tensor_channels)
        self.to_bond_s = ChannelLinear(scalar_channels, scalar_channels)
        self.to_bond_v = ChannelLinear(scalar_channels, vector_channels)
        self.gate_s = ChannelLinear(scalar_channels, scalar_channels)
        self.gate_v = ChannelLinear(scalar_channels, vector_channels)
        self.gate_t = ChannelLinear(scalar_channels, tensor_channels)

    def forward(
        self,
        atom_s: torch.Tensor,
        atom_v: torch.Tensor,
        atom_t: torch.Tensor,
        bond_s: torch.Tensor,
        bond_v: torch.Tensor,
        rel: torch.Tensor,
        dist: torch.Tensor,
        cutoff: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one local interaction update.

        Parameters
        ----------
        atom_s:
            Atom scalar channels ``(batch, atoms, scalar_channels)``.
        atom_v:
            Atom vector channels ``(batch, atoms, 3, vector_channels)``.
        atom_t:
            Atom tensor channels ``(batch, atoms, 3, 3, tensor_channels)``.
        bond_s:
            Bond scalar channels ``(batch, atoms, atoms, scalar_channels)``.
        bond_v:
            Bond vector channels ``(batch, atoms, atoms, 3, vector_channels)``.
        rel:
            Relative vectors ``r_i - r_j``.
        dist:
            Pairwise distances.
        cutoff:
            Smooth pair cutoff.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Updated atom and bond states.
        """
        norm_v = atom_v.square().sum(dim=2).sqrt()
        atom_s1 = self.act(self.atom_pre(torch.cat([atom_s, norm_v], dim=-1)))
        atom_v1 = self.atom_vector(atom_v)
        atom_t1 = self.atom_tensor(atom_t)
        norm_bv = bond_v.square().sum(dim=3).sqrt()
        bond_s1 = self.act(self.bond_pre(torch.cat([bond_s, norm_bv], dim=-1)))

        s_i = atom_s1[:, :, None, :]
        s_j = atom_s1[:, None, :, :]
        v_i = atom_v1[:, :, None, :, :]
        v_j = atom_v1[:, None, :, :, :]
        t_i = atom_t1[:, :, None, :, :, :]
        t_j = atom_t1[:, None, :, :, :, :]
        rel_chan = rel[..., None]
        t_rel_i = torch.einsum("bnmpqc,bnmp->bnmqc", t_i, rel)
        t_rel_j = torch.einsum("bnmpqc,bnmp->bnmqc", t_j, -rel)
        v2_i = v_i + t_rel_i
        v2_j = v_j + t_rel_j

        x0_i = s_i * cutoff.unsqueeze(-1)
        x0_j = s_j * cutoff.unsqueeze(-1)
        x1_i = (v2_i * rel_chan).sum(dim=3) * cutoff.unsqueeze(-1)
        x1_j = (v2_j * -rel_chan).sum(dim=3) * cutoff.unsqueeze(-1)
        x2_i = (v2_i * bond_v).sum(dim=3)
        x2_j = (v2_j * bond_v).sum(dim=3)
        x3 = (v2_i * v2_j).sum(dim=3) * cutoff.unsqueeze(-1)

        ysym = self.sym(torch.cat([x0_i + x0_j, x1_i + x1_j, x2_i + x2_j, x3, bond_s1], dim=-1))
        yasym = self.asym(torch.cat([x0_i - x0_j, x1_i - x1_j, x2_i - x2_j], dim=-1))
        ytot = self.act(ysym) + yasym.square()

        beta_s = self.to_atom_s(ytot) * cutoff.unsqueeze(-1)
        beta_v_coeff = self.to_atom_v(ytot) * cutoff.unsqueeze(-1)
        beta_v = rel_chan * beta_v_coeff.unsqueeze(3)
        beta_t_coeff = self.to_atom_t(ytot) * cutoff.unsqueeze(-1)
        dyad = rel[..., :, None] * rel[..., None, :]
        beta_t = dyad[..., None] * beta_t_coeff.unsqueeze(3).unsqueeze(3)

        atom_s3 = beta_s.sum(dim=2)
        atom_v3 = beta_v.sum(dim=2)
        atom_t3 = beta_t.sum(dim=2)
        bond_s3 = self.to_bond_s(ytot)
        bond_v3 = rel_chan * self.to_bond_v(ytot).unsqueeze(3)

        eye = torch.eye(3, device=atom_s.device, dtype=atom_s.dtype)
        identity_bias = eye[None, None, :, :, None] * self.gate_t(atom_s)[:, :, None, None, :]
        atom_s_out = atom_s + self.atom_scalar(atom_s) + self.gate_s(atom_s) * atom_s3
        atom_v_out = atom_v + atom_v3 * self.gate_v(atom_s).unsqueeze(2)
        atom_t_out = (
            atom_t + atom_t3 * self.gate_t(atom_s).unsqueeze(2).unsqueeze(2) + identity_bias
        )
        return atom_s_out, atom_v_out, atom_t_out, bond_s + bond_s3, bond_v + bond_v3


class TeaNet(nn.Module):
    """Tensor-Embedded Atom Network energy model."""

    def __init__(
        self,
        max_atomic_number: int = 18,
        scalar_channels: int = 16,
        vector_channels: int = 4,
        tensor_channels: int = 4,
        layers: int = 4,
        cutoff_radius: float = 6.0,
    ) -> None:
        """Initialize TeaNet.

        Parameters
        ----------
        max_atomic_number:
            Maximum supported element.
        scalar_channels:
            Scalar channel count.
        vector_channels:
            Vector channel count.
        tensor_channels:
            Tensor channel count.
        layers:
            Number of local interaction blocks.
        cutoff_radius:
            Physical cutoff radius.
        """
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.embedding = nn.Embedding(max_atomic_number + 1, scalar_channels)
        self.bond_init = nn.Sequential(
            nn.Linear(1, scalar_channels), nn.SiLU(), nn.Linear(scalar_channels, scalar_channels)
        )
        self.blocks = nn.ModuleList(
            [TeaNetBlock(scalar_channels, vector_channels, tensor_channels) for _ in range(layers)]
        )
        self.atom_energy = nn.Linear(scalar_channels, 1)
        self.bond_energy = nn.Linear(scalar_channels, 1)

    def _cutoff(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute a smooth compact cutoff.

        Parameters
        ----------
        distances:
            Pairwise distances.

        Returns
        -------
        torch.Tensor
            Cutoff values.
        """
        ratio = (distances / self.cutoff_radius).clamp(0.0, 1.0)
        return 0.5 * (torch.cos(torch.pi * ratio) + 1.0) * (distances < self.cutoff_radius)

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        """Predict total energy from atomic numbers and positions.

        Parameters
        ----------
        atoms:
            Tensor ``(batch, atoms, 4)`` with ``Z, x, y, z``.

        Returns
        -------
        torch.Tensor
            Total energy.
        """
        z = atoms[..., 0].long().clamp(min=0, max=18)
        pos = atoms[..., 1:4]
        atom_s = self.embedding(z)
        batch, num_atoms, _ = atom_s.shape
        atom_v = torch.zeros(batch, num_atoms, 3, 4, device=atoms.device, dtype=atoms.dtype)
        atom_t = torch.zeros(batch, num_atoms, 3, 3, 4, device=atoms.device, dtype=atoms.dtype)
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        dist = rel.norm(dim=-1).clamp_min(1e-5)
        eye = torch.eye(num_atoms, device=atoms.device, dtype=atoms.dtype)
        cutoff = self._cutoff(dist) * (1.0 - eye)
        bond_s = self.bond_init(dist.unsqueeze(-1)) * cutoff.unsqueeze(-1)
        bond_v = torch.zeros(
            batch, num_atoms, num_atoms, 3, 4, device=atoms.device, dtype=atoms.dtype
        )
        for block in self.blocks:
            atom_s, atom_v, atom_t, bond_s, bond_v = block(
                atom_s,
                atom_v,
                atom_t,
                bond_s,
                bond_v,
                rel,
                dist,
                cutoff,
            )
        atom_total = self.atom_energy(atom_s).sum(dim=1)
        bond_total = (self.bond_energy(bond_s) * cutoff.unsqueeze(-1)).sum(dim=(1, 2)) * 0.5
        return (atom_total + bond_total).squeeze(-1)


def build_teanet() -> TeaNet:
    """Build a tiny traceable TeaNet.

    Returns
    -------
    TeaNet
        Tiny TeaNet model.
    """
    return TeaNet()


def example_input_teanet() -> torch.Tensor:
    """Create an atomistic input.

    Returns
    -------
    torch.Tensor
        Atomic numbers and positions.
    """
    return torch.tensor(
        [
            [
                [6.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.1],
                [8.0, 0.8, 0.2, 0.0],
                [14.0, -0.8, -0.1, 0.0],
            ]
        ]
    )


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [("TeaNet", "build_teanet", "example_input_teanet", 2019, "REIMPL")]
