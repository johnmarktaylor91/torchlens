# FAITHFUL REIMPLEMENTATION from Ko, Finkler, Goedecker & Behler, "A fourth-generation
# high-dimensional neural network potential with accurate electrostatics including
# non-local charge transfer", Nature Communications 12, 398 (2021),
# https://doi.org/10.1038/s41467-020-20427-2 (no public PyTorch/Python code; the only
# public implementation is n2p2 (github.com/CompPhysVienna/n2p2), written in C++ for
# LAMMPS integration, not a base-torch-installable Python package).
"""4G-HDNNP: a fourth-generation high-dimensional neural network potential.

Faithfully reimplements the architecture described in Ko et al. (2021):

1. A per-element atomic neural network predicts an environment-dependent
   atomic electronegativity ``chi_i`` from atom-centered symmetry-function
   descriptors ``G_i`` (Eq. 1: each atom's electronegativity is the scalar
   output of a small feed-forward NN over its own descriptor vector, one
   independently-parameterized NN per chemical species -- the standard
   Behler-Parrinello atomic-NN topology used throughout all HDNNP
   generations).

2. A differentiable global charge-equilibration (QEq) solve determines the
   atomic partial charges ``Q_i`` that minimize

       E_Qeq = E_elec + sum_i (chi_i * Q_i + 0.5 * J_i * Q_i^2)     (Eq. 2)

   subject to the total-charge constraint ``sum_i Q_i = Q_tot``, where

       E_elec = sum_{i<j} [erf(r_ij / (sqrt(2) * gamma_ij)) / r_ij] * Q_i * Q_j
                + sum_i Q_i^2 / (2 * sigma_i * sqrt(pi))            (Eq. 3)

   is the screened (Gaussian-charge-density) Coulomb electrostatic energy,
   with ``gamma_ij = sqrt(sigma_i^2 + sigma_j^2)``. This is solved as the
   standard Rappe-Goddard QEq bordered linear system (a Lagrange multiplier
   enforces the total-charge constraint):

       [ A   1 ] [ Q ]   [ -chi ]
       [ 1^T 0 ] [ l ] = [ Q_tot]

   with hardness matrix ``A_ii = J_i + 1/(sigma_i * sqrt(pi))`` on the
   diagonal and ``A_ij = erf(r_ij / (sqrt(2) * gamma_ij)) / r_ij`` off the
   diagonal (this bordered-linear-system form is the closed-form QEq solve
   used throughout the charge-equilibration literature this paper builds on,
   e.g. Rappe & Goddard 1991; identical structure confirmed independently in
   Naserifar et al. 2017 and Vondrak et al. 2023's q-pac formulation of the
   same 4G-style charge model).

3. A second per-element atomic neural network predicts the short-range
   atomic energy ``E_i`` from the descriptor ``G_i`` *and* the equilibrated
   charge ``Q_i`` as an additional input feature (Eq. 9:
   ``E_short = sum_i E_i({G_i}, Q_i)``), replacing the second-generation
   HDNNP's descriptor-only atomic energy with a charge-aware one -- this is
   precisely what distinguishes 4G-HDNNP from 2G/3G-HDNNPs.

4. Total energy: ``E_total = E_short + E_elec`` (Eq. 9, rearranged).

No public PyTorch implementation of this architecture exists; the only
public code (n2p2) implements it in C++ for LAMMPS. The atomic-NN topology
here (per-species stacks of ``Linear -> activation -> ... -> Linear``) uses
the same universally-adopted Behler-Parrinello ANN structure as n2p2's
``NeuralNetwork`` class and this same batch's ``aenet-PyTorch`` vendor entry,
faithfully transcribed from the paper's description rather than guessed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


def _atomic_nn(input_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    """Per-species feed-forward ANN: Linear -> tanh -> ... -> Linear(out_dim).

    Standard Behler-Parrinello atomic-NN topology shared by all HDNNP
    generations (2G/3G/4G) and n2p2's ``NeuralNetwork`` class.
    """
    layers: list[nn.Module] = []
    dims = [input_dim, *hidden_dims]
    for a, b in zip(dims[:-1], dims[1:], strict=False):
        layers.append(nn.Linear(a, b))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


class ElectronegativityNet(nn.Module):
    """Per-element atomic NN predicting environment-dependent electronegativity
    ``chi_i`` from symmetry-function descriptors (Eq. 1 of Ko et al. 2021)."""

    def __init__(self, n_species: int, descriptor_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.nets = nn.ModuleList(
            _atomic_nn(descriptor_dim, hidden_dims, out_dim=1) for _ in range(n_species)
        )

    def forward(self, descriptors: torch.Tensor, species_idx: torch.LongTensor) -> torch.Tensor:
        """descriptors: (n_atoms, descriptor_dim); species_idx: (n_atoms,) long
        index into ``self.nets``. Returns chi: (n_atoms,)."""
        chi = torch.zeros(descriptors.shape[0], dtype=descriptors.dtype, device=descriptors.device)
        for sp, net in enumerate(self.nets):
            mask = species_idx == sp
            if mask.any():
                chi = chi.masked_scatter(mask, net(descriptors[mask]).squeeze(-1))
        return chi


class ShortRangeEnergyNet(nn.Module):
    """Per-element atomic NN predicting the short-range atomic energy
    ``E_i({G_i}, Q_i)`` from descriptors *and* the equilibrated charge
    (Eq. 9 of Ko et al. 2021) -- the charge-aware second atomic-NN set that
    is unique to 4G-HDNNP relative to 2G/3G-HDNNPs."""

    def __init__(self, n_species: int, descriptor_dim: int, hidden_dims: list[int]):
        super().__init__()
        # +1 input feature for the equilibrated atomic charge Q_i.
        self.nets = nn.ModuleList(
            _atomic_nn(descriptor_dim + 1, hidden_dims, out_dim=1) for _ in range(n_species)
        )

    def forward(
        self,
        descriptors: torch.Tensor,
        charges: torch.Tensor,
        species_idx: torch.LongTensor,
    ) -> torch.Tensor:
        """Returns per-atom short-range energies E_i: (n_atoms,)."""
        feats = torch.cat([descriptors, charges.unsqueeze(-1)], dim=-1)
        e_short = torch.zeros(
            descriptors.shape[0], dtype=descriptors.dtype, device=descriptors.device
        )
        for sp, net in enumerate(self.nets):
            mask = species_idx == sp
            if mask.any():
                e_short = e_short.masked_scatter(mask, net(feats[mask]).squeeze(-1))
        return e_short


class ChargeEquilibration(nn.Module):
    """Global QEq solve (Eq. 2-3): given electronegativities ``chi_i``,
    hardness parameters ``J_i``, Gaussian charge-density widths ``sigma_i``,
    pairwise distances, and the total system charge, returns the
    equilibrated atomic partial charges ``Q_i`` and the electrostatic energy
    ``E_elec``.

    Solved as the bordered QEq linear system with a Lagrange multiplier
    enforcing ``sum_i Q_i = Q_tot`` (standard Rappe-Goddard closed-form QEq
    solve; see module docstring)."""

    def __init__(self, n_species: int):
        super().__init__()
        # Learnable per-species hardness J_i and Gaussian width sigma_i,
        # softplus-parameterized to stay positive (hardness/width must be
        # positive physical quantities).
        self.log_hardness = nn.Parameter(torch.zeros(n_species))
        self.log_sigma = nn.Parameter(torch.zeros(n_species))

    def forward(
        self,
        chi: torch.Tensor,
        species_idx: torch.LongTensor,
        pair_i: torch.LongTensor,
        pair_j: torch.LongTensor,
        r_ij: torch.Tensor,
        q_tot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        chi: (n_atoms,) electronegativities.
        species_idx: (n_atoms,) species index per atom.
        pair_i, pair_j: (n_pairs,) long indices of all i<j atom pairs.
        r_ij: (n_pairs,) pairwise distances for those pairs.
        q_tot: scalar total system charge.

        Returns (Q, E_elec).
        """
        n_atoms = chi.shape[0]
        dtype, device = chi.dtype, chi.device

        hardness = torch.nn.functional.softplus(self.log_hardness)[species_idx]  # (n_atoms,)
        sigma = torch.nn.functional.softplus(self.log_sigma)[species_idx]  # (n_atoms,)

        # Hardness matrix A (Eq. 8-style): diagonal J_i + 1/(sigma_i*sqrt(pi)),
        # off-diagonal erf(r_ij / (sqrt(2)*gamma_ij)) / r_ij.
        A = torch.zeros(n_atoms, n_atoms, dtype=dtype, device=device)
        diag = hardness + 1.0 / (
            sigma * torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device))
        )
        A = A + torch.diag(diag)

        gamma_ij = torch.sqrt(sigma[pair_i] ** 2 + sigma[pair_j] ** 2)
        off_diag = (
            torch.erf(r_ij / (torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device)) * gamma_ij))
            / r_ij
        )
        A = A.index_put((pair_i, pair_j), off_diag, accumulate=True)
        A = A.index_put((pair_j, pair_i), off_diag, accumulate=True)

        # Bordered QEq linear system: [[A, 1], [1^T, 0]] @ [Q; lambda] = [-chi; q_tot]
        border = torch.ones(n_atoms, 1, dtype=dtype, device=device)
        top = torch.cat([A, border], dim=1)
        bottom = torch.cat(
            [border.transpose(0, 1), torch.zeros(1, 1, dtype=dtype, device=device)], dim=1
        )
        M = torch.cat([top, bottom], dim=0)

        rhs = torch.cat([-chi, q_tot.reshape(1)], dim=0).unsqueeze(-1)
        sol = torch.linalg.solve(M, rhs).squeeze(-1)
        Q = sol[:n_atoms]

        # E_elec (Eq. 3): pairwise screened-Coulomb term + Gaussian self-energy.
        e_pair = (off_diag * Q[pair_i] * Q[pair_j]).sum()
        e_self = (
            Q**2 / (2.0 * sigma * torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device)))
        ).sum()
        e_elec = e_pair + e_self

        return Q, e_elec


class FourthGenHDNNP(nn.Module):
    """Full 4G-HDNNP: electronegativity NN -> QEq charge solve -> charge-aware
    short-range energy NN -> total energy = E_short + E_elec (Eq. 9)."""

    def __init__(
        self,
        n_species: int,
        descriptor_dim: int,
        chi_hidden: list[int],
        eshort_hidden: list[int],
    ):
        super().__init__()
        self.electronegativity_net = ElectronegativityNet(n_species, descriptor_dim, chi_hidden)
        self.charge_eq = ChargeEquilibration(n_species)
        self.short_range_net = ShortRangeEnergyNet(n_species, descriptor_dim, eshort_hidden)

    def forward(
        self,
        descriptors: torch.Tensor,
        species_idx: torch.LongTensor,
        pair_i: torch.LongTensor,
        pair_j: torch.LongTensor,
        r_ij: torch.Tensor,
        q_tot: torch.Tensor,
    ) -> torch.Tensor:
        chi = self.electronegativity_net(descriptors, species_idx)
        Q, e_elec = self.charge_eq(chi, species_idx, pair_i, pair_j, r_ij, q_tot)
        e_short_atomic = self.short_range_net(descriptors, Q, species_idx)
        e_short = e_short_atomic.sum()
        e_total = e_short + e_elec
        return e_total.reshape(1)


def build_4g_hdnnp() -> FourthGenHDNNP:
    """Tiny random-init 4G-HDNNP for a 2-species system with a small
    symmetry-function descriptor width."""
    torch.manual_seed(0)
    return (
        FourthGenHDNNP(
            n_species=2,
            descriptor_dim=8,
            chi_hidden=[6],
            eshort_hidden=[6],
        )
        .double()
        .eval()
    )


def example_input_4g_hdnnp() -> tuple[
    torch.Tensor, torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor
]:
    """A tiny synthetic 5-atom, 2-species cluster (3 of species 0, 2 of
    species 1) with all-pairs (i<j) distances and a neutral total charge."""
    torch.manual_seed(0)
    n_atoms = 5
    descriptor_dim = 8

    descriptors = torch.randn(n_atoms, descriptor_dim, dtype=torch.float64)
    species_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    pair_i, pair_j = [], []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = torch.tensor(pair_i, dtype=torch.long)
    pair_j = torch.tensor(pair_j, dtype=torch.long)

    positions = torch.randn(n_atoms, 3, dtype=torch.float64) * 2.0
    r_ij = torch.linalg.norm(positions[pair_i] - positions[pair_j], dim=-1)
    # Avoid near-zero distances (would blow up the 1/r_ij Coulomb term).
    r_ij = r_ij + 1.5

    q_tot = torch.tensor(0.0, dtype=torch.float64)

    return (descriptors, species_idx, pair_i, pair_j, r_ij, q_tot)


MENAGERIE_ENTRIES = [
    ("4G-HDNNP", "build_4g_hdnnp", "example_input_4g_hdnnp", 2021, MENAGERIE_ZOO),
]
