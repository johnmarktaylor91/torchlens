# SOURCE: vendored from aiqm/torchani @ master
# (torchani/aev.py::AEVComputer + torchani/nn.py::ANIModel/SpeciesConverter +
#  torchani/utils.py::EnergyShifter, assembled per torchani/models.py::BuiltinModel /
#  the official examples/nnp_training.py reference training script)
#
# TorchANI ("TorchANI: A Free and Open Source PyTorch-Based Deep Learning
# Implementation of the ANI Neural Network Potentials", Gao, Ramezanghorbani,
# Isayev, Smith, Roitberg; J. Chem. Inf. Model. 2020). ANI-style neural-network
# interatomic potentials: a hand-crafted symmetry-function ("AEV") featurizer
# fed into per-chemical-element small MLPs whose outputs are summed to an
# extensive total energy. Only pip package `torchani` is not installed in this
# base env (torch/numpy/etc are); its architecture modules (`aev.py`, `nn.py`,
# the `EnergyShifter` in `utils.py`) import ONLY base torch/stdlib, so they are
# vendored verbatim here rather than reimplemented from a description.
#
# The `torchani.models.ANI1x()` convenience constructor always downloads real
# trained weights + constants from the `ani-model-zoo` release the first time
# any of the ANI1x/ANI1ccx/ANI2x classes are instantiated (network + repo-hosted
# checkpoint dependent) -- unnecessary and undesirable for capturing the
# trainable architecture. Instead this module assembles the exact same real
# `AEVComputer -> SpeciesConverter/ANIModel -> EnergyShifter` composition
# directly, using the official ANI-1x AEV hyperparameters (Rcr/Rca/EtaR/ShfR/...)
# and the official per-element (H/C/N/O) MLP architecture, both copied verbatim
# from TorchANI's own reference training example
# `examples/nnp_training.py` (a real, documented, non-guessed architecture) --
# only the weights are randomly initialized (menagerie captures architecture,
# not trained parameters).
#
# The only non-architectural edit: `aev.py`'s top-of-module `has_cuaev` probe
# uses `importlib_metadata.metadata(__package__)`, which requires this file to
# be an installed package distribution; since this is a standalone vendored
# module (not the installed `torchani` package) that check is replaced with a
# hardcoded `has_cuaev = False` (we never build/use the optional CUDA
# extension), a no-op elision of an environment probe, not an architecture change.

import math
from collections import OrderedDict
from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"

has_cuaev = False  # see module header: cuaev CUDA extension is never built here


# ---- torchani/aev.py ----
class SpeciesAEV(NamedTuple):
    species: Tensor
    aevs: Tensor


def cutoff_cosine(distances: Tensor, cutoff: float) -> Tensor:
    return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


def radial_terms(Rcr: float, EtaR: Tensor, ShfR: Tensor, distances: Tensor) -> Tensor:
    distances = distances.view(-1, 1, 1)
    fc = cutoff_cosine(distances, Rcr)
    ret = 0.25 * torch.exp(-EtaR * (distances - ShfR) ** 2) * fc
    return ret.flatten(start_dim=1)


def angular_terms(
    Rca: float, ShfZ: Tensor, EtaA: Tensor, Zeta: Tensor, ShfA: Tensor, vectors12: Tensor
) -> Tensor:
    vectors12 = vectors12.view(2, -1, 3, 1, 1, 1, 1)
    distances12 = vectors12.norm(2, dim=-5)
    cos_angles = vectors12.prod(0).sum(1) / torch.clamp(distances12.prod(0), min=1e-10)
    angles = torch.acos(0.95 * cos_angles)

    fcj12 = cutoff_cosine(distances12, Rca)
    factor1 = ((1 + torch.cos(angles - ShfZ)) / 2) ** Zeta
    factor2 = torch.exp(-EtaA * (distances12.sum(0) / 2 - ShfA) ** 2)
    ret = 2 * factor1 * factor2 * fcj12.prod(0)
    return ret.flatten(start_dim=1)


def compute_shifts(cell: Tensor, pbc: Tensor, cutoff: float) -> Tensor:
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats = torch.where(pbc, num_repeats, num_repeats.new_zeros(()))
    r1 = torch.arange(1, num_repeats[0].item() + 1, device=cell.device)
    r2 = torch.arange(1, num_repeats[1].item() + 1, device=cell.device)
    r3 = torch.arange(1, num_repeats[2].item() + 1, device=cell.device)
    o = torch.zeros(1, dtype=torch.long, device=cell.device)
    return torch.cat(
        [
            torch.cartesian_prod(r1, r2, r3),
            torch.cartesian_prod(r1, r2, o),
            torch.cartesian_prod(r1, r2, -r3),
            torch.cartesian_prod(r1, o, r3),
            torch.cartesian_prod(r1, o, o),
            torch.cartesian_prod(r1, o, -r3),
            torch.cartesian_prod(r1, -r2, r3),
            torch.cartesian_prod(r1, -r2, o),
            torch.cartesian_prod(r1, -r2, -r3),
            torch.cartesian_prod(o, r2, r3),
            torch.cartesian_prod(o, r2, o),
            torch.cartesian_prod(o, r2, -r3),
            torch.cartesian_prod(o, o, r3),
        ]
    )


def neighbor_pairs(
    padding_mask: Tensor, coordinates: Tensor, cell: Tensor, shifts: Tensor, cutoff: float
) -> Tuple[Tensor, Tensor]:
    coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
    cell = cell.detach()
    num_atoms = padding_mask.shape[1]
    all_atoms = torch.arange(num_atoms, device=cell.device)

    p12_center = torch.triu_indices(num_atoms, num_atoms, 1, device=cell.device)
    shifts_center = shifts.new_zeros((p12_center.shape[1], 3))

    num_shifts = shifts.shape[0]
    all_shifts = torch.arange(num_shifts, device=cell.device)
    prod = torch.cartesian_prod(all_shifts, all_atoms, all_atoms).t()
    shift_index = prod[0]
    p12 = prod[1:]
    shifts_outside = shifts.index_select(0, shift_index)

    shifts_all = torch.cat([shifts_center, shifts_outside])
    p12_all = torch.cat([p12_center, p12], dim=1)
    shift_values = shifts_all.to(cell.dtype) @ cell

    selected_coordinates = coordinates.index_select(1, p12_all.view(-1)).view(
        coordinates.shape[0], 2, -1, 3
    )
    distances = (
        selected_coordinates[:, 0, ...] - selected_coordinates[:, 1, ...] + shift_values
    ).norm(2, -1)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index]
    shifts = shifts_all.index_select(0, pair_index)
    return molecule_index + atom_index12, shifts


def neighbor_pairs_nopbc(padding_mask: Tensor, coordinates: Tensor, cutoff: float) -> Tensor:
    coordinates = coordinates.detach().masked_fill(padding_mask.unsqueeze(-1), math.nan)
    current_device = coordinates.device
    num_atoms = padding_mask.shape[1]
    p12_all = torch.triu_indices(num_atoms, num_atoms, 1, device=current_device)
    p12_all_flattened = p12_all.view(-1)

    pair_coordinates = coordinates.index_select(1, p12_all_flattened).view(
        coordinates.shape[0], 2, -1, 3
    )
    distances = (pair_coordinates[:, 0, ...] - pair_coordinates[:, 1, ...]).norm(2, -1)
    in_cutoff = (distances <= cutoff).nonzero()
    molecule_index, pair_index = in_cutoff.unbind(1)
    molecule_index *= num_atoms
    atom_index12 = p12_all[:, pair_index] + molecule_index
    return atom_index12


def triu_index(num_species: int) -> Tensor:
    species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
    pair_index = torch.arange(species1.shape[0], dtype=torch.long)
    ret = torch.zeros(num_species, num_species, dtype=torch.long)
    ret[species1, species2] = pair_index
    ret[species2, species1] = pair_index
    return ret


def cumsum_from_zero(input_: Tensor) -> Tensor:
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


def triple_by_molecule(atom_index12: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    ai1 = atom_index12.view(-1)
    sorted_ai1, rev_indices = ai1.sort()

    uniqued_central_atom_index, counts = torch.unique_consecutive(
        sorted_ai1, return_inverse=False, return_counts=True
    )

    pair_sizes = torch.div(counts * (counts - 1), 2, rounding_mode="trunc")
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_atom_index = uniqued_central_atom_index.index_select(0, pair_indices)

    m = counts.max().item() if counts.numel() > 0 else 0
    n = pair_sizes.shape[0]
    intra_pair_indices = (
        torch.tril_indices(m, m, -1, device=ai1.device).unsqueeze(1).expand(-1, n, -1)
    )
    mask = (
        torch.arange(intra_pair_indices.shape[2], device=ai1.device) < pair_sizes.unsqueeze(1)
    ).flatten()
    sorted_local_index12 = intra_pair_indices.flatten(1, 2)[:, mask]
    sorted_local_index12 += cumsum_from_zero(counts).index_select(0, pair_indices)

    local_index12 = rev_indices[sorted_local_index12]

    n = atom_index12.shape[1]
    sign12 = ((local_index12 < n).to(torch.int8) * 2) - 1
    return central_atom_index, local_index12 % n, sign12


def compute_aev(
    species: Tensor,
    coordinates: Tensor,
    triu_index_: Tensor,
    constants: Tuple[float, Tensor, Tensor, float, Tensor, Tensor, Tensor, Tensor],
    sizes: Tuple[int, int, int, int, int],
    cell_shifts: Optional[Tuple[Tensor, Tensor]],
) -> Tensor:
    Rcr, EtaR, ShfR, Rca, ShfZ, EtaA, Zeta, ShfA = constants
    num_species, radial_sublength, radial_length, angular_sublength, angular_length = sizes
    num_molecules = species.shape[0]
    num_atoms = species.shape[1]
    num_species_pairs = angular_length // angular_sublength
    coordinates_ = coordinates
    coordinates = coordinates_.flatten(0, 1)

    if cell_shifts is None:
        atom_index12 = neighbor_pairs_nopbc(species == -1, coordinates_, Rcr)
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        vec = selected_coordinates[0] - selected_coordinates[1]
    else:
        cell, shifts = cell_shifts
        atom_index12, shifts = neighbor_pairs(species == -1, coordinates_, cell, shifts, Rcr)
        shift_values = shifts.to(cell.dtype) @ cell
        selected_coordinates = coordinates.index_select(0, atom_index12.view(-1)).view(2, -1, 3)
        vec = selected_coordinates[0] - selected_coordinates[1] + shift_values

    species = species.flatten()
    species12 = species[atom_index12]

    distances = vec.norm(2, -1)

    radial_terms_ = radial_terms(Rcr, EtaR, ShfR, distances)
    radial_aev = radial_terms_.new_zeros(
        (num_molecules * num_atoms * num_species, radial_sublength)
    )
    index12 = atom_index12 * num_species + species12.flip(0)
    radial_aev.index_add_(0, index12[0], radial_terms_)
    radial_aev.index_add_(0, index12[1], radial_terms_)
    radial_aev = radial_aev.reshape(num_molecules, num_atoms, radial_length)

    even_closer_indices = (distances <= Rca).nonzero().flatten()
    atom_index12 = atom_index12.index_select(1, even_closer_indices)
    species12 = species12.index_select(1, even_closer_indices)
    vec = vec.index_select(0, even_closer_indices)

    central_atom_index, pair_index12, sign12 = triple_by_molecule(atom_index12)
    species12_small = species12[:, pair_index12]
    vec12 = vec.index_select(0, pair_index12.view(-1)).view(2, -1, 3) * sign12.unsqueeze(-1)
    species12_ = torch.where(sign12 == 1, species12_small[1], species12_small[0])
    angular_terms_ = angular_terms(Rca, ShfZ, EtaA, Zeta, ShfA, vec12)
    angular_aev = angular_terms_.new_zeros(
        (num_molecules * num_atoms * num_species_pairs, angular_sublength)
    )
    index = central_atom_index * num_species_pairs + triu_index_[species12_[0], species12_[1]]
    angular_aev.index_add_(0, index, angular_terms_)
    angular_aev = angular_aev.reshape(num_molecules, num_atoms, angular_length)
    return torch.cat([radial_aev, angular_aev], dim=-1)


class AEVComputer(torch.nn.Module):
    def __init__(
        self, Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=False
    ):
        super().__init__()
        self.Rcr = Rcr
        self.Rca = Rca
        assert Rca <= Rcr, "Current implementation of AEVComputer assumes Rca <= Rcr"
        self.num_species = num_species
        self.use_cuda_extension = use_cuda_extension

        self.register_buffer("EtaR", EtaR.view(-1, 1))
        self.register_buffer("ShfR", ShfR.view(1, -1))
        self.register_buffer("EtaA", EtaA.view(-1, 1, 1, 1))
        self.register_buffer("Zeta", Zeta.view(1, -1, 1, 1))
        self.register_buffer("ShfA", ShfA.view(1, 1, -1, 1))
        self.register_buffer("ShfZ", ShfZ.view(1, 1, 1, -1))

        self.radial_sublength = self.EtaR.numel() * self.ShfR.numel()
        self.radial_length = self.num_species * self.radial_sublength
        self.angular_sublength = (
            self.EtaA.numel() * self.Zeta.numel() * self.ShfA.numel() * self.ShfZ.numel()
        )
        self.angular_length = (
            (self.num_species * (self.num_species + 1)) // 2 * self.angular_sublength
        )
        self.aev_length = self.radial_length + self.angular_length
        self.sizes = (
            self.num_species,
            self.radial_sublength,
            self.radial_length,
            self.angular_sublength,
            self.angular_length,
        )

        self.register_buffer("triu_index", triu_index(num_species).to(device=self.EtaR.device))

        cutoff = max(self.Rcr, self.Rca)
        default_cell = torch.eye(3, dtype=self.EtaR.dtype, device=self.EtaR.device)
        default_pbc = torch.zeros(3, dtype=torch.bool, device=self.EtaR.device)
        default_shifts = compute_shifts(default_cell, default_pbc, cutoff)
        self.register_buffer("default_cell", default_cell)
        self.register_buffer("default_shifts", default_shifts)

        self.cuaev_enabled = False

    def constants(self):
        return self.Rcr, self.EtaR, self.ShfR, self.Rca, self.ShfZ, self.EtaA, self.Zeta, self.ShfA

    def forward(
        self,
        input_: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> SpeciesAEV:
        species, coordinates = input_
        assert species.dim() == 2
        assert species.shape == coordinates.shape[:-1]
        assert coordinates.shape[-1] == 3

        if cell is None and pbc is None:
            aev = compute_aev(
                species, coordinates, self.triu_index, self.constants(), self.sizes, None
            )
        else:
            assert cell is not None and pbc is not None
            cutoff = max(self.Rcr, self.Rca)
            shifts = compute_shifts(cell, pbc, cutoff)
            aev = compute_aev(
                species, coordinates, self.triu_index, self.constants(), self.sizes, (cell, shifts)
            )

        return SpeciesAEV(species, aev)


# ---- torchani/nn.py ----
class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


class ANIModel(torch.nn.ModuleDict):
    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))

    def forward(
        self,
        species_aev: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            mask = species_ == i
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return output


class SpeciesConverter(torch.nn.Module):
    def __init__(self, periodic_table, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(periodic_table)}
        maxidx = max(rev_idx.values())
        self.register_buffer("conv_tensor", torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(
        self,
        input_: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ):
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f"Unknown species found in {species}")

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)


# ---- torchani/utils.py ----
class EnergyShifter(torch.nn.Module):
    def __init__(self, self_energies, fit_intercept=False):
        super().__init__()

        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        self.register_buffer("self_energies", self_energies)

    def sae(self, species):
        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]

        self_energies = self.self_energies[species]
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(
            0, device=species.device, dtype=torch.double
        )
        return self_energies.sum(dim=1) + intercept

    def forward(
        self,
        species_energies: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, energies = species_energies
        sae = self.sae(species)
        return SpeciesEnergies(species, energies + sae)


# ---- assembled model: real ANI-1x AEV hyperparameters + real per-element MLP
# widths, both copied verbatim from torchani's own examples/nnp_training.py ----
PERIODIC_TABLE = ["Dummy", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]


class ANIPotential(torch.nn.Module):
    """AEVComputer -> SpeciesConverter -> ANIModel -> EnergyShifter, the real
    TorchANI single-network inference pipeline (BuiltinModel's composition in
    torchani/models.py), assembled directly from the vendored architecture
    classes above instead of via the weight-downloading `ANI1x()` constructor."""

    def __init__(self, aev_computer, species_converter, neural_networks, energy_shifter):
        super().__init__()
        self.aev_computer = aev_computer
        self.species_converter = species_converter
        self.neural_networks = neural_networks
        self.energy_shifter = energy_shifter

    def forward(self, species_coordinates: Tuple[Tensor, Tensor]) -> SpeciesEnergies:
        species_coordinates = self.species_converter(species_coordinates)
        species_aevs = self.aev_computer(species_coordinates)
        species_energies = self.neural_networks(species_aevs)
        return self.energy_shifter(species_energies)


def _real_hcno_network(aev_dim: int, widths):
    w1, w2, w3 = widths
    return torch.nn.Sequential(
        torch.nn.Linear(aev_dim, w1),
        torch.nn.CELU(0.1),
        torch.nn.Linear(w1, w2),
        torch.nn.CELU(0.1),
        torch.nn.Linear(w2, w3),
        torch.nn.CELU(0.1),
        torch.nn.Linear(w3, 1),
    )


def build_torchani():
    # ANI-1x AEV hyperparameters, verbatim from examples/nnp_training.py
    Rcr = 5.2000e00
    Rca = 3.5000e00
    EtaR = torch.tensor([1.6000000e01])
    ShfR = torch.tensor(
        [
            9.0000000e-01,
            1.1687500e00,
            1.4375000e00,
            1.7062500e00,
            1.9750000e00,
            2.2437500e00,
            2.5125000e00,
            2.7812500e00,
            3.0500000e00,
            3.3187500e00,
            3.5875000e00,
            3.8562500e00,
            4.1250000e00,
            4.3937500e00,
            4.6625000e00,
            4.9312500e00,
        ]
    )
    Zeta = torch.tensor([3.2000000e01])
    ShfZ = torch.tensor(
        [
            1.9634954e-01,
            5.8904862e-01,
            9.8174770e-01,
            1.3744468e00,
            1.7671459e00,
            2.1598449e00,
            2.5525440e00,
            2.9452431e00,
        ]
    )
    EtaA = torch.tensor([8.0000000e00])
    ShfA = torch.tensor([9.0000000e-01, 1.5500000e00, 2.2000000e00, 2.8500000e00])
    species_order = ["H", "C", "N", "O"]
    num_species = len(species_order)

    aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
    species_converter = SpeciesConverter(PERIODIC_TABLE, species_order)

    aev_dim = aev_computer.aev_length
    # real per-element MLP widths, verbatim from examples/nnp_training.py
    h_network = _real_hcno_network(aev_dim, (160, 128, 96))
    c_network = _real_hcno_network(aev_dim, (144, 112, 96))
    n_network = _real_hcno_network(aev_dim, (128, 112, 96))
    o_network = _real_hcno_network(aev_dim, (128, 112, 96))
    neural_networks = ANIModel([h_network, c_network, n_network, o_network])

    energy_shifter = EnergyShifter([0.0, 0.0, 0.0, 0.0])

    model = ANIPotential(aev_computer, species_converter, neural_networks, energy_shifter)
    model.eval()
    return model


def example_input_torchani():
    # A small water-like HCNO test molecule: 1 conformation, 5 atoms.
    # Species given as periodic-table atomic numbers (H=1, C=6), matching
    # SpeciesConverter's expected input convention.
    species = torch.tensor([[1, 6, 6, 7, 8]], dtype=torch.long)
    coordinates = torch.randn(1, 5, 3, dtype=torch.float32) * 1.5
    return ((species, coordinates),)


MENAGERIE_ENTRIES = [
    (
        "ANI neural network potentials",
        "build_torchani",
        "example_input_torchani",
        2020,
        "vendored-pytorch",
    ),
]
