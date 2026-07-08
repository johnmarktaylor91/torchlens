# SOURCE: vendored from aiqm/torchani @ 2.8.2 (2026-07-01)
# Files combined from the real repo (paths as in upstream), trimmed to the code path
# exercised by a random-init `simple_ani(..., repulsion=True, dispersion=False)` model
# (no pretrained-weight downloads, no PBC, no cuAEV/cell-list/adaptive-list extensions,
# no charge networks, no `torchani.datasets`):
#   torchani/constants.py          (ATOMIC_NUMBER, PERIODIC_TABLE, XTB_REPULSION_ALPHA/YEFF
#                                    -- data tables; the periodic-table values themselves are
#                                    inlined below, extracted verbatim from the package's
#                                    resources/atomic_constants.json since that JSON resource
#                                    file can't be shipped alongside a single staging module)
#   torchani/tuples.py             (SpeciesEnergies, EnergiesScalars, SpeciesAEV, etc.)
#   torchani/cutoffs.py            (Cutoff, CutoffSmooth, CutoffCosine, _parse_cutoff_fn)
#   torchani/utils.py              (linspace, _validate_user_kwargs, map_to_central,
#                                    nonzero_in_chunks, fast_masked_select -- subset)
#   torchani/_core.py              (_ChemModule)
#   torchani/sae.py                (SelfEnergy)
#   torchani/neighbors.py          (Neighbors, Triples, AllPairs/all_pairs + narrow_down +
#                                    discard_* + neighbors_to_triples -- the 'all_pairs'
#                                    strategy only; cell-list/adaptive-list/verlet-cell-list
#                                    strategies dropped, they are unexercised alternates)
#   torchani/aev/_terms.py         (ANIRadial, ANIAngular -- the 2-body/3-body AEV terms)
#   torchani/aev/_computer.py      (AEVComputer -- pyAEV strategy path; cuAEV CUDA-extension
#                                    branches are real code but no-op/unreachable since
#                                    CUAEV_IS_INSTALLED is always False without a compiled ext)
#   torchani/nn/_core.py           (AtomicContainer, AtomicNetwork, TightCELU, parse_activation)
#   torchani/nn/_containers.py     (ANINetworks, SpeciesConverter -- the default atomic-network
#                                    container; SingleNN/ANISharedNetworks/Ensemble dropped,
#                                    unexercised container alternates for ensemble_size=1)
#   torchani/nn/_internal.py       (_ZeroANINetworks -- used internally by Assembler)
#   torchani/potentials/core.py    (Potential, BasePairPotential -- pair-potential base classes)
#   torchani/potentials/xtb.py     (RepulsionXTB -- the paper's short-range repulsion potential)
#   torchani/potentials/nnp.py     (NNPotential -- adapts AEVComputer+ANINetworks as a Potential)
#   torchani/arch.py               (_ANI, ANI, Assembler, simple_ani -- the model class and its
#                                    builder; ANIq (charge-prediction subclass), the QBC/ensemble
#                                    analysis convenience methods, and the pretrained
#                                    ANI1x/ANI2x/etc factory functions in models.py were dropped:
#                                    none are on the random-init forward-pass call path)
#
# TorchANI is the canonical PyTorch implementation of the ANI-style neural network
# interatomic potential family (Roitberg group; Smith, Isayev, Roitberg, Chem. Sci. 2017,
# "ANI-1"). An `AEVComputer` featurizes each atom's local chemical environment via radial +
# angular symmetry functions (2-body and 3-body terms over a neighborlist), and a per-element
# `ANINetworks` container of small MLPs ("atomic networks", one per chemical element) maps
# each atom's AEV to a per-atom energy contribution; contributions are summed (plus a
# self-energy baseline and, optionally, an XTB short-range repulsion correction) to predict
# the total molecular energy. Code is transcribed verbatim from the real repo; only import
# paths were flattened into this single file, unconditional `huggingface_hub` pretrained-
# weight-download machinery was deferred/dropped (unused for random init), and multiple files
# were trimmed to the classes/functions actually reachable from `simple_ani(...)` with
# `repulsion=True, dispersion=False, neighborlist="all_pairs", strategy="pyaev"` -- these
# trims remove unexercised alternates (PBC, cell-list/adaptive-list neighborlists, cuAEV,
# charge networks, DFT-D3 dispersion (needs h5py), ensembles) without altering any of the
# retained architecture code.
#
# Original license: MIT (per repo, LICENSE file).

import math
import typing as tp
import warnings
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.jit import Final

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# torchani/constants.py -- periodic-table data tables, extracted verbatim from
# torchani/resources/atomic_constants.json (package data file; inlined here since a JSON
# resource can't ride alongside a single staging module). Only the subset used downstream
# (ATOMIC_NUMBER for all 118 elements, PERIODIC_TABLE, and XTB repulsion alpha/yeff for the
# elements RepulsionXTB is exercised on) is kept.
# ---------------------------------------------------------------------------
ATOMIC_NUMBER: tp.Dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}  # noqa: E501

ATOMIC_XTB_REPULSION_ALPHA: tp.Dict[str, float] = {
    "H": 2.213717,
    "He": 3.60467,
    "Li": 0.475307,
    "Be": 0.939696,
    "B": 1.373856,
    "C": 1.247655,
    "N": 1.682689,
    "O": 2.165712,
    "F": 2.421394,
    "Ne": 3.318479,
    "Na": 0.572728,
    "Mg": 0.917975,
    "Al": 0.876623,
    "Si": 1.187323,
    "P": 1.143343,
    "S": 1.214553,
    "Cl": 1.577144,
    "Ar": 0.896198,
    "K": 0.482206,
    "Ca": 0.683051,
    "Sc": 0.574299,
    "Ti": 0.723104,
    "V": 0.928532,
    "Cr": 0.966993,
    "Mn": 1.0711,
    "Fe": 1.113422,
    "Co": 1.241717,
    "Ni": 1.077516,
    "Cu": 0.998768,
    "Zn": 1.160262,
    "Ga": 1.122923,
    "Ge": 1.222349,
    "As": 1.249372,
    "Se": 1.230284,
    "Br": 1.296174,
    "Kr": 0.908074,
    "Rb": 0.574054,
    "Sr": 0.697345,
    "Y": 0.706172,
    "Zr": 0.681106,
    "Nb": 0.865552,
    "Mo": 1.034519,
    "Tc": 1.019565,
    "Ru": 1.031669,
    "Rh": 1.094599,
    "Pd": 1.092745,
    "Ag": 0.678344,
    "Cd": 0.936236,
    "In": 1.024007,
    "Sn": 1.139959,
    "Sb": 1.122937,
    "Te": 1.000712,
    "I": 1.017946,
    "Xe": 1.012036,
    "Cs": 0.585257,
    "Ba": 0.716259,
    "La": 0.737643,
    "Ce": 0.72995,
    "Pr": 0.734624,
    "Nd": 0.739299,
    "Pm": 0.743973,
    "Sm": 0.748648,
    "Eu": 0.753322,
    "Gd": 0.757996,
    "Tb": 0.762671,
    "Dy": 0.767345,
    "Ho": 0.77202,
    "Er": 0.776694,
    "Tm": 0.781368,
    "Yb": 0.786043,
    "Lu": 0.790717,
    "Hf": 0.852852,
    "Ta": 0.990234,
    "W": 1.018805,
    "Re": 1.170412,
    "Os": 1.221937,
    "Ir": 1.197148,
    "Pt": 1.204081,
    "Au": 0.91921,
    "Hg": 1.13736,
    "Tl": 1.399312,
    "Pb": 1.179922,
    "Bi": 1.13086,
    "Po": 0.957939,
    "At": 0.963878,
    "Rn": 0.965577,
}  # noqa: E501

ATOMIC_XTB_REPULSION_YEFF: tp.Dict[str, float] = {
    "H": 1.105388,
    "He": 1.094283,
    "Li": 1.289367,
    "Be": 4.221216,
    "B": 7.192431,
    "C": 4.231078,
    "N": 5.242592,
    "O": 5.784415,
    "F": 7.021486,
    "Ne": 11.041068,
    "Na": 5.244917,
    "Mg": 18.083164,
    "Al": 17.867328,
    "Si": 40.001111,
    "P": 19.683502,
    "S": 14.99509,
    "Cl": 17.353134,
    "Ar": 7.266606,
    "K": 10.439482,
    "Ca": 14.786701,
    "Sc": 8.004267,
    "Ti": 12.036336,
    "V": 15.677873,
    "Cr": 19.517914,
    "Mn": 18.760605,
    "Fe": 20.360089,
    "Co": 27.127744,
    "Ni": 10.533269,
    "Cu": 9.913846,
    "Zn": 22.099503,
    "Ga": 31.14675,
    "Ge": 42.100144,
    "As": 39.147587,
    "Se": 27.426779,
    "Br": 32.845361,
    "Kr": 17.363803,
    "Rb": 44.338211,
    "Sr": 34.365525,
    "Y": 17.326237,
    "Zr": 24.263093,
    "Nb": 30.562732,
    "Mo": 48.312796,
    "Tc": 44.779882,
    "Ru": 28.070247,
    "Rh": 38.035941,
    "Pd": 28.6747,
    "Ag": 6.493286,
    "Cd": 26.226628,
    "In": 63.85424,
    "Sn": 80.053438,
    "Sb": 77.05756,
    "Te": 48.614745,
    "I": 63.319176,
    "Xe": 51.188398,
    "Cs": 67.249039,
    "Ba": 46.984607,
    "La": 50.927529,
    "Ce": 48.676714,
    "Pr": 47.669448,
    "Nd": 46.662183,
    "Pm": 45.654917,
    "Sm": 44.647651,
    "Eu": 43.640385,
    "Gd": 42.63312,
    "Tb": 41.625854,
    "Dy": 40.618588,
    "Ho": 39.611322,
    "Er": 38.604057,
    "Tm": 37.596791,
    "Yb": 36.589525,
    "Lu": 35.582259,
    "Hf": 40.186772,
    "Ta": 54.666156,
    "W": 55.899801,
    "Re": 80.410086,
    "Os": 62.809871,
    "Ir": 56.045639,
    "Pt": 53.881425,
    "Au": 14.711475,
    "Hg": 51.577544,
    "Tl": 58.801614,
    "Pb": 102.368258,
    "Bi": 132.896832,
    "Po": 52.301232,
    "At": 81.771063,
    "Rn": 128.13358,
}  # noqa: E501

PERIODIC_TABLE = ("",) + tuple(kv[0] for kv in sorted(ATOMIC_NUMBER.items(), key=lambda x: x[1]))

GSAES: tp.Dict[str, tp.Dict[str, float]] = {
    "wb97x-631gd": {
        "C": -37.8338334,
        "Cl": -460.116700600,
        "F": -99.6949007,
        "H": -0.4993212,
        "N": -54.5732825,
        "O": -75.0424519,
        "S": -398.0814169,
    },
}


def _mapping_to_znumber_indexed_seq(symbols_map: tp.Mapping[str, float]) -> tp.Tuple[float, ...]:
    _symbols_map = dict(symbols_map)
    seq = [math.nan] * (len(ATOMIC_NUMBER) + 1)
    for k, v in _symbols_map.items():
        seq[ATOMIC_NUMBER[k]] = v
    return tuple(seq)


XTB_REPULSION_ALPHA = _mapping_to_znumber_indexed_seq(ATOMIC_XTB_REPULSION_ALPHA)
XTB_REPULSION_YEFF = _mapping_to_znumber_indexed_seq(ATOMIC_XTB_REPULSION_YEFF)

ANGSTROM_TO_BOHR = 1.8897261258369282


# ---------------------------------------------------------------------------
# torchani/tuples.py
# ---------------------------------------------------------------------------
class SpeciesEnergies(tp.NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesAEV(tp.NamedTuple):
    species: Tensor
    aevs: Tensor


class EnergiesScalars(tp.NamedTuple):
    energies: Tensor
    scalars: tp.Optional[Tensor] = None


# ---------------------------------------------------------------------------
# torchani/utils.py (subset)
# ---------------------------------------------------------------------------
def linspace(start: float, stop: float, steps: int) -> tp.Tuple[float, ...]:
    r"""Pure python linspace"""
    return tuple(start + ((stop - start) / steps) * j for j in range(steps))


def _validate_user_kwargs(
    clsname: str,
    names_dict: tp.Dict[str, tp.Sequence[str]],
    kwargs: tp.Dict[str, tp.Union[tp.Tuple, tp.List, float]],
    trainable: tp.Sequence[str],
) -> None:
    _num_tensors = sum(len(seq) for seq in names_dict.values())
    kwargs_set: tp.Set[str] = set()
    for v in names_dict.values():
        kwargs_set = kwargs_set.union(v)

    if len(kwargs_set) != _num_tensors:
        raise ValueError("tensor names must be unique")

    if set(kwargs) != kwargs_set:
        raise ValueError(
            f"Expected arguments '{', '.join(kwargs_set)}'"
            f" but got '{', '.join(kwargs.keys())}'"
            f" Maybe you forgot '*_tensors = [..., 'argname']'"
            f" when defining the class?"
        )


def map_to_central(coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
    r"""Map atoms outside the unit cell into the cell using PBC"""
    frac = torch.remainder((coordinates @ cell.inverse()), cell.new_ones(3))
    frac[frac >= 1.0] -= 1.0
    frac[frac < 0.0] += 1.0
    out_coordinates = torch.matmul(frac, cell)
    out_coordinates[:, :, ~pbc] = coordinates[:, :, ~pbc].clone()
    return out_coordinates


def nonzero_in_chunks(tensor: Tensor, chunk_size: int = 2**31 - 1):
    r"""Flatten a tensor and apply nonzero in chunks (workaround for the INT_MAX limit)"""
    tensor = tensor.view(-1)
    num_splits = math.ceil(tensor.numel() / chunk_size)
    if num_splits <= 1:
        return tensor.nonzero().view(-1)
    offset = 0
    nonzero_chunks: tp.List[Tensor] = []
    for chunk in torch.chunk(tensor, num_splits):
        nonzero_chunks.append(chunk.nonzero() + offset)
        offset += chunk.shape[0]
    return torch.cat(nonzero_chunks).view(-1)


def fast_masked_select(x: Tensor, mask: Tensor, idx: int) -> Tensor:
    r"""Has the same effect as `torch.masked_select` but faster"""
    return x.index_select(idx, nonzero_in_chunks(mask))


def cumsum_from_zero(input_: Tensor) -> Tensor:
    r"""Cumulative sum just like `torch.cumsum`, but result starts from 0"""
    cumsum = torch.zeros_like(input_)
    torch.cumsum(input_[:-1], dim=0, out=cumsum[1:])
    return cumsum


# ---------------------------------------------------------------------------
# torchani/cutoffs.py
# ---------------------------------------------------------------------------
class Cutoff(torch.nn.Module):
    r"""Base class for cutoff functions."""

    _cuaev_name: str

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self._fn_params = args + tuple(kwargs.values())
        self._cuaev_name = ""

    @torch.jit.unused
    def is_same(self, other: object) -> bool:
        if not isinstance(other, Cutoff):
            return False
        if type(self) is not type(other):
            return False
        if not self._fn_params == other._fn_params:
            return False
        return True

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        raise NotImplementedError


class CutoffDummy(Cutoff):
    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return torch.ones_like(distances)


class CutoffCosine(Cutoff):
    def __init__(self) -> None:
        super().__init__()
        self._cuaev_name = "cosine"

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        return 0.5 * torch.cos(distances * (math.pi / cutoff)) + 0.5


class CutoffSmooth(Cutoff):
    r"""Use an infinitely differentiable exponential cutoff"""

    def __init__(self, order: int = 2, eps: float = 1.0e-10) -> None:
        super().__init__(order, eps)
        if order == 2 and eps == 1.0e-10:
            self._cuaev_name = "smooth"
        self.order = order
        self.eps = eps

    def forward(self, distances: Tensor, cutoff: float) -> Tensor:
        e = 1 - 1 / (1 - (distances / cutoff) ** self.order).clamp(min=self.eps)
        return torch.exp(e)

    def extra_repr(self) -> str:
        return f"order={self.order}, eps={self.eps:.1e}"


CutoffArg = tp.Union[tp.Literal["global", "dummy", "cosine", "smooth"], Cutoff]


def _parse_cutoff_fn(cutoff_fn: CutoffArg, global_cutoff: tp.Optional[Cutoff] = None) -> Cutoff:
    if cutoff_fn == "global":
        assert global_cutoff is not None
        cutoff_fn = global_cutoff
    if cutoff_fn == "dummy":
        cutoff_fn = CutoffDummy()
    elif cutoff_fn == "cosine":
        cutoff_fn = CutoffCosine()
    elif cutoff_fn == "smooth":
        cutoff_fn = CutoffSmooth()
    elif not isinstance(cutoff_fn, Cutoff):
        raise ValueError(f"Unsupported cutoff fn: {cutoff_fn}")
    return tp.cast(Cutoff, cutoff_fn)


# ---------------------------------------------------------------------------
# torchani/_core.py
# ---------------------------------------------------------------------------
class _ChemModule(torch.nn.Module):
    atomic_numbers: Tensor
    _conv_tensor: Tensor

    def __init__(self, symbols: tp.Sequence[str] = ()) -> None:
        super().__init__()
        atomic_numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)
        conv_tensor = -torch.ones(118 + 2, dtype=torch.long)
        for i, znum in enumerate(atomic_numbers):
            conv_tensor[znum] = i

        self.register_buffer("atomic_numbers", atomic_numbers, persistent=False)
        self.register_buffer("_conv_tensor", conv_tensor, persistent=False)

    @torch.jit.unused
    def _validate_elem_seq(
        self,
        name: str,
        seq: tp.Sequence[float],
        default: tp.Sequence[float] = (),
        pair: bool = False,
    ) -> tp.Sequence[float]:
        if not pair:
            if not seq and default:
                seq = [default[j] for j in self.atomic_numbers]

        if not all(isinstance(v, float) for v in seq):
            raise ValueError(f"Some values in {name} are not floats")
        num_elem = len(self.symbols)
        num_expect = num_elem if not pair else num_elem * (num_elem + 1) // 2
        if not len(seq) == num_expect:
            raise ValueError(f"{name} and symbols should have the same len")
        return seq

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)


# ---------------------------------------------------------------------------
# torchani/sae.py
# ---------------------------------------------------------------------------
class SelfEnergy(_ChemModule):
    self_energies: Tensor
    _enabled: bool

    def __init__(self, symbols: tp.Sequence[str], self_energies: tp.Sequence[float]):
        super().__init__(symbols)
        self_energies = self._validate_elem_seq("self_energies", self_energies)
        self.register_buffer("self_energies", torch.tensor(self_energies))
        self._enabled = True

    @staticmethod
    def _sorted_gsaes(symbols: tp.Sequence[str], functional: str, basis_set: str) -> tp.List[float]:
        gsaes = GSAES[f"{functional.lower()}-{basis_set.lower()}"]
        return [gsaes[e] for e in symbols]

    @classmethod
    def with_gsaes(cls, symbols: tp.Sequence[str], functional: str, basis_set: str):
        return cls(symbols, cls._sorted_gsaes(symbols, functional, basis_set))

    def forward(self, elem_idxs: Tensor, atomic: bool = False) -> Tensor:
        self_atomic_energies = self.self_energies[elem_idxs]
        self_atomic_energies = self_atomic_energies.masked_fill(elem_idxs == -1, 0.0)
        if atomic:
            return self_atomic_energies
        return self_atomic_energies.sum(dim=-1)


# ---------------------------------------------------------------------------
# torchani/neighbors.py (subset -- 'all_pairs' strategy only)
# ---------------------------------------------------------------------------
class Neighbors(tp.NamedTuple):
    indices: Tensor
    distances: Tensor
    diff_vectors: Tensor


class Triples(tp.NamedTuple):
    central_idxs: Tensor
    side_idxs: Tensor
    diff_signs: Tensor
    distances: Tensor
    diff_vectors: Tensor


def discard_inter_molecule_pairs(neighbors: Neighbors, molecule_idxs: Tensor) -> Neighbors:
    molecule_neighbor_idxs = molecule_idxs[neighbors.indices]
    internal_idxs = (
        (molecule_neighbor_idxs[0, :] == molecule_neighbor_idxs[1, :]).nonzero().view(-1)
    )
    indices = neighbors.indices.index_select(1, internal_idxs)
    distances = neighbors.distances.index_select(0, internal_idxs)
    diff_vectors = neighbors.diff_vectors.index_select(0, internal_idxs)
    return Neighbors(indices, distances, diff_vectors)


def discard_outside_cutoff(neighbors: Neighbors, cutoff: float) -> Neighbors:
    closer_indices = (neighbors.distances <= cutoff).nonzero().flatten()
    indices = neighbors.indices.index_select(1, closer_indices)
    distances = neighbors.distances.index_select(0, closer_indices)
    diff_vectors = neighbors.diff_vectors.index_select(0, closer_indices)
    return Neighbors(indices, distances, diff_vectors)


def narrow_down(
    cutoff: float,
    elem_idxs: Tensor,
    coords: Tensor,
    neighbor_idxs: Tensor,
    shifts: tp.Optional[Tensor] = None,
) -> Neighbors:
    mask = elem_idxs == -1
    if not torch.compiler.is_compiling() and mask.any():
        mask = mask.view(-1)[neighbor_idxs.view(-1)].view(2, -1)
        non_dummy_pairs = (~torch.any(mask, dim=0)).nonzero().flatten()
        neighbor_idxs = neighbor_idxs.index_select(1, non_dummy_pairs)
        if shifts is not None:
            shifts = shifts.index_select(0, non_dummy_pairs)

    coords = coords.view(-1, 3)
    if cutoff == math.inf:
        if shifts is not None:
            raise ValueError("PBC can't use an infinite cutoff")
    else:
        _coords = coords.detach()
        _coords0 = _coords.index_select(0, neighbor_idxs[0])
        _coords1 = _coords.index_select(0, neighbor_idxs[1])
        _diff_vectors = _coords0 - _coords1
        if shifts is not None:
            _diff_vectors += shifts
        in_cutoff = (_diff_vectors.norm(2, -1) <= cutoff).nonzero().flatten()
        neighbor_idxs = neighbor_idxs.index_select(1, in_cutoff)
        if shifts is not None:
            shifts = shifts.index_select(0, in_cutoff)

    coords0 = coords.index_select(0, neighbor_idxs[0])
    coords1 = coords.index_select(0, neighbor_idxs[1])
    diff_vectors = coords0 - coords1
    if shifts is not None:
        diff_vectors += shifts
    distances = diff_vectors.norm(2, -1)
    return Neighbors(neighbor_idxs, distances, diff_vectors)


class Neighborlist(torch.nn.Module):
    r"""Base class for modules that compute pairs of neighbors."""

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        raise NotImplementedError("Must be implemented by subclasses")


class AllPairs(Neighborlist):
    r"""Compute pairs of neighbors. Naive O(N^2) algorithm."""

    def forward(
        self,
        cutoff: float,
        species: Tensor,
        coords: Tensor,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Neighbors:
        return all_pairs(cutoff, species, coords, cell, pbc)


def all_pairs(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor] = None,
    pbc: tp.Optional[Tensor] = None,
) -> Neighbors:
    _validate_inputs(cutoff, species, coords, cell, pbc)

    if pbc is not None:
        raise NotImplementedError("PBC not vendored in this staging module")
    molecs, atoms = species.shape
    device = species.device
    neighbor_idxs = torch.triu_indices(atoms, atoms, 1, device=device)
    if molecs > 1:
        neighbor_idxs = neighbor_idxs.unsqueeze(1).repeat(1, molecs, 1)
        neighbor_idxs += atoms * torch.arange(molecs, device=device).view(1, -1, 1)
        neighbor_idxs = neighbor_idxs.view(-1).view(2, -1)
    return narrow_down(cutoff, species, coords, neighbor_idxs)


NeighborlistArg = tp.Union[
    tp.Literal["all_pairs", "base"],
    Neighborlist,
]


def _parse_neighborlist(neighborlist: NeighborlistArg = "base") -> Neighborlist:
    if neighborlist == "all_pairs":
        neighborlist = AllPairs()
    elif neighborlist == "base":
        neighborlist = Neighborlist()
    elif not isinstance(neighborlist, Neighborlist):
        raise ValueError(f"Unsupported neighborlist: {neighborlist}")
    return tp.cast(Neighborlist, neighborlist)


def _validate_inputs(
    cutoff: float,
    species: Tensor,
    coords: Tensor,
    cell: tp.Optional[Tensor],
    pbc: tp.Optional[Tensor],
    supports_batches: bool = True,
    supports_individual_pbc: bool = True,
):
    if torch.compiler.is_compiling() or torch.jit.is_scripting():
        return
    if cutoff <= 0.0:
        raise ValueError("Cutoff must be a strictly positive float")
    if not supports_batches and coords.shape[0] != 1:
        raise ValueError("This neighborlist doesn't support batches")
    if pbc is not None:
        if cell is None:
            raise ValueError("If pbc is not None, cell should be present")
    else:
        if cell is not None:
            raise ValueError("Cell is not supported if not using pbc")


def _unique_and_counts(sorted_flat_idxs: Tensor) -> tp.Tuple[Tensor, Tensor]:
    if torch.compiler.is_compiling():
        return torch.unique(sorted_flat_idxs, return_counts=True)
    return torch.unique_consecutive(sorted_flat_idxs, return_counts=True)


def neighbors_to_triples(neighbors: Neighbors) -> Triples:
    r"""Converts output of a neighborlist calculation into triples of atoms"""
    sorted_flat_idxs, rev_idxs = neighbors.indices.view(-1).sort()
    unique_central_idxs, counts = _unique_and_counts(sorted_flat_idxs)

    pair_sizes = (counts * (counts - 1)).div(2, rounding_mode="floor")
    pair_indices = torch.repeat_interleave(pair_sizes)
    central_idxs = unique_central_idxs.index_select(0, pair_indices)

    zcounts = torch.cat((counts.new_zeros(1), counts))
    dev = zcounts.device
    counts_max: int = int(zcounts.max())
    max_local_pairs = torch.tril_indices(counts_max, counts_max, -1, device=dev)
    mask = torch.arange(max_local_pairs.shape[1], device=dev) < pair_sizes.view(-1, 1)
    sort_local_pairs = max_local_pairs.repeat(1, pair_sizes.shape[0])[:, mask.view(-1)]
    sort_local_pairs += torch.cumsum(zcounts, dim=0).index_select(0, pair_indices)

    local_pairs = rev_idxs[sort_local_pairs]

    num_neigh = neighbors.indices.shape[1]
    sign12 = (local_pairs < num_neigh).to(torch.int8) * 2 - 1
    side_idxs = local_pairs % num_neigh

    flat_diff_vectors = neighbors.diff_vectors.index_select(0, side_idxs.view(-1))
    diff_vectors = flat_diff_vectors.view(2, -1, 3) * sign12.view(2, -1, 1)
    distances = neighbors.distances.index_select(0, side_idxs.view(-1)).view(2, -1)
    return Triples(central_idxs, side_idxs, sign12, distances, diff_vectors)


# ---------------------------------------------------------------------------
# torchani/aev/_terms.py
# ---------------------------------------------------------------------------
class _Term(torch.nn.Module):
    cutoff: float
    num_feats: int

    def __init__(self, cutoff: float, cutoff_fn: CutoffArg = "cosine") -> None:
        super().__init__()
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.cutoff = cutoff
        self.num_feats = 0


class _BaseAngular(_Term):
    def forward(self, tri_distances: Tensor, tri_vectors: Tensor) -> Tensor:
        assert tri_vectors.shape == (2, tri_distances.shape[1], 3)
        assert tri_distances.shape == (2, tri_distances.shape[1])

        tri_factor = self.cutoff_fn(tri_distances, self.cutoff)

        tri_vectors = tri_vectors.view(2, -1, 3, 1)
        tri_distances = tri_distances.view(2, -1, 1)
        cos_angles = (tri_vectors[0] * tri_vectors[1]).sum(1) / torch.clamp(
            tri_distances[0] * tri_distances[1], min=1e-10
        )
        _angular = self.compute_cos_angles(cos_angles)
        _radial = self.compute_radial(tri_distances[0], tri_distances[1])
        terms = (_radial.unsqueeze(2) * _angular.unsqueeze(1)).view(-1, self.num_feats)
        return terms * (tri_factor[0] * tri_factor[1]).view(-1, 1)

    def compute_radial(self, distances_ji: Tensor, distances_jk) -> Tensor:
        raise NotImplementedError

    def compute_cos_angles(self, cos_angles: Tensor) -> Tensor:
        raise NotImplementedError


class _BaseRadial(_Term):
    def forward(self, distances: Tensor) -> Tensor:
        assert distances.shape == (distances.shape[0],)
        factor = self.cutoff_fn(distances, self.cutoff).view(-1, 1)
        return self.compute(distances.view(-1, 1)) * factor

    def compute(self, distances: Tensor) -> Tensor:
        raise NotImplementedError


class ANIRadial(_BaseRadial):
    shifts: Tensor
    eta: Tensor

    def __init__(
        self, eta: float, shifts: tp.Sequence[float], cutoff: float, cutoff_fn: CutoffArg = "cosine"
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        dtype = torch.float
        self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.num_feats = len(shifts)

    def compute(self, distances: Tensor) -> Tensor:
        return 0.25 * torch.exp(-self.eta * (distances - self.shifts.view(1, -1)) ** 2)

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 5.2,
        eta: float = 19.7,
        num_shifts: int = 16,
        cutoff_fn: CutoffArg = "cosine",
    ):
        shifts = linspace(start, cutoff, num_shifts)
        return cls(eta, shifts, cutoff, cutoff_fn)


class ANIAngular(_BaseAngular):
    shifts: Tensor
    eta: Tensor
    zeta: Tensor
    sections: Tensor

    def __init__(
        self,
        eta: float,
        zeta: float,
        shifts: tp.Sequence[float],
        sections: tp.Sequence[float],
        cutoff: float,
        cutoff_fn: CutoffArg = "cosine",
    ):
        super().__init__(cutoff=cutoff, cutoff_fn=cutoff_fn)
        dtype = torch.float
        self.register_buffer("eta", torch.tensor([eta], dtype=dtype))
        self.register_buffer("zeta", torch.tensor([zeta], dtype=dtype))
        self.register_buffer("shifts", torch.tensor(shifts, dtype=dtype))
        self.register_buffer("sections", torch.tensor(sections, dtype=dtype))
        self.num_feats = len(shifts) * len(sections)

    def compute_radial(self, distances_ji: Tensor, distances_jk) -> Tensor:
        mean_dists = (distances_ji + distances_jk) / 2
        return torch.exp(-self.eta * (mean_dists - self.shifts.view(1, -1)) ** 2)

    def compute_cos_angles(self, cos_angles: Tensor) -> Tensor:
        angles = torch.acos(0.95 * cos_angles)
        angle_deviations = angles - self.sections.view(1, -1)
        return 2 * ((1 + torch.cos(angle_deviations)) / 2) ** self.zeta

    @classmethod
    def cover_linearly(
        cls,
        start: float = 0.9,
        cutoff: float = 3.5,
        eta: float = 12.5,
        zeta: float = 14.1,
        num_shifts: int = 8,
        num_sections: int = 4,
        cutoff_fn: CutoffArg = "cosine",
    ):
        shifts = linspace(start, cutoff, num_shifts)
        angle_start = math.pi / num_sections / 2
        sections = linspace(angle_start, math.pi + angle_start, num_sections)
        return cls(eta, zeta, shifts, sections, cutoff, cutoff_fn)


_Models = tp.Literal["ani1x", "ani2x", "ani1ccx"]
AngularArg = tp.Union[_Models, _BaseAngular]
RadialArg = tp.Union[_Models, _BaseRadial]


def _parse_angular_term(angular_term: AngularArg) -> _BaseAngular:
    if not isinstance(angular_term, _BaseAngular):
        raise ValueError(f"Unsupported angular term: {angular_term}")
    return angular_term


def _parse_radial_term(radial_term: RadialArg) -> _BaseRadial:
    if not isinstance(radial_term, _BaseRadial):
        raise ValueError(f"Unsupported radial term: {radial_term}")
    return radial_term


# ---------------------------------------------------------------------------
# torchani/aev/_computer.py (pyAEV strategy path; cuAEV branches kept but always
# no-op since CUAEV_IS_INSTALLED is False without the compiled CUDA extension)
# ---------------------------------------------------------------------------
CUAEV_IS_INSTALLED = False


class AEVComputer(torch.nn.Module):
    r"""Computes local atomic features (AEVs) given a batch of molecules."""

    num_species: Final[int]
    num_species_pairs: Final[int]
    angular_len: Final[int]
    radial_len: Final[int]
    out_dim: Final[int]

    triu_index: Tensor
    _strategy: str

    def __init__(
        self,
        radial: RadialArg,
        angular: AngularArg,
        num_species: int,
        strategy: str = "pyaev",
        cutoff_fn: tp.Optional[CutoffArg] = None,
        neighborlist: NeighborlistArg = "all_pairs",
    ):
        super().__init__()
        self.num_species = num_species
        self.num_species_pairs = num_species * (num_species + 1) // 2
        self.register_buffer("triu_index", self._calculate_triu_index(num_species))

        self.radial = _parse_radial_term(radial)
        self.angular = _parse_angular_term(angular)
        if cutoff_fn is not None:
            _cutoff_fn = _parse_cutoff_fn(cutoff_fn)
            self.radial.cutoff_fn = _cutoff_fn
            self.angular.cutoff_fn = _cutoff_fn

        if not (self.angular.cutoff_fn.is_same(self.radial.cutoff_fn)):
            raise ValueError("Cutoff fn must be the same for angular and radial terms")
        if self.angular.cutoff > self.radial.cutoff:
            raise ValueError("Angular cutoff should be smaller than radial cutoff")

        self.neighborlist = _parse_neighborlist(neighborlist)

        self.radial_len = self.radial.num_feats * self.num_species
        self.angular_len = self.angular.num_feats * self.num_species_pairs
        self.out_dim = self.radial_len + self.angular_len

        if strategy != "pyaev":
            raise ValueError("Only 'pyaev' strategy is vendored in this staging module")
        self._strategy = strategy

    @staticmethod
    def _calculate_triu_index(num_species: int) -> Tensor:
        species1, species2 = torch.triu_indices(num_species, num_species).unbind(0)
        pair_index = torch.arange(species1.shape[0], dtype=torch.long)
        ret = torch.zeros(num_species, num_species, dtype=torch.long)
        ret[species1, species2] = pair_index
        ret[species2, species1] = pair_index
        return ret

    def forward(
        self,
        elem_idxs: Tensor,
        coords: tp.Optional[Tensor] = None,
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
    ) -> Tensor:
        assert coords is not None
        assert elem_idxs.dim() == 2
        assert coords.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)
        assert self.angular.cutoff < self.radial.cutoff

        neighbors = self.neighborlist(self.radial.cutoff, elem_idxs, coords, cell, pbc)
        return self._pyaev_compute_from_neighbors(elem_idxs, coords, neighbors)

    def compute_from_neighbors(
        self, elem_idxs: Tensor, coords: Tensor, neighbors: Neighbors
    ) -> Tensor:
        return self._pyaev_compute_from_neighbors(elem_idxs, coords, neighbors)

    def _pyaev_compute_from_neighbors(
        self, elem_idxs: Tensor, coords: Tensor, neighbors: Neighbors
    ) -> Tensor:
        terms = self.radial(neighbors.distances)
        radial_aev = self._collect_radial(elem_idxs, neighbors.indices, terms)

        neighbors = discard_outside_cutoff(neighbors, self.angular.cutoff)
        triples = neighbors_to_triples(neighbors)

        terms = self.angular(triples.distances, triples.diff_vectors)
        angular_aev = self._collect_angular(
            elem_idxs,
            neighbors.indices,
            triples.central_idxs,
            triples.side_idxs,
            triples.diff_signs,
            terms,
        )
        return torch.cat([radial_aev, angular_aev], dim=-1)

    def _collect_angular(
        self,
        elem_idxs: Tensor,
        neighbor_idxs: Tensor,
        central_idx: Tensor,
        side_idxs: Tensor,
        sign12: Tensor,
        terms: Tensor,
    ) -> Tensor:
        num_molecs, num_atoms = elem_idxs.shape
        neighbor_elem_idxs = elem_idxs.view(-1)[neighbor_idxs]

        species12_small = neighbor_elem_idxs[:, side_idxs]
        triple_element_side_idxs = torch.where(sign12 == 1, species12_small[1], species12_small[0])
        angular_aev = terms.new_zeros(
            (num_molecs * num_atoms * self.num_species_pairs, self.angular.num_feats)
        )
        index = central_idx * self.num_species_pairs + self.triu_index[
            triple_element_side_idxs[0], triple_element_side_idxs[1]
        ].to(torch.long)
        angular_aev.index_add_(0, index, terms)
        return angular_aev.reshape(num_molecs, num_atoms, self.angular_len)

    def _collect_radial(self, elem_idxs: Tensor, neighbor_idxs: Tensor, terms: Tensor) -> Tensor:
        num_molecs, num_atoms = elem_idxs.shape
        neighbor_elem_idxs = elem_idxs.view(-1)[neighbor_idxs]
        radial_aev = terms.new_zeros(
            (num_molecs * num_atoms * self.num_species, self.radial.num_feats)
        )
        index12 = neighbor_idxs * self.num_species + neighbor_elem_idxs.flip(0)
        radial_aev.index_add_(0, index12[0], terms)
        radial_aev.index_add_(0, index12[1], terms)
        return radial_aev.reshape(num_molecs, num_atoms, self.radial_len)


# ---------------------------------------------------------------------------
# torchani/nn/_core.py
# ---------------------------------------------------------------------------
class AtomicContainer(torch.nn.Module):
    r"""Base class for ANI modules that contain Atomic Neural Networks"""

    num_species: int
    total_members_num: int
    active_members_idxs: tp.List[int]
    atomic_numbers: Tensor

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.total_members_num = 1
        self.active_members_idxs = [0]
        self.num_species = 0
        atomic_numbers = torch.tensor([0], dtype=torch.long)
        self.register_buffer("atomic_numbers", atomic_numbers, persistent=False)

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert aevs is not None
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @torch.jit.export
    def get_active_members_num(self) -> int:
        return len(self.active_members_idxs)

    @torch.jit.export
    def set_active_members(self, idxs: tp.List[int]) -> None:
        for idx in idxs:
            if not (0 <= idx < self.total_members_num):
                raise IndexError(f"Idx {idx} should be 0 <= idx < {self.total_members_num}")
        self.active_members_idxs = idxs

    @torch.jit.unused
    def to_infer_model(self, use_mnp: bool = False) -> "AtomicContainer":
        return self


class AtomicNetwork(torch.nn.Module):
    def __init__(
        self,
        layer_dims: tp.Sequence[int],
        activation: tp.Union[str, torch.nn.Module] = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if any(d <= 0 for d in layer_dims):
            raise ValueError("Layer dims must be strict positive integers")

        dims = tuple(layer_dims)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(_in, _out, bias=bias) for _in, _out in zip(dims[:-2], dims[1:-1])]
        )
        self.final_layer = torch.nn.Linear(dims[-2], dims[-1], bias=bias)
        self.activation = parse_activation(activation)
        self.has_biases = bias

    def __getitem__(self, idx: int) -> torch.nn.Module:
        if idx in [-1, len(self.layers)]:
            return self.final_layer
        if idx < -1:
            idx += 1
        return self.layers[idx]

    def forward(self, features: Tensor) -> Tensor:
        for layer in self.layers:
            features = self.activation(layer(features))
        return self.final_layer(features)


class TightCELU(torch.nn.Module):
    r"""CELU activation function with alpha=0.1"""

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.celu(x, alpha=0.1)


def parse_activation(module: tp.Union[str, torch.nn.Module]) -> torch.nn.Module:
    if module == "gelu":
        return torch.nn.GELU()
    if module == "celu":
        return TightCELU()
    assert isinstance(module, torch.nn.Module)
    return module


# ---------------------------------------------------------------------------
# torchani/nn/_containers.py (subset: ANINetworks + SpeciesConverter)
# ---------------------------------------------------------------------------
class ANINetworks(AtomicContainer):
    r"""Predict molecular or atomic scalars from a set of element-specific networks"""

    def __init__(self, modules: tp.Dict[str, AtomicNetwork], alias: bool = False):
        super().__init__()
        if any(s not in PERIODIC_TABLE for s in modules):
            raise ValueError("All modules should be mapped to valid chemical symbols")
        if not alias and len(set(id(m) for m in modules.values())) != len(modules):
            raise ValueError("Symbols map to same module. If intended use `alias=True`")
        self.atomics = torch.nn.ModuleDict(modules)
        self.num_species = len(self.atomics)
        atomic_numbers = torch.tensor([ATOMIC_NUMBER[e] for e in modules], dtype=torch.long)
        self.register_buffer("atomic_numbers", atomic_numbers, persistent=False)

        final_layer = next(iter(self.atomics.values())).final_layer
        self.out_dim: int = final_layer.out_features  # type: ignore

    def __getitem__(self, idx: str) -> AtomicNetwork:
        return tp.cast(AtomicNetwork, self.atomics[idx])

    def forward(
        self,
        elem_idxs: Tensor,
        aevs: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert aevs is not None
        assert elem_idxs.shape == aevs.shape[:-1]
        flat_elem_idxs = elem_idxs.flatten()
        aev = aevs.flatten(0, 1)
        scalars = aev.new_zeros(flat_elem_idxs.shape + (self.out_dim,))
        for i, m in enumerate(self.atomics.values()):
            selected_idxs = (flat_elem_idxs == i).nonzero().view(-1)
            if selected_idxs.shape[0] > 0:
                input_ = aev.index_select(0, selected_idxs)
                scalars.index_add_(0, selected_idxs, m(input_).view(-1, self.out_dim))
        scalars = scalars.view(elem_idxs.shape[0], elem_idxs.shape[1], self.out_dim)
        scalars = scalars.squeeze(-1)
        if atomic:
            return scalars
        return scalars.sum(dim=1)

    @classmethod
    def build(
        cls,
        symbols: tp.Sequence[str],
        in_dim: int,
        dims: tp.Dict[str, tp.Tuple[int, ...]],
        out_dim: int = 1,
        activation: tp.Union[str, torch.nn.Module] = "gelu",
        bias: bool = False,
        default_dims: tp.Tuple[int, ...] = (),
    ):
        modules: tp.Dict[str, AtomicNetwork] = {}
        for s in symbols:
            layer_dims = (in_dim,) + dims.get(s, default_dims) + (out_dim,)
            modules[s] = AtomicNetwork(layer_dims=layer_dims, activation=activation, bias=bias)
        return cls(modules)

    @classmethod
    def like_2x(
        cls,
        symbols: tp.Sequence[str] = ("H", "C", "N", "O", "S", "F", "Cl"),
        in_dim: int = 1008,
        out_dim: int = 1,
        activation: tp.Union[str, torch.nn.Module] = "celu",
        bias: bool = True,
    ):
        default_dims = (160, 128, 96)
        dims: tp.Dict[str, tp.Tuple[int, ...]] = {
            "H": (256, 192, 160),
            "C": (224, 192, 160),
            "N": (192, 160, 128),
            "O": (192, 160, 128),
            "S": (160, 128, 96),
            "F": (160, 128, 96),
            "Cl": (160, 128, 96),
        }
        return cls.build(
            symbols,
            in_dim,
            dims,
            out_dim=out_dim,
            bias=bias,
            activation=activation,
            default_dims=default_dims,
        )

    @classmethod
    def default(
        cls,
        symbols: tp.Sequence[str],
        in_dim: int,
        out_dim: int = 1,
        activation: tp.Union[str, torch.nn.Module] = "gelu",
        bias: bool = False,
    ):
        return cls.like_2x(symbols, in_dim, out_dim, activation, bias)


class SpeciesConverter(torch.nn.Module):
    r"""Convert atomic numbers into internal ANI element indices"""

    conv_tensor: Tensor

    def __init__(self, symbols: tp.Sequence[str]):
        super().__init__()
        if isinstance(symbols, str):
            raise ValueError("Please use 'SpeciesConverter(['H', 'C', ...])' instead")
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer("conv_tensor", torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(symbols):
            self.conv_tensor[rev_idx[s]] = i
        self.atomic_numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)

    def forward(self, atomic_nums: Tensor, nop: bool = False) -> Tensor:
        if nop:
            if atomic_nums.max() >= len(self.atomic_numbers):
                raise ValueError(f"Unsupported element idx in {atomic_nums}")
            return atomic_nums

        elem_idxs = self.conv_tensor.to(torch.long)[atomic_nums]
        if (elem_idxs[atomic_nums != -1] == -1).any():
            raise ValueError(
                f"Model doesn't support some elements in input"
                f" Input elements include: {torch.unique(atomic_nums)}"
                f" Supported elements are: {self.atomic_numbers}"
            )
        return elem_idxs.to(atomic_nums.device)


# ---------------------------------------------------------------------------
# torchani/nn/_internal.py (subset: _ZeroANINetworks, used internally by Assembler)
# ---------------------------------------------------------------------------
class _ZeroANINetworks(ANINetworks):
    def forward(
        self,
        elem_idxs: Tensor,
        aevs: tp.Optional[Tensor] = None,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> Tensor:
        assert aevs is not None
        if atomic:
            return aevs.new_zeros(elem_idxs.shape)
        return aevs.new_zeros(elem_idxs.shape[0])


# ---------------------------------------------------------------------------
# torchani/potentials/core.py (subset: Potential, BasePairPotential)
# ---------------------------------------------------------------------------
class Potential(_ChemModule):
    r"""Base class for all atomic potentials"""

    ANGSTROM_TO_BOHR: float
    cutoff: float
    _enabled: bool

    def __init__(self, symbols: tp.Sequence[str], *, cutoff: float = math.inf):
        super().__init__(symbols)
        self.cutoff = cutoff
        self.ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR
        self._enabled = True

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        raise NotImplementedError("Must be implemented by subclasses")


class BasePairPotential(Potential):
    r"""General base class for all pairwise potentials"""

    def __init__(
        self,
        symbols: tp.Sequence[str],
        *,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff)
        if cutoff != math.inf:
            self.cutoff_fn = _parse_cutoff_fn(cutoff_fn)
        else:
            self.cutoff_fn = CutoffDummy()

    @staticmethod
    def clamp(distances: Tensor) -> Tensor:
        return distances.clamp(min=1e-7)

    def pair_energies(self, elem_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        raise NotImplementedError

    def _pair_energies_wrapper(
        self, elem_idxs: Tensor, neighbors: Neighbors, ghost_flags: tp.Optional[Tensor] = None
    ) -> Tensor:
        pair_energies = self.pair_energies(elem_idxs, neighbors)
        pair_energies = pair_energies * self.cutoff_fn(neighbors.distances, self.cutoff)

        if ghost_flags is not None:
            ghost12 = ghost_flags.view(-1)[neighbors.indices]
            ghost_mask = torch.logical_or(ghost12[0], ghost12[1])
            pair_energies = torch.where(ghost_mask, pair_energies * 0.5, pair_energies)
        return pair_energies

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        pair_energies = self._pair_energies_wrapper(elem_idxs, neighbors, ghost_flags)
        molecs_num, atoms_num = elem_idxs.shape
        if atomic:
            energies = neighbors.distances.new_zeros(molecs_num * atoms_num)
            energies.index_add_(0, neighbors.indices[0], pair_energies / 2)
            energies.index_add_(0, neighbors.indices[1], pair_energies / 2)
            energies = energies.view(molecs_num, atoms_num)
        else:
            energies = neighbors.distances.new_zeros(molecs_num)
            molecs_idxs = torch.div(neighbors.indices[0], elem_idxs.shape[1], rounding_mode="floor")
            energies.index_add_(0, molecs_idxs, pair_energies)
        return EnergiesScalars(energies)


# ---------------------------------------------------------------------------
# torchani/potentials/xtb.py -- the paper's short-range repulsion potential
# ---------------------------------------------------------------------------
class RepulsionXTB(BasePairPotential):
    r"""Calculates the xTB repulsion energy terms for a given molecule (Grimme et al.)"""

    y_ab: Tensor
    sqrt_alpha_ab: Tensor
    k_rep_ab: Tensor

    def __init__(
        self,
        symbols: tp.Sequence[str],
        krep_hydrogen: float = 1.0,
        krep: float = 1.5,
        alpha: tp.Sequence[float] = (),
        yeff: tp.Sequence[float] = (),
        *,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "smooth",
    ):
        super().__init__(symbols, cutoff=cutoff, cutoff_fn=cutoff_fn)
        alpha = self._validate_elem_seq("alpha", alpha, XTB_REPULSION_ALPHA)
        yeff = self._validate_elem_seq("yeff", yeff, XTB_REPULSION_YEFF)

        num_elem = len(symbols)
        k_rep_ab = torch.full((num_elem, num_elem), krep)
        if 1 in self.atomic_numbers:
            hydrogen_idx = (self.atomic_numbers == 1).nonzero().view(-1)
            k_rep_ab[hydrogen_idx, hydrogen_idx] = krep_hydrogen

        _yeff = torch.tensor(yeff)
        self.register_buffer("y_ab", torch.outer(_yeff, _yeff))
        _alpha = torch.tensor(alpha)
        self.register_buffer("sqrt_alpha_ab", torch.outer(_alpha, _alpha).sqrt())
        self.register_buffer("k_rep_ab", k_rep_ab)

    def pair_energies(self, element_idxs: Tensor, neighbors: Neighbors) -> Tensor:
        dists = self.clamp(neighbors.distances) * self.ANGSTROM_TO_BOHR
        species12 = element_idxs.view(-1)[neighbors.indices]
        y_ab = self.y_ab[species12[0], species12[1]]
        sqrt_alpha_ab = self.sqrt_alpha_ab[species12[0], species12[1]]
        k_rep_ab = self.k_rep_ab[species12[0], species12[1]]
        return (y_ab / dists) * torch.exp(-sqrt_alpha_ab * (dists**k_rep_ab))


# ---------------------------------------------------------------------------
# torchani/potentials/nnp.py -- adapts AEVComputer+ANINetworks as a Potential
# ---------------------------------------------------------------------------
class NNPotential(Potential):
    def __init__(self, aev_computer: AEVComputer, neural_networks: AtomicContainer):
        super().__init__(neural_networks.symbols, cutoff=aev_computer.radial.cutoff)
        self.aev_computer = aev_computer
        self.neural_networks = neural_networks

    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        ghost_flags: tp.Optional[Tensor] = None,
    ) -> EnergiesScalars:
        aevs = self.aev_computer.compute_from_neighbors(elem_idxs, coords, neighbors)
        energies = self.neural_networks(elem_idxs, aevs, atomic, ensemble_values)
        return EnergiesScalars(energies)


# ---------------------------------------------------------------------------
# torchani/arch.py -- the ANI model class and its Assembler/simple_ani builder
# ---------------------------------------------------------------------------
AEVComputerCls = tp.Type[AEVComputer]
PotentialCls = tp.Type[Potential]
AtomicContainerCls = tp.Type[AtomicContainer]
ModelCls = tp.Type["ANI"]


class _ANI(torch.nn.Module):
    cutoff: float
    atomic_numbers: Tensor
    periodic_table_index: bool

    def __init__(
        self,
        symbols: tp.Sequence[str],
        aev_computer: AEVComputer,
        neural_networks: AtomicContainer,
        energy_shifter: SelfEnergy,
        potentials: tp.Optional[tp.Dict[str, Potential]] = None,
        periodic_table_index: bool = True,
    ):
        super().__init__()
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in symbols], dtype=torch.long)
        self.register_buffer("atomic_numbers", numbers)

        assert len(energy_shifter.self_energies) == len(self.atomic_numbers)
        assert aev_computer.num_species == len(self.atomic_numbers)
        assert neural_networks.num_species == len(self.atomic_numbers)

        self.neighborlist = aev_computer.neighborlist

        device = energy_shifter.self_energies.device
        self.energy_shifter = energy_shifter
        self.species_converter = SpeciesConverter(symbols).to(device)

        self.potentials = torch.nn.ModuleDict(potentials or {})
        self.potentials["nnp"] = NNPotential(aev_computer, neural_networks)
        self.cutoff = max(p.cutoff for p in self.potentials.values())  # type: ignore
        self.periodic_table_index = periodic_table_index
        self._has_extra_potentials = self._check_has_extra_potentials()

    def _check_has_extra_potentials(self) -> bool:
        return any(p._enabled for k, p in self.potentials.items() if k != "nnp")

    @property
    @torch.jit.unused
    def symbols(self) -> tp.Tuple[str, ...]:
        return tuple(PERIODIC_TABLE[z] for z in self.atomic_numbers)

    @property
    def nnp(self):
        return self.potentials["nnp"]

    @property
    def neural_networks(self):  # type: ignore
        return self.nnp.neural_networks

    @property
    def aev_computer(self):
        return self.nnp.aev_computer

    @staticmethod
    def _check_inputs(elem_idxs: Tensor, coords: Tensor, charge: int = 0) -> None:
        assert elem_idxs.dim() == 2
        assert coords.shape == (elem_idxs.shape[0], elem_idxs.shape[1], 3)
        assert charge == 0, "Model only supports neutral molecules"


class ANI(_ANI):
    r"""ANI-style neural network interatomic potential"""

    def forward(
        self,
        species_coordinates: tp.Tuple[Tensor, Tensor],
        cell: tp.Optional[Tensor] = None,
        pbc: tp.Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: tp.Optional[Tensor] = None,
    ) -> SpeciesEnergies:
        species, coords = species_coordinates
        self._check_inputs(species, coords, charge)
        elem_idxs = self.species_converter(species, nop=not self.periodic_table_index)
        if not self._has_extra_potentials and pbc is None and _molecule_idxs is None:
            energies = coords.new_zeros(elem_idxs.shape[0])
            if atomic:
                energies = energies.unsqueeze(1)
            if ensemble_values:
                energies = energies.unsqueeze(0)
            if self.potentials["nnp"]._enabled:
                aevs = self.potentials["nnp"].aev_computer(elem_idxs, coords, cell, pbc)  # type: ignore[operator]
                energies = energies + self.potentials["nnp"].neural_networks(
                    elem_idxs, aevs, atomic, ensemble_values
                )  # type: ignore[operator]
            if self.energy_shifter._enabled:
                energies = energies + self.energy_shifter(elem_idxs, atomic=atomic)
            return SpeciesEnergies(elem_idxs, energies)

        neighbors = self.neighborlist(self.cutoff, elem_idxs, coords, cell, pbc)
        if _molecule_idxs is not None:
            if coords.shape[0] != 1:
                raise ValueError("molecule_idxs expects only one conformation")
            neighbors = discard_inter_molecule_pairs(neighbors, _molecule_idxs)

        result = self.compute_from_neighbors(
            elem_idxs, coords, neighbors, charge, atomic, ensemble_values
        )
        return SpeciesEnergies(elem_idxs, result.energies)

    @torch.jit.export
    def compute_from_neighbors(
        self,
        elem_idxs: Tensor,
        coords: Tensor,
        neighbors: Neighbors,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
    ) -> EnergiesScalars:
        self._check_inputs(elem_idxs, coords, charge)
        energies = coords.new_zeros(elem_idxs.shape[0])
        if atomic:
            energies = energies.unsqueeze(1)
        if ensemble_values:
            energies = energies.unsqueeze(0)
        first_neighbors = neighbors
        for pot in self.potentials.values():
            if pot._enabled:
                neighbors = discard_outside_cutoff(first_neighbors, pot.cutoff)  # type: ignore[arg-type]
                result = pot.compute_from_neighbors(
                    elem_idxs, coords, neighbors, charge, atomic, ensemble_values
                )  # type: ignore[operator]
                energies = energies + result.energies
        if self.energy_shifter._enabled:
            energies = energies + self.energy_shifter(elem_idxs, atomic=atomic)
        return EnergiesScalars(energies)


@dataclass
class _AEVComputerWrapper:
    def __init__(
        self,
        cls,
        radial: RadialArg,
        angular: AngularArg,
        cutoff_fn: CutoffArg = "global",
        strategy: str = "pyaev",
    ) -> None:
        self.cls = cls
        self.cutoff_fn = cutoff_fn
        self.radial = _parse_radial_term(radial)
        self.angular = _parse_angular_term(angular)
        if self.angular.cutoff > self.radial.cutoff:
            raise ValueError("Angular cutoff must be smaller or equal to radial cutoff")
        if self.angular.cutoff <= 0 or self.radial.cutoff <= 0:
            raise ValueError("Cutoffs must be strictly positive")
        self.strategy = strategy


@dataclass
class _AtomicContainerWrapper:
    cls: AtomicContainerCls
    ctor: str = "separate_networks"
    kwargs: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def make(self, symbols: tp.Sequence[str], in_dim: int) -> AtomicContainer:
        out = getattr(self.cls, self.ctor)(symbols, in_dim, **self.kwargs)
        return tp.cast(AtomicContainer, out)


@dataclass
class _PotentialWrapper:
    cls: PotentialCls
    kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
    cutoff: float = math.inf
    cutoff_fn: CutoffArg = "global"


class Assembler:
    r"""Assembles an `ANI` model"""

    def __init__(
        self,
        symbols: tp.Sequence[str] = (),
        cls: ModelCls = ANI,
        neighborlist: NeighborlistArg = "all_pairs",
        periodic_table_index: bool = True,
    ) -> None:
        self._global_cutoff_fn: tp.Optional[Cutoff] = CutoffSmooth(2)
        self._neighborlist = _parse_neighborlist(neighborlist)
        self._aevcomp: tp.Optional[_AEVComputerWrapper] = None
        self._potentials: tp.Dict[str, _PotentialWrapper] = {}
        self._self_energies: tp.Dict[str, float] = {}
        self._container: tp.Optional[_AtomicContainerWrapper] = None
        self._symbols: tp.Tuple[str, ...] = tuple(symbols)
        self._cls: ModelCls = cls
        self.periodic_table_index = periodic_table_index

    @property
    def symbols(self) -> tp.Tuple[str, ...]:
        return self._symbols

    def set_symbols(self, symbols: tp.Sequence[str]) -> None:
        self._symbols = tuple(symbols)

    @property
    def self_energies(self) -> tp.Dict[str, float]:
        if not self._self_energies:
            raise RuntimeError("Self energies have not been set")
        return self._self_energies

    def set_self_energies(self, value: tp.Mapping[str, float]) -> None:
        self._check_symbols(value.keys())
        self._self_energies = {k: v for k, v in value.items()}

    def set_zeros_as_self_energies(self) -> None:
        self._check_symbols()
        self.set_self_energies({s: 0.0 for s in self.symbols})

    def set_gsaes_as_self_energies(self, lot: str = "") -> None:
        self._check_symbols()
        gsaes = GSAES[lot.lower()]
        self.set_self_energies({s: gsaes[s] for s in self.symbols})

    def _check_symbols(self, symbols: tp.Optional[tp.Iterable[str]] = None) -> None:
        if not self.symbols:
            raise ValueError("Please set symbols before setting the gsaes as self energies")
        if symbols is not None:
            if set(self.symbols) != set(symbols):
                raise ValueError(f"Passed symbols don't match supported elements {self._symbols}")

    def set_atomic_networks(
        self,
        cls: AtomicContainerCls = ANINetworks,
        ctor: str = "ani2x",
        kwargs: tp.Dict[str, tp.Any] = {},
    ) -> None:
        ctor = {"ani2x": "like_2x"}.get(ctor, ctor)
        self._container = _AtomicContainerWrapper(cls, ctor, kwargs)

    @property
    def atomic_networks_container(self) -> _AtomicContainerWrapper:
        if self._container is None:
            raise RuntimeError("Call 'set_atomic_networks(...)' before assembly")
        return self._container

    def set_aev_computer(
        self,
        angular: AngularArg,
        radial: RadialArg,
        cutoff_fn: CutoffArg = "global",
        strategy: str = "pyaev",
        aev_computer_cls=AEVComputer,
    ) -> None:
        self._aevcomp = _AEVComputerWrapper(
            aev_computer_cls, cutoff_fn=cutoff_fn, angular=angular, radial=radial, strategy=strategy
        )

    def set_neighborlist(self, neighborlist: NeighborlistArg) -> None:
        self._neighborlist = _parse_neighborlist(neighborlist)

    def set_global_cutoff_fn(self, cutoff_fn: CutoffArg) -> None:
        self._global_cutoff_fn = _parse_cutoff_fn(cutoff_fn)

    def add_potential(
        self,
        cls: PotentialCls,
        name: str,
        cutoff: float = math.inf,
        cutoff_fn: CutoffArg = "global",
        kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        if name in self._potentials:
            raise ValueError("Potential names must be unique")
        self._potentials[name] = _PotentialWrapper(
            cls, kwargs=kwargs, cutoff=cutoff, cutoff_fn=cutoff_fn
        )

    def assemble(self, ensemble_size: int = 1) -> "ANI":
        if ensemble_size < 0:
            raise ValueError("Ensemble size must be positive")
        if not self.symbols:
            raise RuntimeError("Symbols not set. Call 'set_symbols()' before assembly")
        if self._aevcomp is None:
            raise RuntimeError("AEVComputer not set. Call 'set_aev_computer' before assembly")

        feat_cutoff_fn = _parse_cutoff_fn(self._aevcomp.cutoff_fn, self._global_cutoff_fn)
        self._aevcomp.angular.cutoff_fn = feat_cutoff_fn
        self._aevcomp.radial.cutoff_fn = feat_cutoff_fn
        aevcomp = self._aevcomp.cls(
            neighborlist=self._neighborlist,
            cutoff_fn=feat_cutoff_fn,
            angular=self._aevcomp.angular,
            radial=self._aevcomp.radial,
            num_species=len(self.symbols),
            strategy=self._aevcomp.strategy,
        )

        if ensemble_size != 1:
            raise ValueError("Only ensemble_size=1 is vendored in this staging module")
        neural_networks: AtomicContainer = self.atomic_networks_container.make(
            self.symbols, aevcomp.out_dim
        )

        self_energies = self.self_energies
        shifter = SelfEnergy(
            symbols=self.symbols, self_energies=tuple(self_energies[k] for k in self.symbols)
        )
        kwargs: tp.Dict[str, tp.Any] = {}
        if self._potentials:
            potentials: tp.Dict[str, Potential] = {}
            for pot_name, pot in self._potentials.items():
                pot_kwargs = pot.kwargs if pot.kwargs is not None else {}
                potentials[pot_name] = pot.cls(
                    symbols=self.symbols,
                    **pot_kwargs,
                    cutoff=pot.cutoff,
                    cutoff_fn=_parse_cutoff_fn(pot.cutoff_fn, self._global_cutoff_fn),
                )
            kwargs.update({"potentials": potentials})

        return self._cls(
            symbols=self.symbols,
            aev_computer=aevcomp,
            energy_shifter=shifter,
            neural_networks=neural_networks,
            periodic_table_index=self.periodic_table_index,
            **kwargs,
        )


def simple_ani(
    symbols: tp.Sequence[str],
    lot: str,
    radial_start: float = 0.9,
    angular_start: float = 0.9,
    radial_cutoff: float = 5.2,
    angular_cutoff: float = 3.5,
    radial_shifts: int = 16,
    angular_shifts: int = 8,
    sections: int = 4,
    radial_precision: float = 19.7,
    angular_precision: float = 12.5,
    angular_zeta: float = 14.1,
    cutoff_fn: CutoffArg = "smooth",
    repulsion: bool = True,
    container: str = "ANINetworks",
    activation: tp.Union[str, torch.nn.Module] = "gelu",
    bias: bool = False,
    strategy: str = "auto",
    periodic_table_index: bool = True,
    neighborlist: NeighborlistArg = "all_pairs",
    repulsion_cutoff: tp.Union[bool, float] = True,
    self_energies: tp.Union[tp.Optional[tp.Dict[str, float]], tp.Literal["zero"]] = None,
) -> ANI:
    r"""Flexible builder to create ANI-style models that predict energies

    Trimmed from the real `torchani.arch.simple_ani`: dispersion (DFT-D3, needs h5py) and
    non-"pyaev" strategies are not vendored in this staging module.
    """
    if strategy == "auto":
        strategy = "pyaev"
    asm = Assembler(periodic_table_index=periodic_table_index)
    asm.set_symbols(symbols)
    asm.set_global_cutoff_fn(cutoff_fn)
    asm.set_aev_computer(
        radial=ANIRadial.cover_linearly(
            start=radial_start, cutoff=radial_cutoff, eta=radial_precision, num_shifts=radial_shifts
        ),
        angular=ANIAngular.cover_linearly(
            start=angular_start,
            eta=angular_precision,
            zeta=angular_zeta,
            num_shifts=angular_shifts,
            num_sections=sections,
            cutoff=angular_cutoff,
        ),
        strategy=strategy,
    )
    asm.set_atomic_networks(
        cls=ANINetworks if container == "ANINetworks" else ANINetworks,
        ctor="default",
        kwargs={"bias": bias, "activation": parse_activation(activation)},
    )
    asm.set_neighborlist(neighborlist)
    if self_energies == "zero":
        asm.set_zeros_as_self_energies()
    elif self_energies is not None:
        asm.set_self_energies(self_energies)
    else:
        asm.set_gsaes_as_self_energies(lot)
    if repulsion:
        asm.add_potential(
            RepulsionXTB,
            name="repulsion_xtb",
            cutoff=radial_cutoff if repulsion_cutoff else math.inf,
        )
    return tp.cast(ANI, asm.assemble(1))


# ---------------------------------------------------------------------------
# Menagerie build/example plumbing
# ---------------------------------------------------------------------------
def build_torchani():
    """Tiny random-init ANI-style model (ANI2x-like AEV, 4-element ANINetworks, XTB repulsion)."""
    torch.manual_seed(0)
    symbols = ("H", "C", "N", "O")
    return simple_ani(
        symbols=symbols,
        lot="wb97x-631gd",
        radial_start=0.8,
        angular_start=0.8,
        radial_cutoff=5.1,
        angular_cutoff=3.5,
        radial_shifts=4,
        angular_shifts=2,
        sections=2,
        cutoff_fn="cosine",
        repulsion=True,
        activation="gelu",
        strategy="pyaev",
        periodic_table_index=True,
        neighborlist="all_pairs",
        self_energies="zero",
    )


def example_input_torchani():
    """A small batch of molecules: atomic numbers + 3D coordinates (methane-sized), as
    the (species, coords) tuple `ANI.forward` expects."""
    torch.manual_seed(0)
    n_molecs, n_atoms = 2, 5
    # Atomic numbers restricted to H(1)/C(6)/N(7)/O(8), the symbols the model supports.
    species = torch.tensor(
        [
            [6, 1, 1, 1, 1],
            [7, 1, 1, 1, 8],
        ],
        dtype=torch.long,
    )
    coords = torch.randn(n_molecs, n_atoms, 3) * 1.5
    return ((species, coords),)


MENAGERIE_ENTRIES = [
    ("TorchANI (simple_ani)", build_torchani, example_input_torchani, 2017, "REAL"),
]
