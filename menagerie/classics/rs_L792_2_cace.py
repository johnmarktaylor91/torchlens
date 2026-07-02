# SOURCE: vendored from https://github.com/BingqingCheng/cace @ main (fetched 2026-07-01)
#
# CACE (Cartesian Atomic Cluster Expansion) is a pure-PyTorch equivariant
# interatomic-potential architecture (Cheng, "Cartesian Atomic Cluster
# Expansion for Machine Learning Interatomic Potentials", npj Comput. Mater.
# 2025 / arXiv:2402.07472). The real `cace` package is pure torch
# (`cace/models/atomistic.py`, `cace/representations/cace_representation.py`,
# `cace/modules/*`), but the top-level package `__init__.py` chain
# (`cace.data` -> `ase.Atoms`, `cace.tools` -> `ase`) requires the `ase` and
# `matscipy` packages, which are not in the installed base-lib set. Rather
# than reimplement, this module vendors the REAL model-defining files
# verbatim (NeuralNetworkPotential/AtomisticModel, Cace representation,
# Atomwise readout, and every cace/modules/* class the forward path touches:
# NodeEncoder/NodeEmbedding/EdgeEncoder, AngularComponent, BesselRBF,
# PolynomialCutoff, SharedRadialLinearTransform, Symmetrizer,
# MessageAr/MessageBchi/NodeMemory, Dense/ResidualBlock/build_mlp,
# scatter_sum, elementwise_multiply_3tensors) with only import paths
# rewritten to resolve within this single file (no `ase`/`matscipy`
# dependency, no vendored `torch_geometric` -- the example input below
# builds the plain tensor-dict the real `Cace.forward`/`Preprocess.forward`
# consume, bypassing `cace.data.AtomicData`/`ase` entirely). No architecture
# altered.

import itertools as _itertools
import math as _math
from collections import OrderedDict as _OrderedDict
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# =====================================================================
# from cace/tools/torch_tools.py (elementwise_multiply_3tensors)
# =====================================================================


@torch.jit.script
def elementwise_multiply_3tensors(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Elementwise multiplication of three 2D tensors
    :param a: (N, A) tensor
    :param b: (N, B) tensor
    :param c: (N, C) tensor
    :return: (N, A, B, C) tensor
    """
    a_expanded = a.unsqueeze(2).unsqueeze(3)
    b_expanded = b.unsqueeze(1).unsqueeze(3)
    c_expanded = c.unsqueeze(1).unsqueeze(2)
    return a_expanded * b_expanded * c_expanded


# =====================================================================
# from cace/tools/scatter.py (scatter_sum)
# =====================================================================


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if dim < 0:
        dim = src.dim() + dim

    if index.dim() != src.dim():
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
        index = index.expand_as(src)

    if out is None:
        size: List[int] = list(src.size())

        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1

        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


# =====================================================================
# from cace/modules/blocks.py
# =====================================================================


def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    residual: bool = False,
    use_batchnorm: bool = False,
    bias: bool = True,
    last_zero_init: bool = False,
) -> nn.Module:
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    if residual:
        if n_layers < 3 or n_layers % 2 == 0:
            raise ValueError(
                "Residual networks require at least 3 layers and an odd number of layers"
            )
        layers = []
        for i in range(0, n_layers - 1, 2):
            in_features = n_neurons[i]
            out_features = n_neurons[min(i + 2, len(n_neurons) - 1)]
            layers.append(
                ResidualBlock(
                    in_features,
                    out_features,
                    activation,
                    skip_interval=2,
                    use_batchnorm=use_batchnorm,
                )
            )
    else:
        layers = [
            Dense(
                n_neurons[i],
                n_neurons[i + 1],
                activation=activation,
                use_batchnorm=use_batchnorm,
                bias=bias,
            )
            for i in range(n_layers - 1)
        ]

    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=bias,
            )
        )
    else:
        layers.append(Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=bias))
    out_net = nn.Sequential(*layers)
    return out_net


class Dense(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = nn.Identity(),
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()
        if self.use_batchnorm:
            self.batchnorm = nn.BatchNorm1d(out_features)

    def forward(self, input: torch.Tensor):
        y = self.linear(input)
        if self.use_batchnorm:
            y = self.batchnorm(y)
        y = self.activation(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation, skip_interval=2, use_batchnorm=True):
        super().__init__()
        self.skip_interval = skip_interval
        self.use_batchnorm = use_batchnorm
        self.layers = nn.ModuleList()

        if in_features != out_features:
            skip_layers = [Dense(in_features, out_features, activation=None)]
            if self.use_batchnorm:
                skip_layers.append(nn.BatchNorm1d(out_features))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

        for _ in range(skip_interval):
            self.layers.append(Dense(in_features, out_features, activation=activation))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

    def forward(self, x):
        identity = self.skip(x)
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if (i + 1) % self.skip_interval == 0:
                out += identity
        return out


# =====================================================================
# from cace/modules/cutoff.py (PolynomialCutoff only)
# =====================================================================


class PolynomialCutoff(nn.Module):
    """
    Klicpera, J.; Groß, J.; Guennemann, S. Directional Message Passing for
    Molecular Graphs; ICLR 2020. Equation (8)
    """

    p: torch.Tensor
    cutoff: torch.Tensor

    def __init__(self, cutoff: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.cutoff, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.cutoff, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.cutoff, self.p + 2)
        )
        return envelope * (x < self.cutoff)


# =====================================================================
# from cace/modules/radial.py (BesselRBF only)
# =====================================================================


class BesselRBF(nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    Klicpera, Gross, Guennemann: Directional message passing for molecular
    graphs. ICLR 2020, Equation (7)
    """

    def __init__(self, cutoff: float, n_rbf=8, trainable=False):
        super().__init__()
        self.n_rbf = n_rbf
        bessel_weights = (
            _math.pi
            / cutoff
            * torch.linspace(start=1.0, end=n_rbf, steps=n_rbf, dtype=torch.get_default_dtype())
        )
        if trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor", torch.tensor(_math.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x)
        return self.prefactor * (numerator / x)


# =====================================================================
# from cace/modules/radial_transform.py
# =====================================================================


class SharedRadialLinearTransform(nn.Module):
    def __init__(
        self,
        max_l: int,
        radial_dim: int,
        radial_embedding_dim: Optional[int] = None,
        channel_dim: Optional[int] = None,
    ):
        super().__init__()
        self.max_l = max_l
        self.radial_dim = radial_dim
        self.radial_embedding_dim = radial_embedding_dim or radial_dim
        self.channel_dim = channel_dim
        self.register_buffer(
            "angular_dim_groups",
            torch.tensor(self._init_angular_dim_groups(max_l), dtype=torch.int64),
        )
        self.weights = self._initialize_weights(radial_dim, self.radial_embedding_dim, channel_dim)

    def _initialize_weights(
        self, radial_dim: int, radial_embedding_dim: int, channel_dim: int
    ) -> nn.ParameterList:
        torch.manual_seed(0)
        if channel_dim is not None:
            return nn.ParameterList(
                [
                    nn.Parameter(torch.rand([radial_dim, radial_embedding_dim, channel_dim]))
                    for _ in self.angular_dim_groups
                ]
            )
        else:
            return nn.ParameterList(
                [
                    nn.Parameter(torch.rand([radial_dim, radial_embedding_dim]))
                    for _ in self.angular_dim_groups
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_nodes, radial_dim, angular_dim, embedding_dim = x.shape

        output = torch.zeros(
            n_nodes,
            self.radial_embedding_dim,
            angular_dim,
            embedding_dim,
            device=x.device,
            dtype=x.dtype,
        )

        for index, weight in enumerate(self.weights):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            group = torch.arange(i_start, i_end)
            group_x = x[:, :, group, :]
            if self.channel_dim:
                transformed_group = torch.einsum("ijkh,jmh->imkh", group_x, weight)
            else:
                transformed_group = torch.einsum("ijkh,jm->imkh", group_x, weight)
            output[:, :, group, :] = transformed_group
        return output

    def _compute_length_lxlylz(self, l):
        return int((l + 1) * (l + 2) / 2)

    def _init_angular_dim_groups(self, max_l):
        angular_dim_groups: List[int] = []
        l_now = 0
        for l in range(max_l + 1):
            l_list_atl = [l_now, l_now + self._compute_length_lxlylz(l)]
            angular_dim_groups.append(l_list_atl)
            l_now += self._compute_length_lxlylz(l)
        return angular_dim_groups


# =====================================================================
# from cace/modules/angular.py (AngularComponent only)
# =====================================================================


class AngularComponent(nn.Module):
    """Angular component of the edge basis functions
    Optimized for CPU usage (use recursive formula)
    """

    def __init__(self, l_max):
        super().__init__()
        self.l_max = l_max
        self.precompute_lxlylz()

    def precompute_lxlylz(self):
        self.lxlylz_dict = _OrderedDict({l: [] for l in range(self.l_max + 1)})
        self.lxlylz_dict[0] = [(0, 0, 0)]
        for l in range(1, self.l_max + 1):
            for prev_lxlylz_combination in self.lxlylz_dict[l - 1]:
                for i in range(3):
                    lxlylz_combination = list(prev_lxlylz_combination)
                    lxlylz_combination[i] += 1
                    lxlylz_combination_tuple = tuple(lxlylz_combination)
                    if lxlylz_combination_tuple not in self.lxlylz_dict[l]:
                        self.lxlylz_dict[l].append(lxlylz_combination_tuple)
        self.lxlylz_list = self._convert_lxlylz_to_list()
        self.lxlylz_index = torch.zeros((self.l_max + 1, 2), dtype=torch.long)
        for l in range(self.l_max + 1):
            self.lxlylz_index[l, 0] = 0 if l == 0 else self.lxlylz_index[l - 1, 1]
            self.lxlylz_index[l, 1] = compute_length_lmax(l)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        computed_values = {
            (0, 0, 0): torch.ones(vectors.size(0), device=vectors.device, dtype=vectors.dtype)
        }
        for l in range(1, self.l_max + 1):
            for lxlylz_combination in self.lxlylz_dict[l]:
                prev_lxlylz_combination = tuple(
                    l - 1 if i == lxlylz_combination.index(max(lxlylz_combination)) else l
                    for i, l in enumerate(lxlylz_combination)
                )
                i = lxlylz_combination.index(max(lxlylz_combination))
                computed_values[lxlylz_combination] = (
                    computed_values[prev_lxlylz_combination] * vectors[:, i]
                )

        computed_values_list = self._convert_computed_values_to_list(computed_values)
        return torch.stack(computed_values_list, dim=1)

    def _convert_lxlylz_to_list(self):
        lxlylz_list = []
        for l, combinations in self.lxlylz_dict.items():
            lxlylz_list.extend(combinations)
        return lxlylz_list

    def _convert_computed_values_to_list(self, computed_values):
        return [computed_values[comb] for comb in self.lxlylz_list]

    def get_lxlylz_list(self):
        return self.lxlylz_list

    def get_lxlylz_dict(self):
        return self.lxlylz_dict

    def get_lxlylz_index(self):
        return self.lxlylz_index


def compute_length_lxlylz(l):
    return int((l + 1) * (l + 2) / 2)


def compute_length_lmax(l_max):
    return int((l_max + 1) * (l_max + 2) * (l_max + 3) / 6)


def l1l2_factorial_coef(l1, l2):
    from math import factorial

    result = 1
    for l1i, l2i in zip(l1, l2):
        result *= factorial(l1i + l2i)
        result /= factorial(l1i)
        result /= factorial(l2i)
    return result


def lxlylz_factorial_coef(lxlylz):
    from math import factorial

    sorted_lxlylz = sorted(lxlylz, reverse=True)
    l = sum(sorted_lxlylz)
    result = factorial(l)
    for lxly in sorted_lxlylz:
        result //= factorial(lxly)
    return result


def make_lxlylz(l):
    lxlylz = []
    for lx in range(l + 1):
        for ly in range(l + 1):
            lz = l - lx - ly
            if lz >= 0:
                lxlylz.append([lx, ly, lz])
    return lxlylz


# =====================================================================
# from cace/modules/angular_tools.py (find_combo_vectors_nu2/3/4 only)
# =====================================================================


def find_combo_vectors_nu2(l_max):
    vec_dict = {}
    L_list = range(1, l_max + 1)
    for i, L in enumerate(L_list):
        for lxlylz_now in make_lxlylz(L):
            prefactor = lxlylz_factorial_coef(lxlylz_now)
            key = L
            vec_dict[key] = vec_dict.get(key, []) + [(lxlylz_now, lxlylz_now, prefactor)]

    vectors = []
    prefactors = []
    vector_idx = []
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now, lxlylz_now2, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2]))
            prefactors.append(prefactor)

    if not vectors:
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)

    return (
        vec_dict,
        stacked_vectors,
        torch.tensor(vector_idx, dtype=torch.int64),
        torch.tensor(prefactors, dtype=torch.int64),
        len(vec_dict),
    )


def find_combo_vectors_nu3(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in _itertools.product(range(l_max + 1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < (lx1 + ly1 + lz1) <= l_max:
            for lx2, ly2, lz2 in _itertools.product(range(l_max + 1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if (lx1 + ly1 + lz1) <= (lx2 + ly2 + lz2) <= l_max:
                    lx3, ly3, lz3 = lx1 + lx2, ly1 + ly2, lz1 + lz2
                    if (lx3 + ly3 + lz3) <= l_max:
                        prefactor = lxlylz_factorial_coef([lx1, ly1, lz1]) * lxlylz_factorial_coef(
                            [lx2, ly2, lz2]
                        )
                        key = (l1, l2)
                        vec_dict[key] = vec_dict.get(key, []) + [
                            ([lx1, ly1, lz1], [lx2, ly2, lz2], [lx3, ly3, lz3], prefactor)
                        ]

    vectors = []
    prefactors = []
    vector_idx = []
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now1, lxlylz_now2, lxlylz_now3, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now1, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            lxlylz_tensor3 = torch.tensor(lxlylz_now3, dtype=torch.int64)
            vectors.append(torch.stack([lxlylz_tensor1, lxlylz_tensor2, lxlylz_tensor3]))
            prefactors.append(prefactor)

    if not vectors:
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)

    return (
        vec_dict,
        stacked_vectors,
        torch.tensor(vector_idx, dtype=torch.int64),
        torch.tensor(prefactors, dtype=torch.int64),
        len(vec_dict),
    )


def find_combo_vectors_nu4(l_max):
    vec_dict = {}
    for lx1, ly1, lz1 in _itertools.product(range(l_max + 1), repeat=3):
        l1 = lx1 + ly1 + lz1
        if 0 < l1 <= l_max:
            for lx2, ly2, lz2 in _itertools.product(range(l_max + 1), repeat=3):
                l2 = lx2 + ly2 + lz2
                if l1 < l2 <= l_max:
                    for dx, dy, dz in _itertools.product(range(l_max + 1), repeat=3):
                        dl = dx + dy + dz
                        if dl >= 1:
                            lx3, ly3, lz3 = lx1 + dx, ly1 + dy, lz1 + dz
                            lx4, ly4, lz4 = lx2 + dx, ly2 + dy, lz2 + dz
                            if (lx3 + ly3 + lz3) <= l_max and (lx4 + ly4 + lz4) <= l_max:
                                prefactor = (
                                    lxlylz_factorial_coef([lx1, ly1, lz1])
                                    * lxlylz_factorial_coef([lx2, ly2, lz2])
                                    * lxlylz_factorial_coef([dx, dy, dz])
                                )
                                key = (l1, l2, dl)
                                vec_dict[key] = vec_dict.get(key, []) + [
                                    (
                                        [lx1, ly1, lz1],
                                        [lx2, ly2, lz2],
                                        [lx3, ly3, lz3],
                                        [lx4, ly4, lz4],
                                        prefactor,
                                    )
                                ]

    vectors = []
    prefactors = []
    vector_idx = []
    for i, (key, vec_lists) in enumerate(vec_dict.items()):
        for value in vec_lists:
            [lxlylz_now1, lxlylz_now2, lxlylz_now3, lxlylz_now4, prefactor] = value
            vector_idx.append(i)
            lxlylz_tensor1 = torch.tensor(lxlylz_now1, dtype=torch.int64)
            lxlylz_tensor2 = torch.tensor(lxlylz_now2, dtype=torch.int64)
            lxlylz_tensor3 = torch.tensor(lxlylz_now3, dtype=torch.int64)
            lxlylz_tensor4 = torch.tensor(lxlylz_now4, dtype=torch.int64)
            vectors.append(
                torch.stack([lxlylz_tensor1, lxlylz_tensor2, lxlylz_tensor3, lxlylz_tensor4])
            )
            prefactors.append(prefactor)

    if not vectors:
        stacked_vectors = torch.tensor([])
    else:
        stacked_vectors = torch.stack(vectors)

    return (
        vec_dict,
        stacked_vectors,
        torch.tensor(vector_idx, dtype=torch.int64),
        torch.tensor(prefactors, dtype=torch.int64),
        len(vec_dict),
    )


# =====================================================================
# from cace/modules/symmetrize_basis.py (Symmetrizer only)
# =====================================================================


class Symmetrizer(nn.Module):
    def __init__(self, max_nu: int, max_l: int, l_list: list):
        super().__init__()
        if max_nu >= 5:
            raise NotImplementedError

        self.max_nu = max_nu
        self.max_l = max_l

        l_list_tuples = [tuple(l) for l in l_list]
        self.l_list_indices = {l_tuple: i for i, l_tuple in enumerate(l_list_tuples)}

        if max_nu > 4:
            raise NotImplementedError("max_nu > 4 is not supported yet.")
        self.vec_dict_allnu = {}
        if max_nu >= 2:
            self.vec_dict_allnu[2] = find_combo_vectors_nu2(self.max_l)[0]
        if max_nu >= 3:
            self.vec_dict_allnu[3] = find_combo_vectors_nu3(self.max_l)[0]
        if max_nu == 4:
            self.vec_dict_allnu[4] = find_combo_vectors_nu4(self.max_l)[0]

        self.indice_dict_allnu = None
        self._get_indices_allnu()

    def _get_indices_allnu(self):
        self.indice_dict_allnu = {}
        for nu in range(2, self.max_nu + 1):
            self.indice_dict_allnu[nu] = {}
            for i, (l_key, lxlylz_list) in enumerate(self.vec_dict_allnu[nu].items()):
                self.indice_dict_allnu[nu][l_key] = []
                for item in lxlylz_list:
                    prefactor = item[-1]
                    indices = [self.l_list_indices[tuple(lxlylz)] for lxlylz in item[:-1]]
                    self.indice_dict_allnu[nu][l_key].append([indices, prefactor])

    def forward(self, node_attr: torch.Tensor):
        try:
            self.indice_dict_allnu
        except AttributeError:
            self._get_indices_allnu()

        num_nodes, n_radial, _, n_chanel = node_attr.size()
        n_angular_sym = 1 + sum(len(self.vec_dict_allnu[nu]) for nu in range(2, self.max_nu + 1))
        sym_node_attr = torch.zeros(
            (num_nodes, n_radial, n_angular_sym, n_chanel),
            dtype=node_attr.dtype,
            device=node_attr.device,
        )

        sym_node_attr[:, :, 0, :] = node_attr[:, :, 0, :]
        n_sym_node_attr = 1

        for nu in range(2, self.max_nu + 1):
            for i, (_, indices_list) in enumerate(self.indice_dict_allnu[nu].items()):
                for item in indices_list:
                    indices, prefactor = item[0], item[-1]
                    product = torch.prod(node_attr[:, :, indices, :], dim=2)
                    sym_node_attr[:, :, i + n_sym_node_attr, :] += prefactor * product
            n_sym_node_attr += len(self.indice_dict_allnu[nu])

        return sym_node_attr


# =====================================================================
# from cace/modules/type.py (NodeEncoder, NodeEmbedding, EdgeEncoder only)
# =====================================================================


def get_edge_node_type(
    edge_index: torch.Tensor,
    node_type: torch.Tensor,
    node_type_2: torch.Tensor = None,
):
    if node_type_2 is None:
        node_type_2 = node_type
    sender_type = node_type[edge_index[0]]
    receiver_type = node_type_2[edge_index[1]]
    return sender_type, receiver_type


class NodeEncoder(nn.Module):
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.register_buffer(
            "index_map",
            torch.tensor(
                [zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)], dtype=torch.int64
            ),
        )

    def forward(self, atomic_numbers) -> torch.Tensor:
        device = atomic_numbers.device
        indices = self.index_map[atomic_numbers]
        if (indices < 0).any():
            raise ValueError(f"Atomic numbers out of range: {atomic_numbers[indices < 0]}")
        one_hot_encoding = self.to_one_hot(
            indices.unsqueeze(-1), num_classes=self.num_classes, device=device
        )
        return one_hot_encoding

    def to_one_hot(
        self, indices: torch.Tensor, num_classes: int, device=torch.device
    ) -> torch.Tensor:
        shape = indices.shape[:-1] + (num_classes,)
        oh = torch.zeros(shape, device=device)
        oh.scatter_(dim=-1, index=indices, value=1)
        return oh


class NodeEmbedding(nn.Module):
    def __init__(self, node_dim: int, embedding_dim: int, trainable=True, random_seed=42):
        super().__init__()
        embedding_weights = torch.Tensor(node_dim, embedding_dim)
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.reset_parameters(embedding_weights)

        if trainable:
            self.embedding_weights = nn.Parameter(embedding_weights)
        else:
            self.register_buffer("embedding_weights", embedding_weights)

    def reset_parameters(self, embedding_weights):
        nn.init.xavier_uniform_(embedding_weights)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mm(data, self.embedding_weights)


class EdgeEncoder(nn.Module):
    def __init__(self, directed=True):
        super().__init__()
        self.directed = directed

    def forward(
        self,
        edge_index: torch.Tensor,
        node_type: torch.Tensor,
        node_type_2: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        node1, node2 = get_edge_node_type(edge_index, node_type, node_type_2)

        if self.directed:
            encoded_edges = torch.einsum("ki,kj->kij", node1, node2).flatten(start_dim=1)
        else:
            min_node, max_node = torch.min(node1, node2), torch.max(node1, node2)
            encoded_edges = torch.einsum("ki,kj->kij", min_node, max_node).flatten(start_dim=1)

        return encoded_edges


# =====================================================================
# from cace/modules/interaction.py (MessageAr, MessageBchi, NodeMemory only)
# =====================================================================


def _init_angular_dim_groups(max_l):
    angular_dim_groups: List[int] = []
    l_now = 0
    for l in range(max_l + 1):
        length_at_l = compute_length_lxlylz(l)
        l_list_atl = [l_now, l_now + length_at_l]
        angular_dim_groups.append(l_list_atl)
        l_now += length_at_l
    return angular_dim_groups


class MessageAr(nn.Module):
    r"""
    Interaction layer for the message passing network.
    Dependent on radial and channel dimensions, shared L channels.
    """

    def __init__(
        self,
        cutoff: float,
        max_l: int,
        radial_embedding_dim: int,
        channel_dim: int,
    ):
        super().__init__()
        self.register_buffer(
            "angular_dim_groups", torch.tensor(_init_angular_dim_groups(max_l), dtype=torch.int64)
        )

        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        self.prefactor = nn.ParameterList(
            [
                nn.Parameter(torch.rand(radial_embedding_dim, channel_dim))
                for _ in self.angular_dim_groups
            ]
        )

        self.invr0 = nn.ParameterList(
            [
                nn.Parameter((1.0 / cutoff) * (torch.rand(radial_embedding_dim, channel_dim) + 0.5))
                for _ in self.angular_dim_groups
            ]
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_lengths: torch.Tensor,
        radial_cutoff_fn: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        assert radial_dim == self.radial_embedding_dim
        assert channel_dim == self.channel_dim

        n_edges = edge_index.shape[1]
        sender_features = node_feat[edge_index[0]]

        message = torch.zeros(
            (n_edges, radial_dim, angular_dim, channel_dim),
            device=node_feat.device,
            dtype=node_feat.dtype,
        )

        for index, (prefactor, invr0) in enumerate(zip(self.prefactor, self.invr0)):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            group = torch.arange(i_start, i_end)

            radial_decay = (
                torch.exp(-1.0 * edge_lengths.view(n_edges, 1, 1) * invr0[None, :, :])
                * prefactor[None, :, :]
                * radial_cutoff_fn.view(n_edges, 1, 1)
            )
            message[:, :, group, :] = sender_features[:, :, group, :] * radial_decay[:, :, None, :]

        return message


class MessageBchi(nn.Module):
    r"""another message passing mechanism -- shared radial+channel gate on the B basis"""

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 1,
        shared_channels: bool = True,
        shared_l: bool = True,
        n_out: Optional[int] = None,
        lxlylz_index: Optional[torch.Tensor] = None,
        activation: Callable = F.silu,
        residual: bool = False,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        self.shared_channels = shared_channels
        self.shared_l = shared_l

        if shared_channels:
            self.nc = 1

        if shared_l is False and lxlylz_index is None:
            raise ValueError("lxlylz_index must be provided if shared_l is False")

        if shared_l:
            self.nl = 1
        else:
            self.nl = len(lxlylz_index)
            self.nlxlylz = lxlylz_index[-1, 1]
            l_matrix = torch.zeros(self.nl, self.nlxlylz)
            for i, index_now in enumerate(lxlylz_index):
                l_matrix[i, index_now[0] : index_now[1]] = 1
            self.register_buffer("l_matrix", l_matrix)

        if shared_channels and shared_l:
            n_out = 1

        if n_in is not None and n_out is not None:
            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
            )
        else:
            self.hnet = None

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_attri: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        features = node_feat.reshape(n_nodes, -1)
        n_edges = edge_index.shape[1]

        if not hasattr(self, "shared_l"):
            self.shared_l = True
            self.nl = 1
            self.nc = 1

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.hnet == None:
            if self.shared_channels:
                n_out = 1
            else:
                self.nc = channel_dim
                n_out = channel_dim
            if self.shared_l is False:
                n_out *= self.nl

            self.hnet = build_mlp(
                n_in=self.n_in,
                n_out=n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
            )
            self.hnet = self.hnet.to(features.device)

        node_weight = self.hnet(features).reshape(n_nodes, self.nl, self.nc)
        if self.shared_l is False:
            node_weight = torch.einsum("lm,ilk->imk", self.l_matrix, node_weight)
        edge_weight = node_weight[edge_index[0]]
        message = edge_attri * edge_weight[:, None, :, :]
        return message


class NodeMemory(nn.Module):
    """Compute the memory of the node during message passing"""

    def __init__(
        self,
        max_l: int,
        radial_embedding_dim: int,
        channel_dim: int,
        memory_coef_init: torch.Tensor = torch.tensor(0.25),
    ):
        super().__init__()
        self.max_l = max_l
        self.register_buffer(
            "angular_dim_groups", torch.tensor(_init_angular_dim_groups(max_l), dtype=torch.int64)
        )
        self.radial_embedding_dim = radial_embedding_dim
        self.channel_dim = channel_dim

        self.memory_coef = nn.ParameterList(
            [
                nn.Parameter(torch.ones(radial_embedding_dim, channel_dim) * memory_coef_init)
                for _ in self.angular_dim_groups
            ]
        )

    def forward(
        self,
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes, radial_dim, angular_dim, channel_dim = node_feat.shape
        assert radial_dim == self.radial_embedding_dim
        assert channel_dim == self.channel_dim

        node_memory = torch.zeros_like(node_feat)

        for index, memory_coef in enumerate(self.memory_coef):
            i_start = self.angular_dim_groups[index, 0]
            i_end = self.angular_dim_groups[index, 1]
            group = torch.arange(i_start, i_end)
            node_memory[:, :, group, :] = node_feat[:, :, group, :] * memory_coef[None, :, None, :]

        return node_memory


# =====================================================================
# from cace/modules/utils.py (get_edge_vectors_and_lengths only)
# =====================================================================


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    shifts: torch.Tensor,
    normalize: bool = False,
    eps: float = 1e-9,
):
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths
    return vectors, lengths


# =====================================================================
# from cace/representations/cace_representation.py (Cace only)
# =====================================================================


class Cace(nn.Module):
    def __init__(
        self,
        zs: Sequence[int],
        n_atom_basis: int,
        cutoff: float,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        max_l: int,
        max_nu: int,
        num_message_passing: int,
        node_encoder: Optional[nn.Module] = None,
        edge_encoder: Optional[nn.Module] = None,
        type_message_passing: List[str] = ["M", "Ar", "Bchi"],
        args_message_passing: Dict = {"M": {}, "Ar": {}, "Bchi": {}},
        embed_receiver_nodes: bool = False,
        atom_embedding_random_seed: List[int] = [42, 42],
        n_radial_basis: Optional[int] = None,
        avg_num_neighbors: float = 10.0,
        device: torch.device = torch.device("cpu"),
        timeit: bool = False,
        keep_node_features_A: bool = False,
        max_l_out: int = 0,
        forward_features: List[str] = [],
        charge_spin_key: Optional[str] = None,
    ):
        super().__init__()
        self.zs = zs
        self.nz = len(zs)
        self.n_atom_basis = n_atom_basis
        self.cutoff = cutoff
        self.max_l = max_l
        self.max_l_out = max_l_out
        self.max_nu = max_nu
        self.mp_norm_factor = 1.0 / (avg_num_neighbors) ** 0.5
        self.keep_node_features_A = max_l_out > 0
        self.forward_features = forward_features

        if node_encoder is None:
            self.node_onehot = NodeEncoder(self.zs)
            self.nz = len(zs)
        else:
            self.node_onehot = node_encoder
            self.nz = node_encoder.embedding_dim

        self.node_embedding_sender = NodeEmbedding(
            node_dim=self.nz,
            embedding_dim=self.n_atom_basis,
            random_seed=atom_embedding_random_seed[0],
        )
        if embed_receiver_nodes:
            self.node_embedding_receiver = NodeEmbedding(
                node_dim=self.nz,
                embedding_dim=self.n_atom_basis,
                random_seed=atom_embedding_random_seed[1],
            )
        else:
            self.node_embedding_receiver = self.node_embedding_sender

        self.charge_spin_key = charge_spin_key
        if charge_spin_key is not None:
            self.charge_spin_embedding = NodeEmbedding(
                node_dim=1,
                embedding_dim=self.n_atom_basis,
                random_seed=atom_embedding_random_seed[0],
            )

        if edge_encoder is not None:
            self.edge_coding = edge_encoder
        else:
            self.edge_coding = EdgeEncoder(directed=True)

        self.n_edge_channels = n_atom_basis**2

        self.radial_basis = radial_basis
        self.n_radial_func = self.radial_basis.n_rbf
        self.n_radial_basis = n_radial_basis or self.radial_basis.n_rbf
        self.cutoff_fn = cutoff_fn
        self.angular_basis = AngularComponent(self.max_l)
        radial_transform = SharedRadialLinearTransform(
            max_l=self.max_l,
            radial_dim=self.n_radial_func,
            radial_embedding_dim=self.n_radial_basis,
            channel_dim=self.n_edge_channels,
        )
        self.radial_transform = radial_transform

        self.l_list = self.angular_basis.get_lxlylz_list()
        self.symmetrizer = Symmetrizer(self.max_nu, self.max_l, self.l_list)

        self.num_message_passing = num_message_passing
        self.message_passing_list = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        NodeMemory(
                            max_l=self.max_l,
                            radial_embedding_dim=self.n_radial_basis,
                            channel_dim=self.n_edge_channels,
                            **args_message_passing["M"] if "M" in args_message_passing else {},
                        )
                        if "M" in type_message_passing
                        else None,
                        MessageAr(
                            cutoff=cutoff,
                            max_l=self.max_l,
                            radial_embedding_dim=self.n_radial_basis,
                            channel_dim=self.n_edge_channels,
                            **args_message_passing["Ar"] if "Ar" in args_message_passing else {},
                        )
                        if "Ar" in type_message_passing
                        else None,
                        MessageBchi(
                            lxlylz_index=self.angular_basis.get_lxlylz_index(),
                            **args_message_passing["Bchi"]
                            if "Bchi" in args_message_passing
                            else {},
                        )
                        if "Bchi" in type_message_passing
                        else None,
                    ]
                )
                for _ in range(self.num_message_passing)
            ]
        )

        self.device = device

    def forward(self, data: Dict[str, torch.Tensor]):
        n_nodes = data["positions"].shape[0]
        if data["batch"] is None:
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=self.device)
        else:
            batch_now = data["batch"]

        node_feats_list = []
        node_feats_A_list = []

        node_one_hot = self.node_onehot(data["atomic_numbers"])

        node_embedded_sender = self.node_embedding_sender(node_one_hot)
        node_embedded_receiver = self.node_embedding_receiver(node_one_hot)

        if hasattr(self, "charge_spin_key") and self.charge_spin_key is not None:
            charge_spin_feat = self.charge_spin_embedding(data[self.charge_spin_key].view(-1, 1))
            node_embedded_sender = node_embedded_sender + charge_spin_feat
            node_embedded_receiver = node_embedded_receiver + charge_spin_feat

        encoded_edges = self.edge_coding(
            edge_index=data["edge_index"],
            node_type=node_embedded_sender,
            node_type_2=node_embedded_receiver,
            data=data,
        )

        edge_vectors, edge_lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            normalize=True,
        )
        radial_component = self.radial_basis(edge_lengths)
        radial_cutoff = self.cutoff_fn(edge_lengths)
        angular_component = self.angular_basis(edge_vectors)

        edge_attri = elementwise_multiply_3tensors(
            radial_component * radial_cutoff, angular_component, encoded_edges
        )

        node_feat_A = scatter_sum(
            src=edge_attri, index=data["edge_index"][1], dim=0, dim_size=n_nodes
        )

        node_feat_A = self.radial_transform(node_feat_A)
        if hasattr(self, "keep_node_features_A") and self.keep_node_features_A:
            node_feats_A_list.append(node_feat_A)

        node_feat_B = self.symmetrizer(node_attr=node_feat_A)
        node_feats_list.append(node_feat_B)

        for nm, mp_Ar, mp_Bchi in self.message_passing_list:
            if nm is not None:
                momeory_now = nm(node_feat=node_feat_A)
            else:
                momeory_now = 0.0

            if mp_Bchi is not None:
                message_Bchi = mp_Bchi(
                    node_feat=node_feat_B,
                    edge_attri=edge_attri,
                    edge_index=data["edge_index"],
                )
                node_feat_A_Bchi = scatter_sum(
                    src=message_Bchi, index=data["edge_index"][1], dim=0, dim_size=n_nodes
                )
                node_feat_A_Bchi = self.radial_transform(node_feat_A_Bchi)
            else:
                node_feat_A_Bchi = 0.0

            if mp_Ar is not None:
                message_Ar = mp_Ar(
                    node_feat=node_feat_A,
                    edge_lengths=edge_lengths,
                    radial_cutoff_fn=radial_cutoff,
                    edge_index=data["edge_index"],
                )

                node_feat_Ar = scatter_sum(
                    src=message_Ar, index=data["edge_index"][1], dim=0, dim_size=n_nodes
                )
            else:
                node_feat_Ar = 0.0

            node_feat_A = node_feat_Ar + node_feat_A_Bchi
            node_feat_A *= self.mp_norm_factor
            node_feat_A += momeory_now
            if hasattr(self, "keep_node_features_A") and self.keep_node_features_A:
                node_feats_A_list.append(node_feat_A)
            node_feat_B = self.symmetrizer(node_attr=node_feat_A)
            node_feats_list.append(node_feat_B)

        node_feats_out = torch.stack(node_feats_list, dim=-1)

        l_feats_out = {}

        try:
            displacement = data["displacement"]
        except Exception:
            displacement = None

        output = {
            "positions": data["positions"],
            "cell": data["cell"],
            "displacement": displacement,
            "batch": batch_now,
            "node_feats": node_feats_out,
            "node_feats_l": l_feats_out,
        }

        if hasattr(self, "forward_features") and len(self.forward_features) > 0:
            for key in self.forward_features:
                if key in data:
                    output[key] = data[key]

        return output


# =====================================================================
# from cace/modules/atomwise.py (Atomwise only)
# =====================================================================


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    def __init__(
        self,
        n_in: Optional[int] = None,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        bias: bool = True,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        feature_key: Union[str, Sequence[int]] = "node_feats",
        output_key: str = "energy",
        per_atom_output_key: Optional[str] = None,
        descriptor_output_key: Optional[str] = None,
        residual: bool = False,
        use_batchnorm: bool = False,
        add_linear_nn: bool = False,
        post_process: Optional[Callable] = None,
        output_scale: Optional[float] = None,
    ):
        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.descriptor_output_key = descriptor_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        if self.descriptor_output_key is not None:
            self.model_outputs.append(self.descriptor_output_key)

        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.aggregation_mode = aggregation_mode
        self.residual = residual
        self.use_batchnorm = use_batchnorm
        self.add_linear_nn = add_linear_nn
        self.post_process = post_process
        self.bias = bias
        self.feature_key = feature_key
        self.output_scale = output_scale

        if n_in is not None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                bias=self.bias,
            )
            if self.add_linear_nn:
                self.linear_nn = Dense(
                    self.n_in,
                    self.n_out,
                    bias=self.bias,
                    activation=None,
                    use_batchnorm=self.use_batchnorm,
                )

        else:
            self.outnet = None

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = None,
        output_index: int = None,
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "feature_key") or self.feature_key is None:
            self.feature_key = "node_feats"

        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat(
                [data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1
            )

        if self.n_in is None:
            self.n_in = features.shape[1]
        else:
            assert self.n_in == features.shape[1]

        if self.outnet == None:
            self.outnet = build_mlp(
                n_in=self.n_in,
                n_out=self.n_out,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                activation=self.activation,
                residual=self.residual,
                use_batchnorm=self.use_batchnorm,
                bias=self.bias,
            )
            self.outnet = self.outnet.to(features.device)
            if self.add_linear_nn:
                self.linear_nn = Dense(
                    self.n_in,
                    self.n_out,
                    bias=self.bias,
                    activation=None,
                    use_batchnorm=self.use_batchnorm,
                )
                self.linear_nn = self.linear_nn.to(features.device)
            else:
                self.linear_nn = None

        y = self.outnet(features)
        if self.add_linear_nn:
            y += self.linear_nn(features)

        if self.per_atom_output_key is not None:
            data[self.per_atom_output_key] = y

        if hasattr(self, "descriptor_output_key") and self.descriptor_output_key is not None:
            data[self.descriptor_output_key] = features

        if self.aggregation_mode is not None:
            if "batch" in data and data["batch"].numel() > 0:
                nbatch = data["batch"].max() + 1
            else:
                nbatch = 1

            y = scatter_sum(src=y, index=data["batch"], dim=0, dim_size=nbatch)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / torch.bincount(data["batch"], minlength=nbatch)

        if hasattr(self, "post_process") and self.post_process is not None:
            y = self.post_process(y)
        if hasattr(self, "output_scale") and self.output_scale is not None:
            y = y * self.output_scale
        data[self.output_key] = y[:, output_index] if output_index is not None else y
        return data


# =====================================================================
# from cace/models/atomistic.py (AtomisticModel, NeuralNetworkPotential only)
# =====================================================================


class Preprocess(nn.Module):
    """from cace/modules/preprocess.py -- default input_module for NeuralNetworkPotential"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_stress: bool = False,
        compute_virials: bool = False,
    ):
        data["displacement"] = None
        return data


class AtomisticModel(nn.Module):
    """
    Base class for atomistic neural network models.
    """

    def __init__(
        self,
        postprocessors: Optional[List] = None,
        do_postprocessing: bool = False,
    ):
        super().__init__()
        self.do_postprocessing = do_postprocessing
        self.postprocessors = nn.ModuleList(postprocessors or [])
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if hasattr(m, "required_derivatives") and m.required_derivatives is not None:
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:
            if p in data.keys():
                data[p].requires_grad_(True)
        return data

    def postprocess(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.do_postprocessing:
            for pp in self.postprocessors:
                data = pp(data)
        return data

    def extract_outputs(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results = {k: data[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.
    """

    def __init__(
        self,
        representation: nn.Module = None,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List] = None,
        do_postprocessing: bool = False,
        keep_graph: bool = False,
    ):
        super().__init__(
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        if input_modules is None:
            preprocessor = Preprocess()
            input_modules = [preprocessor]
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()

        self.keep_graph = keep_graph

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_stress: bool = True,
        compute_virials: bool = False,
        output_index: int = None,
    ) -> Dict[str, torch.Tensor]:
        data = self.initialize_derivatives(data)

        if "stress" in self.model_outputs or "CACE_stress" in self.model_outputs:
            compute_stress = True
        for m in self.input_modules:
            data = m(data, compute_stress=compute_stress, compute_virials=compute_virials)

        if self.representation is not None:
            data = self.representation(data)

        for m in self.output_modules:
            if hasattr(self, "keep_graph"):
                training = training or self.keep_graph
            data = m(data, training=training, output_index=output_index)

        data = self.postprocess(data)

        results = self.extract_outputs(data)

        return results


# =====================================================================
# staging build/example functions (menagerie recipe-scale construction)
# =====================================================================


def _toy_water_dimer_data():
    """
    Build a small random-init tensor-dict input matching the fields that
    `Preprocess.forward` and `Cace.forward` (both vendored above,
    unmodified) consume: `positions`, `atomic_numbers`, `edge_index`,
    `shifts`, `cell`, `batch`, `ptr`. This replicates what
    `cace.data.AtomicData` + `cace.data.get_neighborhood` would build from
    an `ase.Atoms` object, but constructed directly as plain tensors so no
    `ase`/`matscipy` import is required. Two water molecules (O,H,H each,
    6 atoms total) with a fully-connected (all-to-all, no PBC) neighbor
    list within each molecule -- a standard toy probe geometry for this
    class of interatomic-potential model, matching the real repo's
    `examples/water_train.py` (zs=[1, 8]) system.
    """
    torch.manual_seed(0)
    # 2 molecules x (O, H, H) = 6 atoms; atomic numbers restricted to zs=[1, 8]
    atomic_numbers = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int64)
    positions = torch.randn(6, 3) * 1.5

    # build an all-to-all (excluding self) edge_index within molecule 0 (atoms 0-2)
    # and within molecule 1 (atoms 3-5) -- a small, cutoff-free neighbor list
    edges = []
    for mol_atoms in ([0, 1, 2], [3, 4, 5]):
        for i in mol_atoms:
            for j in mol_atoms:
                if i != j:
                    edges.append((i, j))
    edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()  # [2, n_edges]
    n_edges = edge_index.shape[1]
    shifts = torch.zeros(n_edges, 3)  # no periodic images

    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    cell = torch.zeros(2 * 3, 3)  # non-periodic cell placeholder, 2 graphs x 3 rows

    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "shifts": shifts,
        "cell": cell,
        "batch": batch,
    }


def build_cace():
    """
    Real `cace.models.atomistic.NeuralNetworkPotential` wrapping the real
    `cace.representations.Cace` representation + real `cace.modules.Atomwise`
    energy head, matching the shrunk configuration style used across the
    menagerie recipes (small `n_atom_basis`/`max_l`/`max_nu`/message-passing
    depth; real repo defaults from `examples/water_train.py` are
    n_atom_basis=3, max_l=3, max_nu=3, num_message_passing=1).
    """
    torch.manual_seed(0)
    cutoff = 5.5
    radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
    cutoff_fn = PolynomialCutoff(cutoff=cutoff)

    cace_representation = Cace(
        zs=[1, 8],
        n_atom_basis=3,
        embed_receiver_nodes=True,
        cutoff=cutoff,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        n_radial_basis=8,
        max_l=2,
        max_nu=2,
        num_message_passing=1,
        type_message_passing=["Bchi"],
        device=torch.device("cpu"),
        timeit=False,
    )

    atomwise = Atomwise(
        n_layers=3,
        output_key="CACE_energy",
        n_hidden=[16, 8],
        use_batchnorm=False,
        add_linear_nn=True,
    )

    model = NeuralNetworkPotential(
        input_modules=None,
        representation=cace_representation,
        output_modules=[atomwise],
    )
    model.eval()
    return model


def example_input_cace():
    return _toy_water_dimer_data()


MENAGERIE_ENTRIES = [
    (
        "CACE (Cartesian Atomic Cluster Expansion)",
        "build_cace",
        "example_input_cace",
        2024,
        MENAGERIE_ZOO,
    ),
]
