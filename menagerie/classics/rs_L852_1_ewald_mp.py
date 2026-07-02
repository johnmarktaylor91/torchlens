# SOURCE: vendored from arthurkosmala/EwaldMP @ main (2026-07-01)
# Files combined from the real repo (paths as in upstream):
#   ocpmodels/models/painn/painn.py           (PaiNN backbone: PaiNN, PaiNNMessage, PaiNNUpdate,
#                                               PaiNNOutput, GatedEquivariantBlock)
#   ocpmodels/models/painn/utils.py           (repeat_blocks, get_edge_id)
#   ocpmodels/models/ewald_block.py           (EwaldBlock -- the Ewald message passing long-range block)
#   ocpmodels/models/base.py                  (BaseModel.generate_graph)
#   ocpmodels/models/gemnet/initializers.py   (he_orthogonal_init)
#   ocpmodels/models/gemnet/layers/base_layers.py       (Dense, ScaledSiLU, SiQU, ResidualLayer)
#   ocpmodels/models/gemnet/layers/embedding_block.py   (AtomEmbedding)
#   ocpmodels/models/gemnet/layers/radial_basis.py      (RadialBasis + envelope/rbf basis classes)
#   ocpmodels/modules/scaling/scale_factor.py           (ScaleFactor)
#   ocpmodels/modules/scaling/compat.py                 (load_scales_compat)
#   ocpmodels/common/utils.py (subset: compute_neighbors, conditional_grad, get_k_index_product_set,
#                               get_k_voxel_grid, get_pbc_distances, pos_svd_frame, radius_graph_pbc,
#                               x_to_k_cell -- the free-function subset painn.py imports; the rest of
#                               that 40KB module is OCP-trainer/registry/matplotlib plumbing unrelated
#                               to the model architecture and is not needed to run PaiNN+Ewald forward)
#
# This is the reference "PaiNN+Ewald" combination from the paper "Ewald-based Long-Range Message
# Passing for Molecular Graphs" (Kosmala, Gasteiger, Gao, Gunnemann; ICML 2023). The Ewald block
# (EwaldBlock) is the paper's architectural contribution: a long-range message-passing block that
# computes atom-atom interactions via structure factors evaluated on a reciprocal-space (Fourier) k-
# point grid, in addition to the short-range PaiNN message passing. Code is transcribed verbatim from
# the real repo; only import paths were flattened into this single file and unrelated OCP-trainer
# plumbing (registry/distutils/matplotlib helpers not on the forward-pass call path) was dropped.
#
# Original license: MIT (Facebook, Inc. and affiliates; extended by arthurkosmala for the Ewald parts).

import itertools
import logging
import math
from contextlib import contextmanager
from typing import Callable, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.special import binom
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_scatter import scatter, segment_coo, segment_csr

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# ocpmodels/models/gemnet/initializers.py
# ---------------------------------------------------------------------------
def _standardize(kernel):
    """Makes sure that N*Var(W) = 1 and E[W] = 0"""
    eps = 1e-6
    if len(kernel.shape) == 3:
        axis = [0, 1]
    else:
        axis = 1
    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """He (Kaiming)-scaled (semi-)orthogonal weight init, as used throughout GemNet/PaiNN/Ewald."""
    tensor = torch.nn.init.orthogonal_(tensor)
    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]
    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5
    return tensor


# ---------------------------------------------------------------------------
# ocpmodels/models/gemnet/layers/base_layers.py
# ---------------------------------------------------------------------------
class Dense(torch.nn.Module):
    """Combines a dense layer with scaling for swish activation."""

    def __init__(self, in_features, out_features, bias=False, activation=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError("Activation function not implemented for GemNet (yet).")

    def reset_parameters(self, initializer=he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x):
        return x * self._activation(x)


class ResidualLayer(torch.nn.Module):
    """Residual block with output scaled by 1/sqrt(2)."""

    def __init__(self, units: int, nLayers: int = 2, layer=Dense, **layer_kwargs):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(in_features=units, out_features=units, bias=False, **layer_kwargs)
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x


# ---------------------------------------------------------------------------
# ocpmodels/models/gemnet/layers/embedding_block.py (AtomEmbedding only)
# ---------------------------------------------------------------------------
class AtomEmbedding(torch.nn.Module):
    """Initial atom embeddings based on the atom type."""

    def __init__(self, emb_size, num_elements):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


# ---------------------------------------------------------------------------
# ocpmodels/models/gemnet/layers/radial_basis.py
# ---------------------------------------------------------------------------
class PolynomialEnvelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(np.pi * np.arange(1, num_radial + 1, dtype=np.float32)),
            requires_grad=True,
        )

    def forward(self, d_scaled):
        return self.norm_const / d_scaled[:, None] * torch.sin(self.frequencies * d_scaled[:, None])


class BernsteinBasis(torch.nn.Module):
    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor", torch.tensor(prefactor, dtype=torch.float), persistent=False
        )
        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float), requires_grad=True
        )
        self.softplus = torch.nn.Softplus()
        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)


class RadialBasis(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]
        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(start=0, stop=1, num_gaussians=num_radial, **rbf_hparams)
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(num_radial=num_radial, cutoff=cutoff, **rbf_hparams)
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff
        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)


# ---------------------------------------------------------------------------
# ocpmodels/modules/scaling/scale_factor.py + compat.py
# ---------------------------------------------------------------------------
class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str):
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\nActual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor
    name: Optional[str] = None
    index_fn: Optional[IndexFn] = None
    stats: Optional[_Stats] = None

    def __init__(self, name: Optional[str] = None, enforce_consistency: bool = True):
        super().__init__()
        self.name = name
        self.index_fn = None
        self.stats = None
        self.scale_factor = nn.parameter.Parameter(torch.tensor(0.0), requires_grad=False)
        if enforce_consistency:
            self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if not self.fitted:
            return
        persistent_buffers = {
            k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue
            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self):
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self):
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: Union[float, torch.Tensor]):
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, float) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: Optional[IndexFn] = None):
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"
        self.stats["variance_in"] = self.stats["variance_in"] / self.stats["n_samples"]
        self.stats["variance_out"] = self.stats["variance_out"] / self.stats["n_samples"]
        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)
        self.set_(value)
        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return
        n_samples = x.shape[0]
        self.stats["variance_out"] += torch.mean(torch.var(x, dim=0)).item() * n_samples
        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += torch.mean(torch.var(ref, dim=0)).item() * n_samples
        self.stats["n_samples"] += n_samples

    def forward(self, x: torch.Tensor, *, ref: Optional[torch.Tensor] = None):
        if self.index_fn is not None:
            self.index_fn()
        if self.fitted:
            x = x * self.scale_factor
        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)
        return x


def load_scales_compat(module: nn.Module, scale_file):
    # scale_file is None in this recipe (random init, no pretrained scaling json) -> no-op.
    if not scale_file:
        return


# ---------------------------------------------------------------------------
# ocpmodels/common/utils.py (subset of free functions painn.py imports)
# ---------------------------------------------------------------------------
def compute_neighbors(data, edge_index):
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data.natoms.sum())
    image_indptr = torch.zeros(data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


def conditional_grad(dec):
    """Decorator to enable/disable grad depending on whether force/energy predictions are made."""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if self.regress_forces and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def get_k_index_product_set(num_k_x, num_k_y, num_k_z):
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1, dtype=torch.float),
        torch.arange(-num_k_y, num_k_y + 1, dtype=torch.float),
        torch.arange(-num_k_z, num_k_z + 1, dtype=torch.float),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    k_index_product_set = k_index_product_set[k_index_product_set.shape[0] // 2 + 1 :]
    num_k_degrees_of_freedom = k_index_product_set.shape[0]
    return k_index_product_set, num_k_degrees_of_freedom


def get_k_voxel_grid(k_cutoff, delta_k, num_k_rbf):
    num_k = k_cutoff / delta_k
    k_index_product_set, _ = get_k_index_product_set(num_k, num_k, num_k)
    k_cell = torch.tensor([[delta_k, 0, 0], [0, delta_k, 0], [0, 0, delta_k]], dtype=torch.float)
    k_grid = torch.matmul(k_index_product_set, k_cell)
    k_grid = k_grid[torch.sum(k_grid**2, dim=-1) <= k_cutoff**2]
    k_offset = 0.1 if num_k_rbf <= 48 else 0.25
    k_rbf_values = RadialBasis(
        num_radial=num_k_rbf,
        cutoff=k_cutoff + k_offset,
        rbf={"name": "gaussian"},
        envelope={"name": "polynomial", "exponent": 5},
    )(torch.linalg.norm(k_grid, dim=-1))
    num_k_degrees_of_freedom = k_rbf_values.shape[-1]
    return k_grid, k_rbf_values, num_k_degrees_of_freedom


def pos_svd_frame(data):
    pos = data.pos
    batch = data.batch
    batch_size = int(batch.max()) + 1
    with torch.cuda.amp.autocast(False):
        rotated_pos_list = []
        for i in range(batch_size):
            pos_batch = pos[batch == i]
            pos_batch = pos_batch - pos_batch.mean(0)
            if pos_batch.shape[0] > 2:
                U, S, V = torch.svd(pos_batch)
                rotated_pos_batch = torch.matmul(pos_batch, V)
            else:
                rotated_pos_batch = pos_batch
            rotated_pos_list.append(rotated_pos_batch)
        pos = torch.cat(rotated_pos_list)
    return pos


def x_to_k_cell(cells):
    cross_a2a3 = torch.cross(cells[:, 1], cells[:, 2], dim=-1)
    cross_a3a1 = torch.cross(cells[:, 2], cells[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cells[:, 0], cells[:, 1], dim=-1)
    vol = torch.sum(cells[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    b1 = 2 * np.pi * cross_a2a3 / vol
    b2 = 2 * np.pi * cross_a3a1 / vol
    b3 = 2 * np.pi * cross_a1a2 / vol
    bcells = torch.stack((b1, b2, b3), dim=1)
    return (bcells, vol[:, 0])


def get_pbc_distances(
    pos, edge_index, cell, cell_offsets, neighbors, return_offsets=False, return_distance_vec=False
):
    row, col = edge_index
    distance_vectors = pos[row] - pos[col]
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets
    distances = distance_vectors.norm(dim=-1)
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]
    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]
    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]
    return out


def radius_graph_pbc(data, radius, max_num_neighbors_threshold, pbc=[True, True, True]):
    device = data.pos.device
    batch_size = len(data.natoms)
    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. "
                    "This is not currently supported."
                )
    atom_pos = data.pos
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)
    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)
    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float) for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms_per_image_sqr, dim=0)

    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    pos2 = pos2 + pbc_offsets_per_atom
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    ).view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=num_atoms_per_image,
        index=index1,
        atom_distance_sqr=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )
    if not torch.all(mask_num_neighbors):
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell, mask_num_neighbors.view(-1, 1).expand(-1, 3)
        ).view(-1, 3)

    edge_index = torch.stack((index2, index1))
    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(natoms, index, atom_distance_sqr, max_num_neighbors_threshold):
    device = natoms.device
    num_atoms = natoms.sum()

    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors = segment_coo(torch.ones(len(index), device=device), index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    if max_num_neighbors <= max_num_neighbors_threshold or max_num_neighbors_threshold <= 0:
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(index_neighbor_offset, num_neighbors)
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    index_sort = (
        index_sort
        + index_neighbor_offset.view(-1, 1).expand(-1, max_num_neighbors_threshold)[mask_finite]
    )
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


# ---------------------------------------------------------------------------
# ocpmodels/models/painn/utils.py
# ---------------------------------------------------------------------------
def repeat_blocks(sizes, repeats, continuous_indexing=True, start_idx=0, block_inc=0, repeat_inc=0):
    assert sizes.dim() == 1
    assert all(sizes >= 0)
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    r1 = torch.repeat_interleave(torch.arange(len(sizes), device=sizes.device), repeats)
    N = (sizes * repeats).sum()
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        insert_val[idx] += block_inc

    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    id_ar[insert_index] = insert_val
    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1
    id_ar[0] += start_idx
    res = id_ar.cumsum(0)
    return res


def get_edge_id(edge_idx, cell_offsets, num_atoms):
    cell_basis = cell_offsets.max() - cell_offsets.min() + 1
    cell_id = (
        (cell_offsets * cell_offsets.new_tensor([[1, cell_basis, cell_basis**2]])).sum(-1).long()
    )
    edge_id = edge_idx[0] + edge_idx[1] * num_atoms + cell_id * num_atoms**2
    return edge_id


# ---------------------------------------------------------------------------
# ocpmodels/models/ewald_block.py -- the paper's architectural contribution
# ---------------------------------------------------------------------------
class EwaldBlock(torch.nn.Module):
    """Long-range block from the Ewald message passing method."""

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        activation=None,
        name=None,
        use_pbc: bool = True,
        delta_k: float = None,
        k_rbf_values: torch.Tensor = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.use_pbc = use_pbc
        self.return_k_params = return_k_params
        self.delta_k = delta_k
        self.k_rbf_values = k_rbf_values

        self.down = shared_downprojection
        self.up = Dense(downprojection_size, emb_size_atom, activation=None, bias=False)
        self.pre_residual = ResidualLayer(emb_size_atom, nLayers=2, activation=activation)
        self.ewald_layers = self.get_mlp(emb_size_atom, emb_size_atom, num_hidden, activation)
        if name is not None:
            self.ewald_scale_sum = ScaleFactor(name + "_sum")
        else:
            self.ewald_scale_sum = None

    def get_mlp(self, units_in, units, num_hidden, activation):
        dense1 = Dense(units_in, units, activation=activation, bias=False)
        mlp = [dense1]
        res = [ResidualLayer(units, nLayers=2, activation=activation) for i in range(num_hidden)]
        mlp += res
        return torch.nn.ModuleList(mlp)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        k: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        dot: torch.Tensor = None,
        sinc_damping: torch.Tensor = None,
    ):
        hres = self.pre_residual(h)
        if dot is None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)
        if sinc_damping is None:
            if self.use_pbc is False:
                sinc_damping = (
                    torch.sinc(0.5 * self.delta_k * x[:, 0].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 1].unsqueeze(-1))
                    * torch.sinc(0.5 * self.delta_k * x[:, 2].unsqueeze(-1))
                )
                sinc_damping = sinc_damping.expand(-1, k.shape[-2])
            else:
                sinc_damping = 1

        if self.use_pbc:
            self.kfilter = (
                torch.matmul(self.up.linear.weight, self.down.linear.weight)
                .T.unsqueeze(0)
                .expand(num_batch, -1, -1)
            )
        else:
            self.k_rbf_values = self.k_rbf_values.to(x.device)
            self.kfilter = (
                self.up(self.down(self.k_rbf_values)).unsqueeze(0).expand(num_batch, -1, -1)
            )

        sf_real = hres.new_zeros(num_batch, dot.shape[-1], hres.shape[-1]).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.cos(dot)).unsqueeze(-1).expand(-1, -1, hres.shape[-1]),
        )
        sf_imag = hres.new_zeros(num_batch, dot.shape[-1], hres.shape[-1]).index_add_(
            0,
            batch_seg,
            hres.unsqueeze(-2).expand(-1, dot.shape[-1], -1)
            * (sinc_damping * torch.sin(dot)).unsqueeze(-1).expand(-1, -1, hres.shape[-1]),
        )

        h_update = 0.01 * torch.sum(
            torch.index_select(sf_real * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.cos(dot)).unsqueeze(-1).expand(-1, -1, hres.shape[-1])
            + torch.index_select(sf_imag * self.kfilter, 0, batch_seg)
            * (sinc_damping * torch.sin(dot)).unsqueeze(-1).expand(-1, -1, hres.shape[-1]),
            dim=1,
        )

        if self.ewald_scale_sum is not None:
            h_update = self.ewald_scale_sum(h_update, ref=h)

        for layer in self.ewald_layers:
            h_update = layer(h_update)

        if self.return_k_params:
            return h_update, dot, sinc_damping
        else:
            return h_update


# ---------------------------------------------------------------------------
# ocpmodels/models/base.py (trimmed BaseModel: only generate_graph, which PaiNN uses)
# ---------------------------------------------------------------------------
class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(self, data, cutoff=None, max_neighbors=None, use_pbc=None, otf_graph=None):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if not otf_graph:
            try:
                edge_index = data.edge_index
                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors
            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(data, cutoff, max_neighbors)
            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )
            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_neighbors
                )
            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]
            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
            neighbors = compute_neighbors(data, edge_index)

        return edge_index, edge_dist, distance_vec, cell_offsets, cell_offset_distances, neighbors


# ---------------------------------------------------------------------------
# ocpmodels/models/painn/painn.py -- PaiNN backbone + Ewald wiring (the model class)
# ---------------------------------------------------------------------------
class PaiNN(BaseModel):
    r"""PaiNN model (Schuett et al. 2021) with the Ewald long-range message-passing block
    (Kosmala et al., ICML 2023) spliced in via `ewald_hyperparams`."""

    def __init__(
        self,
        num_atoms,
        bond_feat_dim,
        num_targets,
        hidden_channels=512,
        num_layers=6,
        num_rbf=128,
        cutoff=12.0,
        max_neighbors=50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        regress_forces=True,
        direct_forces=True,
        use_pbc=True,
        otf_graph=True,
        num_elements=83,
        scale_file: Optional[str] = None,
        ewald_hyperparams=None,
        atom_to_atom_cutoff=None,
    ):
        super(PaiNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.regress_forces = regress_forces
        self.direct_forces = direct_forces
        self.otf_graph = otf_graph
        self.use_pbc = use_pbc

        self.use_ewald = ewald_hyperparams is not None
        self.atom_to_atom_cutoff = atom_to_atom_cutoff
        self.use_atom_to_atom_mp = atom_to_atom_cutoff is not None

        if self.use_ewald:
            if self.use_pbc:
                self.num_k_x = ewald_hyperparams["num_k_x"]
                self.num_k_y = ewald_hyperparams["num_k_y"]
                self.num_k_z = ewald_hyperparams["num_k_z"]
                self.delta_k = None
            else:
                self.k_cutoff = ewald_hyperparams["k_cutoff"]
                self.delta_k = ewald_hyperparams["delta_k"]
                self.num_k_rbf = ewald_hyperparams["num_k_rbf"]
            self.downprojection_size = ewald_hyperparams["downprojection_size"]
            self.num_hidden = ewald_hyperparams["num_hidden"]

        if self.use_ewald:
            if self.use_pbc:
                (self.k_index_product_set, self.num_k_degrees_of_freedom) = get_k_index_product_set(
                    self.num_k_x, self.num_k_y, self.num_k_z
                )
                self.k_rbf_values = None
                self.delta_k = None
            else:
                (self.k_grid, self.k_rbf_values, self.num_k_degrees_of_freedom) = get_k_voxel_grid(
                    self.k_cutoff, self.delta_k, self.num_k_rbf
                )

            self.down = Dense(
                self.num_k_degrees_of_freedom, self.downprojection_size, activation=None, bias=False
            )
            self.ewald_blocks = torch.nn.ModuleList(
                [
                    EwaldBlock(
                        self.down,
                        hidden_channels,
                        self.downprojection_size,
                        self.num_hidden,
                        activation="silu",
                        use_pbc=self.use_pbc,
                        delta_k=self.delta_k,
                        k_rbf_values=self.k_rbf_values,
                    )
                    for i in range(self.num_layers)
                ]
            )

        if self.use_atom_to_atom_mp:
            if self.use_pbc:
                self.max_neighbors_at = int((self.atom_to_atom_cutoff / 6.0) ** 3 * 50)
            else:
                self.max_neighbors_at = 100
            from torch_geometric.nn.models.schnet import InteractionBlock

            self.interactions_at = torch.nn.ModuleList(
                [
                    InteractionBlock(hidden_channels, 200, 256, self.atom_to_atom_cutoff)
                    for i in range(self.num_layers)
                ]
            )
            self.distance_expansion_at = GaussianSmearing(0.0, self.atom_to_atom_cutoff, 200)

        self.skip_connection_factor = (
            2.0 + float(self.use_ewald) + float(self.use_atom_to_atom_mp)
        ) ** (-0.5)

        self.symmetric_edge_symmetrization = False

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)
        self.radial_basis = RadialBasis(
            num_radial=num_rbf, cutoff=self.cutoff, rbf=rbf, envelope=envelope
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for i in range(num_layers):
            self.message_layers.append(PaiNNMessage(hidden_channels, num_rbf).jittable())
            self.update_layers.append(PaiNNUpdate(hidden_channels))
            setattr(self, "upd_out_scalar_scale_%d" % i, ScaleFactor())

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        if self.regress_forces is True and self.direct_forces is True:
            self.out_forces = PaiNNOutput(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)

        self.reset_parameters()
        load_scales_compat(self, scale_file)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.out_energy[0].weight)
        self.out_energy[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_energy[2].weight)
        self.out_energy[2].bias.data.fill_(0)

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def symmetrize_edges(
        self,
        edge_index,
        cell_offsets,
        neighbors,
        batch_idx,
        reorder_tensors,
        reorder_tensors_invneg,
    ):
        num_atoms = batch_idx.shape[0]

        if self.symmetric_edge_symmetrization:
            edge_index_bothdir = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            cell_offsets_bothdir = torch.cat([cell_offsets, -cell_offsets], dim=0)
            edge_ids = get_edge_id(edge_index_bothdir, cell_offsets_bothdir, num_atoms)
            unique_ids, unique_inv = torch.unique(edge_ids, return_inverse=True)
            perm = torch.arange(
                unique_inv.size(0), dtype=unique_inv.dtype, device=unique_inv.device
            )
            unique_idx = scatter(
                perm, unique_inv, dim=0, dim_size=unique_ids.shape[0], reduce="min"
            )
            edge_index_new = edge_index_bothdir[:, unique_idx]
            edge_index_order = torch.argsort(edge_index_new[1])
            edge_index_new = edge_index_new[:, edge_index_order]
            unique_idx = unique_idx[edge_index_order]
            cell_offsets_new = cell_offsets_bothdir[unique_idx]
            reorder_tensors = [
                self.symmetrize_tensor(tensor, unique_idx, False) for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.symmetrize_tensor(tensor, unique_idx, True)
                for tensor in reorder_tensors_invneg
            ]
            ones = edge_index_new.new_ones(1).expand_as(edge_index_new[1])
            neighbors_per_atom = segment_coo(ones, edge_index_new[1], dim_size=num_atoms)
            neighbors_per_image = segment_coo(
                neighbors_per_atom, batch_idx, dim_size=neighbors.shape[0]
            )
        else:
            mask_sep_atoms = edge_index[0] < edge_index[1]
            cell_earlier = (
                (cell_offsets[:, 0] < 0)
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
                | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
            )
            mask_same_atoms = edge_index[0] == edge_index[1]
            mask_same_atoms &= cell_earlier
            mask = mask_sep_atoms | mask_same_atoms

            edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)
            edge_index_cat = torch.cat([edge_index_new, edge_index_new.flip(0)], dim=1)

            batch_edge = torch.repeat_interleave(
                torch.arange(neighbors.size(0), device=edge_index.device), neighbors
            )
            batch_edge = batch_edge[mask]
            ones = batch_edge.new_ones(1).expand_as(batch_edge)
            neighbors_per_image = 2 * segment_coo(ones, batch_edge, dim_size=neighbors.size(0))

            edge_reorder_idx = repeat_blocks(
                torch.div(neighbors_per_image, 2, rounding_mode="floor"),
                repeats=2,
                continuous_indexing=True,
                repeat_inc=edge_index_new.size(1),
            )

            edge_index_new = edge_index_cat[:, edge_reorder_idx]
            cell_offsets_new = self.select_symmetric_edges(
                cell_offsets, mask, edge_reorder_idx, True
            )
            reorder_tensors = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, False)
                for tensor in reorder_tensors
            ]
            reorder_tensors_invneg = [
                self.select_symmetric_edges(tensor, mask, edge_reorder_idx, True)
                for tensor in reorder_tensors_invneg
            ]

        edge_ids = get_edge_id(edge_index_new, cell_offsets_new, num_atoms)
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(edge_index_new.flip(0), -cell_offsets_new, num_atoms)
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_per_image,
            reorder_tensors,
            reorder_tensors_invneg,
            id_swap,
        )

    def generate_graph_values(self, data):
        (edge_index, edge_dist, distance_vec, cell_offsets, _, neighbors) = self.generate_graph(
            data
        )

        mask_zero = torch.isclose(edge_dist, torch.tensor(0.0), atol=1e-6)
        edge_dist[mask_zero] = 1.0e-6
        edge_vector = distance_vec / edge_dist[:, None]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError(
                f"An image has no neighbors: id={data.id[empty_image]}, "
                f"sid={data.sid[empty_image]}, fid={data.fid[empty_image]}"
            )

        (edge_index, cell_offsets, neighbors, [edge_dist], [edge_vector], id_swap) = (
            self.symmetrize_edges(
                edge_index, cell_offsets, neighbors, data.batch, [edge_dist], [edge_vector]
            )
        )

        return edge_index, neighbors, edge_dist, edge_vector, id_swap

    def forward(self, data):
        pos = pos_svd_frame(data) if (self.use_ewald and not self.use_pbc) else data.pos
        batch = data.batch
        batch_size = int(batch.max()) + 1
        z = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (edge_index, neighbors, edge_dist, edge_vector, id_swap) = self.generate_graph_values(data)

        if self.use_atom_to_atom_mp:
            (edge_index_at, edge_weight_at, distance_vec_at, cell_offsets_at, _, neighbors_at) = (
                self.generate_graph(
                    data, cutoff=self.atom_to_atom_cutoff, max_neighbors=self.max_neighbors_at
                )
            )
            edge_attr_at = self.distance_expansion_at(edge_weight_at)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)

        if self.use_ewald:
            if self.use_pbc:
                k_cell, _ = x_to_k_cell(data.cell)
                k_grid = torch.matmul(self.k_index_product_set.to(batch.device), k_cell)
            else:
                k_grid = self.k_grid.to(batch.device).unsqueeze(0).expand(batch_size, -1, -1)

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        if self.use_ewald or self.use_atom_to_atom_mp:
            dot = None
            sinc_damping = None
            for i in range(self.num_layers):
                if self.use_ewald:
                    dx_ewald, dot, sinc_damping = self.ewald_blocks[i](
                        x, pos, k_grid, batch_size, batch, dot, sinc_damping
                    )
                else:
                    dx_ewald = 0
                if self.use_atom_to_atom_mp:
                    dx_at = self.interactions_at[i](x, edge_index_at, edge_weight_at, edge_attr_at)
                else:
                    dx_at = 0
                dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)

                x = x + dx + dx_ewald + dx_at
                vec = vec + dvec
                x = x * self.skip_connection_factor

                dx, dvec = self.update_layers[i](x, vec)
                x = x + dx
                vec = vec + dvec
                x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)
        else:
            for i in range(self.num_layers):
                dx, dvec = self.message_layers[i](x, vec, edge_index, edge_rbf, edge_vector)
                x = x + dx
                vec = vec + dvec
                x = x * self.inv_sqrt_2

                dx, dvec = self.update_layers[i](x, vec)
                x = x + dx
                vec = vec + dvec
                x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0)

        if self.regress_forces:
            if self.direct_forces:
                forces = self.out_forces(x, vec)
                return energy, forces
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        x, pos, grad_outputs=torch.ones_like(x), create_graph=True
                    )[0]
                )
                return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"max_neighbors={self.max_neighbors}, "
            f"cutoff={self.cutoff})"
        )


class PaiNNMessage(MessagePassing):
    def __init__(self, hidden_channels, num_rbf):
        super(PaiNNMessage, self).__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels
        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))
        rbfh = self.rbf_proj(edge_rbf)
        dx, dvec = self.propagate(
            edge_index, xh=xh, vec=vec, rbfh_ij=rbfh, r_ij=edge_vector, size=None
        )
        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h
        return x, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h
        x_vec_h = self.xvec_proj(
            torch.cat([x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1)
        )
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)
        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2
        dvec = xvec3.unsqueeze(1) * vec1
        return dx, dvec


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block, as defined in Schuett et al. (2021). Borrowed from TorchMD-Net."""

    def __init__(self, hidden_channels, out_channels):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels
        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, out_channels * 2),
        )
        self.act = ScaledSiLU()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2
        x = self.act(x)
        return x, v


class PaiNNOutput(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(hidden_channels, hidden_channels // 2),
                GatedEquivariantBlock(hidden_channels // 2, 1),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def forward(self, x, vec):
        for layer in self.output_network:
            x, vec = layer(x, vec)
        return vec.squeeze()


# ---------------------------------------------------------------------------
# Menagerie build/example plumbing
# ---------------------------------------------------------------------------
def build_ewald_painn():
    """Tiny random-init PaiNN+Ewald model (non-periodic path, direct force head)."""
    torch.manual_seed(0)
    return PaiNN(
        num_atoms=None,
        bond_feat_dim=None,
        num_targets=1,
        hidden_channels=32,
        num_layers=2,
        num_rbf=16,
        cutoff=6.0,
        max_neighbors=8,
        regress_forces=True,
        direct_forces=True,
        use_pbc=False,
        otf_graph=True,
        num_elements=20,
        scale_file=None,
        ewald_hyperparams={
            "k_cutoff": 0.4,
            "delta_k": 0.2,
            "num_k_rbf": 12,
            "downprojection_size": 8,
            "num_hidden": 1,
        },
    )


def example_input_ewald_painn():
    """A small non-periodic molecular graph (a torch_geometric Data-like object) -- 6 atoms."""
    from torch_geometric.data import Data

    torch.manual_seed(0)
    n_atoms = 6
    pos = torch.randn(n_atoms, 3) * 2.0
    atomic_numbers = torch.randint(1, 10, (n_atoms,))
    batch = torch.zeros(n_atoms, dtype=torch.long)
    natoms = torch.tensor([n_atoms])
    data = Data(pos=pos, atomic_numbers=atomic_numbers, batch=batch)
    data.natoms = natoms
    data.id = torch.tensor([0])
    data.sid = torch.tensor([0])
    data.fid = torch.tensor([0])
    return (data,)


MENAGERIE_ENTRIES = [
    ("EwaldMP (PaiNN+Ewald)", build_ewald_painn, example_input_ewald_painn, 2023, "REAL"),
]
