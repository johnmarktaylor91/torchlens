# SOURCE: vendored from https://github.com/OnlyLoveKFC/Neural_P3M @ 0af68d81ca68
# (src/models/components/schnet_p3m.py + src/models/components/utils.py +
#  src/models/components/radius_utils.py + src/models/components/fno.py +
#  src/models/components/fno_layers.py)
#
# Neural P3M: A Long-Range Interaction Modeling Enhancer for Geometric GNNs
# (Wang, Cheng, Li, Ren, Shao, Liu et al., NeurIPS 2024, arXiv:2409.17622).
# Neural P3M augments geometric GNNs (SchNet/DimeNet++/PaiNN/GemNet-T here) with
# an explicit 3D mesh alongside the atoms, exchanging short-range atom-atom
# message passing with long-range atom-mesh-atom message passing (mesh update
# via an FNO3d Fourier neural operator or multi-head attention over the grid),
# reimagining P3M-style (Particle-Particle Particle-Mesh) long-range interaction
# modeling in a trainable form. This file vendors the SchNet_P3M variant
# (`configs/experiment/schnet_oe62.yaml`, long_type="FNO"), the smallest of the
# four enhanced backbones shipped in the repo. No architecture was altered;
# only relative imports were flattened into this single file and a synthetic
# HeteroData atom+mesh graph is fabricated in place of the repo's own
# OE62/AG/MD22 LMDB data pipeline (which needs ase/lmdb and is not part of the
# model definition).

import math
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.models.schnet import GaussianSmearing
from torch_scatter import scatter, segment_coo, segment_csr

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# src/models/components/fno_layers.py
# (modified from the `neuraloperator` package: https://github.com/neuraloperator/neuraloperator)
# THIS CODE IS HARDCODED FOR 3D GRID DATA
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.silu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0.0 else None
        )
        Conv = getattr(nn, f"Conv{n_dim}d")
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(Conv(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x


class SoftGating(nn.Module):
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.ones_(self.bias)

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x


class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = self.n_modes
        self.n_layers = n_layers

        self.weight = nn.ParameterList(
            [
                nn.Parameter(
                    torch.view_as_real(
                        torch.empty(
                            in_channels, out_channels, *self.max_n_modes, dtype=torch.cfloat
                        )
                    )
                )
                for _ in range(n_layers)
            ]
        )
        self.bias = nn.Parameter(torch.empty(*((n_layers, self.out_channels) + (1,) * self.order)))

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight:
            nn.init.normal_(w, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)
        nn.init.normal_(self.bias, 0, (2 / (self.in_channels + self.out_channels)) ** 0.5)

    def _get_weight(self, index):
        return torch.view_as_complex(self.weight[index])

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    def forward(self, x: torch.Tensor, indices=0):
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))

        x = torch.fft.rfftn(x, norm="backward", dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat
        )
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        slices_w = [slice(None), slice(None)]  # Batch_size, channels
        slices_w += [
            slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]
        ]
        # The last mode already has redundant half removed
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        weight = self._get_weight(indices)[slices_w]

        starts = [
            (size - min(size, n_mode))
            for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))
        ]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [
            slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]
        ]
        # The last mode already has redundant half removed
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)]
        out_fft[slices_x] = torch.einsum("bcxyz,cdxyz->bdxyz", x[slices_x], weight)

        if self.order > 1:
            out_fft = torch.fft.ifftshift(out_fft, dim=fft_dims[:-1])
        x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm="backward")
        x = x + self.bias[indices, ...]
        return x


class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, non_linearity=F.silu):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                SoftGating(
                    self.in_channels,
                    self.out_channels,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.convs.reset_parameters()
        for fno_skip in self.fno_skips:
            fno_skip.reset_parameters()

    def forward(self, x, index=0):
        x_skip_fno = self.fno_skips[index](x)
        x_fno = self.convs(x, index)
        x = x_fno + x_skip_fno
        if index < (self.n_layers - 1):
            x = self.non_linearity(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes


# ---------------------------------------------------------------------------
# src/models/components/fno.py
# (modified from the `neuraloperator` package)
# ---------------------------------------------------------------------------


class FNO(nn.Module):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=1,
        lifting_channels=256,
        projection_channels=256,
        non_linearity=F.silu,
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            non_linearity=non_linearity,
        )

        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
            )
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.lifting.reset_parameters()
        self.fno_blocks.reset_parameters()
        self.projection.reset_parameters()

    def forward(self, x):
        x = self.lifting(x)
        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
        x = self.projection(x)
        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class FNO3d(FNO):
    def __init__(
        self,
        n_modes_height,
        n_modes_width,
        n_modes_depth,
        hidden_channels,
        in_channels=1,
        out_channels=1,
        n_layers=1,
        lifting_channels=256,
        projection_channels=256,
        non_linearity=F.silu,
    ):
        super().__init__(
            n_modes=(n_modes_height, n_modes_width, n_modes_depth),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            non_linearity=non_linearity,
        )
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width
        self.n_modes_depth = n_modes_depth


# ---------------------------------------------------------------------------
# src/models/components/utils.py
# ---------------------------------------------------------------------------


class Scalar(nn.Module):
    def __init__(self, hidden_channels):
        super(Scalar, self).__init__()
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def forward(self, x):
        return self.output_network(x)


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = nn.SiLU()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, dim_size=None):
        x = self.conv(x, edge_index, edge_weight, edge_attr, dim_size)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        mlp: nn.Sequential,
        cutoff: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.mlp = mlp
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr, dim_size=None):
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.mlp(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x_j = torch.index_select(x, 0, edge_index[0])
        x_j = x_j * W
        x = scatter(x_j, edge_index[1], dim=0, reduce="add", dim_size=dim_size)
        x = self.lin2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scaled_scores = scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        values = torch.matmul(attention_weights, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o


# ---------------------------------------------------------------------------
# src/models/components/radius_utils.py (non-PBC path only is exercised here;
# the PBC helpers are vendored too since they live in the same source file,
# but are unreachable when use_pbc=False as in the shipped schnet_oe62 config)
# ---------------------------------------------------------------------------


def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
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


# Borrowed from GemNet.
def select_symmetric_edges(tensor, mask, reorder_idx, inverse_neg):
    tensor_directed = tensor[mask]
    sign = 1 - 2 * inverse_neg
    tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
    tensor_ordered = tensor_cat[reorder_idx]
    return tensor_ordered


def symmetrize_edges(
    edge_index,
    cell_offsets,
    neighbors,
):
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
    cell_offsets_new = select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)

    return edge_index_new, cell_offsets_new, neighbors_per_image


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

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
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold
    else:
        effective_cutoff = distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        is_included = torch.le(distance_sort.T, effective_cutoff)

        distance_sort[~is_included.T] = np.inf

        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(-1, max_num_included)
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def radius_graph_determinstic(
    data,
    radius,
    max_num_neighbors_threshold,
    symmetrize: bool = True,
    enforce_max_neighbors_strictly: bool = False,
):
    device = data.pos.device

    # position of the atoms
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
        + index_offset_expand
    )
    index2 = atom_count_sqr % num_atoms_per_image_expand + index_offset_expand
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))

    if symmetrize:
        edge_index, unit_cell, num_neighbors_image = symmetrize_edges(
            edge_index,
            torch.zeros(edge_index.shape[1], 3, device=edge_index.device),
            num_neighbors_image,
        )

    return edge_index, num_neighbors_image  # GemNet uses num_neighbors_image


def radius_determinstic(
    data_x,
    data_y,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
):
    device = data_x.pos.device

    atom_pos = data_x.pos
    mesh_pos = data_y.pos

    num_atoms_per_image = data_x.natoms
    num_meshs_per_image = data_y.nmeshs
    num_atoms_per_image_sqr = (num_atoms_per_image * num_meshs_per_image).long()

    atom_index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    atom_index_offset_expand = torch.repeat_interleave(atom_index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    mesh_index_offset = torch.cumsum(num_meshs_per_image, dim=0) - num_meshs_per_image
    mesh_index_offset_expand = torch.repeat_interleave(mesh_index_offset, num_atoms_per_image_sqr)

    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, num_atoms_per_image_sqr)
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
        + mesh_index_offset_expand
    )
    index2 = atom_count_sqr % num_atoms_per_image_expand + atom_index_offset_expand

    mesh_pos1 = torch.index_select(mesh_pos, 0, index1)
    atom_pos2 = torch.index_select(atom_pos, 0, index2)

    atom_distance_sqr = torch.sum((mesh_pos1 - atom_pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)

    edge_index = torch.stack((index2, index1))

    return edge_index


def get_distances(
    edge_index,
    pos2,  # src pos
    pos1=None,  # dst pos
    return_distance_vec=True,
):
    j, i = edge_index

    if pos1 is None:
        pos1 = pos2

    distance_vec = pos2[j] - pos1[i]

    edge_dist = distance_vec.norm(dim=-1)

    distance_vec = distance_vec / edge_dist.view(-1, 1)

    if return_distance_vec:
        return edge_dist, distance_vec
    else:
        return edge_dist


# ---------------------------------------------------------------------------
# src/models/components/schnet_p3m.py
# ---------------------------------------------------------------------------


class SchNet_P3M(nn.Module):
    def __init__(
        self,
        regress_forces: bool,
        use_pbc: bool,
        num_layers: int,
        num_rbf: int,
        num_filters: int,
        hidden_channels: int,
        max_z: int,
        atom_cutoff: int,
        max_a2a_neighbors: int,
        grid_cutoff: int,
        max_a2m_neighbors: int,
        num_grids: Union[List[int], int],
        long_type: str,  # 'FNO' or 'MHA'
    ):
        super().__init__()
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.num_filters = num_filters
        self.hidden_channels = hidden_channels
        self.max_z = max_z
        self.atom_cutoff = atom_cutoff
        self.max_a2a_neighbors = max_a2a_neighbors
        self.grid_cutoff = grid_cutoff
        self.max_a2m_neighbors = max_a2m_neighbors

        if isinstance(num_grids, int):
            self.num_grids = [num_grids, num_grids, num_grids]
        else:
            self.num_grids = num_grids

        self.total_num_grids = self.num_grids[0] * self.num_grids[1] * self.num_grids[2]

        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.a2a_distance_expansion = GaussianSmearing(0.0, atom_cutoff, num_rbf)
        self.a2m_distance_expansion = GaussianSmearing(0.0, grid_cutoff, num_rbf)
        self.m2a_distance_expansion = GaussianSmearing(0.0, grid_cutoff, num_rbf)

        self.sl_block = nn.ModuleList()
        for _ in range(num_layers):
            a2m_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            m2a_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, grid_cutoff)
            short_mp = InteractionBlock(hidden_channels, num_rbf, num_filters, atom_cutoff)
            if long_type == "FNO":
                long_mp = FNO3d(
                    *self.num_grids,
                    hidden_channels=hidden_channels // 2,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_layers=1,
                    lifting_channels=hidden_channels // 2,
                    projection_channels=hidden_channels // 2,
                    non_linearity=nn.SiLU(),
                )
            elif long_type == "MHA":
                long_mp = MultiheadAttention(hidden_channels, hidden_channels, 8)
            else:
                raise ValueError(f"Unknown long range interaction type: {long_type}")
            self.sl_block.append(
                ShortLongMixLayer(
                    hidden_channels,
                    self.num_grids,
                    a2m_mp,
                    m2a_mp,
                    short_mp,
                    long_mp,
                )
            )
        self.out_a_norm = nn.LayerNorm(hidden_channels)
        self.out_m_norm = nn.LayerNorm(hidden_channels)
        self.a_output = Scalar(hidden_channels)
        self.m_output = Scalar(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for block in self.sl_block:
            block.reset_parameters()
        self.out_a_norm.reset_parameters()
        self.out_m_norm.reset_parameters()
        self.a_output.reset_parameters()
        self.m_output.reset_parameters()

    def forward(self, mesh_atom_graph):
        if self.regress_forces:
            mesh_atom_graph["atom"].pos.requires_grad_(True)

        bs = mesh_atom_graph.num_graphs if hasattr(mesh_atom_graph["atom"], "batch") else 1

        a_pos = mesh_atom_graph["atom"].pos
        m_pos = mesh_atom_graph["mesh"].pos

        if not self.use_pbc:
            a2a_edge_index, _ = radius_graph_determinstic(
                mesh_atom_graph["atom"],
                self.atom_cutoff,
                self.max_a2a_neighbors,
                symmetrize=True,
            )
            a2a_edge_weights = get_distances(a2a_edge_index, a_pos, return_distance_vec=False)
            a2m_edge_index = radius_determinstic(
                mesh_atom_graph["atom"],
                mesh_atom_graph["mesh"],
                self.grid_cutoff,
                self.max_a2m_neighbors,
            )
            a2m_edge_weights = get_distances(
                a2m_edge_index, a_pos, m_pos, return_distance_vec=False
            )
            m2a_edge_index = a2m_edge_index.flip(0)
            m2a_edge_weights = get_distances(
                m2a_edge_index, m_pos, a_pos, return_distance_vec=False
            )
        else:
            raise NotImplementedError(
                "PBC path (radius_graph_pbc/radius_pbc) is vendored in the real repo "
                "but unreachable for the schnet_oe62 config used here (use_pbc=False)."
            )

        a2a_edge_attr = self.a2a_distance_expansion(a2a_edge_weights)
        a2m_edge_attr = self.a2m_distance_expansion(a2m_edge_weights)
        m2a_edge_attr = self.m2a_distance_expansion(m2a_edge_weights)

        # N_atoms, F
        a_x = self.embedding(mesh_atom_graph["atom"].atomic_numbers)
        # NonTrainable Message Passing For Initial Mesh Embedding
        a_x_j = torch.index_select(a_x, 0, a2m_edge_index[0])
        m_x = scatter(
            a_x_j, a2m_edge_index[1], dim=0, reduce="mean", dim_size=self.total_num_grids * bs
        )

        for i in range(self.num_layers):
            a_x, m_x = self.sl_block[i](
                a_x,
                m_x,
                a2a_edge_index,
                a2m_edge_index,
                m2a_edge_index,
                a2a_edge_weights,
                a2m_edge_weights,
                m2a_edge_weights,
                a2a_edge_attr,
                a2m_edge_attr,
                m2a_edge_attr,
            )

        out_a_x = self.out_a_norm(a_x)
        out_m_x = self.out_m_norm(m_x)

        output_a_x = self.a_output(out_a_x)
        energy_a = scatter(
            output_a_x,
            mesh_atom_graph["atom"].batch
            if hasattr(mesh_atom_graph["atom"], "batch")
            else torch.zeros_like(mesh_atom_graph["atom"].atomic_numbers),
            dim=0,
            reduce="sum",
        )
        output_m_x = self.m_output(out_m_x)

        energy_m = torch.sum(output_m_x.reshape(bs, -1), dim=-1, keepdim=True)
        energy = energy_a + energy_m

        if self.regress_forces:
            forces = (
                -1
                * (
                    torch.autograd.grad(
                        energy,
                        mesh_atom_graph["atom"].pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True,
                    )[0]
                )
            )
            return energy, forces
        else:
            return energy, None


class ShortLongMixLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_grids: List[int],
        a2m_mp: nn.Module,
        m2a_mp: nn.Module,
        short_mp: nn.Module,
        long_mp: nn.Module,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.a2m_mp = a2m_mp
        self.m2a_mp = m2a_mp
        self.short_mp = short_mp
        self.long_mp = long_mp
        self.num_grids = num_grids
        self.a2m_layernorm = nn.LayerNorm(hidden_channels)
        self.m2a_layernorm = nn.LayerNorm(hidden_channels)
        self.short_layernorm = nn.LayerNorm(hidden_channels)
        self.long_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.a2m_mp.reset_parameters()
        self.m2a_mp.reset_parameters()
        self.short_mp.reset_parameters()
        self.long_mp.reset_parameters()
        self.a2m_layernorm.reset_parameters()
        self.m2a_layernorm.reset_parameters()
        self.short_layernorm.reset_parameters()
        self.long_layernorm.reset_parameters()

    def forward(
        self,
        a_x,
        m_x,
        a2a_edge_index,
        a2m_edge_index,
        m2a_edge_index,
        a2a_edge_weights,
        a2m_edge_weights,
        m2a_edge_weights,
        a2a_edge_attr,
        a2m_edge_attr,
        m2a_edge_attr,
    ):
        delta_a_x, delta_m_x = a_x, m_x

        # N_atoms, F
        a_x = self.short_layernorm(a_x)
        a_x = self.short_mp(
            a_x, a2a_edge_index, a2a_edge_weights, a2a_edge_attr, dim_size=a_x.shape[0]
        )

        # N_meshs, F
        m_x = self.long_layernorm(m_x)
        if isinstance(self.long_mp, MultiheadAttention):
            m_x = m_x.reshape(-1, torch.prod(torch.tensor(self.num_grids)), self.hidden_channels)
            m_x = self.long_mp(m_x)
            m_x = m_x.reshape(-1, self.hidden_channels)
        else:
            m_x = m_x.reshape(
                -1, self.num_grids[0], self.num_grids[1], self.num_grids[2], self.hidden_channels
            ).permute(0, 4, 1, 2, 3)
            m_x = self.long_mp(m_x).permute(0, 2, 3, 4, 1).reshape(-1, self.hidden_channels)

        # N_meshs, F
        a2m_message = self.a2m_mp(
            a_x, a2m_edge_index, a2m_edge_weights, a2m_edge_attr, dim_size=m_x.shape[0]
        )
        a2m_message = self.a2m_layernorm(a2m_message)

        # N_atoms, F
        m2a_message = self.m2a_mp(
            m_x, m2a_edge_index, m2a_edge_weights, m2a_edge_attr, dim_size=a_x.shape[0]
        )
        m2a_message = self.m2a_layernorm(m2a_message)

        return a_x + m2a_message + delta_a_x, m_x + a2m_message + delta_m_x


# ---------------------------------------------------------------------------
# build_ / example_input_
# ---------------------------------------------------------------------------


def build_neural_p3m():
    torch.manual_seed(0)
    model = SchNet_P3M(
        regress_forces=False,
        use_pbc=False,
        num_layers=2,
        num_rbf=16,
        num_filters=24,
        hidden_channels=24,
        max_z=100,
        atom_cutoff=6.0,
        max_a2a_neighbors=10,
        grid_cutoff=4.0,
        max_a2m_neighbors=5,
        num_grids=3,  # matches configs/data/oe62.yaml's transforms.num_grids
        long_type="FNO",
    )
    model.eval()
    return model


def example_input_neural_p3m():
    # A tiny synthetic single-molecule HeteroData atom+mesh graph, built the
    # same way `NonPBCAddGrid.__call__` builds it in
    # src/data/components/transform.py (a small water-cluster-sized molecule
    # of 6 atoms plus a 3x3x3 mesh grid); the real repo constructs this from
    # OE62/AG/MD22 LMDB records via `src.data.oe62_datamodule`, which needs
    # ase/lmdb and is a data-loading concern, not part of the model.
    torch.manual_seed(0)
    n_atoms = 6
    num_grids = [3, 3, 3]  # must match build_neural_p3m()'s num_grids

    mesh_atom_graph = HeteroData()
    mesh_atom_graph["atom"].pos = torch.randn(n_atoms, 3)
    mesh_atom_graph["atom"].atomic_numbers = torch.randint(1, 20, (n_atoms,), dtype=torch.long)
    mesh_atom_graph["atom"].natoms = torch.tensor([n_atoms], dtype=torch.long)

    x_centers = torch.linspace(-1.0, 1.0, num_grids[0])
    y_centers = torch.linspace(-1.0, 1.0, num_grids[1])
    z_centers = torch.linspace(-1.0, 1.0, num_grids[2])
    mesh = torch.stack(torch.meshgrid(x_centers, y_centers, z_centers, indexing="ij"), dim=-1)
    mesh_atom_graph["mesh"].pos = mesh.reshape(-1, 3).float()
    mesh_atom_graph["mesh"].nmeshs = torch.tensor(
        [mesh_atom_graph["mesh"].pos.shape[0]], dtype=torch.long
    )

    return (mesh_atom_graph,)


MENAGERIE_ENTRIES = [
    (
        "Neural P3M (SchNet+FNO)",
        "build_neural_p3m",
        "example_input_neural_p3m",
        2024,
        MENAGERIE_ZOO,
    ),
]
