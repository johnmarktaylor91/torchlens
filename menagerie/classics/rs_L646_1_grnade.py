# SOURCE: vendored from chaitjo/geometric-rna-design @ 2453b18778a1981ac25f314790e5395f1e9ab218
# Files: src/layers.py (verbatim) + src/models.py (gRNAde class, verbatim).
# Only change: src/layers.py's `from src.layers import *` in models.py was replaced by
# in-file definitions (this staging module inlines both files); no architectural changes.
################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

import functools
import sys
from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

# Harness note (not part of vendored code): torch_geometric.nn.MessagePassing
# introspects `message()`'s signature via `sys.modules[cls.__module__]` at
# subclass-construction time. When this file is loaded with
# `importlib.util.spec_from_file_location` + `module_from_spec` (as the
# menagerie staging harness does) without also registering the resulting
# module object in `sys.modules`, that lookup raises `KeyError`. Guard by
# registering a module object exposing this file's own globals under our
# `__name__`, so the file is loadable both as a bare staged file and once
# integrated as a normal package import (where `sys.modules[__name__]` is
# already populated by the import machinery and this is a no-op).
if __name__ not in sys.modules:
    import types as _types

    _self_module = _types.ModuleType(__name__)
    _self_module.__dict__.update(globals())
    sys.modules[__name__] = _self_module

MENAGERIE_ZOO = "vendored-pytorch"

#########################################################################


class GVPConvLayer(nn.Module):
    """Full graph convolution / message passing layer with Geometric Vector Perceptrons. Residually
    updates node embeddings with aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        autoregressive=False,
        activations=(F.silu, torch.sigmoid),
        vector_gate=True,
        residual=True,
        norm_first=False,
    ):
        super().__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr="add" if autoregressive else "mean",
            activations=activations,
            vector_gate=vector_gate,
        )
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The current node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward),
            )

            count = (
                scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0))
                .clamp(min=1)
                .unsqueeze(-1)
            )

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            if self.norm_first:
                dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            else:
                dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        if self.norm_first:
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class GVPConv(MessagePassing):
    """Graph convolution / message passing with Geometric Vector Perceptrons. Takes in a graph with
    node and edge embeddings, and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        module_list=None,
        aggr="mean",
        activations=(F.silu, torch.sigmoid),
        vector_gate=True,
    ):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), (self.so, self.vo))
                )
            else:
                module_list.append(GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims))
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x
        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
            edge_attr=edge_attr,
        )
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


#########################################################################


class MultiGVPConvLayer(nn.Module):
    """GVPConvLayer for handling multiple conformations (encoder-only)"""

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        activations=(F.silu, torch.sigmoid),
        vector_gate=True,
        residual=True,
        norm_first=False,
    ):
        super().__init__()
        self.conv = MultiGVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr="mean",
            activations=activations,
            vector_gate=vector_gate,
        )
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        if self.norm_first:
            dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            dh = self.conv(x, edge_index, edge_attr)
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        return x


class MultiGVPConv(MessagePassing):
    """GVPConv for handling multiple conformations."""

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        module_list=None,
        aggr="mean",
        activations=(F.silu, torch.sigmoid),
        vector_gate=True,
    ):
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), (self.so, self.vo))
                )
            else:
                module_list.append(GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims))
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x
        n_conf = x_s.shape[1]

        # x_s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
        x_s = x_s.contiguous().view(x_s.shape[0], x_s.shape[1] * x_s.shape[2])
        # x_v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
        x_v = x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * x_v.shape[2] * 3)

        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)

        return _split_multi(message, self.so, self.vo, n_conf)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        # [n_nodes, n_conf * d] -> [n_nodes, n_conf, d]
        s_i = s_i.view(s_i.shape[0], s_i.shape[1] // self.si, self.si)
        s_j = s_j.view(s_j.shape[0], s_j.shape[1] // self.si, self.si)
        # [n_nodes, n_conf * d * 3] -> [n_nodes, n_conf, d, 3]
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // (self.vi * 3), self.vi, 3)
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // (self.vi * 3), self.vi, 3)

        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge_multi(*message)


#########################################################################


class GVP(nn.Module):
    """Geometric Vector Perceptron. See manuscript and README.md for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self, in_dims, out_dims, h_dim=None, activations=(F.silu, torch.sigmoid), vector_gate=True
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


#########################################################################


class _VDropout(nn.Module):
    """Vector channel dropout where the elements of each vector channel are dropped together."""

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """Combined dropout for tuples (s, V).

    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """Combined LayerNorm for tuples (s, V).

    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


def tuple_sum(*args):
    """Sums any number of tuples (s, V) elementwise."""
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    """Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    :return: (s, V) with s.shape = (n, n_scalar) and V.shape = (n, n_vector, 3)
    """
    return torch.randn(n, dims[0], device=device), torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    """Splits a merged representation of (s, V) back into a tuple. Should be used only with
    `_merge(s, V)` and only if the tuple representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    s = x[..., : -3 * nv]
    v = x[..., -3 * nv :].contiguous().view(x.shape[0], nv, 3)
    return s, v


def _merge(s, v):
    """Merges a tuple (s, V) into a single `torch.Tensor`, where the vector channels are flattened
    and appended to the scalar channels.

    Should be used only if the tuple representation cannot be used. Use `_split(x, nv)` to reverse.
    """
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)


def _split_multi(x, ns, nv, n_conf=5):
    """_split for multiple conformers."""
    s = x[..., : -3 * nv * n_conf].contiguous().view(x.shape[0], n_conf, ns)
    v = x[..., -3 * nv * n_conf :].contiguous().view(x.shape[0], n_conf, nv, 3)
    return s, v


def _merge_multi(s, v):
    """_merge for multiple conformers."""
    # s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
    s = s.contiguous().view(s.shape[0], s.shape[1] * s.shape[2])
    # v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
    v = v.contiguous().view(v.shape[0], v.shape[1] * v.shape[2] * 3)
    return torch.cat([s, v], -1)


class gRNAde(torch.nn.Module):
    """gRNAde: Generative model for RNA inverse design.

    The model accepts flexible multi-modal inputs that can include:
        - Target pseudoknotted secondary structure (2D base pairing patterns)
        - 3D backbone coordinates from experimentally determined or predicted structures
        - Partial sequence constraints (e.g., functional motifs to preserve)
    All three input modalities are optional, allowing the model to design from
    incomplete structural information.

    Architecture:
        The core architecture consists of an SE(3)-equivariant Graph Neural Network
        with Geometric Vector Perceptron (GVP) layers. Each nucleotide is represented
        as a node in a geometric graph with three types of edges:
        - Backbone connectivity: 32 nearest neighbors along the RNA chain (1D)
        - Base pairing: all nucleotides involved in pairs and pseudoknots (2D)
        - Spatial proximity: 32 nearest neighbors in 3D space (if structure provided)

        An encoder-decoder architecture processes these graphs:
        - GVP-GNN encoder layers process the structural information into latent
          representations that capture both scalar and vector (geometric) features
        - Autoregressive GVP-GNN decoder generates sequences from 5' to 3', with
          optional logit biasing to enforce sequence constraints

    Multi-conformation support:
        The model explicitly handles multiple structural conformations (multi-state
        design), pooling features across conformations using masked mean pooling to
        design sequences that fold consistently across different structural states.

    Training:
        Trained on 4,211 diverse RNA structures up to 500 nucleotides from the PDB
        (resolution ≤4.0Å). A novel 3D dropout strategy during training enables
        the model to design even without complete 3D information, relying on implicit
        3D constraints from secondary structure alone.

    Usage:
        The standard forward pass requires sequence information as input and should
        be used for training or evaluating likelihood. For sequence design and
        sampling, use the `sample()` method which generates sequences autoregressively.

    Input format:
        Takes in RNA structure graphs of type `torch_geometric.data.Data` or
        `torch_geometric.data.Batch` and returns a categorical distribution over
        4 nucleotides at each position in a `torch.Tensor` of shape [n_nodes, 4].
        See data/featurizer.py for details on graph construction and features.

    Args:
        node_in_dim (tuple): Input node feature dimensions as (scalar_dim, vector_dim).
            Nodes are featurized using 3-bead coarse-grained representation (P, C4',
            N1/N9 atoms) and local geometric properties. Default: (15, 4).
        node_h_dim (tuple): Hidden node feature dimensions as (scalar_dim, vector_dim)
            for GVP-GNN layers. Default: (128, 16).
        edge_in_dim (tuple): Input edge feature dimensions as (scalar_dim, vector_dim).
            Edges encode edge type, relative distances and orientations between nodes.
            Default: (132, 3).
        edge_h_dim (tuple): Hidden edge feature dimensions as (scalar_dim, vector_dim)
            for GVP-GNN layers. Default: (64, 4).
        num_layers (int): Number of GVP-GNN layers in both encoder and decoder.
            Default: 4.
        drop_rate (float): Dropout rate applied in all dropout layers for
            regularization. Default: 0.5.
        out_dim (int): Output dimension corresponding to number of nucleotide types.
            Default: 4 (A, C, G, U).
    """

    def __init__(
        self,
        node_in_dim=(15, 4),
        node_h_dim=(128, 16),
        edge_in_dim=(132, 3),
        edge_h_dim=(64, 4),
        num_layers=4,
        drop_rate=0.5,
        out_dim=4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)

        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None), vector_gate=True),
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None), vector_gate=True),
        )

        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
            MultiGVPConvLayer(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim + 1, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
            GVPConvLayer(
                self.node_h_dim,
                self.edge_h_dim,
                activations=activations,
                vector_gate=True,
                drop_rate=drop_rate,
                autoregressive=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))

    def forward(self, batch):
        """Forward pass for sequence likelihood computation and training.

        Processes RNA structure graph(s) through encoder layers, pools multi-conformation
        features, and decodes autoregressively to produce logits over nucleotide
        vocabulary at each position. Requires ground truth sequence information.

        Args:
            batch (torch_geometric.data.Data or torch_geometric.data.Batch):
                Input graph(s) containing:
                - node_s: node scalar features
                - node_v: node vector features
                - edge_s: edge scalar features
                - edge_v: edge vector features
                - edge_index: graph connectivity
                - seq: ground truth sequence for autoregressive decoding
                - mask_confs: mask indicating valid conformations

        Returns:
            logits (torch.Tensor): Unnormalized log probabilities over nucleotide
                vocabulary of shape [n_nodes, 4]
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features:
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V

        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = self.W_out(h_V)

        return logits

    @torch.no_grad()
    def sample(
        self,
        batch,
        n_samples,
        temperature: Optional[float] = 0.1,
        logit_bias: Optional[torch.Tensor] = None,
        return_logits: Optional[bool] = False,
    ):
        """Sample sequences autoregressively from the learned distribution.

        Generates multiple sequence designs for a given RNA structure by sampling
        from the categorical distribution at each position. The model encodes the
        structure, pools multi-conformation features, then autoregressively decodes
        nucleotides one position at a time. No gradients are computed during sampling.

        Args:
            batch (torch_geometric.data.Data): Input graph containing a single RNA
                backbone structure to design sequences for, with:
                - node_s: node scalar features
                - node_v: node vector features
                - edge_s: edge scalar features
                - edge_v: edge vector features
                - edge_index: graph connectivity
                - mask_confs: mask indicating valid conformations
            n_samples (int): Number of sequences to sample for the given structure
            temperature (float): Temperature parameter for softmax sampling. Lower
                values (e.g., 0.1) produce more deterministic samples, higher values
                (e.g., 1.0) produce more diverse samples. Default is 0.1.
            logit_bias (torch.Tensor, optional): Additive bias to apply to logits
                during sampling for manual control over specific positions. Shape
                should be [n_nodes, 4] to bias each position's nucleotide logits.
                Can be used to fix certain nucleotides or encourage specific bases.
            return_logits (bool): If True, return both sampled sequences and their
                corresponding logits. If False, return only sequences. Default is False.

        Returns:
            If return_logits is False:
                seq (torch.Tensor): Sampled sequences of shape [n_samples, n_nodes]
                    with integer values corresponding to nucleotide indices based on
                    the model's nucleotide-to-int mapping (typically A=0, C=1, G=2, U=3).

            If return_logits is True:
                tuple: A tuple containing:
                    - seq (torch.Tensor): Sampled sequences of shape [n_samples, n_nodes]
                    - logits (torch.Tensor): Unnormalized log probabilities of shape
                        [n_samples, n_nodes, 4] for each position and nucleotide
        """
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        device = edge_index.device
        num_nodes = h_V[0].shape[0]

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1), h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1), h_E[1].repeat(n_samples, 1, 1))

        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph

        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])

            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats

            for j, layer in enumerate(self.decoder_layers):
                out = layer(
                    h_V_cache[j],
                    edge_index_,
                    h_E_,
                    autoregressive_x=h_V_cache[0],
                    node_mask=node_mask,
                )

                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats

                if j < len(self.decoder_layers) - 1:
                    h_V_cache[j + 1][0][i::num_nodes] = out[0]
                    h_V_cache[j + 1][1][i::num_nodes] = out[1]

            lgts = self.W_out(out)
            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:
            return seq.view(n_samples, num_nodes)

    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        """Pool multi-conformation features using masked mean pooling.

        Aggregates features across multiple structural conformations by computing
        a masked average. Handles both scalar and vector features in GVP format.
        If only one conformation is present, returns features directly without pooling.

        Args:
            h_V (tuple): Node features as (scalar_features, vector_features) where:
                - scalar_features: shape (n_nodes, n_conf, d_s)
                - vector_features: shape (n_nodes, n_conf, d_v, 3)
            h_E (tuple): Edge features as (scalar_features, vector_features) where:
                - scalar_features: shape (n_edges, n_conf, d_se)
                - vector_features: shape (n_edges, n_conf, d_ve, 3)
            mask_confs (torch.Tensor): Boolean mask indicating valid conformations
                of shape (n_nodes, n_conf), where True indicates a valid conformation
            edge_index (torch.Tensor): Graph connectivity of shape (2, n_edges)

        Returns:
            tuple: A tuple containing:
                - h_V (tuple): Pooled node features as (scalar, vector) where:
                    - scalar: shape (n_nodes, d_s)
                    - vector: shape (n_nodes, d_v, 3)
                - h_E (tuple): Pooled edge features as (scalar, vector) where:
                    - scalar: shape (n_edges, d_se)
                    - vector: shape (n_edges, d_ve, 3)
        """
        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])

        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)

        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]

        # Average pooling multi-conformation features
        h_V = (
            h_V0.sum(dim=1) / n_conf_true,  # (n_nodes, d_s)
            h_V1.sum(dim=1) / n_conf_true.unsqueeze(2),
        )  # (n_nodes, d_v, 3)
        h_E = (
            h_E0.sum(dim=1) / n_conf_true[edge_index[0]],  # (n_edges, d_se)
            h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2),
        )  # (n_edges, d_ve, 3)

        return h_V, h_E


#########################################################################
# --- staging harness ---
#########################################################################


def build_grnade():
    """Tiny gRNAde for tracing: 1 GVP-GNN layer, small hidden dims."""
    torch.manual_seed(0)
    return gRNAde(
        node_in_dim=(15, 4),
        node_h_dim=(16, 4),
        edge_in_dim=(132, 3),
        edge_h_dim=(16, 2),
        num_layers=1,
        drop_rate=0.0,
        out_dim=4,
    )


def example_input_grnade():
    """Builds a small single-conformation RNA backbone graph batch."""
    torch.manual_seed(0)
    n_nodes = 8
    n_conf = 1
    node_in_dim = (15, 4)
    edge_in_dim = (132, 3)

    # Simple ring graph (backbone connectivity), directed both ways.
    src = torch.arange(n_nodes)
    dst = (src + 1) % n_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    n_edges = edge_index.shape[1]

    node_s = torch.randn(n_nodes, n_conf, node_in_dim[0])
    node_v = torch.randn(n_nodes, n_conf, node_in_dim[1], 3)
    edge_s = torch.randn(n_edges, n_conf, edge_in_dim[0])
    edge_v = torch.randn(n_edges, n_conf, edge_in_dim[1], 3)
    seq = torch.randint(0, 4, (n_nodes,))
    mask_confs = torch.ones(n_nodes, n_conf, dtype=torch.bool)

    data = torch_geometric.data.Data(
        node_s=node_s,
        node_v=node_v,
        edge_s=edge_s,
        edge_v=edge_v,
        edge_index=edge_index,
        seq=seq,
        mask_confs=mask_confs,
    )
    return data


MENAGERIE_ENTRIES = [
    ("gRNAde", "build_grnade", "example_input_grnade", 2024, "vendored-pytorch"),
]
