# SOURCE: vendored from https://github.com/BioColLab/KcatNet @ main (accessed 2026-07-01)
#
# KcatNet (BioColLab, 2024) predicts enzyme turnover numbers (kcat) from a
# protein-ligand pair. It is a dual-graph interaction network: a custom
# Principal Neighbourhood Aggregation (PNA, Corso et al. 2020) convolution
# over the protein residue-contact graph (features = pretrained ESM + ProtT5
# embeddings), a learned soft-clustering / dense-mincut pooling stage that
# coarsens the residue graph into a handful of "pocket" clusters each layer,
# a multi-head motif-attention pool over the ligand atom graph, and a
# bidirectional cross-attention ("InterConv") that lets ligand atoms and
# protein clusters exchange information across `total_layer` rounds before a
# 3-layer MLP regression head predicts log10(kcat). All building blocks are
# pure PyTorch / torch_geometric / torch_scatter / numpy / scipy -- no extra
# deps beyond our base env -- so the real repo code is vendored unmodified
# (only local `models.*` imports flattened into this single file).
#
# Vendored files (unmodified logic, only local imports resolved):
#   models/scaler.py       -> DegreeScalerAggregation (PNA degree-scaler aggregation)
#   models/pna.py           -> PNAConv (custom local copy of torch_geometric's PNAConv)
#   models/layers.py        -> GCNCluster, PosLinear, MLP, Protein_PNAConv, InterConv
#   models/Mol_pool.py       -> MotifPool (ligand atom multi-head attention pool)
#   models/protein_pool.py  -> dense_mincut_pool (+ its _rank3_trace/_rank3_diag helpers)
#   models/model_kcat.py     -> KcatNet, _rbf (top-level module)
#
# The real dataset construction (utils/protein_init.py / utils/ligand_init.py)
# needs a live ESM-2 + ProtT5 embedding pipeline plus RDKit featurization to
# turn raw sequences/SMILES into node/edge tensors. We do not vendor that (out
# of scope for a forward-pass architecture trace); example_input_kcatnet()
# instead builds synthetic tensors matching the EXACT dimensionalities the
# real pred_kcat.py / config_KcatNet.json hard-code: ligand atom_idx (Long,
# Embedding(20,.)), ligand atom_feature (43-dim, config mol_in_channels),
# ligand mol_total_fea (1024-dim global fingerprint feature), protein
# residue ESM-2 embeddings (1280-dim, config prot_in_channels), protein
# residue ProtT5 embeddings (1024-dim, config prot_evo_channels), and a
# residue contact-graph edge_index/edge_weight (edge_weight in [0,1], the
# real contact-probability the model RBF-expands). `prot_deg` (the PNA
# in-degree histogram, computed from the training set by the real
# PNAConv.get_degree_histogram) is built from the same synthetic graph so
# `KcatNet.__init__` gets a real, correctly-shaped degree tensor rather than
# a stub.

import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import APPNP, GCNConv, MessagePassing, SAGEConv, SGConv
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    degree,
    segregate_self_loops,
    softmax,
    subgraph,
    to_dense_adj,
    to_dense_batch,
    to_scipy_sparse_matrix,
)
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"

EPS = 1e-15


# ---- vendored from models/scaler.py ----


class DegreeScalerAggregation(Aggregation):
    """Combines aggregators and rescales by in-degree, as in PNA (Corso et al. 2020)."""

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(
                "Only strings, list, tuples and instances of "
                f"`torch_geometric.nn.aggr.Aggregation` are valid aggregation schemes (got '{type(aggr)}')"
            )

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg: Dict[str, float] = {
            "lin": float((bin_degrees * deg).sum()) / num_nodes,
            "log": float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            "exp": float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype).clamp_(1)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for scaler in self.scaler:
            if scaler == "identity":
                out_scaler = out
            elif scaler == "amplification":
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg["log"])
            elif scaler == "attenuation":
                out_scaler = out * (self.avg_deg["log"] / torch.log(deg + 1))
            elif scaler == "exponential":
                out_scaler = out * (torch.exp(deg) / self.avg_deg["exp"])
            elif scaler == "linear":
                out_scaler = out * (deg / self.avg_deg["lin"])
            elif scaler == "inverse_linear":
                out_scaler = out * (self.avg_deg["lin"] / deg)
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


# ---- vendored from models/pna.py (local copy of PNAConv, extended with edge_dim/towers) ----


class PNAConv(MessagePassing):
    """Principal Neighbourhood Aggregation graph convolution (Corso et al. 2020)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        divide_input: bool = False,
        **kwargs,
    ):
        aggr = DegreeScalerAggregation(aggregators, scalers, deg)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            self.edge_encoder = PyGLinear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [PyGLinear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [PyGLinear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels_post = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [PyGLinear(in_channels_post, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [PyGLinear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = PyGLinear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn_ in self.pre_nns:
            reset(nn_)
        for nn_ in self.post_nns:
            reset(nn_)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor:
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn_(out[:, i]) for i, nn_ in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn_(h[:, i]) for i, nn_ in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, towers={self.towers}, "
            f"edge_dim={self.edge_dim})"
        )

    @staticmethod
    def get_degree_histogram(loader) -> Tensor:
        max_degree = 0
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        deg_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())
        return deg_histogram


# ---- vendored from models/layers.py ----


class GCNCluster(torch.nn.Module):
    def __init__(self, dims, out_norm=True, in_norm=False):
        super().__init__()
        list_Conv_layers = [GCNConv(dims[idx - 1], dims[idx]) for idx in range(1, len(dims))]
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers + 1):
            self.Conv_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x, edge_index):
        y = x
        if self.in_norm:
            y = self.in_ln(y)

        for idx in range(self.hidden_layers):
            y = self.Conv_layers[idx](y, edge_index)
            y = F.relu(y)
        y = self.Conv_layers[-1](y, edge_index)

        if self.out_norm:
            y = self.out_ln(y)

        return y


class PosLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_value=0.2,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        lower_bound = init_value / 2
        upper_bound = init_value
        weight = nn.init.uniform_(
            torch.empty((out_features, in_features), **factory_kwargs), a=lower_bound, b=upper_bound
        )
        weight = torch.abs(weight)
        self.weight = nn.Parameter(weight.log())
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.exp(), self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class MLP(nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False, bias=True):
        super().__init__()
        list_FC_layers = [
            nn.Linear(dims[idx - 1], dims[idx], bias=bias) for idx in range(1, len(dims))
        ]
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.hidden_layers = len(dims) - 2

        self.out_norm = out_norm
        self.in_norm = in_norm

        if self.out_norm:
            self.out_ln = nn.LayerNorm(dims[-1])
        if self.in_norm:
            self.in_ln = nn.LayerNorm(dims[0])

    def reset_parameters(self):
        for idx in range(self.hidden_layers + 1):
            self.FC_layers[idx].reset_parameters()
        if self.out_norm:
            self.out_ln.reset_parameters()
        if self.in_norm:
            self.in_ln.reset_parameters()

    def forward(self, x):
        y = x

        for idx in range(self.hidden_layers):
            y = self.FC_layers[idx](y)
            y = F.relu(y)
        y = self.FC_layers[-1](y)

        if self.out_norm:
            y = self.out_ln(y)

        return y


class Protein_PNAConv(nn.Module):
    def __init__(
        self,
        prot_deg,
        hidden_channels,
        edge_channels,
        pre_layers=2,
        post_layers=2,
        aggregators=["sum", "mean", "min", "max", "std"],
        scalers=["identity", "amplification", "attenuation"],
        num_towers=4,
        dropout=0.1,
    ):
        super(Protein_PNAConv, self).__init__()

        self.conv = PNAConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_channels,
            aggregators=aggregators,
            scalers=scalers,
            deg=prot_deg,
            pre_layers=pre_layers,
            post_layers=post_layers,
            towers=num_towers,
            divide_input=True,
        )

        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x, prot_edge_index, prot_edge_attr=None):
        x_in = x
        x = x_in + F.relu(self.conv(x, prot_edge_index, prot_edge_attr))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class InterConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        atom_channels: int,
        residue_channels: int,
        heads: int = 1,
        t=0.2,
        dropout_attn_score=0.2,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super(InterConv, self).__init__(node_dim=0, **kwargs)

        assert residue_channels % heads == 0
        assert atom_channels % heads == 0

        self.residue_out_channels = residue_channels // heads
        self.atom_out_channels = atom_channels // heads
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_key = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        self.lin_query = nn.Linear(atom_channels, heads * self.atom_out_channels, bias=False)
        self.lin_value = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * self.atom_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter("lin_edge", None)

        self.lin_atom_value = nn.Linear(atom_channels, heads * self.residue_out_channels)

        self.drug_in_norm = torch.nn.LayerNorm(atom_channels)
        self.residue_in_norm = torch.nn.LayerNorm(residue_channels)

        self.drug_out_norm = torch.nn.LayerNorm(heads * self.atom_out_channels)
        self.residue_out_norm = torch.nn.LayerNorm(heads * self.residue_out_channels)

        self.clique_mlp = nn.Linear(atom_channels, atom_channels)
        self.clique_mlp1 = MLP([atom_channels, atom_channels * 2, atom_channels], out_norm=True)
        self.residue_mlp = MLP(
            [residue_channels * 2, residue_channels * 2, residue_channels], out_norm=True
        )
        self.residue_mlp1 = nn.Linear(residue_channels, residue_channels)

        self.t = t
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_atom_value.reset_parameters()
        self.drug_in_norm.reset_parameters()
        self.residue_in_norm.reset_parameters()
        self.drug_out_norm.reset_parameters()
        self.residue_out_norm.reset_parameters()
        self.clique_mlp.reset_parameters()
        self.residue_mlp.reset_parameters()

    def forward(self, drug_x, residue_x, edge_index: Adj):
        H, aC = self.heads, self.atom_out_channels
        residue_hx = residue_x
        query = self.lin_query(drug_x).view(-1, H, aC)
        key = self.lin_key(residue_hx).view(-1, H, aC)
        value = self.lin_value(residue_hx).view(-1, H, aC)

        drug_out = self.propagate(
            edge_index, query=query, key=key, value=value, edge_attr=None, size=None
        )
        alpha = self._alpha
        self._alpha = None
        drug_out = drug_out.view(-1, H * aC)
        drug_out = drug_x + F.relu(self.clique_mlp(drug_out))

        H, rC = self.heads, self.residue_out_channels
        drug_hx = drug_x
        residue_value = self.lin_atom_value(drug_hx).view(-1, H, rC)[edge_index[1]]
        residue_out = residue_value * alpha.view(-1, H, 1)
        residue_out = residue_out.view(-1, H * rC)
        residue_out = self.residue_out_norm(residue_out)
        residue_x = residue_x + F.relu(self.residue_mlp1(residue_out))
        return drug_out, residue_x, (edge_index, alpha)

    def message(
        self,
        query_i: Tensor,
        key_j: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.atom_out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout_attn_score, training=self.training)
        out = value_j
        out = out * alpha.view(-1, self.heads, 1)

        return out


def dropout_node(edge_index, p, num_nodes, batch, training):
    """Randomly drops nodes from the adjacency matrix with probability p."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 (got {p}")

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    deg_batch = degree(batch, dtype=torch.long).tolist()
    unbatched_node_mask = node_mask.split(deg_batch, 0)
    node_mask_list = []

    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)

    node_mask = torch.cat(node_mask_list)

    edge_index, _, edge_mask = subgraph(
        node_mask, edge_index, num_nodes=num_nodes, return_edge_mask=True
    )
    return edge_index, edge_mask, node_mask


# ---- vendored from models/Mol_pool.py ----


class MotifPool(torch.nn.Module):
    def __init__(self, hidden_dim, heads, dropout_attn_score=0, dropout_node_proba=0):
        super().__init__()
        assert hidden_dim % heads == 0

        self.lin_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        hidden_dim = hidden_dim // heads

        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads):
            self.score_proj.append(MLP([hidden_dim, hidden_dim * 2, 1]))

        self.heads = heads
        self.hidden_dim = hidden_dim
        self.dropout_node_proba = dropout_node_proba
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        for m in self.score_proj:
            m.reset_parameters()

    def forward(self, x, mol_batch):
        H = self.heads
        C = self.hidden_dim

        atom_feature = x.view(-1, H, C)
        score = torch.cat(
            [mlp(atom_feature[:, i]) for i, mlp in enumerate(self.score_proj)], dim=-1
        )
        alpha = softmax(score, mol_batch)
        drug_feat = atom_feature.view(-1, H, C) * alpha.view(-1, H, 1)
        drug_feat = global_add_pool(drug_feat.view(-1, H * C), mol_batch)

        return drug_feat, alpha


# ---- vendored from models/protein_pool.py ----


def dense_mincut_pool(x, adj, s, mask=None, cluster_drop_node=None):
    """MinCut pooling (Bianchi et al. 2020)."""
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)

        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum("ijk->ij", adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2)
    )
    ortho_loss = torch.mean(ortho_loss)

    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s, out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x):
    return torch.einsum("ijj->i", x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


# ---- vendored from models/model_kcat.py ----


def _rbf(D, D_min=0.0, D_max=1.0, D_count=16, device="cpu"):
    """Radial Basis Function embedding of D. From https://github.com/jingraham/neurips19-graph-protein-design"""
    D = torch.where(D < D_max, D, torch.tensor(D_max).float().to(device))
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


class KcatNet(torch.nn.Module):
    def __init__(
        self,
        prot_deg,
        mol_in_channels=43,
        prot_in_channels=40,
        prot_evo_channels=1280,
        hidden_channels=200,
        pre_layers=2,
        post_layers=1,
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "linear"],
        total_layer=3,
        K=[10, 15, 20],
        t=1,
        heads=5,
        dropout=0,
        dropout_attn_score=0.2,
        drop_atom=0,
        device="cpu",
    ):
        super(KcatNet, self).__init__()

        self.hidden_channels = hidden_channels
        self.atom_type_encoder = Embedding(20, hidden_channels)

        self.prot_convs = torch.nn.ModuleList()
        self.atom_update = torch.nn.ModuleList()
        self.inter_convs = torch.nn.ModuleList()
        self.num_cluster = K
        self.cluster = torch.nn.ModuleList()

        self.mol_pools = torch.nn.ModuleList()
        self.res_update = torch.nn.ModuleList()
        self.mol_update = torch.nn.ModuleList()
        self.atom_embed_total = torch.nn.ModuleList()
        self.atom_embed_total2 = torch.nn.ModuleList()

        self.total_layer = total_layer
        self.prot_edge_dim = hidden_channels

        for idx in range(total_layer):
            self.prot_convs.append(
                Protein_PNAConv(
                    prot_deg,
                    hidden_channels,
                    edge_channels=hidden_channels,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    aggregators=aggregators,
                    scalers=scalers,
                    num_towers=heads,
                    dropout=dropout,
                )
            )
            self.cluster.append(
                GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]])
            )

            self.atom_update.append(Linear(hidden_channels, hidden_channels))
            self.mol_pools.append(MotifPool(hidden_channels, heads, dropout_attn_score, drop_atom))
            self.atom_embed_total.append(Linear(hidden_channels * 2, hidden_channels))
            self.atom_embed_total2.append(Linear(hidden_channels, hidden_channels))

            self.inter_convs.append(
                InterConv(
                    atom_channels=hidden_channels,
                    residue_channels=hidden_channels,
                    heads=heads,
                    t=t,
                    dropout_attn_score=dropout_attn_score,
                )
            )

            self.res_update.append(Linear(hidden_channels, hidden_channels))
            self.mol_update.append(Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.device = device

        self.seq_embed_esm = torch.nn.Linear(prot_in_channels, hidden_channels * 2)
        self.seq_embed_prot5 = torch.nn.Linear(prot_evo_channels, hidden_channels * 2)
        self.seq_embed = torch.nn.Linear(hidden_channels * 4, hidden_channels)
        self.seq_embed_evo2 = torch.nn.Linear(hidden_channels * 2, hidden_channels)

        self.atom_feat_embed = Linear(mol_in_channels, hidden_channels)
        self.atom_type_embed = Embedding(20, hidden_channels)
        self.atom_type_embed2 = Linear(hidden_channels // 2, hidden_channels // 2)
        self.atom_feat_embed2 = Linear(hidden_channels // 2, hidden_channels)
        self.mol_embed = torch.nn.Linear(1024, hidden_channels)
        self.mol_embed2 = torch.nn.Linear(hidden_channels, hidden_channels)

        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.GN = GraphNorm(hidden_channels)

        self.inter_attn_lin = PosLinear(heads, 1, bias=False, init_value=1 / heads)
        self.inter_attn_lin2 = PosLinear(heads, 1, bias=False, init_value=1 / heads)

        self.mol_fea_update = Linear(hidden_channels * total_layer, hidden_channels)
        self.res_fea_update = Linear(hidden_channels * total_layer, hidden_channels)
        self.res_fea_update2 = Linear(hidden_channels * total_layer, hidden_channels)
        self.cluster_fea_update2 = Linear(hidden_channels * total_layer, hidden_channels)

        self.classifier = nn.Linear(hidden_channels * 4, 512)
        self.classifier1 = nn.Linear(512, 128)
        self.classifier2 = nn.Linear(128, 1)
        # NOTE: the real repo also builds self.optimizer = torch.optim.AdamW(...)
        # inside __init__. We intentionally omit that (an optimizer over
        # not-yet-materialized/randomly-reinitialized params has no role in a
        # forward-pass architecture trace) -- no nn.Module/layer is affected.

    def forward(
        self,
        mol_x,
        mol_x_feat,
        mol_total_fea,
        residue_esm,
        residue_prot5,
        residue_edge_index,
        residue_edge_weight,
        mol_batch=None,
        prot_batch=None,
    ):
        cluster_loss = torch.tensor(0.0).to(self.device)
        residue_scores = []

        residue_ini = torch.cat(
            [F.relu(self.seq_embed_prot5(residue_prot5)), F.relu(self.seq_embed_esm(residue_esm))],
            dim=-1,
        )
        residue_x = F.relu(self.seq_embed(residue_ini))
        residue_edge_attr = _rbf(
            residue_edge_weight, D_max=1.0, D_count=self.prot_edge_dim, device=self.device
        )

        atom_x = self.atom_type_embed(mol_x.squeeze()) + F.relu(self.atom_feat_embed(mol_x_feat))
        mol_total_fea = F.relu(self.mol_embed(mol_total_fea))
        mol_total_fea = self.norm(self.mol_embed2(mol_total_fea))

        res_feas = []
        res_feas2 = []
        mol_feas = []
        cluster_feas = []

        for idx in range(self.total_layer):
            residue_x = self.GN(residue_x, prot_batch)
            atom_x = self.GN(atom_x, mol_batch)

            residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)
            residue_max = global_max_pool(residue_x, prot_batch)
            res_feas.append(residue_max)

            s = self.cluster[idx](residue_x, residue_edge_index)
            s, _ = to_dense_batch(s, prot_batch)
            residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)
            residue_adj = to_dense_adj(residue_edge_index, prot_batch)
            s, cluster_x, residue_adj, cl_loss, _ = dense_mincut_pool(
                residue_hx, residue_adj, s, residue_mask, None
            )
            cluster_x = self.norm(cluster_x)
            cluster_loss = cluster_loss + cl_loss

            atom_x = F.relu(self.atom_update[idx](atom_x))
            mol_x_pooled, _ = self.mol_pools[idx](atom_x, mol_batch)
            mol_x_pooled = self.norm(mol_x_pooled)

            mol_x_pooled = torch.cat([mol_total_fea, mol_x_pooled], dim=-1)
            mol_x_pooled = F.relu(self.atom_embed_total[idx](mol_x_pooled))
            mol_x_pooled = self.atom_embed_total2[idx](mol_x_pooled)
            mol_x_pooled = self.norm(mol_x_pooled)

            batch_size = s.size(0)
            cluster_x = cluster_x.reshape(batch_size * self.num_cluster[idx], -1)
            cluster_residue_batch = (
                torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]).to(self.device)
            )
            p2m_edge_index = torch.stack(
                [
                    torch.arange(batch_size * self.num_cluster[idx]),
                    torch.arange(batch_size).repeat_interleave(self.num_cluster[idx]),
                ]
            ).to(self.device)

            mol_x_pooled, cluster_x, inter_attn = self.inter_convs[idx](
                mol_x_pooled, cluster_x, p2m_edge_index
            )
            mol_feas.append(mol_x_pooled)
            inter_attn = inter_attn[1]

            atom_x = atom_x + F.relu(self.mol_update[idx](mol_x_pooled)[mol_batch])
            cluster_score = softmax(self.inter_attn_lin(inter_attn), cluster_residue_batch)
            pool_cluster = self.norm(
                global_max_pool(cluster_x * cluster_score, cluster_residue_batch)
            )

            cluster_feas.append(pool_cluster)

            cluster_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)
            inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)

            residue_x = residue_x + F.relu((self.res_update[idx]((s @ cluster_hx)[residue_mask])))

            residue_score = self.inter_attn_lin2((s @ inter_attn)[residue_mask])
            residue_score = softmax(residue_score, prot_batch)
            residue_scores.append(residue_score)

            pool_enz = self.norm(global_max_pool(residue_x * residue_score, prot_batch))
            res_feas2.append(pool_enz)

        mol_feas = torch.cat(mol_feas, dim=-1)
        res_feas = torch.cat(res_feas, dim=-1)
        res_feas2 = torch.cat(res_feas2, dim=-1)
        clu_fea = torch.cat(cluster_feas, dim=-1)

        mol_x_out = F.relu(self.mol_fea_update(mol_feas))
        res_feas = F.relu(self.res_fea_update(res_feas))
        res_feas2 = F.relu(self.res_fea_update2(res_feas2))
        clu_fea = F.relu(self.cluster_fea_update2(clu_fea))
        mol_prot_feat = torch.cat([res_feas, res_feas2, clu_fea, mol_x_out], dim=-1)

        reg_pred = F.relu((self.classifier(mol_prot_feat)))
        reg_pred = F.relu((self.classifier1(reg_pred)))
        reg_pred = self.classifier2(reg_pred)

        return reg_pred, cluster_loss


# ---- menagerie staging harness ----


def build_kcatnet():
    """Real KcatNet at its published config_KcatNet.json hyperparameters
    (hidden_channels=200, PNA aggregators/scalers, 3 interaction layers with
    K=[3,10,30] pocket clusters, 5-head cross-attention). All dims are
    architecture constants from the real config; only the synthetic input
    graph below is kept tiny (a few protein residues / ligand atoms) so the
    real-sized network traces quickly. prot_deg (the PNA in-degree histogram
    the real training pipeline pre-computes and saves to Dataset/degree.pt)
    is computed here from the same synthetic residue-contact graph used by
    example_input_kcatnet(), via the real PNAConv.get_degree_histogram-style
    bincount over in-degrees."""
    torch.manual_seed(0)
    n_pro = 6
    edge_index = _protein_edge_index(n_pro)
    deg = degree(edge_index[1], num_nodes=n_pro, dtype=torch.long)
    prot_deg = torch.bincount(deg, minlength=int(deg.max()) + 1)

    model = KcatNet(
        prot_deg,
        mol_in_channels=43,
        prot_in_channels=1280,
        prot_evo_channels=1024,
        hidden_channels=200,
        pre_layers=2,
        post_layers=1,
        aggregators=["mean", "min", "max", "std"],
        scalers=["identity", "amplification", "linear"],
        total_layer=3,
        K=[3, 10, 30],
        heads=5,
        dropout=0,
        dropout_attn_score=0.2,
        device="cpu",
    )
    return model.eval()


def _protein_edge_index(n_pro):
    rows, cols = [], []
    for i in range(n_pro):
        for j in range(n_pro):
            rows.append(i)
            cols.append(j)
    return torch.tensor([rows, cols], dtype=torch.long)


def example_input_kcatnet():
    """A synthetic single-pair (protein, ligand) input matching the exact
    tensor shapes real utils/Kcat_Dataset.py / pred_kcat.py feed into
    KcatNet.forward: ligand atom indices (Long, Embedding(20,.)), ligand
    43-dim atom features (config mol_in_channels), a 1024-dim ligand global
    fingerprint feature, protein 1280-dim ESM-2 residue embeddings (config
    prot_in_channels), protein 1024-dim ProtT5 residue embeddings (config
    prot_evo_channels), and a fully-connected residue contact-graph
    edge_index with edge_weight in [0,1] (the real contact-probability the
    model RBF-expands via `_rbf`). K=[3,10,30] pocket clusters need at least
    that many residues per protein graph to keep the dense-mincut pooling's
    assignment matrix non-degenerate, so n_pro=32 (each protein cluster count
    strictly less than n_pro, matching real training-graph sizes)."""
    torch.manual_seed(0)
    n_pro = 32
    n_atoms = 8

    mol_x = torch.randint(0, 20, (n_atoms, 1)).long()
    mol_x_feat = torch.rand(n_atoms, 43)
    mol_total_fea = torch.rand(1, 1024)
    mol_batch = torch.zeros(n_atoms, dtype=torch.long)

    residue_esm = torch.randn(n_pro, 1280)
    residue_prot5 = torch.randn(n_pro, 1024)
    residue_edge_index = _protein_edge_index(n_pro)
    residue_edge_weight = torch.rand(residue_edge_index.size(1))
    prot_batch = torch.zeros(n_pro, dtype=torch.long)

    return (
        mol_x,
        mol_x_feat,
        mol_total_fea,
        residue_esm,
        residue_prot5,
        residue_edge_index,
        residue_edge_weight,
        mol_batch,
        prot_batch,
    )


MENAGERIE_ENTRIES = [
    (
        "KcatNet",
        "build_kcatnet",
        "example_input_kcatnet",
        2024,
        "vendored-pytorch",
    ),
]
