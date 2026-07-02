# SOURCE: vendored from idrugLab/hignn @ main (source/model.py)
#
# HiGNN (Zhu et al., 2023, "Harnessing Hierarchical Molecular Graphs for Drug Discovery")
# is a hierarchical molecular-graph transformer for property prediction. It runs two
# parallel NTNConv (neural-tensor-network message-passing) stacks with GRU-style gated
# residual updates: one over the plain atom graph, one over a BRICS-fragment-contracted
# graph built by clustering atoms into their BRICS retrosynthetic fragments. Per-molecule
# and per-fragment representations are pooled (global add pool) and fused through a
# GATConv cross-attention layer between fragment and molecule embeddings before the final
# readout MLP. Vendored verbatim (architecture-relevant classes/functions only, dropping
# the yacs-config `build_model` factory and yaml-driven CLI in favor of a directly
# parameterized constructor). No architectural changes.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
from torch import nn
from torch.nn import Bilinear, Linear, Parameter, Sequential
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------
# Attention layers
# ---------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, batch, size=None):
        max_result = scatter(x, batch, dim=0, dim_size=size, reduce="max")
        sum_result = scatter(x, batch, dim=0, dim_size=size, reduce="sum")
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.sigmoid(max_out + sum_out)
        y = y[batch]
        return x * y


# ---------------------------------------
# Neural tensor networks conv
# ---------------------------------------
class NTNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, slices, dropout, edge_dim=None, **kwargs):
        kwargs.setdefault("aggr", "add")
        super(NTNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slices = slices
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
        if edge_dim is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_dim, out_channels))
        else:
            self.weight_edge = self.register_parameter("weight_edge", None)

        self.bilinear = Bilinear(out_channels, out_channels, slices, bias=False)

        if self.edge_dim is not None:
            self.linear = Linear(3 * out_channels, slices)
        else:
            self.linear = Linear(2 * out_channels, slices)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        glorot(self.weight_edge)
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        x = torch.matmul(x, self.weight_node)

        if self.weight_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.weight_edge)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        score = self.bilinear(x_i, x_j)
        if edge_attr is not None:
            vec = torch.cat((x_i, edge_attr, x_j), 1)
            block_score = self.linear(vec)  # bias already included
        else:
            vec = torch.cat((x_i, x_j), 1)
            block_score = self.linear(vec)
        scores = score + block_score
        alpha = torch.tanh(scores)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dim_split = self.out_channels // self.slices
        out = torch.max(x_j, edge_attr).view(-1, self.slices, dim_split)

        out = out * alpha.view(-1, self.slices, 1)
        out = out.view(-1, self.out_channels)
        return out

    def __repr__(self):
        return "{}({}, {}, slices={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.slices
        )


# ---------------------------------------
# HiGNN backbone
# ---------------------------------------
class HiGNN(torch.nn.Module):
    """Hierarchical informative graph neural network for molecular representation."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim,
        num_layers,
        slices,
        dropout,
        f_att=False,
        r=4,
        brics=True,
        cl=False,
    ):
        super(HiGNN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_att = f_att
        self.brics = brics
        self.cl = cl

        # atom feature transformation
        self.lin_a = Linear(in_channels, hidden_channels)
        self.lin_b = Linear(edge_dim, hidden_channels)

        # convs block
        self.atom_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = NTNConv(
                hidden_channels,
                hidden_channels,
                slices=slices,
                dropout=dropout,
                edge_dim=hidden_channels,
            )
            self.atom_convs.append(conv)

        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)

        if self.f_att:
            self.feature_att = FeatureAttention(channels=hidden_channels, reduction=r)

        if self.brics:
            # mol-fra attention
            self.cross_att = GATConv(
                hidden_channels,
                hidden_channels,
                heads=4,
                dropout=dropout,
                add_self_loops=False,
                negative_slope=0.01,
                concat=False,
            )

        if self.brics:
            self.out = Linear(2 * hidden_channels, out_channels)
        else:
            self.out = Linear(hidden_channels, out_channels)

        if self.cl:
            self.lin_project = Linear(hidden_channels, int(hidden_channels / 2))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_a.reset_parameters()
        self.lin_b.reset_parameters()

        for conv in self.atom_convs:
            conv.reset_parameters()

        self.lin_gate.reset_parameters()

        if self.f_att:
            self.feature_att.reset_parameters()

        if self.brics:
            self.cross_att.reset_parameters()

        self.out.reset_parameters()

        if self.cl:
            self.lin_project.reset_parameters()

    def forward(self, data):
        # get mol input
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = F.relu(self.lin_a(x))  # (N, 46) -> (N, hidden_channels)
        edge_attr = F.relu(self.lin_b(edge_attr))  # (N, 10) -> (N, hidden_channels)

        # mol conv block
        for i in range(0, self.num_layers):
            h = F.relu(self.atom_convs[i](x, edge_index, edge_attr))
            beta = self.lin_gate(torch.cat([x, h, x - h], 1)).sigmoid()
            x = beta * x + (1 - beta) * h
            if self.f_att:
                x = self.feature_att(x, batch)

        mol_vec = global_add_pool(x, batch).relu_()

        if self.brics:
            # get fragment input
            fra_x = data.x
            fra_edge_index = data.fra_edge_index
            fra_edge_attr = data.fra_edge_attr
            cluster = data.cluster_index

            fra_x = F.relu(self.lin_a(fra_x))  # (N, 46) -> (N, hidden_channels)
            fra_edge_attr = F.relu(self.lin_b(fra_edge_attr))  # (N, 10) -> (N, hidden_channels)

            # fragment convs block
            for i in range(0, self.num_layers):
                fra_h = F.relu(self.atom_convs[i](fra_x, fra_edge_index, fra_edge_attr))
                beta = self.lin_gate(torch.cat([fra_x, fra_h, fra_x - fra_h], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * fra_h
                if self.f_att:
                    fra_x = self.feature_att(fra_x, cluster)

            fra_x = global_add_pool(fra_x, cluster).relu_()

            # get fragment batch
            cluster, perm = consecutive_cluster(cluster)
            fra_batch = pool_batch(perm, data.batch)

            # molecule-fragment attention
            row = torch.arange(fra_batch.size(0), device=batch.device)
            mol_fra_index = torch.stack([row, fra_batch], dim=0)
            fra_vec = self.cross_att((fra_x, mol_vec), mol_fra_index).relu_()

            vectors_concat = list()
            vectors_concat.append(mol_vec)
            vectors_concat.append(fra_vec)

            out = torch.cat(vectors_concat, 1)

            # molecule-fragment contrastive
            if self.cl:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return (
                    self.out(out),
                    self.lin_project(mol_vec).relu_(),
                    self.lin_project(fra_vec).relu_(),
                )
            else:
                out = F.dropout(out, p=self.dropout, training=self.training)
                return self.out(out)

        else:
            assert self.cl is False
            out = F.dropout(mol_vec, p=self.dropout, training=self.training)
            return self.out(out)


def build_hignn():
    torch.manual_seed(0)
    return HiGNN(
        in_channels=46,
        hidden_channels=16,
        out_channels=1,
        edge_dim=10,
        num_layers=2,
        dropout=0.1,
        slices=2,
        f_att=True,
        r=2,
        brics=True,
        cl=False,
    )


def example_input_hignn():
    # HiGNN.forward reads a PyG `Data` batch with the atom-level graph (x/edge_index/
    # edge_attr/batch, 46-dim CGCNN-style atom features, 10-dim bond features -- matching the
    # real `dataset.py` featurizer) plus a BRICS-fragment-contracted graph (fra_edge_index/
    # fra_edge_attr/cluster_index, built by BRICS-decomposing the molecule and contracting
    # each fragment's atoms into a single cluster id) that the real pipeline pre-computes with
    # rdkit. cluster_index assigns every atom to its BRICS fragment id within the molecule
    # (values are re-consecutivized per-batch inside forward()).
    torch.manual_seed(0)
    n_atoms, n_bonds = 9, 16
    x = torch.randn(n_atoms, 46)
    edge_index = torch.randint(0, n_atoms, (2, n_bonds))
    edge_attr = torch.randn(n_bonds, 10)
    batch = torch.zeros(n_atoms, dtype=torch.long)

    # 3 BRICS fragments spanning the 9 atoms; fragment-graph reuses the same atom count with
    # a smaller (intra + inter-fragment stub) bond set, as in the real dataset's `frag_data`.
    n_fra_bonds = 10
    fra_edge_index = torch.randint(0, n_atoms, (2, n_fra_bonds))
    fra_edge_attr = torch.randn(n_fra_bonds, 10)
    cluster_index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        fra_edge_index=fra_edge_index,
        fra_edge_attr=fra_edge_attr,
        cluster_index=cluster_index,
    )
    return (data,)


MENAGERIE_ENTRIES = [
    ("HiGNN", "build_hignn", "example_input_hignn", 2023, "vendored-pytorch"),
]
