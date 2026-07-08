# SOURCE: vendored from https://github.com/DeepGraphLearning/ConfGF @ main (38aeb6c7719343d13fa867f4b17b02ed45d09bd0)
# (confgf/models/scorenet.py: DistanceScoreMatch; confgf/layers/gin.py:
# GINEConv/GraphIsomorphismNetwork; confgf/layers/common.py: MeanReadout/
# SumReadout/MultiLayerPerceptron -- the real ConfGF "Learning Neural Generative
# Dynamics for Molecular Conformation Generation" (ICML 2021) score network,
# unmodified. torch_scatter and torch_sparse (the repo's own graph-op deps) are
# both present in this environment and used exactly as in the original code.
#
# The one RDKit touchpoint in the original repo is `confgf/utils/chem.py`:
# `BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}`, a plain data
# constant (the count of RDKit's `BondType` enum members) used only as an
# integer offset in `extend_graph` -- RDKit is not otherwise imported or called
# anywhere in the traced nn.Module forward path. RDKit is not installed in this
# environment, so `NUM_BOND_TYPES = 22` is hardcoded here as the (version-stable)
# count of `rdkit.Chem.rdchem.BondType.names`, verified directly against the
# rdkit 2026.3.3 wheel. The score-network architecture below is untouched.
import sys
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import dense_to_sparse, to_dense_adj

# torch_geometric's MessagePassing.inspector resolves `message()`'s type-hinted params
# (e.g. `Optional[torch.Tensor]`) via `sys.modules[self.__module__].__dict__`. If this file
# is loaded via importlib.util.module_from_spec() without the resulting module object also
# being registered in sys.modules under its own name, that lookup KeyErrors at
# GINEConv construction time. Registering the currently-executing module object here is
# load-time scaffolding only (mirrors what a normal `import` statement always does) -- it
# does not touch the vendored GINEConv architecture below.
if __name__ not in sys.modules:
    import types as _types

    sys.modules[__name__] = _types.ModuleType(__name__)

# Number of RDKit BondType enum members (rdkit.Chem.rdchem.BondType.names), the
# only quantity the real repo's `utils.BOND_TYPES` constant is used for here.
NUM_BOND_TYPES = 22


# ---------------------------------------------------------------------------
# confgf/layers/common.py (verbatim)
# ---------------------------------------------------------------------------
class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.

    Note there is no activation or dropout in the last layer.
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# confgf/layers/gin.py (verbatim)
# ---------------------------------------------------------------------------
class GINEConv(MessagePassing):
    def __init__(
        self,
        nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        activation="softplus",
        **kwargs,
    ):
        super(GINEConv, self).__init__(aggr="add", **kwargs)
        self.nn = nn
        self.initial_eps = eps

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.activation:
            return self.activation(x_j + edge_attr)
        else:
            return x_j + edge_attr

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class GraphIsomorphismNetwork(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_convs=3,
        activation="softplus",
        readout="sum",
        short_cut=False,
        concat_hidden=False,
    ):
        super(GraphIsomorphismNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(
                GINEConv(
                    MultiLayerPerceptron(
                        hidden_dim, [hidden_dim, hidden_dim], activation=activation
                    ),
                    activation=activation,
                )
            )

        if readout == "sum":
            self.readout = SumReadout()
        elif readout == "mean":
            self.readout = MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, data, node_attr, edge_attr):
        hiddens = []
        conv_input = node_attr  # (num_node, hidden)

        for conv_idx, conv in enumerate(self.convs):
            hidden = conv(conv_input, data.edge_index, edge_attr)
            if conv_idx < len(self.convs) - 1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape:
                hidden += conv_input

            hiddens.append(hidden)
            conv_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        graph_feature = self.readout(data, node_feature)

        return {"graph_feature": graph_feature, "node_feature": node_feature}


# ---------------------------------------------------------------------------
# confgf/models/scorenet.py (verbatim: DistanceScoreMatch)
# ---------------------------------------------------------------------------
class DistanceScoreMatch(torch.nn.Module):
    def __init__(self, config):
        super(DistanceScoreMatch, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = MultiLayerPerceptron(
            1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act
        )
        self.output_mlp = MultiLayerPerceptron(
            2 * self.hidden_dim,
            [self.hidden_dim, self.hidden_dim // 2, 1],
            activation=self.config.model.mlp_act,
        )

        self.model = GraphIsomorphismNetwork(
            hidden_dim=self.hidden_dim,
            num_convs=self.config.model.num_convs,
            activation=self.config.model.gnn_act,
            readout="sum",
            short_cut=self.config.model.short_cut,
            concat_hidden=self.config.model.concat_hidden,
        )
        sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(self.config.model.sigma_begin),
                    np.log(self.config.model.sigma_end),
                    self.config.model.num_noise_level,
                )
            ),
            dtype=torch.float32,
        )
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)  # (num_noise_level)

    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):
        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            adj_mats = [
                torch.eye(adj.size(0), dtype=torch.long, device=adj.device),
                binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device)),
            ]

            for i in range(2, order + 1):
                adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order + 1):
                order_mat += (adj_mats[i] - adj_mats[i - 1]) * i

            return order_mat

        num_types = NUM_BOND_TYPES

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)  # (N, N)
        type_highorder = torch.where(
            adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)
        )
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(
            new_edge_index, new_edge_type.long(), N, N
        )  # modify data
        edge_index_1, data.edge_order = coalesce(
            new_edge_index, edge_order.long(), N, N
        )  # modify data
        data.is_bond = data.edge_type < num_types
        assert (data.edge_index == edge_index_1).all()

        return data

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1)  # (num_edge, 1)
        data.edge_length = d
        return data

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        self.device = self.sigmas.device
        data = self.extend_graph(data, self.order)
        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]

        # sample noise level
        noise_level = torch.randint(
            0, self.sigmas.size(0), (data.num_graphs,), device=self.device
        )  # (num_graph)
        used_sigmas = self.sigmas[noise_level]  # (num_graph)
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1)  # (num_edge, 1)

        # perturb
        d = data.edge_length  # (num_edge, 1)

        if self.noise_type == "symmetry":
            num_nodes = scatter_add(
                torch.ones(data.num_nodes, dtype=torch.long, device=self.device), node2graph
            )  # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0)  # (num_graph)
            node_offset = num_cum_nodes - num_nodes  # (num_graph)
            edge_offset = node_offset[edge2graph]  # (num_edge)

            num_nodes_square = num_nodes**2  # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1)  # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square  # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

            symm_noise = torch.zeros(all_len, device=self.device).normal_()
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1)  # (num_edge, 1)

        elif self.noise_type == "rand":
            d_noise = torch.randn_like(d)
        else:
            raise NotImplementedError("noise type must in [distance_symm, distance_rand]")
        assert d_noise.shape == d.shape
        perturbed_d = d + d_noise * used_sigmas

        # get target, origin_d minus perturbed_d
        target = -1 / (used_sigmas**2) * (perturbed_d - d)  # (num_edge, 1)

        # estimate scores
        node_attr = self.node_emb(data.atom_type)  # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type)  # (num_edge, hidden)
        d_emb = self.input_mlp(perturbed_d)  # (num_edge, hidden)
        edge_attr = d_emb * edge_attr  # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = (
            output["node_feature"][data.edge_index[0]],
            output["node_feature"][data.edge_index[1]],
        )  # (num_edge, hidden)

        distance_feature = torch.cat([h_row * h_col, edge_attr], dim=-1)  # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature)  # (num_edge, 1)
        scores = scores * (
            1.0 / used_sigmas
        )  # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        target = target.view(-1)  # (num_edge)
        scores = scores.view(-1)  # (num_edge)
        loss = (
            0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power)
        )  # (num_edge)
        loss = scatter_add(loss, edge2graph)  # (num_graph)
        return loss


# ---------------------------------------------------------------------------
# Menagerie staging glue: tiny config + synthetic molecule-graph batch
# ---------------------------------------------------------------------------
class _Namespace:
    """Minimal attribute-dict, mirroring the yacs/easydict-style `config` object
    the real repo constructs from its `config/*.yml` files (only the fields
    `DistanceScoreMatch.__init__`/`forward` actually reads are populated)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, _Namespace(**v) if isinstance(v, dict) else v)


def _confgf_config() -> _Namespace:
    # Mirrors config/qm9_default.yml (confgf repo), sized down for a tiny model.
    return _Namespace(
        train=dict(anneal_power=2.0),
        model=dict(
            hidden_dim=16,
            num_convs=2,
            sigma_begin=10,
            sigma_end=0.01,
            num_noise_level=5,
            order=3,
            mlp_act="relu",
            gnn_act="relu",
            short_cut=True,
            concat_hidden=False,
            noise_type="symmetry",
        ),
    )


def _synthetic_qm9_batch() -> Data:
    """Two tiny toy molecules batched into one torch_geometric Data object,
    matching the fields `DistanceScoreMatch.forward` reads: `atom_type`,
    `edge_index`, `edge_type`, `pos`, `batch`."""
    torch.manual_seed(0)
    # molecule 0: 4 atoms in a ring; molecule 1: 3 atoms in a path
    edge_index_0 = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    edge_index_1 = (
        torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long) + 4
    )  # offset past mol 0's 4 atoms

    edge_index = torch.cat([edge_index_0, edge_index_1], dim=1)
    edge_type = torch.randint(1, 4, (edge_index.size(1),), dtype=torch.long)  # SINGLE/DOUBLE/TRIPLE
    atom_type = torch.randint(1, 9, (7,), dtype=torch.long)  # 7 total atoms, small atomic numbers
    pos = torch.randn(7, 3)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    data = Data(
        atom_type=atom_type, edge_index=edge_index, edge_type=edge_type, pos=pos, batch=batch
    )
    data.num_graphs = 2
    return data


def build_confgf():
    return DistanceScoreMatch(_confgf_config())


def example_input_confgf():
    return (_synthetic_qm9_batch(),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("ConfGF", build_confgf, example_input_confgf, 2021, "SOURCE_AVAILABLE"),
]
