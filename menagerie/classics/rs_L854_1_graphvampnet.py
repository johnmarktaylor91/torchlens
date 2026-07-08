# SOURCE: vendored from https://github.com/xuhuihuang/graphvampnets @ main
# https://github.com/xuhuihuang/graphvampnets/blob/main/graphvampnets/layers/graph.py
#
# GraphVAMPNet: a graph-neural-network "lobe" (the traceable nn.Module plugged into the
# VAMPNet/SRVNet estimators in graphvampnets/vamp/vampnet.py and srvnet.py) that learns
# slow collective variables of molecular self-assembly from a molecular-dynamics
# trajectory graph representation (RBF-expanded edge distances -> attention-weighted
# continuous-filter graph convolution -> atom-embedding pooling -> graph embedding).
# The VAMPNet_Estimator / VAMPNet classes around it are training-loop orchestration, not
# nn.Module architecture, so only graph.py (the actual network) is vendored here.
#
# Copied verbatim from the real repo (only this staging-entrypoint section added at the
# bottom); no architectural changes.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ------------------------------------------------------------------
# graphvampnets/layers/graph.py  (verbatim)
# ------------------------------------------------------------------
def LinearLayer(d_in, d_out, bias=True, activation=None, dropout=0):
    seq = [nn.Linear(d_in, d_out, bias=bias)]

    # activate is a function, e.g. nn.ELU()
    if activation is not None:
        if isinstance(activation, nn.Module):
            seq += [activation]
        else:
            raise TypeError("Activation {} is not a valid torch.nn.Module".format(str(activation)))

    if dropout > 0:
        seq += [nn.Dropout(dropout)]

    # "with torch.no_grad()" already included in nn.init.funcs()
    torch.nn.init.xavier_uniform_(seq[0].weight)

    return seq


class RBFExpansion(nn.Module):
    def __init__(self, dmin, dmax, step, var=None):
        super(RBFExpansion, self).__init__()

        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = torch.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var
        self.edge_emb_dim = len(self.filter)

    def forward(self, edge_dist):
        # edge_dist: [num_edges, 1]
        # edge_emb: [num_edges, edge_emb_dim]
        edge_emb = torch.exp(
            -((edge_dist - self.filter.to(device=edge_dist.device)) ** 2) / self.var**2
        )
        return edge_emb


class ContinuousFilterConv(nn.Module):
    def __init__(self, edge_emb_dim, hidden_emb_dim, activation=nn.Tanh()):
        super(ContinuousFilterConv, self).__init__()

        filter_layers = LinearLayer(edge_emb_dim, hidden_emb_dim, bias=True, activation=activation)
        filter_layers += LinearLayer(
            hidden_emb_dim, hidden_emb_dim, bias=True
        )  # no activation here
        self.filter_generator = nn.Sequential(*filter_layers)

        self.linear_coef = nn.Parameter(torch.Tensor(hidden_emb_dim, 1))
        nn.init.xavier_uniform_(self.linear_coef.data, gain=1.414)

        self.hidden_emb_dim = hidden_emb_dim

    def forward(self, hidden_atom_emb, edge_emb, edge_list):
        # hidden_atom_emb: [num_atoms_over_batches, hidden_emb_dim]
        # edge_emb: [num_edges, edge_emb_dim]
        # edge_list: [num_edges, 2]

        # [num_edges, hidden_emb_dim]
        hidden_edge_emb = self.filter_generator(edge_emb)

        # [num_edges, hidden_emb_dim]
        hidden_conv_nbr_edge_emb = hidden_atom_emb[edge_list[:, 1]] * hidden_edge_emb
        # [num_edges, 1]
        attn_without_norm = torch.exp(torch.matmul(hidden_conv_nbr_edge_emb, self.linear_coef))

        # [num_atoms_over_batches, num_atoms_over_batches, hidden_emb_dim]
        sparse_mat_attn_without_norm = torch.sparse_coo_tensor(
            edge_list.t(),
            values=attn_without_norm.repeat(1, self.hidden_emb_dim),
            size=(hidden_atom_emb.shape[0], hidden_atom_emb.shape[0], self.hidden_emb_dim),
        )
        # [num_atoms_over_batches, num_atoms_over_batches, hidden_emb_dim]
        sparse_mat_hidden_conv_nbr_edge_emb = torch.sparse_coo_tensor(
            edge_list.t(),
            values=hidden_conv_nbr_edge_emb,
            size=(hidden_atom_emb.shape[0], hidden_atom_emb.shape[0], self.hidden_emb_dim),
        )
        # [num_atoms_over_batches, hidden_emb_dim]
        hidden_conv_atom_emb_without_norm = torch.sparse.sum(
            sparse_mat_hidden_conv_nbr_edge_emb * sparse_mat_attn_without_norm, dim=1
        ).to_dense()
        # [num_atoms_over_batches, hidden_emb_dim]
        norm_coef = torch.sparse.sum(sparse_mat_attn_without_norm, dim=1).to_dense()
        # [num_atoms_over_batches, hidden_emb_dim]
        hidden_conv_atom_emb = torch.where(
            norm_coef > 0,
            hidden_conv_atom_emb_without_norm / norm_coef,
            hidden_conv_atom_emb_without_norm,
        )

        return hidden_conv_atom_emb


class InteractionBlock(nn.Module):
    def __init__(self, atom_emb_dim, edge_emb_dim, hidden_emb_dim, activation=nn.Tanh()):
        super(InteractionBlock, self).__init__()

        self.initial_dense = nn.Sequential(
            *LinearLayer(atom_emb_dim, hidden_emb_dim, bias=False, activation=None)
        )
        self.cfconv = ContinuousFilterConv(
            edge_emb_dim=edge_emb_dim, hidden_emb_dim=hidden_emb_dim, activation=activation
        )

        output_layers = LinearLayer(
            hidden_emb_dim, hidden_emb_dim, bias=True, activation=activation
        )
        output_layers += LinearLayer(hidden_emb_dim, hidden_emb_dim, bias=True)
        self.output_dense = nn.Sequential(*output_layers)

    def forward(self, atom_emb, edge_emb, edge_list):
        hidden_atom_emb = self.initial_dense(atom_emb)
        hidden_conv_atom_emb = self.cfconv(hidden_atom_emb, edge_emb, edge_list)
        output_atom_emb = self.output_dense(hidden_conv_atom_emb)

        return output_atom_emb


class GraphVAMPNetLayer(nn.Module):
    def __init__(
        self,
        num_atoms,
        num_conv,
        atom_emb_dim,
        dmin,
        dmax,
        step,
        graph_emb_dim=None,
        conv_activation=nn.ReLU(),
        residual=True,
        atom_class_idx=None,
        pretrained_atom_emb=None,
        atom_emb_init_type="normal",
        freeze=False,
    ):
        super(GraphVAMPNetLayer, self).__init__()

        self._num_atoms = num_atoms
        self._num_conv = num_conv

        self._atom_emb_dim = atom_emb_dim
        self._graph_emb_dim = graph_emb_dim
        self._dmin = dmin
        self._dmax = dmax
        self._step = step
        self._rbf = RBFExpansion(dmin, dmax, step)
        self._edge_emb_dim = self._rbf.edge_emb_dim

        self.convs = nn.ModuleList(
            [
                InteractionBlock(
                    atom_emb_dim=self._atom_emb_dim,
                    edge_emb_dim=self._edge_emb_dim,
                    hidden_emb_dim=self._atom_emb_dim,
                    activation=nn.Tanh(),
                )
                for _ in range(self._num_conv)
            ]
        )
        if isinstance(conv_activation, nn.Module):
            self._conv_activation = conv_activation
        else:
            raise TypeError(
                "Activation {} is not a valid torch.nn.Module".format(str(conv_activation))
            )
        self._residual = residual

        if atom_class_idx is None:
            self._atom_class_idx = torch.arange(self._num_atoms)
        else:
            self._atom_class_idx = atom_class_idx
        self._num_atom_classes = torch.max(self._atom_class_idx) + 1
        if pretrained_atom_emb is None:
            self._atom_emb = nn.Embedding(
                num_embeddings=self._num_atom_classes, embedding_dim=self._atom_emb_dim
            )
            self._atom_emb_init_type = atom_emb_init_type
            self.init_emb()
        else:
            self._atom_emb = nn.Embedding.from_pretrained(
                pretrained_atom_emb
            )  # should be defined based on atom classes
        self._atom_emb.weight.requires_grad = not freeze

        if self._graph_emb_dim is not None:
            self.fc_graph = nn.Linear(self._atom_emb_dim, self._graph_emb_dim)

        self._atom_emb_last = None
        self._graph_emb = None

    def init_emb(self):
        if self._atom_emb_init_type == "normal":
            self._atom_emb.weight.data.normal_()

        elif self._atom_emb_init_type == "uniform":
            self._atom_emb.weight.data.uniform_()

        else:
            raise ValueError("Only normal and uniform distributions are supported")

    def pooling(self, atom_emb):
        summed = torch.sum(atom_emb, dim=1)
        return summed / self._num_atoms

    @property
    def atom_emb_last(self):
        if self._atom_emb_last is None:
            raise ValueError("Please feed the data first")
        return self._atom_emb_last.detach()

    @property
    def graph_emb(self):
        if self._graph_emb is None:
            raise ValueError("Please feed the data first")
        return self._graph_emb.detach()

    def forward(self, data):
        num_nodes = data[-1, -1]
        num_graphs = int(num_nodes // self._num_atoms)

        edge_dist = data[:-1, -1].reshape((-1, 1))  # (num_edges, 1)
        edge_list = data[:-1, :2].to(torch.int64)

        batch_atom_class_idx = self._atom_class_idx.repeat(num_graphs).to(device=edge_dist.device)
        atom_emb = self._atom_emb(batch_atom_class_idx)  # (num_atoms_over_batches, atom_emb_dim)
        edge_emb = self._rbf(edge_dist)  # (num_edges, edge_emb_dim)

        for i in range(self._num_conv):
            tmp_conv = self.convs[i](atom_emb=atom_emb, edge_emb=edge_emb, edge_list=edge_list)
            if self._residual:
                atom_emb = atom_emb + tmp_conv
            else:
                atom_emb = tmp_conv

        self._atom_emb_last = self._conv_activation(atom_emb)

        if self._graph_emb_dim is None:
            self._graph_emb = self.pooling(
                self._atom_emb_last.reshape((-1, self._num_atoms, self._atom_emb_dim))
            )
        else:
            self._graph_emb = self.fc_graph(
                self.pooling(self._atom_emb_last.reshape((-1, self._num_atoms, self._atom_emb_dim)))
            )

        return self._graph_emb


# ------------------------------------------------------------------
# Menagerie staging entrypoints
# ------------------------------------------------------------------
def build_graphvampnet():
    torch.manual_seed(0)
    # Small synthetic system: 6 atoms, 2 conv layers, tiny RBF grid.
    return GraphVAMPNetLayer(
        num_atoms=6,
        num_conv=2,
        atom_emb_dim=8,
        dmin=0.0,
        dmax=1.0,
        step=0.2,
        graph_emb_dim=4,
    )


def example_input_graphvampnet():
    torch.manual_seed(0)
    # The real repo's graphloader packs each MD-frame graph as a flat 2D tensor: one row
    # per edge holding [src_idx, dst_idx, edge_dist], plus a trailing sentinel row whose
    # last entry ([-1, -1]) encodes num_nodes = num_atoms * num_graphs_in_batch (see
    # graphloader.py / GraphVAMPNetLayer.forward). Build one batch of 2 graphs over the
    # 6-atom system used in build_graphvampnet, with a small fully-connected edge set per
    # graph so distances stay within [dmin, dmax).
    num_atoms = 6
    num_graphs = 2
    edges = []
    for g in range(num_graphs):
        offset = g * num_atoms
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                dist = 0.1 + 0.05 * abs(i - j)
                edges.append([offset + i, offset + j, dist])
    edges_t = torch.tensor(edges, dtype=torch.float32)
    sentinel = torch.tensor([[-1.0, -1.0, float(num_atoms * num_graphs)]], dtype=torch.float32)
    data = torch.cat([edges_t, sentinel], dim=0)
    return (data,)


MENAGERIE_ENTRIES = [
    ("graphvampnet", build_graphvampnet, example_input_graphvampnet, 2023, MENAGERIE_ZOO),
]
