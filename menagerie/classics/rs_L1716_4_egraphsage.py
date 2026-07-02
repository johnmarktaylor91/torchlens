# FAITHFUL PORT of https://github.com/waimorris/E-GraphSAGE @ master
# (E-GraphSAGE/standard/BoT-IoT/E-GraphSAGE-BoT-IoT-mean-agg_multiclass.ipynb,
# cells defining `SAGELayer`, `SAGE`, `MLPPredictor`, `Model`)
# (original framework: PyTorch + DGL)
#
# E-GraphSAGE (Lo, Layeghy, Sarhan, Gallagher & Portmann, NOMS 2022,
# "E-GraphSAGE: A Graph Neural Network based Intrusion Detection System for
# IoT") extends GraphSAGE to EDGE classification for network-flow intrusion
# detection: each network flow is a graph edge (not a node) carrying NetFlow
# feature vectors, and the model must produce an edge-level prediction. Two
# `SAGELayer`s do edge-feature-aware mean-neighborhood message passing
# (`W_msg(concat(h_src_node, h_edge))`, mean-aggregated per destination node,
# then `W_apply(concat(h_dst_node, h_neigh))` + ReLU); an `MLPPredictor` then
# scores each edge from the concatenation of its two endpoint node
# embeddings after graph convolution.
#
# The real repo's notebooks build this on DGL (`dgl.DGLGraph`,
# `g.update_all(message_func, fn.mean('m','h_neigh'))`,
# `graph.apply_edges(...)`), which is not an installed base library here.
# DGL's `update_all`/mean-reduce over incoming edges is a well-defined,
# mechanical graph primitive (scatter-mean over an edge list keyed by
# destination node) with no discretionary modeling choice in it, so this
# port replaces ONLY that primitive with an equivalent explicit
# `edge_index` + `torch.index_add_`-based mean aggregation in plain torch;
# every learned layer (`W_msg`, `W_apply`, the two stacked `SAGELayer`s, the
# dropout between them, the `.sum(1)` over the vestigial feature-window
# axis, and `MLPPredictor`'s edge-endpoint-concatenation head) is identical
# to the real notebook code. Training-loop/data-loading/confusion-matrix
# cells are not part of the architecture and were dropped.

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


def _mean_aggregate(messages, dst_index, num_nodes):
    """Faithful torch replacement for DGL's
    `g.update_all(message_func, fn.mean('m', 'h_neigh'))`: for each
    destination node, average the incoming messages (zero if a node has no
    incoming edges, matching DGL's mean-reduce default of 0 for empty
    mailboxes)."""
    feat_shape = messages.shape[1:]
    summed = messages.new_zeros((num_nodes, *feat_shape))
    counts = messages.new_zeros((num_nodes,))

    flat_messages = messages.reshape(messages.shape[0], -1)
    flat_summed = summed.reshape(num_nodes, -1)
    flat_summed.index_add_(0, dst_index, flat_messages)
    counts.index_add_(0, dst_index, torch.ones_like(dst_index, dtype=messages.dtype))

    counts = counts.clamp(min=1).view(num_nodes, *([1] * len(feat_shape)))
    return flat_summed.view(num_nodes, *feat_shape) / counts


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        # force to output fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        # apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def forward(self, nfeats, efeats, src_index, dst_index):
        # nfeats: N x 1 x D (vestigial window axis, matches real notebook's
        #   G.ndata['h'] reshape to (num_nodes, 1, edim))
        # efeats: E x 1 x D
        # src_index/dst_index: E (long) edge endpoints
        num_nodes = nfeats.shape[0]

        # Eq4: message_func -- W_msg(concat(h_src, h_edge)), one message per
        # edge, mean-reduced at the destination node (fn.mean('m','h_neigh'))
        h_src = nfeats[src_index]
        messages = self.W_msg(torch.cat([h_src, efeats], dim=2))
        h_neigh = _mean_aggregate(messages, dst_index, num_nodes)

        # Eq5
        h = self.activation(self.W_apply(torch.cat([nfeats, h_neigh], dim=2)))
        return h


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nfeats, efeats, src_index, dst_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(nfeats, efeats, src_index, dst_index)
        return nfeats.sum(1)


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def forward(self, h, src_index, dst_index):
        h_u = h[src_index]
        h_v = h[dst_index]
        score = self.W(torch.cat([h_u, h_v], 1))
        return score


class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout, n_classes=5):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, n_classes)

    def forward(self, nfeats, efeats, src_index, dst_index):
        h = self.gnn(nfeats, efeats, src_index, dst_index)
        return self.pred(h, src_index, dst_index)


def build_egraphsage():
    # ndim_in=efeats.shape[-1] (real notebook seeds node features from
    # NetFlow edge-feature width, Eq1: G.ndata['h'] = ones(N, edim)),
    # ndim_out=16, edim=8 NetFlow feature width, dropout=0.2 (real default).
    return Model(ndim_in=8, ndim_out=16, edim=8, activation=F.relu, dropout=0.2, n_classes=5)


def example_input_egraphsage():
    num_nodes = 12
    num_edges = 20
    edim = 8

    # node features: ones, shape (N, 1, edim), matching Eq1 + the notebook's
    # reshape to insert the vestigial window axis.
    nfeats = torch.ones(num_nodes, 1, edim)
    # edge (NetFlow) features, shape (E, 1, edim)
    efeats = torch.randn(num_edges, 1, edim)

    src_index = torch.randint(0, num_nodes, (num_edges,))
    dst_index = torch.randint(0, num_nodes, (num_edges,))

    return (nfeats, efeats, src_index, dst_index)


MENAGERIE_ENTRIES = [
    (
        "E-GraphSAGE",
        build_egraphsage,
        example_input_egraphsage,
        2022,
        MENAGERIE_ZOO,
    ),
]
