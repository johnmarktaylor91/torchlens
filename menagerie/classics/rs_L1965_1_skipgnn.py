# SOURCE: vendored from kexinhuang12345/SkipGNN @ master
# https://raw.githubusercontent.com/kexinhuang12345/SkipGNN/master/SkipGNN/models.py
# https://raw.githubusercontent.com/kexinhuang12345/SkipGNN/master/SkipGNN/layers.py
#
# "SkipGNN: predicting molecular interactions with skip-graph networks"
# (Huang et al., Scientific Reports 2020). The model jointly propagates over
# the ORIGINAL interaction graph and a derived "skip graph" (2-hop neighbors),
# cross-feeding each graph's first-layer representation into the other
# graph's convolution before decoding a pairwise interaction score. Both the
# GraphConvolution layer (SkipGNN/layers.py) and the SkipGNN model
# (SkipGNN/models.py) are transcribed VERBATIM below; only the relative
# `from layers import GraphConvolution` import was inlined into this single
# file (no architectural change).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---- SkipGNN/layers.py (verbatim) ----
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


# ---- SkipGNN/models.py (verbatim) ----
def reset_parameters(w):
    stdv = 1.0 / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)


class SkipGNN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(SkipGNN, self).__init__()

        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)

        # original graph for skip update
        self.o_gc1_s = GraphConvolution(nhid1, nhid1)

        # skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid1)

        # skip graph for original update
        self.s_gc1_o = GraphConvolution(nfeat, nhid1)
        self.s_gc2_o = GraphConvolution(nhid1, nhid2)

        self.dropout = dropout

        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, o_adj, s_adj, idx):
        o_x = F.relu(self.o_gc1(x, o_adj) + self.s_gc1_o(x, s_adj))
        s_x = F.relu(self.s_gc1(x, s_adj) + self.o_gc1_s(o_x, o_adj))

        o_x = F.dropout(o_x, self.dropout, training=self.training)
        s_x = F.dropout(s_x, self.dropout, training=self.training)

        x = self.o_gc2(o_x, o_adj) + self.s_gc2_o(s_x, s_adj)

        feat_p1 = x[idx[0]]  # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]]  # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim=1)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o, x


def build_skipgnn():
    torch.manual_seed(0)
    return SkipGNN(nfeat=16, nhid1=12, nhid2=8, nhid_decode1=6, dropout=0.0)


def example_input_skipgnn():
    torch.manual_seed(0)
    n_nodes = 10
    nfeat = 16
    x = torch.randn(n_nodes, nfeat)

    def _random_adj(n):
        # torch.spmm accepts either a sparse or dense first operand; use a
        # dense adjacency here so the example input stays a plain strided
        # tensor (TorchLens capture assumes dense strided layouts).
        density = 0.3
        dense = (torch.rand(n, n) < density).float()
        dense = ((dense + dense.t()) > 0).float()
        dense.fill_diagonal_(1.0)
        return dense

    o_adj = _random_adj(n_nodes)
    s_adj = _random_adj(n_nodes)
    idx = (torch.randint(0, n_nodes, (4,)), torch.randint(0, n_nodes, (4,)))
    return (x, o_adj, s_adj, idx)


MENAGERIE_ENTRIES = [
    ("SkipGNN", "build_skipgnn", "example_input_skipgnn", 2020, "vendored"),
]
