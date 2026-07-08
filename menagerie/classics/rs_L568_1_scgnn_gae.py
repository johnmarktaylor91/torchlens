# SOURCE: vendored from juexinwang/scGNN @ 09ac50ba0a2bbf87535613a2d284872074804d02
# https://github.com/juexinwang/scGNN/blob/master/gae/model.py
# https://github.com/juexinwang/scGNN/blob/master/gae/layers.py
#
# scGNN (Wang et al. 2021, "scGNN is a novel graph neural network framework
# for single-cell RNA-Seq analyses") builds its cell-cell graph autoencoder
# stage on a Kipf-style Graph Convolutional Network VAE/AE (`gae/model.py`),
# with a hand-rolled `GraphConvolution` layer (`gae/layers.py`) that consumes
# a normalized sparse adjacency matrix via `torch.spmm`. Both files import
# only `torch`, so they are vendored here verbatim (only the module import
# path `from gae.layers import GraphConvolution` is inlined into this single
# file; no architecture was changed).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# --- verbatim from gae/layers.py ---
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
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


# --- verbatim from gae/model.py ---
class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj, encode=False):
        z = self.encode(x, adj)
        return z, z, None


# --- menagerie staging glue ---
def _dense_normalized_adj(n):
    """Small dense stand-in for scGNN's row-normalized sparse cell graph
    adjacency (built upstream from a KNN graph in the real pipeline);
    `torch.spmm` accepts a dense-strided second operand fine for tracing."""
    adj = torch.eye(n) + 0.1 * torch.rand(n, n)
    row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return adj / row_sum


def build_scgnn_gcn_vae():
    m = GCNModelVAE(input_feat_dim=32, hidden_dim1=16, hidden_dim2=8, dropout=0.0)
    m.eval()
    return m


def example_input_scgnn_gcn_vae():
    n_cells = 12
    return (torch.randn(n_cells, 32), _dense_normalized_adj(n_cells))


def build_scgnn_gcn_ae():
    m = GCNModelAE(input_feat_dim=32, hidden_dim1=16, hidden_dim2=8, dropout=0.0)
    m.eval()
    return m


def example_input_scgnn_gcn_ae():
    n_cells = 12
    return (torch.randn(n_cells, 32), _dense_normalized_adj(n_cells))


MENAGERIE_ENTRIES = [
    (
        "scGNN-GCNModelVAE",
        "build_scgnn_gcn_vae",
        "example_input_scgnn_gcn_vae",
        2021,
        "vendored-pytorch",
    ),
    (
        "scGNN-GCNModelAE",
        "build_scgnn_gcn_ae",
        "example_input_scgnn_gcn_ae",
        2021,
        "vendored-pytorch",
    ),
]
