# SOURCE: vendored from yuanqm55/GraphPPIS @ 6b5d3126978b577c8d817393f562532681ac283a
# (GraphPPIS_model.py::GraphConvolution/deepGCN/GraphPPIS)
"""GraphPPIS: deep graph convolutional network for protein-protein interaction
site prediction (Yuan et al., Bioinformatics 2021). The real model is a "deepGCN"
stack of GCNII-style (Chen et al. 2020) ``GraphConvolution`` layers -- identity
mapping + initial residual connection over a per-residue contact-map graph --
followed by two FC projection layers (input embedding and output classification
head).

Code below is copied verbatim from the official repo's ``GraphConvolution``,
``deepGCN``, and ``GraphPPIS`` classes (only the unused dataset-loading /
feature-embedding helpers, which require on-disk PSSM/HMM/DSSP/distance-map
feature files not present in this environment, are dropped -- they are I/O
preprocessing, not part of the architecture). The optimizer/loss attributes
GraphPPIS.__init__ constructs are also untouched (harmless at trace time).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


class GraphConvolution(nn.Module):
    """GCNII graph-convolution layer (Chen et al. 2020): identity mapping +
    initial residual connection, as used verbatim by GraphPPIS's deepGCN."""

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):  # noqa: E741 (verbatim upstream param name)
        theta = min(1, math.log(lamda / l + 1))
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:  # speed up convergence of the training process
            output = output + input
        return output


class deepGCN(nn.Module):
    """Stack of GraphConvolution (GCNII) layers with FC input/output projections."""

    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(deepGCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant, residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(
                con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1)
            )
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class GraphPPIS(nn.Module):
    """Protein-protein interaction site predictor: deepGCN over a per-residue
    contact-map graph, with a per-residue [not-bind, bind] classification head."""

    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GraphPPIS, self).__init__()

        self.deep_gcn = deepGCN(
            nlayers=nlayers,
            nfeat=nfeat,
            nhidden=nhidden,
            nclass=nclass,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
            variant=variant,
        )
        self.criterion = (
            nn.CrossEntropyLoss()
        )  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=0)

    def forward(self, x, adj):  # x.shape = (seq_len, FEATURE_DIM); adj.shape = (seq_len, seq_len)
        x = x.float()
        output = self.deep_gcn(x, adj)  # output.shape = (seq_len, NUM_CLASSES)
        return output


# --- staging harness -------------------------------------------------------


def build_graphppis():
    # Real hyperparameters from GraphPPIS_model.py's module-level constants
    # (EMBEDDING="e" -> INPUT_DIM=54, HIDDEN_DIM=256, LAYER=8, DROPOUT=0.1,
    # ALPHA=0.7, LAMBDA=1.5, VARIANT=True, NUM_CLASSES=2), sized down to a
    # tiny hidden width for tracing.
    return GraphPPIS(
        nlayers=2,
        nfeat=54,
        nhidden=16,
        nclass=2,
        dropout=0.1,
        lamda=1.5,
        alpha=0.7,
        variant=True,
    ).eval()


def example_input_graphppis():
    # A single protein's per-residue feature matrix (seq_len=10, feat_dim=54)
    # and a symmetric-normalized dense contact-map adjacency matrix, matching
    # the real ``normalize(adjacency_matrix)`` output shape/dtype consumed by
    # ``torch.spmm`` in GraphConvolution.forward (real repo uses a dense numpy
    # array cast to a torch tensor here, not a torch.sparse tensor).
    seq_len = 10
    feat_dim = 54
    x = torch.rand(seq_len, feat_dim)
    raw_adj = (torch.rand(seq_len, seq_len) < 0.3).float()
    adj = raw_adj + raw_adj.t() + torch.eye(seq_len)
    adj = (adj > 0).float()
    rowsum = adj.sum(1)
    r_inv = rowsum.pow(-0.5)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    norm_adj = r_mat_inv @ adj @ r_mat_inv
    return (x, norm_adj)


MENAGERIE_ENTRIES = [
    ("GraphPPIS", "build_graphppis", "example_input_graphppis", 2021, "vendored"),
]
