# SOURCE: vendored from QuKunLab/SPACEL @ main
#
# SPACEL ships several independent torch sub-tools sharing one package. This staging
# module vendors the two that are genuine torch nn.Module architectures with no
# non-base dependency in the model-definition code itself:
#
#   1. Spoint's `PredictionModel` (SPACEL/Spoint/base_model.py) -- an autoencoder-style
#      MLP that maps normalized spot gene expression into a cell-type latent space and
#      back (encoder / pred / decoder heads), used for spatial-transcriptomics cell-type
#      deconvolution.
#   2. Splane's `Splane_GCN` + `Splane_Disc` (SPACEL/Splane/base_model.py) -- a spectral
#      graph-convolutional autoencoder (custom `GraphConvolution` layer using Chebyshev-
#      style multi-support graph convolutions, as in Kipf & Welling 2017) plus an
#      adversarial discriminator MLP, used for spatial-domain identification across
#      multi-slice spatial transcriptomics.
#
# Both classes are transcribed verbatim from the real repo; only the relative-package
# imports (`from . import ...`) were dropped since this file vendors just the model
# definitions, not the full training/data pipeline. No architecture code was altered.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# ---------------------------------------------------------------------------
# Spoint: SPACEL/Spoint/base_model.py
# ---------------------------------------------------------------------------


class PredictionModel(nn.Module):
    def __init__(
        self,
        input_dims,
        latent_dims,
        hidden_dims,
        celltype_dims,
        dropout,
    ):
        super(PredictionModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, latent_dims),
        )
        self.decoder = nn.Sequential(
            nn.Linear(celltype_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Linear(hidden_dims, input_dims),
        )
        self.pred = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, celltype_dims),
            nn.Softmax(dim=1),
        )

        nn.init.kaiming_normal_(self.encoder[0].weight)
        nn.init.kaiming_normal_(self.encoder[4].weight)
        nn.init.kaiming_normal_(self.decoder[0].weight)
        nn.init.kaiming_normal_(self.decoder[3].weight)
        nn.init.kaiming_normal_(self.decoder[6].weight)
        nn.init.xavier_uniform_(self.decoder[-1].weight)
        nn.init.kaiming_normal_(self.pred[0].weight)
        nn.init.xavier_uniform_(self.pred[4].weight)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.pred(z)
        decoded = self.decoder(pred)
        return z, pred, decoded


# ---------------------------------------------------------------------------
# Splane: SPACEL/Splane/base_model.py
# ---------------------------------------------------------------------------


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, support, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.support = support
        self.weight = Parameter(torch.FloatTensor(in_features * support, out_features))
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

    def forward(self, features, basis):
        supports = list()
        for i in range(self.support):
            supports.append(basis[i].matmul(features))
        supports = torch.cat(supports, dim=1)
        output = torch.spmm(supports, self.weight)
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


class Splane_GCN(nn.Module):
    def __init__(self, feature_dims, support, latent_dims=8, hidden_dims=64, dropout=0.8):
        super(Splane_GCN, self).__init__()
        self.feature_dims = feature_dims
        self.support = support
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.encode_gc1 = GraphConvolution(feature_dims, hidden_dims, support)
        self.encode_gc2 = GraphConvolution(hidden_dims, latent_dims, support)
        self.decode_gc1 = GraphConvolution(latent_dims, hidden_dims, support)
        self.decode_gc2 = GraphConvolution(hidden_dims, feature_dims, support)

        nn.init.kaiming_normal_(self.encode_gc1.weight)
        nn.init.xavier_uniform_(self.encode_gc2.weight)
        nn.init.kaiming_normal_(self.decode_gc1.weight)
        nn.init.xavier_uniform_(self.decode_gc2.weight)

    @staticmethod
    def l2_activate(x, dim):
        def scale(z):
            zmax = z.max(1, keepdims=True).values
            zmin = z.min(1, keepdims=True).values
            z_std = torch.nan_to_num(torch.div(z - zmin, (zmax - zmin)), 0)
            return z_std

        x = scale(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.encode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.encode_gc2(x, adj)
        return self.l2_activate(x, dim=1)

    def decode(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.decode_gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.decode_gc2(x, adj)
        return x

    def forward(self, x, adj):
        z = self.encode(x, adj)
        x_ = self.decode(z, adj)
        return z, x_


class Splane_Disc(nn.Module):
    def __init__(self, label, latent_dims=8, hidden_dims=64, dropout=0.5):
        super(Splane_Disc, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.class_num = label.shape[1]
        self.disc = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dims),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, self.class_num),
        )

    def forward(self, x):
        x = self.disc(x)
        y = F.softmax(x, dim=1)
        return y


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_spoint_prediction_model():
    # Real usage: input_dims = #genes (100s-1000s), celltype_dims = #cell types,
    # latent_dims=128, hidden_dims=512 (see Spoint/model.py init_model defaults).
    # Shrunk here to keep the trace small/fast.
    return PredictionModel(
        input_dims=64,
        latent_dims=16,
        hidden_dims=32,
        celltype_dims=10,
        dropout=0.5,
    )


def example_input_spoint_prediction_model():
    return (torch.rand(4, 64),)


def build_splane_gcn():
    # Real usage: feature_dims = #cell types after filtering, support=k+1 (k-order graph
    # neighbors, default k=2 -> support=3), latent_dims=16, hidden_dims=64 (see
    # Splane/model.py init_model defaults). Shrunk here to keep the trace small/fast.
    return Splane_GCN(feature_dims=10, support=3, latent_dims=4, hidden_dims=16, dropout=0.0)


def example_input_splane_gcn():
    n_spots, feature_dims, support = 12, 10, 3
    x = torch.rand(n_spots, feature_dims)
    # `basis` is a list of `support` sparse (here: dense, for tracing) adjacency-like
    # matrices as produced by Splane/graph.py's Chebyshev polynomial expansion.
    adj = [torch.eye(n_spots) for _ in range(support)]
    return (x, adj)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Spoint",
        "build_spoint_prediction_model",
        "example_input_spoint_prediction_model",
        2023,
        "vendored",
    ),
    ("Splane-GCN", "build_splane_gcn", "example_input_splane_gcn", 2023, "vendored"),
]
