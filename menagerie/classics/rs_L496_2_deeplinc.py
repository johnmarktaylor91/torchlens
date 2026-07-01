# FAITHFUL PORT of xryanglab/DeepLinc @ main (original framework: TensorFlow 1.x)
# https://github.com/xryanglab/DeepLinc/blob/main/deeplinc/models.py
# https://github.com/xryanglab/DeepLinc/blob/main/deeplinc/layers.py
#
# DeepLinc: reconstructing cell interaction landscapes from spatial
# transcriptomics via an adversarially regularized graph variational
# autoencoder (Li & Yang, Genome Biology 2022, xryanglab/DeepLinc).
#
# The real repo is TensorFlow 1.x (tf.variable_scope, tf.placeholder,
# tf.sparse_tensor_dense_matmul -- API removed in modern TF/no TF1 available
# in this environment), so it cannot be vendored/run as-is. This module
# faithfully transcribes the real architecture from deeplinc/layers.py and
# deeplinc/models.py into self-contained torch: the two-layer sparse-input
# GraphConvolution encoder (GraphConvolutionSparse -> GraphConvolution mean /
# log-std heads), the reparameterization + InnerProductDecoder link-prediction
# head (class `Deeplinc`), and the latent-space adversarial `Discriminator`
# MLP used for the adversarial-regularization term -- every mechanism mirrors
# the original TF1 code (same layer structure, same reparameterization, same
# decoder), only the graph-execution TF1 API is replaced by eager torch ops
# (`torch.sparse.mm`/`torch.matmul` in place of
# `tf.sparse_tensor_dense_matmul`/`tf.matmul`).

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionSparse(nn.Module):
    """Graph convolution layer for sparse inputs (deeplinc/layers.py::GraphConvolutionSparse)."""

    def __init__(self, input_dim, output_dim, dropout=0.0, act=F.relu):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(x, self.weight) if x.is_sparse else torch.matmul(x, self.weight)
        x = torch.sparse.mm(adj, x) if adj.is_sparse else torch.matmul(adj, x)
        return self.act(x)


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph (deeplinc/layers.py::GraphConvolution)."""

    def __init__(self, input_dim, output_dim, dropout=0.0, act=lambda x: x):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = dropout
        self.act = act

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.matmul(x, self.weight)
        x = torch.sparse.mm(adj, x) if adj.is_sparse else torch.matmul(adj, x)
        return self.act(x)


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction (deeplinc/layers.py::InnerProductDecoder)."""

    def __init__(self, dropout=0.0, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, p=self.dropout, training=self.training)
        x = torch.matmul(z, z.t())
        return self.act(x.reshape(-1))


class Deeplinc(nn.Module):
    """Encoder + link-prediction decoder (deeplinc/models.py::Deeplinc).

    Mirrors the original ``_build`` graph: a sparse-input GraphConvolution
    ("e_dense_1") feeds parallel mean/log-std GraphConvolution heads
    ("e_dense_2"/"e_dense_3"), reparameterized into a latent code ``z`` that
    is fed to the InnerProductDecoder for adjacency reconstruction.
    """

    def __init__(self, num_features, hidden1_dim=125, hidden2_dim=125, dropout=0.0):
        super().__init__()
        self.gc_sparse = GraphConvolutionSparse(
            num_features, hidden1_dim, dropout=dropout, act=F.relu
        )
        self.gc_mean = GraphConvolution(hidden1_dim, hidden2_dim, dropout=dropout, act=lambda x: x)
        self.gc_logstd = GraphConvolution(
            hidden1_dim, hidden2_dim, dropout=dropout, act=lambda x: x
        )
        self.decoder = InnerProductDecoder(dropout=dropout, act=lambda x: x)

    def forward(self, features, adj):
        h1 = self.gc_sparse(features, adj)
        z_mean = self.gc_mean(h1, adj)
        z_log_std = self.gc_logstd(h1, adj)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(z_log_std)
        reconstructions = self.decoder(z)
        return reconstructions, z_mean, z_log_std, z


class Discriminator(nn.Module):
    """Adversarial-regularization discriminator on the latent code (deeplinc/models.py::Discriminator).

    Mirrors the original 2-hidden-layer ``dense`` stack (dc_den1 -> dc_den2
    -> dc_output) used to regularize the encoder's latent distribution
    towards a Gaussian prior.
    """

    def __init__(self, input_dim, dc_hidden1_dim=150, dc_hidden2_dim=125):
        super().__init__()
        self.dc_den1 = nn.Linear(input_dim, dc_hidden1_dim)
        self.dc_den2 = nn.Linear(dc_hidden1_dim, dc_hidden2_dim)
        self.dc_output = nn.Linear(dc_hidden2_dim, 1)

    def forward(self, z):
        x = F.relu(self.dc_den1(z))
        x = F.relu(self.dc_den2(x))
        return self.dc_output(x)


MENAGERIE_ZOO = "ported-pytorch"

_N_NODES = 40
_N_FEATURES = 60


class _DeeplincWithAdj(nn.Module):
    """Staging wrapper: bakes a fixed dense adjacency in so the traced module
    takes a single node-feature tensor input (TorchLens recipe/module inputs
    are concrete tensors; the real repo passes ``adj`` as a second explicit
    input via TF placeholders)."""

    def __init__(
        self, num_features=_N_FEATURES, n_nodes=_N_NODES, hidden1_dim=125, hidden2_dim=125
    ):
        super().__init__()
        self.core = Deeplinc(
            num_features, hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, dropout=0.0
        )
        # Fixed identity+self-loop adjacency (dense) for tracing purposes only.
        self.register_buffer("adj", torch.eye(n_nodes))

    def forward(self, features):
        return self.core(features, self.adj)


def build_deeplinc():
    return _DeeplincWithAdj()


def example_input_deeplinc():
    return torch.randn(_N_NODES, _N_FEATURES)


def build_deeplinc_discriminator():
    return Discriminator(input_dim=125)


def example_input_deeplinc_discriminator():
    return torch.randn(_N_NODES, 125)


MENAGERIE_ENTRIES = [
    (
        "DeepLinc",
        "build_deeplinc",
        "example_input_deeplinc",
        2022,
        MENAGERIE_ZOO,
    ),
    (
        "DeepLinc-Discriminator",
        "build_deeplinc_discriminator",
        "example_input_deeplinc_discriminator",
        2022,
        MENAGERIE_ZOO,
    ),
]
