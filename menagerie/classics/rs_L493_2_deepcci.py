# SOURCE: vendored from JiangBioLab/DeepCCI @ main
# (Cluster_model/GNN.py, Cluster_model/Cluster.py)
#
# DeepCCI cell-clustering model: a dual-path autoencoder + graph-convolutional-network
# (SDCN-style "structural deep clustering network") that fuses a fully-connected
# denoising-autoencoder branch (AE_train) with a stack of GNNLayer graph-convolutions
# (GNN.py) via additive skip connections at every encoder/decoder stage, then produces a
# Student's-t soft cluster assignment `q` from the fused latent `z`. Copied verbatim aside
# from: stripping the CLI/argparse/training driver code (imports of preprocess/utils/eva/
# umap/matplotlib that only exist for the standalone training script), and replacing the
# `self.ae.load_state_dict(torch.load(args.pretrain_path, ...))` checkpoint-restore call
# with a plain random init (the architecture is identical; we only need the graph, not
# pretrained weights). `torch.spmm` in GNNLayer.forward is real, unmodified repo code --
# it accepts a dense adjacency tensor natively (no sparse-tensor construction required),
# so this traces with a dense random adjacency matrix.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

MENAGERIE_ZOO = "vendored-pytorch"


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class AE_train(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z):
        super(AE_train, self).__init__()
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class ClusterModel(nn.Module):
    def __init__(
        self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z, n_clusters, v=1
    ):
        super(ClusterModel, self).__init__()

        # autoencoder for intra information
        self.ae = AE_train(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
        )
        # NOTE: real repo restores pretrained AE weights here via
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'));
        # skipped for a from-scratch trace (architecture is unaffected).

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # GCN Module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(h1, adj)
        h3 = self.gnn_3(h2, adj)
        h4 = self.gnn_4(h3, adj)
        h5 = self.gnn_5(h4, adj, active=False)
        predict = F.softmax(h5, dim=1)

        enc_h1 = F.relu(self.ae.enc_1(x))
        enc_h2 = F.relu(self.ae.enc_2(enc_h1 + h1))
        enc_h3 = F.relu(self.ae.enc_3(enc_h2 + h2))
        z = self.ae.z_layer(enc_h3 + h3)

        dec_h1 = F.relu(self.ae.dec_1(z + h4))
        dec_h2 = F.relu(self.ae.dec_2(dec_h1 + h3))
        dec_h3 = F.relu(self.ae.dec_3(dec_h2 + h2))
        x_bar = self.ae.x_bar_layer(dec_h3 + h1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


class DeepCCIClusterStaging(nn.Module):
    """Thin wrapper: real ClusterModel forward returns a 4-tuple; expose it directly."""

    def __init__(self, cluster_model):
        super().__init__()
        self.cluster_model = cluster_model

    def forward(self, x, adj):
        return self.cluster_model(x, adj)


def build_deepcci():
    n_input = 32
    n_clusters = 4
    cluster_model = ClusterModel(
        16,
        12,
        12,
        12,
        12,
        16,
        n_input=n_input,
        n_z=5,
        n_clusters=n_clusters,
    )
    return DeepCCIClusterStaging(cluster_model)


def example_input_deepcci():
    n_cells = 10
    n_input = 32
    x = torch.randn(n_cells, n_input)
    adj = torch.rand(n_cells, n_cells)
    return (x, adj)


MENAGERIE_ENTRIES = [
    ("DeepCCI", build_deepcci, example_input_deepcci, 2023, "SOURCE_AVAILABLE"),
]
