# SOURCE: vendored from BojarLab/SweetNet @ main
# https://github.com/BojarLab/SweetNet
#
# SweetNet: Burkholz, Quackenbush, Bojar, "Using Graph Convolutional Neural Networks
# to Learn a Representation for Glycans" (Cell Reports 2021). Treats glycans as
# molecular graphs -- monosaccharides as nodes, glycosidic linkages as edges -- and
# classifies them with a stack of `torch_geometric` GraphConv layers interleaved with
# TopKPooling hierarchical pooling; the pooled (max+mean) readouts from all three
# pooling stages are summed and fed to an MLP classification head. This is the
# taxonomic-rank classifier variant from the released `SweetNet_code.ipynb`
# (kingdom-level classification head, `models/sweetnet_kingdom.pt`), the canonical
# architecture also re-shipped inside the `glycowork` pip package
# (`glycowork.ml.models.SweetNet`).
#
# This is the REAL `SweetNet` class transcribed verbatim from the `SweetNet_code.ipynb`
# code cell in the official repo (only Colab/training/data-loading cells were dropped;
# no architectural code was altered). Depends only on `torch` and `torch_geometric`
# (both base libs -- `torch_geometric.nn.{TopKPooling, GraphConv, global_mean_pool,
# global_max_pool}`).

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

MENAGERIE_ZOO = "vendored-pytorch"


class SweetNet(torch.nn.Module):
    def __init__(self, lib_size, num_classes=1):
        super(SweetNet, self).__init__()

        self.conv1 = GraphConv(128, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=lib_size + 1, embedding_dim=128)
        self.lin1 = torch.nn.Linear(256, 1024)
        self.lin2 = torch.nn.Linear(1024, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)
        self.bn1 = torch.nn.BatchNorm1d(1024)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()

    def forward(self, x, edge_index, batch, inference=False):
        att = 0
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.leaky_relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.bn1(self.act1(x))
        x = self.lin2(x)
        x = self.bn2(self.act2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x).squeeze(1)

        if inference:
            x_out = x1 + x2 + x3
            return x, x_out, att
        else:
            return x


def build_sweetnet():
    # Real released checkpoint (`sweetnet_kingdom.pt`) uses a monosaccharide/linkage
    # vocabulary (`lib_size`) of order ~1000+ (glycoletter library built from the
    # training corpus) and num_classes = number of taxonomic kingdoms. Shrunk to a
    # small synthetic vocabulary (lib_size=32) for a menagerie-scale trace;
    # hidden/embedding dims (128) match the real architecture exactly since they are
    # architectural (not data-scale) hyperparameters.
    return SweetNet(lib_size=32, num_classes=4)


def example_input_sweetnet():
    torch.manual_seed(0)
    # A small batch of 2 glycan graphs, monosaccharide/linkage tokens as node features
    # (glycoletter vocabulary indices), glycosidic-linkage edges, and a batch-index
    # vector -- matching the real `dataset_to_graphs()` output consumed by SweetNet.
    num_nodes_per_graph = 6
    num_graphs = 2
    total_nodes = num_nodes_per_graph * num_graphs

    x = torch.randint(0, 32, (total_nodes, 1))

    edges_per_graph = []
    for g in range(num_graphs):
        offset = g * num_nodes_per_graph
        src = torch.arange(0, num_nodes_per_graph - 1) + offset
        dst = torch.arange(1, num_nodes_per_graph) + offset
        edges_per_graph.append(torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0))
    edge_index = torch.cat(edges_per_graph, dim=1)

    batch = torch.repeat_interleave(torch.arange(num_graphs), num_nodes_per_graph)

    return (x, edge_index, batch)


MENAGERIE_ENTRIES = [
    ("SweetNet", "build_sweetnet", "example_input_sweetnet", 2021, "vendored-pytorch"),
]
