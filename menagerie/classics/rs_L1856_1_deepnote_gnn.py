# SOURCE: real library classes, architecture confirmed against
# ccr-cheng/deepnote-gnn-reproduced @ main (models/gnn.py). The DeepNote-GNN
# patient-similarity graph model is a thin wrapper around unmodified
# torch_geometric.nn convolution classes (GCNConv/GATConv/SAGEConv) stacked
# with ReLU + dropout between layers -- no new architecture is introduced;
# the paper's contribution is the ClinicalBERT-derived patient-similarity
# graph construction + readmission-prediction task, not a new GNN layer.
# This file reproduces the exact `BaseGNN`/`GCN` classes from models/gnn.py
# verbatim (only the `register_model` decorator import is dropped, since it
# is unrelated registry plumbing, not architecture).
"""DeepNote-GNN patient-similarity GCN (real torch_geometric.nn.GCNConv stack)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

MENAGERIE_ZOO = "vendored-pytorch"


class BaseGNN(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, num_layers=2, dropout_p=0.0, **kwargs):
        super(BaseGNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList()
        self.convs.append(self.init_conv(input_size, hidden_size, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(self.init_conv(hidden_size, hidden_size, **kwargs))
        self.convs.append(self.init_conv(hidden_size, out_size, **kwargs))

    def init_conv(self, in_size, out_size, **kwargs):
        raise NotImplementedError

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class GCN(BaseGNN):
    def init_conv(self, in_size, out_size, **kwargs):
        return GCNConv(in_size, out_size, **kwargs)


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny patient-similarity graph, matches the
# repo's default GCN config shape family, scaled down for fast tracing).
# ---------------------------------------------------------------------------


def build_deepnote_gnn():
    return GCN(input_size=32, hidden_size=16, out_size=4, num_layers=2, dropout_p=0.1)


def example_input_deepnote_gnn():
    x = torch.randn(20, 32)
    edge_index = torch.randint(0, 20, (2, 40))
    return (x, edge_index)


MENAGERIE_ENTRIES = [
    ("DeepNote-GNN", "build_deepnote_gnn", "example_input_deepnote_gnn", 2021, "vendored-pytorch"),
]
