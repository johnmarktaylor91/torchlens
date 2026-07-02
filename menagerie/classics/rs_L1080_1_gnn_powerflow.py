# SOURCE: vendored from mukhlishga/gnn-powerflow @ main
#
# Graph Neural Simulator for power grids: a message-passing GNN for power flow
# prediction on the IEEE 14-bus test system. Vendored verbatim (only import
# paths adjusted for standalone staging) from:
#   jupyter notebook/[PyG] [14 bus] GNN GNN NN.ipynb  (class My_GNN_GNN_NN)
#
# The notebook defines two variants (My_GNN_NN with 1 GCNConv layer, and
# My_GNN_GNN_NN with 2 stacked GCNConv layers). This vendors the deeper
# My_GNN_GNN_NN variant, which is architecturally representative of the
# family (stacked torch_geometric.nn.GCNConv message passing over the bus
# graph, followed by a flatten + 2-layer MLP readout to predict per-bus
# power-flow quantities).
#
# No architectural changes were made; only the constructor defaults were
# kept as in the original code, and a tiny build_/example_input_ pair was
# added for menagerie tracing.

import torch
import torch.nn as nn
from torch.nn import Linear

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

MENAGERIE_ZOO = "vendored-pytorch"


class My_GNN_GNN_NN(torch.nn.Module):
    def __init__(
        self,
        node_size=None,
        feat_in=None,
        feat_size1=None,
        feat_size2=None,
        hidden_size1=None,
        output_size=None,
    ):
        super(My_GNN_GNN_NN, self).__init__()
        self.feat_in = feat_in if feat_in is not None else 2
        self.feat_size1 = feat_in if feat_in is not None else 5
        self.feat_size2 = feat_in if feat_in is not None else 4
        self.hidden_size1 = hidden_size1 if hidden_size1 is not None else 38
        self.output_size = output_size if output_size is not None else 18

        self.conv1 = GCNConv(feat_in, feat_size1)
        self.conv2 = GCNConv(feat_size1, feat_size2)
        self.lin1 = Linear(node_size * feat_size2, hidden_size1)
        self.lin2 = Linear(hidden_size1, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.tanh(x)

        x = self.conv2(x, edge_index)
        x = torch.tanh(x)

        x = x.flatten(start_dim=0)
        x = self.lin1(x)
        x = torch.tanh(x)

        x = self.lin2(x)

        return x

    def save_weights(self, model, name):
        torch.save(model, name)


# ---------------------------------------------------------------------------
# Staging build/example helpers. Original notebook uses the real IEEE 14-bus
# grid (14 nodes); shrunk here to a tiny 5-node ring graph with matching
# feature/hidden dims for a fast CPU trace, same architecture shape.
# ---------------------------------------------------------------------------
def build_gnn_powerflow():
    torch.manual_seed(0)
    model = My_GNN_GNN_NN(
        node_size=5,
        feat_in=2,
        feat_size1=6,
        feat_size2=4,
        hidden_size1=16,
        output_size=8,
    )
    model.eval()
    return model


def example_input_gnn_powerflow():
    torch.manual_seed(0)
    num_nodes = 5
    x = torch.randn(num_nodes, 2)
    # small ring graph (directed both ways) over 5 buses
    src = list(range(num_nodes))
    dst = [(i + 1) % num_nodes for i in range(num_nodes)]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


MENAGERIE_ENTRIES = [
    (
        "GNN-PowerFlow-14Bus",
        "build_gnn_powerflow",
        "example_input_gnn_powerflow",
        2022,
        MENAGERIE_ZOO,
    ),
]
