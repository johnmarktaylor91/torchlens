# SOURCE: vendored from yuzilan/HeteroTCR @ f4f265dde518504adce0fa0e553c0ae51b4bf454
# https://raw.githubusercontent.com/yuzilan/HeteroTCR/main/code/HeteroModel.py
#
# Yu, Zilan et al. 2024 (Communications Biology) "HeteroTCR: A heterogeneous graph
# neural network for predicting peptide-TCR interaction". Heterogeneous GNN over a
# bipartite `cdr3b` <-> `peptide` graph: `HeteroGNN` stacks `num_layers` PyG
# `HeteroConv` blocks (selectable per-edge-type conv: `SAGEConv`, `TransformerConv`,
# or `FiLMConv`, default 'SAGE') with LeakyReLU between blocks to produce per-node-type
# embeddings; `MLP` decodes a scored edge by concatenating the `cdr3b`/`peptide`
# endpoint embeddings for each `edge_label_index` pair and passing them through a
# 3-layer linear head with a final sigmoid to a binding-probability scalar.
# `HeteroTCR` wires `HeteroGNN` (encoder) + `MLP` (decoder) together. All three
# classes are copied verbatim from the real `code/HeteroModel.py`; no architectural
# changes were made (the commented-out BatchNorm/ReLU lines in `MLP.forward` are
# preserved as dead code exactly as in the original).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear, TransformerConv, FiLMConv


class HeteroTCR(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type="SAGE"):
        super().__init__()
        self.encoder = HeteroGNN(metadata, hidden_channels, num_layers, net_type)
        self.decoder = MLP(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=1024, num_layers=3, net_type="SAGE"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        if net_type == "SAGE":
            for _ in range(num_layers):
                conv = HeteroConv(
                    {edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in metadata[1]}
                )
                self.convs.append(conv)
        elif net_type == "TF":
            for _ in range(num_layers):
                conv = HeteroConv(
                    {edge_type: TransformerConv(-1, hidden_channels) for edge_type in metadata[1]}
                )
                self.convs.append(conv)
        elif net_type == "FiLM":
            for _ in range(num_layers):
                conv = HeteroConv(
                    {edge_type: FiLMConv((-1, -1), hidden_channels) for edge_type in metadata[1]}
                )
                self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return x_dict


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels=1024):
        super().__init__()

        self.lin1 = Linear(hidden_channels * 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.lin2 = Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.lin3 = Linear(256, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x_dict["cdr3b"][row], x_dict["peptide"][col]], dim=-1)

        x = self.lin1(x).relu()
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.lin2(x).relu()
        # x = self.bn2(x)
        # x = self.relu(x)
        x = self.lin3(x)
        x = self.sigmoid(x)
        return x.view(-1)


# --- staging harness (tiny sizes; not part of the real repo) ---


def _example_graph():
    # A tiny HeteroData graph: 6 cdr3b nodes, 4 peptide nodes, each with an 8-dim
    # feature vector (the real repo uses learned CNN features of similar width;
    # SAGEConv's lazy (-1, -1) in-channels makes the exact width immaterial), plus
    # a small edge_index for the 'CBindA' relation and an edge_label_index of pairs
    # to score -- exactly the three positional args HeteroTCR.forward expects, as
    # called in run_Hetero.py: model(data.x_dict, data.edge_index_dict, edge_label_index).
    # data_process.py's TCRDataset_global applies `ToUndirected()` to the built graph
    # right after assigning the ('cdr3b', 'CBindA', 'peptide') edges -- this both makes
    # message passing symmetric AND is load-bearing for HeteroGNN's multi-layer stack:
    # without the reverse edge type, 'cdr3b' nodes never receive an update in layer 1
    # and HeteroConv drops them from x_dict for layer 2's lazy SAGEConv input.
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import ToUndirected

    data = HeteroData()
    data["cdr3b"].x = torch.randn(6, 8)
    data["peptide"].x = torch.randn(4, 8)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    data["cdr3b", "CBindA", "peptide"].edge_index = edge_index
    data = ToUndirected()(data)
    edge_label_index = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 0],
        ],
        dtype=torch.long,
    )
    return data, edge_label_index


def build_heterotcr():
    # hidden_channels shrunk from the real default (1024) to 16 for a fast trace;
    # num_layers=2 (real default 3) and net_type='SAGE' (real default) preserved.
    # metadata is read off the real (post-ToUndirected) example graph: node types
    # 'cdr3b'/'peptide', edge types ('cdr3b', 'CBindA', 'peptide') plus the
    # transform-added reverse ('peptide', 'rev_CBindA', 'cdr3b') -- exactly what
    # data_train.metadata() returns in the real run_Hetero.py before constructing
    # HeteroTCR(data_train.metadata(), hc, nl, nt).
    data, _ = _example_graph()
    model = HeteroTCR(data.metadata(), hidden_channels=16, num_layers=2, net_type="SAGE")

    # Lazy modules (SAGEConv((-1, -1), ...)) need a warm-up pass to materialize
    # their parameter shapes before the traced forward call, exactly as
    # run_Hetero.py does with `with torch.no_grad(): out = model(...)` right after
    # construction and before training/inference.
    data, edge_label_index = _example_graph()
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict, edge_label_index)
    return model


def example_input_heterotcr():
    data, edge_label_index = _example_graph()
    return (data.x_dict, data.edge_index_dict, edge_label_index)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("HeteroTCR", "build_heterotcr", "example_input_heterotcr", 2024, "vendored"),
]
