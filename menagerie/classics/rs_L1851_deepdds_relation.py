# SOURCE: vendored from Sinwang404/DeepDDs @ master
# https://raw.githubusercontent.com/Sinwang404/DeepDDs/master/models/gat.py
#
# Wang, Liu, Luo, Zhao, Zhu, Zhang 2021/2022 (Briefings in Bioinformatics) "DeepDDS: deep graph
# neural network with attention mechanism to predict synergistic drug combinations"
# (arXiv:2107.02467). `GATNet` is the real DeepDDS GAT-based drug-drug-synergy model: two GATConv
# stacks (shared weights `drug1_gcn1`/`drug1_gcn2`) independently embed the two drugs' molecular
# graphs, global-max-pooled and passed through a shared FC layer; a separate MLP ("reduction")
# embeds the cell-line gene-expression context vector; the three embeddings are L2-normalized,
# concatenated, and fed through a final FC stack to a synergy-classification head. This is the
# same "relational graph attention DDI" model AstraZeneca/chemicalx's `DeepDDS` module documents
# in its docstring as "This implementation follows the code on github ... Sinwang404/DeepDDs" --
# chemicalx's version needs the `torchdrug` package (not installed here), so this vendors the
# ORIGINAL upstream GATNet directly (only base libs: torch + torch_geometric + numpy + pandas).
#
# Copied verbatim from models/gat.py's `GATNet` class: layer names/shapes/order/activation
# choices/dropout rates are untouched. The ONLY changes are minimal import-compatibility fixes
# for the installed torch_geometric version (>=2.x): (1) `GATConv.forward` in the original
# (torch_geometric ~1.x era) repo returns `(x, attention_weights)`; the installed torch_geometric
# 2.x `GATConv` returns a plain Tensor by default, so the `x1, arr = self.drug1_gcn1(...)`
# unpacking calls are replaced with plain `x1 = self.drug1_gcn1(...)` (the `arr` attention-weight
# variable was only used by the dropped, disabled-by-default visualization/case-study code path
# in the original `forward` -- see the large commented-out `if len(batch1) < 1000:` block in the
# real file -- never by the architecture itself); (2) the unused `from heatmap import get_map`
# import (a local case-study plotting helper module not part of the model, only referenced by
# the dead/commented-out visualization block) is dropped since `heatmap.py` is not part of the
# model definition. No layer, connection, or computation is added/removed/reordered.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp


# GAT model (real class name: GATNet)
class GATNet(torch.nn.Module):
    def __init__(
        self,
        num_features_xd=78,
        n_output=2,
        num_features_xt=954,
        output_dim=128,
        dropout=0.2,
        file=None,
    ):
        super(GATNet, self).__init__()

        # graph drug layers
        self.drug1_gcn1 = GATConv(num_features_xd, output_dim, heads=10, dropout=dropout)
        self.drug1_gcn2 = GATConv(output_dim * 10, output_dim, dropout=dropout)
        self.drug1_fc_g1 = nn.Linear(output_dim, output_dim)
        self.filename = file

        # DL cell features
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2),
            nn.ReLU(),
        )

        # combined layers
        self.fc1 = nn.Linear(output_dim * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def get_col_index(self, x):
        row_size = len(x[:, 0])
        row = np.zeros(row_size)
        col_size = len(x[0, :])
        for i in range(col_size):
            row[np.argmax(x[:, i])] += 1
        return row

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv("data/case_study/" + path + "_index.csv", header=0, index=0)

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch

        # deal drug1
        x1 = self.drug1_gcn1(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.drug1_gcn2(x1, edge_index1)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1 = gmp(x1, batch1)  # global max pooling

        x1 = self.drug1_fc_g1(x1)
        x1 = self.relu(x1)

        # deal drug2
        x2 = self.drug1_gcn1(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x2 = self.drug1_gcn2(x2, edge_index2)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x2 = gmp(x2, batch2)  # global max pooling

        x2 = self.drug1_fc_g1(x2)
        x2 = self.relu(x2)

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


def _make_drug_graph(num_features_xd: int, num_nodes: int, num_cells: int) -> Data:
    x = torch.randn(num_nodes, num_features_xd)
    # small ring graph so every node has at least one edge
    src = torch.arange(num_nodes)
    dst = torch.roll(src, -1)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    cell = torch.rand(1, num_cells)
    return Data(x=x, edge_index=edge_index, cell=cell)


def build_deepdds_relation():
    torch.manual_seed(0)
    return GATNet(num_features_xd=16, n_output=2, num_features_xt=32, output_dim=8, dropout=0.2)


def example_input_deepdds_relation():
    torch.manual_seed(0)
    num_features_xd = 16
    num_cells = 32
    drug1 = Batch.from_data_list(
        [
            _make_drug_graph(num_features_xd, 6, num_cells),
            _make_drug_graph(num_features_xd, 5, num_cells),
        ]
    )
    drug2 = Batch.from_data_list(
        [
            _make_drug_graph(num_features_xd, 4, num_cells),
            _make_drug_graph(num_features_xd, 7, num_cells),
        ]
    )
    return [drug1, drug2]


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepDDS-RELATION",
        "build_deepdds_relation",
        "example_input_deepdds_relation",
        2021,
        "vendored",
    ),
]
