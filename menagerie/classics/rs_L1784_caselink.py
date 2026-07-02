# FAITHFUL PORT of yanran-tang/CaseLink @ main (original framework: torch + dgl.nn.GATConv)
#
# Original file: CaseLink_model.py (https://github.com/yanran-tang/CaseLink), a legal-case
# retrieval "Global Case Graph" (GCG) inductive GNN: 1-3 stacked graph-attention layers
# (dgl.nn.GATConv) with ReLU + mean-over-heads between layers, and a final residual add of
# the original input features. `dgl` is not an installed base library here (heavy,
# non-base dependency), so the graph-attention primitive is ported 1:1 to
# `torch_geometric.nn.GATConv` (with `concat=False`, matching DGL's per-layer
# "mean over attention heads" behavior) while preserving every other architectural
# choice from the real CaseLink_model.py forward() unchanged: layer count, ReLU
# placement, inter-layer head-averaging, and the trailing `mean(h, dim=1) + in_feat`
# residual connection.
#
# Reference (real DGL forward, transcribed faithfully with GATConv swapped in):
#
#     def forward(self, g, in_feat):
#         if self.layer_num == 3:
#             h = self.GATconv1(g, in_feat); h = F.relu(h); h = th.mean(h, dim=1)
#             h = self.GATconv2(g, h);       h = F.relu(h); h = th.mean(h, dim=1)
#             h = self.GATconv3(g, h)
#         elif self.layer_num == 2:
#             h = self.GATconv1(g, in_feat); h = F.relu(h); h = th.mean(h, dim=1)
#             h = self.GATconv2(g, h)
#         elif self.layer_num == 1:
#             h = self.GATconv1(g, in_feat)
#         h = th.mean(h, dim=1) + in_feat
#         return h

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

MENAGERIE_ZOO = "ported-pytorch"


class CaseLink(nn.Module):
    """Faithful port of CaseLink's Global Case Graph GNN encoder (3-layer default)."""

    def __init__(self, in_dim, h_dim, out_dim, dropout, layer_num, num_heads):
        super().__init__()
        self.hidden_size = h_dim
        self.layer_num = layer_num

        # torch_geometric.nn.GATConv with concat=False averages the per-head outputs
        # internally, mirroring DGL's `th.mean(h, dim=1)` head-pooling between layers.
        self.GATconv1 = GATConv(
            in_dim,
            h_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.GATconv2 = GATConv(
            h_dim,
            out_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.GATconv3 = GATConv(
            h_dim,
            out_dim,
            heads=num_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.reset_parameters()

    def forward(self, in_feat, edge_index):
        if self.layer_num == 3:
            h = self.GATconv1(in_feat, edge_index)
            h = F.relu(h)

            h = self.GATconv2(h, edge_index)
            h = F.relu(h)

            h = self.GATconv3(h, edge_index)
        elif self.layer_num == 2:
            h = self.GATconv1(in_feat, edge_index)
            h = F.relu(h)

            h = self.GATconv2(h, edge_index)
        else:
            h = self.GATconv1(in_feat, edge_index)

        h = h + in_feat
        return h

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


def build_caselink():
    return CaseLink(in_dim=32, h_dim=32, out_dim=32, dropout=0.0, layer_num=3, num_heads=2)


def example_input_caselink():
    torch.manual_seed(0)
    num_nodes = 12
    in_feat = torch.randn(num_nodes, 32)
    # small ring + random extra edges as a stand-in inductive case graph
    src = torch.arange(num_nodes)
    dst = torch.roll(src, -1)
    extra_src = torch.randint(0, num_nodes, (10,))
    extra_dst = torch.randint(0, num_nodes, (10,))
    edge_index = torch.stack([torch.cat([src, extra_src]), torch.cat([dst, extra_dst])], dim=0)
    return (in_feat, edge_index)


MENAGERIE_ENTRIES = [
    ("CaseLink", build_caselink, example_input_caselink, 2024, "ported-pytorch"),
]
