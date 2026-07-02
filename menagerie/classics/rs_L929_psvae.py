# SOURCE: vendored from https://github.com/THUNLP-MT/PS-VAE @ main
# Files: src/modules/encoder.py, src/modules/common_nn.py
#
# PS-VAE (Principal Subgraph Variational Autoencoder) uses a GINEConv-based
# graph neural network encoder to embed molecular graphs into a latent
# condition vector, which a piece-level VAE decoder (RNN + edge predictor,
# RDKit-dependent for molecule reconstruction) then decodes. The decoder's
# `inference()` path performs iterative RDKit molecule construction and is
# not a pure-tensor forward; the Encoder below is the traceable, and
# architecturally distinctive, contribution: t-step GINEConv message passing
# over node/edge features with an embed_node + embed_graph readout, exactly
# as defined in the real repo (imports/module paths adjusted only).

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from src/modules/common_nn.py ---
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, act_func, num_layers):
        super(MLP, self).__init__()
        assert num_layers > 0
        if num_layers == 1:
            self.seq = nn.Linear(dim_in, dim_out)
        else:
            seq = [nn.Linear(dim_in, dim_hidden), act_func()]
            for i in range(num_layers - 2):
                seq.append(nn.Linear(dim_hidden, dim_hidden))
                seq.append(act_func())
            seq.append(nn.Linear(dim_hidden, dim_out))
            self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


# --- vendored from src/modules/encoder.py ---
class Encoder(nn.Module):
    def __init__(self, dim_in, num_edge_type, dim_hidden, dim_out, t=4):
        super(Encoder, self).__init__()
        self.num_edge_type = num_edge_type
        self.t = t  # number of iterations
        self.node_trans = nn.Linear(dim_in, dim_hidden)
        self.edge_trans = nn.Linear(num_edge_type, dim_hidden)
        self.conv = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))
        self.linear = nn.Linear(dim_hidden * self.t, dim_out)

    def embed_node(self, x, edge_index, edge_attr):
        x = self.node_trans(x.float())  # [total_num_nodes, dim_hidden]
        edge_attr = self.edge_trans(edge_attr.float()).squeeze(1)  # [total_num_edges, dim_hidden]
        all_x = []
        for _ in range(self.t):
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            all_x.append(x)
        all_x = torch.cat(all_x, dim=-1)  # [total_num_nodes, dim_hidden * t]
        return x, all_x

    def embed_graph(self, all_x, graph_ids, node_mask=None):
        res = torch.zeros(
            (graph_ids[-1] + 1, all_x.shape[-1]), device=all_x.device
        )  # [num_graphs, dim_out]
        if node_mask is not None:
            graph_ids, all_x = graph_ids[~node_mask], all_x[~node_mask]
        res.index_add_(0, graph_ids, all_x)
        res = self.linear(res)  # to dim out
        return res

    def forward(self, batch, return_x=False):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x, all_x = self.embed_node(x, edge_index, edge_attr)
        res = torch.zeros(
            (batch.num_graphs, all_x.shape[-1]), device=all_x.device
        )  # [num_graphs, dim_out]
        res.index_add_(0, batch.batch, all_x)
        res = self.linear(res)  # to dim out

        if return_x:
            return res, x
        return res


# --- staging harness ---
def build_psvae_encoder():
    return Encoder(dim_in=8, num_edge_type=4, dim_hidden=16, dim_out=32, t=2)


def example_input_psvae_encoder():
    # a tiny molecular-graph-like batch: 2 graphs, small node/edge counts
    g1 = Data(
        x=torch.randn(5, 8),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
        ),
        edge_attr=torch.randn(8, 4),
    )
    g2 = Data(
        x=torch.randn(4, 8),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long),
        edge_attr=torch.randn(6, 4),
    )
    batch = Batch.from_data_list([g1, g2])
    return (batch,)


MENAGERIE_ENTRIES = [
    (
        "PSVAE_Encoder",
        "build_psvae_encoder",
        "example_input_psvae_encoder",
        2021,
        "vendored-pytorch",
    ),
]
