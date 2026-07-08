# SOURCE: vendored from ostrokach/proteinsolver @ 69ef0965a3fc3bf191804035b539720a06e58ba6
# Files: proteinsolver/models/proteinnet.py, proteinsolver/nn/edge_conv_mod.py
# Minimal changes from upstream: `torch_geometric.utils.scatter_(reduce_name, ...)` (removed in
# modern PyG) is replaced with the modern `torch_geometric.utils.scatter(..., reduce=reduce_name)`
# equivalent. No architectural change -- same layers, same forward-pass structure.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops, scatter

MENAGERIE_ZOO = "vendored-pytorch"


# --- proteinsolver/nn/edge_conv_mod.py (verbatim architecture, scatter_ -> scatter shim) ---
class EdgeConvMod(torch.nn.Module):
    def __init__(self, nn, aggr="max"):
        super().__init__()
        self.nn = nn
        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        if edge_attr is None:
            out = torch.cat([x[row], x[col]], dim=-1)
        else:
            out = torch.cat([x[row], x[col], edge_attr], dim=-1)
        out = self.nn(out)
        x = scatter(out, row, dim=0, dim_size=x.size(0), reduce=self.aggr)

        return x, out

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)


class EdgeConvBatch(nn.Module):
    def __init__(self, gnn, hidden_size, batch_norm=True, dropout=0.2):
        super().__init__()

        self.gnn = gnn

        x_post_modules = []
        edge_attr_post_modules = []

        if batch_norm is not None:
            x_post_modules.append(nn.LayerNorm(hidden_size))
            edge_attr_post_modules.append(nn.LayerNorm(hidden_size))

        if dropout:
            x_post_modules.append(nn.Dropout(dropout))
            edge_attr_post_modules.append(nn.Dropout(dropout))

        self.x_postprocess = nn.Sequential(*x_post_modules)
        self.edge_attr_postprocess = nn.Sequential(*edge_attr_post_modules)

    def forward(self, x, edge_index, edge_attr=None):
        x, edge_attr = self.gnn(x, edge_index, edge_attr)
        x = self.x_postprocess(x)
        edge_attr = self.edge_attr_postprocess(edge_attr)
        return x, edge_attr


# --- proteinsolver/models/proteinnet.py (verbatim) ---
def get_graph_conv_layer(input_size, hidden_size, output_size):
    mlp = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    gnn = EdgeConvMod(nn=mlp, aggr="add")
    graph_conv = EdgeConvBatch(gnn, output_size, batch_norm=True, dropout=0.2)
    return graph_conv


class ProteinNet(nn.Module):
    """Graph neural network that predicts amino-acid identity at each residue from a
    protein contact-graph (ProteinSolver, Strokach et al. 2020, Cell Systems)."""

    def __init__(self, x_input_size, adj_input_size, hidden_size, output_size):
        super().__init__()

        self.embed_x = nn.Sequential(
            nn.Embedding(x_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        if adj_input_size:
            self.embed_adj = nn.Sequential(
                nn.Linear(adj_input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.embed_adj = None

        self.graph_conv_1 = get_graph_conv_layer(
            (2 + bool(adj_input_size)) * hidden_size, 2 * hidden_size, hidden_size
        )
        self.graph_conv_2 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv_3 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.graph_conv_4 = get_graph_conv_layer(3 * hidden_size, 2 * hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.embed_x(x)
        edge_index, _ = remove_self_loops(edge_index)
        edge_attr = self.embed_adj(edge_attr) if edge_attr is not None else None

        x_out, edge_attr_out = self.graph_conv_1(x, edge_index, edge_attr)
        x += x_out
        edge_attr = (edge_attr + edge_attr_out) if edge_attr is not None else edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_2(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_3(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = F.relu(x)
        edge_attr = F.relu(edge_attr)
        x_out, edge_attr_out = self.graph_conv_4(x, edge_index, edge_attr)
        x += x_out
        edge_attr += edge_attr_out

        x = self.linear_out(x)
        return x


def build_proteinsolver():
    # Real pretrained checkpoints use x_input_size=21 (20 amino acids + "unknown" mask
    # token), adj_input_size=2 (contact-graph edge features: e.g. distance + angle),
    # output_size=20 (amino-acid class logits). hidden_size shrunk from the released 128
    # to 8 for a menagerie-scale trace.
    return ProteinNet(x_input_size=21, adj_input_size=2, hidden_size=8, output_size=20)


def example_input_proteinsolver():
    torch.manual_seed(0)
    num_nodes = 12
    # A small random contact graph (undirected, stored as two directed edge_index columns).
    src = torch.randint(0, num_nodes, (20,))
    dst = torch.randint(0, num_nodes, (20,))
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    x = torch.randint(0, 21, (num_nodes,))
    edge_attr = torch.rand(edge_index.size(1), 2)
    return (x, edge_index, edge_attr)


MENAGERIE_ENTRIES = [
    (
        "ProteinSolver",
        "build_proteinsolver",
        "example_input_proteinsolver",
        2020,
        "vendored-pytorch",
    ),
]
