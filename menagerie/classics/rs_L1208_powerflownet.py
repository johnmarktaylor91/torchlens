# SOURCE: vendored from StavrosOrf/PoweFlowNet @ main (networks/MPN.py)
#
# https://raw.githubusercontent.com/StavrosOrf/PoweFlowNet/main/networks/MPN.py
#
# PowerFlowNet ("PowerFlowNet: Power Flow Approximation Using Message Passing Graph Neural
# Networks", Electric Power Systems Research 2024) approximates AC power-flow solutions with
# a message-passing GNN: a learned edge-feature-aggregating message-passing layer
# (`EdgeAggregation`, "PowerFlowConv" in the paper) feeding a stack of `TAGConv` graph
# convolutions, plus a learned per-node mask embedding that tells the network which node
# features are known vs. to-be-predicted. `MaskEmbdMultiMPN` is the paper's headline model
# (mask embedding + interleaved EdgeAggregation/TAGConv message passing, no plain-conv-only
# fallback) -- vendored here verbatim rather than the plainer `GCN`/`MPN` baselines also
# defined in the same file, since it is the architecture the repo's README and paper title
# refer to as "PowerFlowNet" itself (config: configs/standard.json).
#
# Code below is copied verbatim from the source file (imports untouched: torch, torch.nn,
# torch_geometric.nn (MessagePassing, TAGConv, GCNConv, ChebConv), torch_geometric.utils.degree
# -- all base-lib). Only the unrelated `GCN`/`MPN`/`SkipMPN`/`MaskEmbdMPN`/`MultiMPN`/
# `MaskEmbdMultiMPN_NoMP`/`WrappedMultiConv`/`MultiConvNet`/`MPN_simplenet` classes in the same
# file (alternate ablations from the paper, not the headline model) are dropped as
# non-architectural to this entry; `SlackAggregation` (defined but unused by
# `MaskEmbdMultiMPN.forward`, commented out in the source) is likewise dropped.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.utils import degree

MENAGERIE_ZOO = "vendored-pytorch"


class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features"""

    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr="add")
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # self.linear = nn.Linear(nfeature_dim, output_dim)
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def message(self, x_i, x_j, edge_attr):
        """
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        """
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1))  # PNAConv style

    def forward(self, x, edge_index, edge_attr):
        """
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)

        output:
            out:        shape (N, num_nodes, output_dim,)
        """
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0)) # no self loop because NO EDGE ATTR FOR SELF LOOP

        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Feature transformation.
        # x = self.linear(x) # no feature transformation

        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here

        return out


class MaskEmbdMultiMPN(nn.Module):
    """Wrapped Message Passing Network
    - Mask Embedding
    - Multi-step mixed MP+Conv
    - No convolution layers
    """

    def __init__(
        self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate
    ):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        # self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
        # self.convs = nn.ModuleList()
        self.layers = nn.ModuleList()

        if n_gnn_layers == 1:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.layers.append(EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers - 2):  # noqa: E741 (kept for parity with source file)
            self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, hidden_dim))
            self.layers.append(TAGConv(hidden_dim, hidden_dim, K=K))

        # self.layers.append(TAGConv(hidden_dim, output_dim, K=K))
        # self.slack_aggr = SlackAggregation(hidden_dim, hidden_dim, 'to_slack')
        # self.slack_propagate = SlackAggregation(hidden_dim, hidden_dim, 'from_slack')
        self.layers.append(EdgeAggregation(hidden_dim, efeature_dim, hidden_dim, output_dim))

        self.mask_embd = nn.Sequential(
            nn.Linear(nfeature_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, nfeature_dim)
        )
        self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

    def is_directed(self, edge_index):
        "determine if a graph id directed by reading only one edge"
        if edge_index.shape[1] == 0:
            # no edge at all, only single nodes. automatically undirected
            return False
        # next line: if there is the reverse of the first edge does not exist, then directed.
        return edge_index[0, 0] not in edge_index[1, edge_index[0, :] == edge_index[1, 0]]

    def undirect_graph(self, edge_index, edge_attr):
        if self.is_directed(edge_index):
            edge_index_dup = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)  # (2, E)
            edge_index = torch.cat([edge_index, edge_index_dup], dim=1)  # (2, 2*E)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)  # (2*E, fe)

            return edge_index, edge_attr
        else:
            return edge_index, edge_attr

    def forward(self, data):
        # assert data.x.shape[-1] == self.nfeature_dim * 2 + 4 # features and their mask + one-hot node type embedding
        # x = data.x[:, 4:4+self.nfeature_dim] # first four features: node type. not elegant at all this way. just saying.
        assert data.x.shape[-1] == 4
        x = data.x  # (N, 4)
        input_x = x  # noqa: F841 (kept for parity with source file; unused as in original)
        bus_type = data.bus_type.long()  # noqa: F841 (kept for parity with source file; unused as in original)
        batch = data.batch  # noqa: F841 (kept for parity with source file; unused as in original)
        mask = data.pred_mask.float()  # indicating which features to predict (==1)
        edge_index = data.edge_index
        edge_features = data.edge_attr

        x = self.mask_embd(mask) + x

        edge_index, edge_features = self.undirect_graph(edge_index, edge_features)

        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], EdgeAggregation):
                x = self.layers[i](x=x, edge_index=edge_index, edge_attr=edge_features)
            else:
                x = self.layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            x = nn.ReLU()(x)

        # x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_attr)
        if isinstance(self.layers[-1], EdgeAggregation):
            x = self.layers[-1](x=x, edge_index=edge_index, edge_attr=edge_features)
        else:
            x = self.layers[-1](x=x, edge_index=edge_index)

        return x


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


class _DataLike:
    """Minimal attribute-bag standing in for the `torch_geometric.data.Data`/`Batch` object
    the real `MaskEmbdMultiMPN.forward` reads (`data.x`, `.bus_type`, `.batch`, `.pred_mask`,
    `.edge_index`, `.edge_attr`) -- built with `configs/standard.json`'s shapes
    (nfeature_dim=6->efeature_dim=5) so this stays a single-positional-arg model input while
    exercising every attribute the real forward pass touches.
    """

    def __init__(self, x, bus_type, batch, pred_mask, edge_index, edge_attr):
        self.x = x
        self.bus_type = bus_type
        self.batch = batch
        self.pred_mask = pred_mask
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def build_powerflownet():
    torch.manual_seed(0)
    # configs/standard.json in the source repo, except nfeature_dim/output_dim pinned to 4
    # (MaskEmbdMultiMPN.forward hard-asserts data.x.shape[-1] == 4: 4 raw node features, no
    # concatenated mask/one-hot node type -- the mask instead flows in via `pred_mask` and is
    # embedded separately, unlike the plainer MPN/GCN variants in the same source file).
    return MaskEmbdMultiMPN(
        nfeature_dim=4,
        efeature_dim=5,
        output_dim=4,
        hidden_dim=32,
        n_gnn_layers=4,
        K=3,
        dropout_rate=0.2,
    )


def example_input_powerflownet():
    torch.manual_seed(0)
    num_nodes, num_edges = 14, 20
    x = torch.randn(num_nodes, 4)
    bus_type = torch.randint(0, 3, (num_nodes,))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    pred_mask = torch.randint(0, 2, (num_nodes, 4)).float()
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 5)
    return _DataLike(x, bus_type, batch, pred_mask, edge_index, edge_attr)


MENAGERIE_ENTRIES = [
    (
        "PowerFlowNet (MaskEmbdMultiMPN)",
        "build_powerflownet",
        "example_input_powerflownet",
        2024,
        MENAGERIE_ZOO,
    ),
]
