# SOURCE: vendored from PowerGraph-Datasets/PowerGraph-Graph @ main
# https://github.com/PowerGraph-Datasets/PowerGraph-Graph/blob/main/code/gnn/model.py
# The `GNN_basic`/`GCN` classes (graph-classification/regression GNN used to benchmark
# and explain PowerGraph, the NeurIPS 2024 Datasets & Benchmarks power-grid cascading-
# failure dataset) are transcribed VERBATIM from code/gnn/model.py. Only change: the
# GNNPool readout dispatch is inlined to a fixed "max" readout (global_max_pool, the
# config default used for every released PowerGraph dataset split in
# config/dataset.yaml) so the staging module needs no external readout-name plumbing;
# no architectural layer was added, removed, or altered.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_max_pool

MENAGERIE_ZOO = "vendored-pytorch"


# --- code/gnn/model.py (verbatim architecture, GCN branch) ---
class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r"""Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                else:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if hasattr(data, "edge_weight") and data.edge_weight is not None:
                    edge_weight = data.edge_weight
                else:
                    edge_weight = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )
            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_attr = torch.ones(
                    (edge_index.shape[1], self.edge_dim),
                    dtype=torch.float32,
                    device=x.device,
                )
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            elif len(args) == 3:
                x, edge_index, edge_attr = args[0], args[1], args[2]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            elif len(args) == 4:
                x, edge_index, edge_attr, batch = args[0], args[1], args[2], args[3]
                edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=x.device)
            else:
                raise ValueError(
                    f"forward's args should take 1, 2 or 3 arguments but got {len(args)}"
                )
        else:
            data = kwargs.get("data")
            if not data:
                x = kwargs.get("x")
                edge_index = kwargs.get("edge_index")
                edge_weight = kwargs.get("edge_weight")
                assert x is not None
                assert edge_index is not None
                edge_attr = kwargs.get("edge_attr")
                if edge_attr is None:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                batch = kwargs.get("batch")
                if batch is None:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                if edge_weight is None:
                    edge_weight = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )
            else:
                x = data.x
                edge_index = data.edge_index
                edge_attr = getattr(data, "edge_attr", None)
                if edge_attr is None:
                    edge_attr = torch.ones(
                        (edge_index.shape[1], self.edge_dim),
                        dtype=torch.float32,
                        device=x.device,
                    )
                batch = getattr(data, "batch", None)
                if batch is None:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_weight = getattr(data, "edge_weight", None)
                if edge_weight is None:
                    edge_weight = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )
        return x, edge_index, edge_attr, edge_weight, batch


class GNN_basic(GNNBase):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        super(GNN_basic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = model_params["edge_dim"]
        self.num_layers = model_params["num_layers"]
        self.hidden_dim = model_params["hidden_dim"]
        self.dropout = model_params["dropout"]
        self.readout = model_params["readout"]
        self.get_layers()
        self.graph_regression = graph_regression

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for _ in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        mlp_dim = current_dim * 2 if self.readout == "cat_max_sum" else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return

    def forward(self, *args, **kwargs):
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        emb = self.get_emb(*args, **kwargs)
        x = global_max_pool(emb, batch)
        self.logits = self.mlps(x)
        if self.graph_regression:
            return self.logits
        else:
            self.probs = F.log_softmax(self.logits, dim=-1)
            return self.probs

    def get_emb(self, *args, **kwargs):
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)
        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params, graph_regression):
        super().__init__(input_dim, output_dim, model_params, graph_regression)

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for _ in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        mlp_dim = current_dim * 2 if self.readout == "cat_max_sum" else current_dim
        self.mlps = nn.Linear(mlp_dim, self.output_dim)
        return

    def get_emb(self, *args, **kwargs):
        x, edge_index, _, _, _ = self._argsparse(*args, **kwargs)
        for layer in self.convs:
            x = layer(x, edge_index)
            x = F.dropout(x, self.dropout, training=self.training)
        return x


def build_powergraph_gcn():
    # Tiny menagerie-scale config: matches config/dataset.yaml `ieee24` split shape
    # (num_layers=3, readout='max') but shrinks hidden_dim to 8 and node feature width
    # to 4 (real released PowerGrid node features are bus-level electrical quantities);
    # graph_regression=True mirrors the "of_reg.mat" regression target released with
    # the dataset (cascading-failure severity regression head).
    model_params = {
        "edge_dim": 4,
        "num_layers": 3,
        "hidden_dim": 8,
        "dropout": 0.0,
        "readout": "max",
    }
    return GCN(input_dim=4, output_dim=1, model_params=model_params, graph_regression=True)


def example_input_powergraph_gcn():
    torch.manual_seed(0)
    num_nodes = 12
    x = torch.randn(num_nodes, 4)
    # simple ring + a few chords, mirroring a small power-grid topology
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return (x, edge_index)


MENAGERIE_ENTRIES = [
    (
        "PowerGraph-GCN",
        "build_powergraph_gcn",
        "example_input_powergraph_gcn",
        2024,
        "vendored-pytorch",
    ),
]
