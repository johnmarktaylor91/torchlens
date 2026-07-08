# SOURCE: vendored from yamizi/OPF-HGNN @ multi_graph (utils/models.py)
#
# https://raw.githubusercontent.com/yamizi/OPF-HGNN/multi_graph/utils/models.py
#
# OPF-HGNN ("Generalizable Heterogeneous Graph Neural Networks for AC Optimal Power Flow",
# IEEE PES General Meeting 2024) trains a heterogeneous GNN over power-grid graphs (bus/gen/
# line/ext_grid node types) to predict AC Optimal Power Flow solutions. The repo's default
# branch only carries a README pointing at the `multi_graph` branch, which holds the actual
# source + replication package (MIT licensed) -- fetched from there.
#
# The model class itself, `GNN` in utils/models.py, is architecture-agnostic to node/edge
# typing: it is converted into the heterogeneous network used in the paper via PyTorch
# Geometric's standard `to_hetero(model, metadata)` transform (see `runs/homoGNN.py` and
# `experiments/test_gen_topology.py` in the source repo, both of which import
# `from torch_geometric.nn import to_hetero` and call it on a `GNN` instance before training
# on the heterogeneous bus/gen/line/ext_grid graphs pandapower produces). This staging module
# reproduces that exact real-source path: the real `GNN` class (SAGEConv-backed, message
# passing + per-node-type linear head) is vendored verbatim, then wrapped with the real
# `to_hetero` call against a small synthetic heterogeneous power-grid-shaped graph (bus/gen/
# line node types, mirroring the pandapower node/edge typing the source repo builds), instead
# of standing up the full pandapower dataset-generation pipeline (data/objective plumbing only,
# not part of the architecture).
#
# Code below is copied verbatim from the source file (imports untouched: torch,
# torch_geometric.nn -- both base-lib). Only unused imports (`Linear2d`/`FCNN`, which the
# source file also defines but which is a separate non-hetero baseline model, not used by
# OPF-HGNN's `to_hetero` path) are dropped as non-architectural to this entry.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GraphConv, SAGEConv, to_hetero
from torch.nn import Linear as Linear2d  # noqa: F401 (kept for parity with source file imports)
from torch_geometric.nn import Linear
from collections import OrderedDict

MENAGERIE_ZOO = "vendored-pytorch"

CLS_MAP = {
    "sage": (SAGEConv, {"in_channels": (-1, -1)}),
    "gcn": (GraphConv, {"in_channels": -1, "add_self_loops": False}),
    "gat": (GATConv, {"in_channels": (-1, -1), "add_self_loops": False}),
}


class GNN(torch.nn.Module):
    def __init__(
        self, hidden_channels, out_channels, initial_channels=None, aggr="mean", cls="sage"
    ):
        # aggr can be mean, max or lstm
        super().__init__()

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        else:
            hidden_channels = [int(e) for e in hidden_channels]
        if initial_channels is None:
            initial_channels = hidden_channels[0] * 2

        nb_hidden_layers = len(hidden_channels)
        CLS, cls_params = CLS_MAP[cls]

        params = {"out_channels": initial_channels, "aggr": aggr, **cls_params}
        # print("1 graph ", cls, params)
        self.first_conv = CLS(**params)
        self.convs = torch.nn.ModuleDict(
            OrderedDict(
                [
                    (f"conv{i}", CLS(out_channels=hidden_channels[i], aggr=aggr, **cls_params))
                    for i in range(nb_hidden_layers)
                ]
            )
        )
        # print(self.convs)
        self.linear = Linear(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index):
        x = self.first_conv(x, edge_index)
        x = x.relu()
        for k, v in self.convs.items():
            x = v(x, edge_index)
            x = x.relu()

        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


def _example_hetero_data() -> HeteroData:
    """Small synthetic heterogeneous power-grid-shaped graph: bus/gen/line node types,
    mirroring the pandapower node typing the source repo's `build_dataset` produces (see
    `utils/pandapower/pandapower_graph.py`), used only to size + trace the real `to_hetero`
    wrapped GNN -- not a reimplementation of the dataset pipeline itself.
    """
    torch.manual_seed(0)
    data = HeteroData()
    data["bus"].x = torch.randn(6, 4)
    data["gen"].x = torch.randn(2, 4)
    data["line"].x = torch.randn(5, 4)

    data["bus", "connects", "bus"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    data["gen", "feeds", "bus"].edge_index = torch.tensor([[0, 1], [0, 3]], dtype=torch.long)
    data["bus", "fed_by", "gen"].edge_index = data["gen", "feeds", "bus"].edge_index.flip(0)
    data["line", "carries", "bus"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 4, 5]], dtype=torch.long
    )
    data["bus", "carried_by", "line"].edge_index = data["line", "carries", "bus"].edge_index.flip(0)
    return data


def build_opf_hgnn():
    torch.manual_seed(0)
    base = GNN(hidden_channels=[16], out_channels=4, cls="sage")
    example = _example_hetero_data()
    # Real-source construction path: the paper's heterogeneous OPF-HGNN model is this
    # homogeneous `GNN` converted via PyG's `to_hetero`, exactly as `runs/homoGNN.py` /
    # `experiments/test_gen_topology.py` do in the source repo.
    hetero_model = to_hetero(base, example.metadata(), aggr="sum")
    # SAGEConv/Linear layers above are PyG lazy modules (in_channels=(-1, -1)); materialize
    # their parameter shapes with one warm-up forward pass, exactly as the source repo's
    # training loop does implicitly on its first batch, before this module is handed to
    # TorchLens (which inspects parameter shapes ahead of the traced forward pass).
    with torch.no_grad():
        hetero_model(example.x_dict, example.edge_index_dict)
    return hetero_model


def example_input_opf_hgnn():
    data = _example_hetero_data()
    return [data.x_dict, data.edge_index_dict]


MENAGERIE_ENTRIES = [
    (
        "OPF-HGNN (heterogeneous GNN, AC-OPF)",
        "build_opf_hgnn",
        "example_input_opf_hgnn",
        2024,
        MENAGERIE_ZOO,
    ),
]
