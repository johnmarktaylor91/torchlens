# SOURCE: vendored from yamizi/OPF-HGNN @ multi_graph
#   utils/models.py (GNN, CLS_MAP)
# LG-HGNN: "OPF-HGNN: Generalizable Heterogeneous Graph Neural Networks for AC Optimal
# Power Flow" (Chatzivasileiadis lab / FNR LEAP project, IEEE PES GM 2024). The real
# architecture is the paper's own production path (see runs/diff_distribution.py,
# runs/homoGNN.py in the same branch): a homogeneous message-passing GNN (`GNN`, stacked
# SAGEConv/GraphConv/GATConv layers) that is converted into a heterogeneous, per-node/
# per-edge-type model via `torch_geometric.nn.to_hetero(model, data.metadata(), aggr="sum")`
# -- PyG's standard "local" (per-relation conv) + "global" (cross-type aggregation via
# to_hetero) heterogeneous-GNN transform over the AC-OPF power-grid graph (bus/gen/
# ext_grid/sgen/line/... node and edge types). The GAT variant (`cls="gat"`, used by the
# paper's attention-based configuration) supplies the "resistance-biased attention" head
# referenced in the paper title/abstract via edge-feature-conditioned attention over the
# line-impedance-derived graph. Vendored verbatim (only import path fixed); only base
# libs (torch, torch_geometric) required.
import torch
from torch_geometric.nn import GATConv, GraphConv, Linear, SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from collections import OrderedDict

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


class FCNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = Linear(input_channels, hidden_channels)
        self.conv2 = Linear(hidden_channels, hidden_channels)
        self.linear = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.linear(x)
        return x


# --- staging build/example helpers (not part of the vendored source) -------------------

MENAGERIE_ZOO = "vendored-pytorch"


def _tiny_opf_hetero_data() -> HeteroData:
    """Minimal AC-OPF-style heterogeneous power-grid graph: bus/gen/ext_grid node
    types connected by (gen -> bus), (ext_grid -> bus), (bus -> bus, via line) and
    their reverses -- matching the node/edge-type vocabulary produced by the real
    repo's utils/pandapower/pandapower_graph.py PandapowerDataset (case9-scale)."""
    torch.manual_seed(0)
    data = HeteroData()
    data["bus"].x = torch.randn(6, 4)
    data["gen"].x = torch.randn(2, 3)
    data["ext_grid"].x = torch.randn(1, 3)

    data["bus", "line", "bus"].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    data["gen", "connects", "bus"].edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    data["ext_grid", "connects", "bus"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data["bus", "rev_connects", "gen"].edge_index = torch.tensor([[0, 2], [0, 1]], dtype=torch.long)
    data["bus", "rev_connects", "ext_grid"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    return data


class _OPFHGNNTraceWrapper(torch.nn.Module):
    """Unpacks the HeteroData x_dict/edge_index_dict tuple the real training code
    passes to the to_hetero-wrapped model (see runs/diff_distribution.py: `model(data.x_dict,
    data.edge_index_dict)`)."""

    def __init__(self, hetero_model: torch.nn.Module) -> None:
        super().__init__()
        self.hetero_model = hetero_model

    def forward(self, x_dict, edge_index_dict):
        return self.hetero_model(x_dict, edge_index_dict)


def build_opf_hgnn() -> torch.nn.Module:
    data = _tiny_opf_hetero_data()
    base = GNN(hidden_channels=[8], out_channels=2, aggr="mean", cls="gat")
    hetero_model = to_hetero(base, data.metadata(), aggr="sum")
    # to_hetero (like PyG's lazy Linear/-1 in_channels) defers parameter materialization
    # until the first real forward call -- warm up with the same tiny graph before tracing.
    with torch.no_grad():
        hetero_model(data.x_dict, data.edge_index_dict)
    hetero_model.eval()
    return _OPFHGNNTraceWrapper(hetero_model)


def example_input_opf_hgnn():
    data = _tiny_opf_hetero_data()
    return (data.x_dict, data.edge_index_dict)


MENAGERIE_ENTRIES = [
    (
        "LG-HGNN (Local-Global Heterogeneous GNN for OPF)",
        build_opf_hgnn,
        example_input_opf_hgnn,
        2024,
        MENAGERIE_ZOO,
    ),
]
