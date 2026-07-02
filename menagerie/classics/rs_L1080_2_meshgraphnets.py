# SOURCE: vendored from echowve/meshGraphNets_pytorch @ master
#
# MeshGraphNets (mesh-based CFD simulation): the encode-process-decode graph
# network architecture (Encoder / GnBlock message-passing processor stack /
# Decoder) plus its EdgeBlock/NodeBlock message-passing primitives. Vendored
# verbatim (only import paths flattened into this single module; no
# architectural changes) from:
#   model/model.py    (build_mlp, Encoder, GnBlock, Decoder, EncoderProcesserDecoder)
#   model/blocks.py   (EdgeBlock, NodeBlock)
#
# The repo's top-level Simulator (model/simulator.py) wraps this network with
# training-time noise injection / running-statistics normalization business
# logic that is orthogonal to the network architecture itself; this module
# vendors the EncoderProcesserDecoder network directly (the actual GNN being
# simulated), which is the real, unmodified mesh-graph message-passing
# architecture used across the MeshGraphNets / GNS family.

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import Data

MENAGERIE_ZOO = "vendored-pytorch"


def build_mlp(in_size, hidden_size, out_size, lay_norm=True):
    module = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    if lay_norm:
        return nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))
    return module


class EdgeBlock(nn.Module):
    def __init__(self, custom_func: nn.Module):
        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        node_attr = graph.x
        senders_idx, receivers_idx = graph.edge_index
        edge_attr = graph.edge_attr

        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)

        edge_attr = self.net(collected_edges)  # Update

        return Data(x=node_attr, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):
    def __init__(self, custom_func: nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class Encoder(nn.Module):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):
        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class GnBlock(nn.Module):
    def __init__(self, hidden_size=128):
        super(GnBlock, self).__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        x = graph.x.clone()
        edge_attr = graph.edge_attr.clone()

        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr

        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):
    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):
        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
        )

        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):
        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded


# ---------------------------------------------------------------------------
# Staging build/example helpers. Original repo defaults to hidden_size=128
# with a configurable message_passing_num (paper uses 15); shrunk here to a
# tiny hidden_size + 2 message-passing steps over a small mesh graph for a
# fast CPU trace, same architecture shape (encode -> N x GnBlock -> decode).
# ---------------------------------------------------------------------------
def build_meshgraphnets():
    torch.manual_seed(0)
    model = EncoderProcesserDecoder(
        message_passing_num=2,
        node_input_size=8,
        edge_input_size=6,
        hidden_size=16,
    )
    model.eval()
    return model


def example_input_meshgraphnets():
    torch.manual_seed(0)
    num_nodes = 6
    num_edges = 10
    x = torch.randn(num_nodes, 8)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, 6)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


MENAGERIE_ENTRIES = [
    (
        "MeshGraphNets-CFD",
        "build_meshgraphnets",
        "example_input_meshgraphnets",
        2021,
        MENAGERIE_ZOO,
    ),
]
