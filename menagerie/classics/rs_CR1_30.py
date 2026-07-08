# FAITHFUL REIMPLEMENTATION from https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/examples/cfd/external_aerodynamics/aero_graph_net/README.html (no public code)
"""Toy AeroGraphNet MeshGraphNet-style external aerodynamics model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


class AttentionMessageLayer(nn.Module):
    """Attention-weighted message-passing layer."""

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the message layer.

        Parameters
        ----------
        hidden_dim:
            Hidden feature dimension.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn = nn.Linear(3 * hidden_dim, 1)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Apply attention-weighted edge aggregation.

        Parameters
        ----------
        nodes:
            Node states.
        edges:
            Edge states.
        edge_index:
            Directed edge indices.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated nodes and edges.
        """
        senders, receivers = edge_index
        pair = torch.cat([nodes[senders], nodes[receivers], edges], dim=-1)
        new_edges = edges + self.edge_mlp(pair)
        weights = torch.sigmoid(self.attn(pair))
        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, receivers, weights * new_edges)
        return nodes + self.node_mlp(torch.cat([nodes, aggregated], dim=-1)), new_edges


class AeroGraphNet(nn.Module):
    """MeshGraphNet-based model for surface pressure, wall shear stress, and drag."""

    def __init__(self, node_dim: int = 6, edge_dim: int = 4, hidden_dim: int = 16) -> None:
        """Initialize AeroGraphNet.

        Parameters
        ----------
        node_dim:
            Surface node feature dimension.
        edge_dim:
            Mesh edge feature dimension.
        hidden_dim:
            Hidden message-passing width.
        """
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [AttentionMessageLayer(hidden_dim), AttentionMessageLayer(hidden_dim)]
        )
        self.field_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4)
        )
        self.drag_decoder = nn.Linear(hidden_dim, 1)

    def forward(self, sample: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Predict aerodynamic surface fields and drag coefficient.

        Parameters
        ----------
        sample:
            Node features, edge features, and edge index.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-node pressure plus wall shear stress and graph-level drag.
        """
        node_features, edge_features, edge_index = sample
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, edge_index)
        fields = self.field_decoder(nodes)
        drag = self.drag_decoder(F.relu(nodes).mean(dim=0, keepdim=True))
        return fields, drag


def build_aerographnet() -> AeroGraphNet:
    """Build a tiny AeroGraphNet model.

    Returns
    -------
    AeroGraphNet
        Model instance.
    """
    return AeroGraphNet()


def example_input_aerographnet() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small surface-mesh graph input.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Node features, edge features, and directed edge indices.
    """
    node_features = torch.randn(5, 6)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 2, 4]], dtype=torch.long)
    edge_features = torch.randn(edge_index.shape[1], 4)
    return node_features, edge_features, edge_index


MENAGERIE_ENTRIES = [
    ("AeroGraphNet", build_aerographnet, example_input_aerographnet, 2026, "REIMPL")
]
