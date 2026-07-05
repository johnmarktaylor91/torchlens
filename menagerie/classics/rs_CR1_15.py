# FAITHFUL REIMPLEMENTATION from https://www.mdpi.com/2227-7390/12/18/2933 (no public code)
"""Toy Adaptnet dual-GNN mesh generation and adaptation model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class MessageBlock(nn.Module):
    """Encode-process-decode message-passing block."""

    def __init__(self, hidden_dim: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        hidden_dim:
            Hidden feature width.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, nodes: Tensor, edges: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Run one edge-then-node message passing update.

        Parameters
        ----------
        nodes:
            Node features of shape ``(num_nodes, hidden_dim)``.
        edges:
            Edge features of shape ``(num_edges, hidden_dim)``.
        edge_index:
            Directed edges as ``(2, num_edges)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node and edge features.
        """
        senders, receivers = edge_index
        edge_input = torch.cat([nodes[senders], nodes[receivers], edges], dim=-1)
        new_edges = edges + self.edge_mlp(edge_input)
        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, receivers, new_edges)
        new_nodes = nodes + self.node_mlp(torch.cat([nodes, aggregated], dim=-1))
        return new_nodes, new_edges


class EncoderProcessorDecoder(nn.Module):
    """Small graph network used for both Meshnet and Graphnet."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, hidden_dim: int = 16) -> None:
        """Initialize the graph network.

        Parameters
        ----------
        node_dim:
            Input node feature dimension.
        edge_dim:
            Input edge feature dimension.
        out_dim:
            Output feature dimension.
        hidden_dim:
            Hidden message-passing width.
        """
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.blocks = nn.ModuleList([MessageBlock(hidden_dim), MessageBlock(hidden_dim)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor) -> Tensor:
        """Predict graph outputs.

        Parameters
        ----------
        node_features:
            Input node features.
        edge_features:
            Input edge features.
        edge_index:
            Directed graph connectivity.

        Returns
        -------
        Tensor
            Per-node predictions.
        """
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for block in self.blocks:
            nodes, edges = block(nodes, edges, edge_index)
        return self.decoder(nodes)


class Adaptnet(nn.Module):
    """Two-GNN Adaptnet with Meshnet and Graphnet branches."""

    def __init__(self) -> None:
        """Initialize Meshnet and Graphnet."""
        super().__init__()
        self.meshnet = EncoderProcessorDecoder(node_dim=5, edge_dim=3, out_dim=2)
        self.graphnet = EncoderProcessorDecoder(node_dim=7, edge_dim=3, out_dim=3)

    def forward(self, sample: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Predict initial mesh parameters and Hessian-metric components.

        Parameters
        ----------
        sample:
            Tuple of node features, edge features, and edge index.

        Returns
        -------
        tuple[Tensor, Tensor]
            Mesh-size/generator parameters and symmetric metric components.
        """
        node_features, edge_features, edge_index = sample
        mesh_params = self.meshnet(node_features, edge_features, edge_index)
        graph_nodes = torch.cat([node_features, mesh_params], dim=-1)
        metric = self.graphnet(graph_nodes, edge_features, edge_index)
        return mesh_params, metric


def build_adaptnet() -> Adaptnet:
    """Build a tiny Adaptnet model.

    Returns
    -------
    Adaptnet
        Model instance.
    """
    return Adaptnet()


def example_input_adaptnet() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small triangular mesh graph input.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Node features, edge features, and directed edge index.
    """
    nodes = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 2, 2, 3, 0], [1, 2, 0, 3, 0, 2]], dtype=torch.long)
    edges = torch.randn(edge_index.shape[1], 3)
    return nodes, edges, edge_index


MENAGERIE_ENTRIES = [("Adaptnet", build_adaptnet, example_input_adaptnet, 2024, "REIMPL")]
