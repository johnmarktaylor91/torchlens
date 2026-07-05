# FAITHFUL REIMPLEMENTATION from arXiv:2210.11164 (no public code) -- A/B codex
"""Trainable-adjacency GNN for multivariate sensor fault diagnosis."""

from __future__ import annotations

import torch
from torch import nn


class TrainableAdjacency(nn.Module):
    """Graph structure learning layer using the paper's tanh(W) parameterization."""

    def __init__(self, num_nodes: int, alpha: float = 0.1, top_k: int | None = None) -> None:
        """Initialize a trainable weighted adjacency matrix.

        Parameters
        ----------
        num_nodes:
            Number of sensor nodes.
        alpha:
            Saturation coefficient used in ``A = tanh(alpha * W)``.
        top_k:
            Optional number of strongest outgoing edges to retain per node.
        """
        super().__init__()
        self.alpha = alpha
        self.top_k = top_k
        self.weight = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.02)

    def forward(self) -> torch.Tensor:
        """Return the learned adjacency matrix.

        Returns
        -------
        torch.Tensor
            Weighted adjacency matrix with self loops removed before GCN normalization.
        """
        adjacency = torch.tanh(self.alpha * self.weight)
        adjacency = adjacency - torch.diag_embed(torch.diagonal(adjacency))
        if self.top_k is None:
            return adjacency
        scores = adjacency.abs()
        _, indices = torch.topk(scores, k=self.top_k, dim=-1)
        mask = torch.zeros_like(adjacency).scatter(-1, indices, 1.0)
        return adjacency * mask


class GraphConvolution(nn.Module):
    """Kipf-Welling graph convolution used by the FDD model."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize a graph convolution.

        Parameters
        ----------
        in_features:
            Input feature dimension per sensor node.
        out_features:
            Output feature dimension per sensor node.
        """
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply normalized graph convolution.

        Parameters
        ----------
        node_features:
            Tensor of shape ``(batch, nodes, features)``.
        adjacency:
            Weighted adjacency matrix of shape ``(nodes, nodes)``.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        support = self.proj(node_features)
        eye = torch.eye(adjacency.shape[0], device=adjacency.device, dtype=adjacency.dtype)
        adjacency_with_self = adjacency + eye
        degree = adjacency_with_self.abs().sum(dim=-1).clamp_min(1e-6)
        inv_sqrt = degree.rsqrt()
        norm = inv_sqrt[:, None] * adjacency_with_self * inv_sqrt[None, :]
        return torch.matmul(norm, support)


class GNNModule(nn.Module):
    """Two-layer GCN module with min readouts."""

    def __init__(
        self, num_nodes: int, window_size: int, hidden_dim: int, top_k: int | None
    ) -> None:
        """Initialize one FDD GNN module.

        Parameters
        ----------
        num_nodes:
            Number of graph nodes.
        window_size:
            Sensor history length used as node features.
        hidden_dim:
            Hidden channels in each GCN layer.
        top_k:
            Optional sparse edge cap per node.
        """
        super().__init__()
        self.adjacency = TrainableAdjacency(num_nodes=num_nodes, top_k=top_k)
        self.gcn1 = GraphConvolution(window_size, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def _readout(self, node_features: torch.Tensor, norm: nn.BatchNorm1d) -> torch.Tensor:
        """Apply batch normalization followed by min graph readout.

        Parameters
        ----------
        node_features:
            Tensor of shape ``(batch, nodes, hidden)``.
        norm:
            Batch normalization module.

        Returns
        -------
        torch.Tensor
            Graph representation of shape ``(batch, hidden)``.
        """
        batch, nodes, channels = node_features.shape
        normalized = norm(node_features.reshape(batch * nodes, channels))
        normalized = normalized.reshape(batch, nodes, channels)
        return normalized.min(dim=1).values

    def forward(self, sensor_window: torch.Tensor) -> torch.Tensor:
        """Encode a multivariate sensor window.

        Parameters
        ----------
        sensor_window:
            Tensor of shape ``(batch, nodes, window)``.

        Returns
        -------
        torch.Tensor
            Concatenated readouts from both GCN layers.
        """
        adjacency = self.adjacency()
        hidden1 = torch.relu(self.gcn1(sensor_window, adjacency))
        readout1 = self._readout(hidden1, self.bn1)
        hidden2 = torch.relu(self.gcn2(hidden1, adjacency))
        readout2 = self._readout(hidden2, self.bn2)
        return torch.cat([readout1, readout2], dim=-1)


class GNNTrainableAdjacencyFDD(nn.Module):
    """Fault classifier with several parallel trainable-adjacency GNN modules."""

    def __init__(
        self,
        num_nodes: int = 8,
        window_size: int = 10,
        hidden_dim: int = 16,
        num_modules: int = 3,
        num_classes: int = 4,
    ) -> None:
        """Initialize the classifier.

        Parameters
        ----------
        num_nodes:
            Number of sensor variables.
        window_size:
            Time window used as node features.
        hidden_dim:
            Hidden channels per GCN layer.
        num_modules:
            Number of parallel graph structure learning modules.
        num_classes:
            Fault classes.
        """
        super().__init__()
        self.modules_list = nn.ModuleList(
            [GNNModule(num_nodes, window_size, hidden_dim, top_k=3) for _ in range(num_modules)]
        )
        self.output = nn.Linear(num_modules * hidden_dim * 2, num_classes)

    def forward(self, sensor_window: torch.Tensor) -> torch.Tensor:
        """Classify a multivariate sensor window.

        Parameters
        ----------
        sensor_window:
            Tensor of shape ``(batch, nodes, window)``.

        Returns
        -------
        torch.Tensor
            Fault logits.
        """
        graph_reprs = [module(sensor_window) for module in self.modules_list]
        return self.output(torch.cat(graph_reprs, dim=-1))


def build_gnn_adj_fdd() -> GNNTrainableAdjacencyFDD:
    """Build a tiny traceable trainable-adjacency FDD GNN.

    Returns
    -------
    GNNTrainableAdjacencyFDD
        Tiny model instance.
    """
    return GNNTrainableAdjacencyFDD()


def example_input_gnn_adj_fdd() -> torch.Tensor:
    """Create a sensor-window example input.

    Returns
    -------
    torch.Tensor
        Example tensor with shape ``(2, 8, 10)``.
    """
    return torch.randn(2, 8, 10)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "GNN-Trainable-Adjacency FDD",
        "build_gnn_adj_fdd",
        "example_input_gnn_adj_fdd",
        2022,
        "REIMPL",
    )
]
