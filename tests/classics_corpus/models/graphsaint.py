"""GraphSAINT: subgraph-sampling GCN training architecture.

Paper: "GraphSAINT: Graph Sampling Based Inductive Learning Method", Zeng et
al., ICLR 2020.

This compact classic fixes one pre-sampled induced subgraph and applies the
paper's core idea: run a complete GCN on the sampled subgraph with explicit edge
and node normalization terms, rather than expanding layer-wise neighborhoods.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAINTConv(nn.Module):
    """Normalized GCN layer used inside a sampled GraphSAINT subgraph."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the normalized graph convolution.

        Parameters
        ----------
        in_dim:
            Input feature dimension.
        out_dim:
            Output feature dimension.
        """

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor, edge_norm: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate normalized sampled-neighbor features.

        Parameters
        ----------
        x:
            Sampled node features with shape ``(batch, sampled_nodes, in_dim)``.
        adjacency:
            Sampled adjacency matrix with shape ``(sampled_nodes, sampled_nodes)``.
        edge_norm:
            Edge-normalization weights with the same shape as ``adjacency``.

        Returns
        -------
        torch.Tensor
            Updated sampled-node features.
        """

        norm_adj = adjacency * edge_norm
        degree = norm_adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        aggregated = torch.matmul(norm_adj / degree, x)
        return self.linear(aggregated)


class GraphSAINTNet(nn.Module):
    """Compact GraphSAINT classifier on one fixed sampled subgraph."""

    def __init__(self, in_dim: int = 5, hidden_dim: int = 12, classes: int = 3) -> None:
        """Initialize the sampled-subgraph GCN.

        Parameters
        ----------
        in_dim:
            Input node-feature dimension.
        hidden_dim:
            Hidden node-feature dimension.
        classes:
            Number of graph-level classes.
        """

        super().__init__()
        adjacency = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=torch.float32,
        )
        sampled_nodes = torch.tensor([0, 1, 2, 4, 5, 7], dtype=torch.long)
        node_norm = torch.tensor([1.25, 1.10, 0.90, 1.15, 0.95, 1.30], dtype=torch.float32)
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("sampled_nodes", sampled_nodes)
        self.register_buffer("node_norm", node_norm)
        self.conv1 = GraphSAINTConv(in_dim, hidden_dim)
        self.conv2 = GraphSAINTConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, classes)

    def _sample_subgraph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the fixed induced subgraph and GraphSAINT normalizers.

        Parameters
        ----------
        x:
            Full-graph node features with shape ``(batch, nodes, in_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Sampled features, sampled adjacency, and edge-normalization matrix.
        """

        sub_x = x.index_select(1, self.sampled_nodes)
        sub_adj = self.adjacency.index_select(0, self.sampled_nodes).index_select(
            1, self.sampled_nodes
        )
        edge_norm = torch.outer(self.node_norm, self.node_norm).to(x.device)
        return sub_x, sub_adj.to(x.device), edge_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a complete GCN on a sampled GraphSAINT minibatch.

        Parameters
        ----------
        x:
            Full-graph node features with shape ``(batch, 8, in_dim)``.

        Returns
        -------
        torch.Tensor
            Graph logits with shape ``(batch, classes)``.
        """

        sub_x, sub_adj, edge_norm = self._sample_subgraph(x)
        hidden = F.relu(self.conv1(sub_x, sub_adj, edge_norm))
        hidden = F.relu(self.conv2(hidden, sub_adj, edge_norm))
        weighted = hidden * self.node_norm.to(x.device).view(1, -1, 1)
        return self.classifier(weighted.mean(dim=1))


def build() -> nn.Module:
    """Build a compact GraphSAINT network.

    Returns
    -------
    nn.Module
        Randomly initialized GraphSAINT classifier.
    """

    return GraphSAINTNet()


def example_input() -> torch.Tensor:
    """Create a tiny full-graph feature batch.

    Returns
    -------
    torch.Tensor
        Random input with shape ``(1, 8, 5)``.
    """

    return torch.randn(1, 8, 5)


MENAGERIE_ENTRIES = [
    (
        "GraphSAINT (subgraph-sampling GCN)",
        "build",
        "example_input",
        "2020",
        "graph/geometric",
    ),
]
