"""SMPNN scalable message-passing neural network, 2026, Borde et al.

Paper: 2026, "Scalable Message Passing Neural Networks."
Transformer-style pre-normalized residual blocks replace attention with dense
message-passing over a provided adjacency matrix; PyG data objects are represented by one
packed tensor for trace-clean standalone execution.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SMPNNBlock(nn.Module):
    """Pre-LN residual message-passing block."""

    def __init__(self, dim: int) -> None:
        """Initialize normalization and MLP layers.

        Parameters
        ----------
        dim
            Node feature width.
        """
        super().__init__()
        self.norm_msg = nn.LayerNorm(dim)
        self.message = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        """Apply one message-passing residual block.

        Parameters
        ----------
        nodes
            Node features with shape ``(batch, nodes, dim)``.
        adjacency
            Dense adjacency with shape ``(batch, nodes, nodes)``.

        Returns
        -------
        Tensor
            Updated node features.
        """
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        messages = torch.bmm(adjacency, self.message(self.norm_msg(nodes))) / degree
        nodes = nodes + messages
        return nodes + self.ff(self.norm_ff(nodes))


class SMPNN(nn.Module):
    """Attention-free scalable message-passing network."""

    def __init__(self, n_nodes: int = 8, dim: int = 12, depth: int = 3) -> None:
        """Initialize stacked SMPNN blocks.

        Parameters
        ----------
        n_nodes
            Number of graph nodes in the packed example.
        dim
            Node feature width.
        depth
            Number of message-passing blocks.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        self.blocks = nn.ModuleList([SMPNNBlock(dim) for _ in range(depth)])
        self.readout = nn.Linear(dim, 5)

    def forward(self, packed: Tensor) -> Tensor:
        """Run message passing on packed node features and adjacency.

        Parameters
        ----------
        packed
            Tensor with shape ``(batch, 8, 20)``; first 12 channels are node features and
            the remaining 8 channels encode dense adjacency rows.

        Returns
        -------
        Tensor
            Graph-level logits.
        """
        nodes = packed[:, :, : self.dim]
        adjacency = torch.sigmoid(packed[:, :, self.dim : self.dim + self.n_nodes])
        for block in self.blocks:
            nodes = block(nodes, adjacency)
        return self.readout(nodes.mean(dim=1))


MENAGERIE_ENTRIES = [
    ("SMPNN (Scalable Message Passing NN)", "build", "example_input", "2026", "DA")
]


def build() -> nn.Module:
    """Build an SMPNN module.

    Returns
    -------
    nn.Module
        Configured SMPNN module.
    """
    return SMPNN()


def example_input() -> Tensor:
    """Create packed graph data.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8, 20)``.
    """
    return torch.randn(1, 8, 20)
