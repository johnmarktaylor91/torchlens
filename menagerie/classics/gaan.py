"""GaAN: gated attention networks for graphs.

Paper: "GaAN: Gated Attention Networks for Learning on Large and Spatiotemporal
Graphs", Zhang et al., UAI 2018.

This compact reimplementation keeps GaAN's distinctive operation: multi-head
key/value graph attention where a separate graph aggregation gate controls the
importance of each attention head at each node. The graph is tiny and fixed so
TorchLens can trace and draw it quickly on CPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaANLayer(nn.Module):
    """Gated multi-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4) -> None:
        """Initialize the GaAN layer.

        Parameters
        ----------
        in_dim:
            Input node-feature dimension.
        out_dim:
            Output node-feature dimension.
        num_heads:
            Number of key/value attention heads.
        """

        super().__init__()
        if out_dim % num_heads != 0:
            msg = "out_dim must be divisible by num_heads"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_heads),
            nn.Sigmoid(),
        )
        self.output = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply gated key/value attention over graph neighbors.

        Parameters
        ----------
        x:
            Node features with shape ``(batch, nodes, in_dim)``.
        adjacency:
            Binary adjacency matrix with self-loops, shape ``(nodes, nodes)``.

        Returns
        -------
        torch.Tensor
            Updated node features with shape ``(batch, nodes, out_dim)``.
        """

        batch, nodes, _ = x.shape
        q = self.query(x).view(batch, nodes, self.num_heads, self.head_dim)
        k = self.key(x).view(batch, nodes, self.num_heads, self.head_dim)
        v = self.value(x).view(batch, nodes, self.num_heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        mask = adjacency.to(dtype=torch.bool, device=x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        heads = torch.einsum("bhij,bjhd->bihd", attn, v)

        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0).to(x.device)
        neighbor_mean = torch.matmul(adjacency.to(x.device), x) / degree
        gates = self.gate(torch.cat([x, neighbor_mean], dim=-1))
        gated_heads = (heads * gates.unsqueeze(-1)).reshape(batch, nodes, -1)
        return F.relu(self.output(gated_heads) + self.skip(x))


class GaANClassifier(nn.Module):
    """Small node-classification network built from GaAN layers."""

    def __init__(self, in_dim: int = 6, hidden_dim: int = 16, classes: int = 3) -> None:
        """Initialize the compact GaAN classifier.

        Parameters
        ----------
        in_dim:
            Input node-feature dimension.
        hidden_dim:
            Hidden feature dimension.
        classes:
            Number of output classes per graph.
        """

        super().__init__()
        adjacency = torch.tensor(
            [
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 1, 1],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("adjacency", adjacency)
        self.layer1 = GaANLayer(in_dim, hidden_dim)
        self.layer2 = GaANLayer(hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a tiny graph from node features.

        Parameters
        ----------
        x:
            Node features with shape ``(batch, 6, in_dim)``.

        Returns
        -------
        torch.Tensor
            Graph logits with shape ``(batch, classes)``.
        """

        x = self.layer1(x, self.adjacency)
        x = self.layer2(x, self.adjacency)
        return self.readout(x.mean(dim=1))


def build() -> nn.Module:
    """Build a compact GaAN graph classifier.

    Returns
    -------
    nn.Module
        Randomly initialized GaAN classifier.
    """

    return GaANClassifier()


def example_input() -> torch.Tensor:
    """Create a tiny graph feature batch.

    Returns
    -------
    torch.Tensor
        Random input with shape ``(1, 6, 6)``.
    """

    return torch.randn(1, 6, 6)


MENAGERIE_ENTRIES = [
    (
        "GaAN (Gated Attention Network)",
        "build",
        "example_input",
        "2018",
        "graph/geometric",
    ),
]
