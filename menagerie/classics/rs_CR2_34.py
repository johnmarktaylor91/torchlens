# FAITHFUL REIMPLEMENTATION from arXiv:2310.13037 (no public code)
from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "reimpl-pytorch"


class GraphSageLayer(nn.Module):
    """GraphSAGE mean aggregation layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize self and neighbor projections."""
        super().__init__()
        self.proj = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Aggregate neighboring plot features with a normalized adjacency."""
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.matmul(adjacency / degree, x)
        return torch.relu(self.proj(torch.cat((x, neigh), dim=-1)))


class AgriGNN(nn.Module):
    """Genotypic-topological GraphSAGE yield predictor."""

    def __init__(self, node_features: int = 6, hidden: int = 12) -> None:
        """Initialize the four-layer Agri-GNN toy model."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphSageLayer(node_features, hidden),
                GraphSageLayer(hidden, hidden),
                GraphSageLayer(hidden, hidden),
                GraphSageLayer(hidden, hidden),
            ]
        )
        self.head = nn.Linear(hidden, 1)

    def build_adjacency(self, coords: torch.Tensor, genotype: torch.Tensor) -> torch.Tensor:
        """Construct spatial plus genotypic similarity edges."""
        spatial = torch.cdist(coords, coords)
        spatial_weight = torch.exp(-spatial)
        genotype_sim = torch.matmul(genotype, genotype.transpose(1, 2)) / genotype.shape[-1]
        adjacency = spatial_weight + torch.relu(genotype_sim)
        eye = torch.eye(adjacency.shape[-1], device=adjacency.device).unsqueeze(0)
        return adjacency * (1.0 - eye)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        genotype: torch.Tensor,
    ) -> torch.Tensor:
        """Predict yield from plot features and genotypic-topological graph."""
        adjacency = self.build_adjacency(coords, genotype)
        x = features
        for layer in self.layers:
            x = layer(x, adjacency)
        return self.head(x).squeeze(-1)


def build_agri_gnn() -> AgriGNN:
    """Build the toy Agri-GNN."""
    return AgriGNN()


def example_input_agri_gnn() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return toy plot features, coordinates, and genotype vectors."""
    return torch.randn(1, 5, 6), torch.randn(1, 5, 2), torch.randn(1, 5, 4)


MENAGERIE_ENTRIES = [("Agri-GNN", build_agri_gnn, example_input_agri_gnn, 2023, "REIMPL")]
