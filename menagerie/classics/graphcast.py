"""GraphCast encoder-processor-decoder weather graph network.

Paper: Lam et al. 2023, "Learning skillful medium-range global weather
forecasting." GraphCast maps latitude-longitude grid variables to an
icosahedral multi-mesh, performs mesh message passing, then decodes back to grid
weather tendencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeMessageBlock(nn.Module):
    """GraphCast-style edge/node message-passing block."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        """Initialize edge and node update MLPs.

        Parameters
        ----------
        node_dim:
            Node feature width.
        edge_dim:
            Edge feature width.
        """

        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim), nn.SiLU(), nn.Linear(edge_dim, edge_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update directed edges and receiver nodes.

        Parameters
        ----------
        nodes:
            Node states of shape ``(batch, nodes, node_dim)``.
        edges:
            Edge states of shape ``(batch, edges, edge_dim)``.
        senders:
            Sender node indices.
        receivers:
            Receiver node indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated node and edge states.
        """

        msg_in = torch.cat([nodes[:, senders], nodes[:, receivers], edges], dim=-1)
        new_edges = edges + self.edge_mlp(msg_in)
        agg = torch.zeros(nodes.shape[0], nodes.shape[1], new_edges.shape[-1], device=nodes.device)
        agg.index_add_(1, receivers, new_edges)
        new_nodes = nodes + self.node_mlp(torch.cat([nodes, agg], dim=-1))
        return new_nodes, new_edges


class GraphCastNet(nn.Module):
    """Compact GraphCast weather forecaster."""

    def __init__(self, grid: int = 6, mesh: int = 5, variables: int = 8, dim: int = 32) -> None:
        """Initialize grid-mesh encoder, mesh processor, and decoder.

        Parameters
        ----------
        grid:
            Grid points.
        mesh:
            Mesh nodes.
        variables:
            Weather variables per grid point.
        dim:
            Hidden width.
        """

        super().__init__()
        self.grid_to_mesh = nn.Parameter(torch.randn(grid, mesh))
        self.mesh_to_grid = nn.Parameter(torch.randn(mesh, grid))
        self.grid_encoder = nn.Linear(variables * 2, dim)
        self.mesh_encoder = nn.Linear(dim, dim)
        self.edge_encoder = nn.Linear(3, dim)
        self.processor = nn.ModuleList([EdgeMessageBlock(dim, dim) for _ in range(3)])
        self.decoder = nn.Linear(dim, variables)
        self.register_buffer("senders", torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long))
        self.register_buffer("receivers", torch.tensor([1, 2, 3, 4, 0, 2, 3, 4], dtype=torch.long))
        self.register_buffer("edge_attr", torch.randn(8, 3))

    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Predict next-step weather increments from two previous states.

        Parameters
        ----------
        current:
            Current grid weather state.
        previous:
            Previous grid weather state.

        Returns
        -------
        torch.Tensor
            Next-step grid tendency.
        """

        grid_feat = self.grid_encoder(torch.cat([current, previous], dim=-1))
        mesh = torch.einsum("bgd,gm->bmd", grid_feat, torch.softmax(self.grid_to_mesh, dim=0))
        nodes = self.mesh_encoder(mesh)
        edges = self.edge_encoder(self.edge_attr).unsqueeze(0).expand(current.shape[0], -1, -1)
        for block in self.processor:
            nodes, edges = block(nodes, edges, self.senders, self.receivers)
        decoded = self.decoder(nodes)
        return torch.einsum("bmv,mg->bgv", decoded, torch.softmax(self.mesh_to_grid, dim=0))


def build() -> nn.Module:
    """Build compact GraphCast.

    Returns
    -------
    nn.Module
        GraphCast module.
    """

    return GraphCastNet()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create current and previous weather states.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        GraphCast inputs.
    """

    return torch.randn(1, 6, 8), torch.randn(1, 6, 8)


MENAGERIE_ENTRIES = [
    ("graphcast", "build", "example_input", "2023", "E7"),
    ("GraphCast", "build", "example_input", "2023", "E7"),
    ("graphcast.GraphCastNet", "build", "example_input", "2023", "E7"),
]
