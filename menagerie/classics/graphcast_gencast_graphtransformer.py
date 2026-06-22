"""GenCast graph-transformer denoiser for ensemble weather forecasting.

Paper: Price et al. 2023/2024, "GenCast: Diffusion-based ensemble forecasting
for medium-range weather"; official DeepMind GraphCast repository.

This compact model keeps GenCast's encoder-processor-decoder graph structure:
grid variables are encoded to mesh nodes, a graph-transformer processor performs
attention-biased message passing, and a decoder maps mesh states back to grid
weather variables inside a diffusion denoiser.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphTransformerBlock(nn.Module):
    """Graph transformer block with learned edge bias."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize graph attention and feed-forward sublayers.

        Parameters
        ----------
        dim:
            Node feature width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.edge_bias = nn.Linear(1, heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Apply edge-biased graph attention.

        Parameters
        ----------
        nodes:
            Mesh node features.
        dist:
            Pairwise mesh distance matrix.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        bias = self.edge_bias(dist.unsqueeze(-1)).permute(2, 0, 1).repeat(nodes.shape[0], 1, 1)
        attn, _ = self.attn(self.norm(nodes), self.norm(nodes), self.norm(nodes), attn_mask=bias)
        nodes = nodes + attn
        return nodes + self.ff(nodes)


class GenCastGraphTransformer(nn.Module):
    """Compact GenCast graph-transformer diffusion denoiser."""

    def __init__(self, grid: int = 6, mesh: int = 5, variables: int = 8, dim: int = 32) -> None:
        """Initialize grid-to-mesh encoder, processor, and decoder.

        Parameters
        ----------
        grid:
            Number of grid points.
        mesh:
            Number of mesh nodes.
        variables:
            Number of weather variables.
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.grid_to_mesh = nn.Parameter(torch.randn(grid, mesh))
        self.mesh_to_grid = nn.Parameter(torch.randn(mesh, grid))
        self.encoder = nn.Linear(variables + 1, dim)
        self.processor = nn.ModuleList([GraphTransformerBlock(dim) for _ in range(2)])
        self.decoder = nn.Linear(dim, variables)

    def forward(self, noisy_grid: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Denoise grid weather variables through mesh graph processing.

        Parameters
        ----------
        noisy_grid:
            Noisy weather state with shape ``(batch, grid, variables)``.
        sigma:
            Diffusion noise level with shape ``(batch, 1)``.

        Returns
        -------
        torch.Tensor
            Denoised grid weather variables.
        """

        sig = sigma.view(sigma.shape[0], 1, 1).expand(-1, noisy_grid.shape[1], 1)
        grid_feat = torch.cat([noisy_grid, sig], dim=-1)
        mesh_feat = torch.einsum("bgc,gm->bmc", grid_feat, torch.softmax(self.grid_to_mesh, dim=0))
        nodes = self.encoder(mesh_feat)
        coords = torch.linspace(0, 1, nodes.shape[1], device=nodes.device)
        dist = (coords[:, None] - coords[None, :]).abs()
        for block in self.processor:
            nodes = block(nodes, dist)
        mesh_out = self.decoder(nodes)
        return torch.einsum("bmv,mg->bgv", mesh_out, torch.softmax(self.mesh_to_grid, dim=0))


def build() -> nn.Module:
    """Build the compact GenCast graph transformer.

    Returns
    -------
    nn.Module
        Random-initialized GenCast graph-transformer module.
    """

    return GenCastGraphTransformer()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy weather variables and a diffusion noise level.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weather grid and sigma tensors.
    """

    return torch.randn(1, 6, 8), torch.ones(1, 1)


MENAGERIE_ENTRIES = [
    ("graphcast.GenCast-graphtransformer", "build", "example_input", "2023", "E6"),
]
