"""tiny-cuda-nn HashGrid and FullyFusedMLP compact reconstructions.

tiny-cuda-nn provides CUDA kernels for multiresolution hash-grid encodings and
fully fused MLPs. These pure-torch components preserve the public architectural
operations while omitting only the fused-kernel implementation detail.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HashGridEncoding(nn.Module):
    """Multiresolution 2D hash-grid encoding."""

    def __init__(self, levels: int = 4, features: int = 2, table_size: int = 64) -> None:
        """Initialize hash tables.

        Parameters
        ----------
        levels:
            Number of grid levels.
        features:
            Features per level.
        table_size:
            Entries per hash table.
        """
        super().__init__()
        self.levels = levels
        self.features = features
        self.table_size = table_size
        self.tables = nn.ParameterList(
            [nn.Parameter(torch.randn(table_size, features) * 0.01) for _ in range(levels)]
        )

    def _hash(self, coords: Tensor) -> Tensor:
        """Hash integer 2D coordinates.

        Parameters
        ----------
        coords:
            Integer coordinates.

        Returns
        -------
        Tensor
            Hash-table indices.
        """
        return ((coords[..., 0] * 1_540_863) ^ (coords[..., 1] * 1_251_557)) % self.table_size

    def forward(self, xy: Tensor) -> Tensor:
        """Encode points with bilinear hash-grid interpolation.

        Parameters
        ----------
        xy:
            Points in ``[0, 1]``.

        Returns
        -------
        Tensor
            Concatenated hash-grid features.
        """
        corners = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=xy.device)
        outs: list[Tensor] = []
        for level, table in enumerate(self.tables):
            res = 4 * (2**level)
            scaled = xy * res
            base = torch.floor(scaled).long()
            frac = scaled - base.float()
            coords = base.unsqueeze(1) + corners.unsqueeze(0)
            feats = table[self._hash(coords).reshape(-1)].reshape(xy.shape[0], 4, self.features)
            cw = corners.float().unsqueeze(0)
            weights = (cw * frac.unsqueeze(1) + (1 - cw) * (1 - frac).unsqueeze(1)).prod(
                dim=-1, keepdim=True
            )
            outs.append((feats * weights).sum(dim=1))
        return torch.cat(outs, dim=-1)


class FullyFusedMLP(nn.Module):
    """Small MLP with a packed fully-fused hidden-layer primitive."""

    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 4) -> None:
        """Initialize dense layers.

        Parameters
        ----------
        in_dim:
            Input dimension.
        hidden:
            Hidden width.
        out_dim:
            Output dimension.
        """
        super().__init__()
        self.in_weight = nn.Parameter(torch.randn(in_dim, hidden) * 0.05)
        self.in_bias = nn.Parameter(torch.zeros(hidden))
        self.hidden_weight = nn.Parameter(torch.randn(2, hidden, hidden) * 0.05)
        self.hidden_bias = nn.Parameter(torch.zeros(2, hidden))
        self.out_weight = nn.Parameter(torch.randn(hidden, out_dim) * 0.05)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def fused_fully_fused_kernel(self, x: Tensor) -> Tensor:
        """Apply packed hidden layers as one fused tiny-cuda-nn-style primitive.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        Tensor
            Hidden activations after packed fused layers.
        """
        h = torch.relu(x @ self.in_weight + self.in_bias)
        for layer in range(self.hidden_weight.shape[0]):
            h = torch.relu(h @ self.hidden_weight[layer] + self.hidden_bias[layer])
        return h

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fused MLP.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        Tensor
            Output features.
        """
        h = self.fused_fully_fused_kernel(x)
        return h @ self.out_weight + self.out_bias


class HashGridMLP(nn.Module):
    """HashGrid encoding followed by a fully fused-style MLP."""

    def __init__(self) -> None:
        """Initialize encoding and MLP."""
        super().__init__()
        self.encoding = HashGridEncoding()
        self.mlp = FullyFusedMLP()

    def forward(self, xy: Tensor) -> Tensor:
        """Encode coordinates and apply MLP.

        Parameters
        ----------
        xy:
            Coordinate tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        return self.mlp(self.encoding(xy))


def build_hashgrid() -> nn.Module:
    """Build standalone HashGrid encoding.

    Returns
    -------
    nn.Module
        HashGrid encoding.
    """
    return HashGridEncoding()


def build_mlp() -> nn.Module:
    """Build standalone fully fused-style MLP.

    Returns
    -------
    nn.Module
        Fully fused-style MLP.
    """
    return FullyFusedMLP()


def build_hashgrid_mlp() -> nn.Module:
    """Build HashGrid plus MLP model.

    Returns
    -------
    nn.Module
        HashGrid MLP.
    """
    return HashGridMLP()


def example_input() -> Tensor:
    """Return normalized 2D points.

    Returns
    -------
    Tensor
        Coordinate tensor.
    """
    return torch.rand(16, 2)


def example_mlp_input() -> Tensor:
    """Return dense MLP inputs.

    Returns
    -------
    Tensor
        Feature tensor.
    """
    return torch.randn(16, 8)


MENAGERIE_ENTRIES = [
    ("tinycudann_HashGridEncoding", "build_hashgrid", "example_input", "2021", "E7"),
    ("tinycudann_FullyFusedMLP", "build_mlp", "example_mlp_input", "2021", "E7"),
]
