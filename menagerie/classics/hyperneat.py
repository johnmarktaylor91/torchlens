"""HyperNEAT substrate with CPPN weight painting, 2009, Stanley, D'Ambrosio, and Gauci.

Paper: "A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks."
A compact CPPN is queried over source-target coordinates to paint substrate weights,
then the painted substrate performs the actual forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HyperNEATSubstrate(nn.Module):
    """CPPN-painted two-layer substrate network."""

    def __init__(self, n_in: int = 25, n_hidden: int = 10, n_out: int = 3) -> None:
        """Initialize CPPN and substrate coordinate buffers.

        Parameters
        ----------
        n_in
            Number of substrate input units.
        n_hidden
            Number of hidden substrate units.
        n_out
            Number of substrate output units.
        """
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.cppn = nn.Sequential(nn.Linear(4, 12), nn.Tanh(), nn.Linear(12, 1))
        self.register_buffer("input_coords", self._grid_coords(n_in))
        self.register_buffer("hidden_coords", self._line_coords(n_hidden, y=0.0))
        self.register_buffer("output_coords", self._line_coords(n_out, y=1.0))

    def _grid_coords(self, n_points: int) -> Tensor:
        """Create square-grid substrate coordinates.

        Parameters
        ----------
        n_points
            Number of grid points.

        Returns
        -------
        Tensor
            Coordinate tensor with shape ``(n_points, 2)``.
        """
        side = int(n_points**0.5)
        axis = torch.linspace(-1.0, 1.0, side)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)[:n_points]

    def _line_coords(self, n_points: int, y: float) -> Tensor:
        """Create one-dimensional substrate coordinates at a fixed height.

        Parameters
        ----------
        n_points
            Number of points on the line.
        y
            Vertical coordinate.

        Returns
        -------
        Tensor
            Coordinate tensor with shape ``(n_points, 2)``.
        """
        x = torch.linspace(-1.0, 1.0, n_points)
        return torch.stack((x, torch.full_like(x, y)), dim=-1)

    def _paint(self, source: Tensor, target: Tensor) -> Tensor:
        """Query the CPPN over all source-target coordinate pairs.

        Parameters
        ----------
        source
            Source coordinates of shape ``(n_source, 2)``.
        target
            Target coordinates of shape ``(n_target, 2)``.

        Returns
        -------
        Tensor
            Painted weight matrix of shape ``(n_source, n_target)``.
        """
        source_expanded = source[:, None, :].expand(source.shape[0], target.shape[0], 2)
        target_expanded = target[None, :, :].expand(source.shape[0], target.shape[0], 2)
        pairs = torch.cat((source_expanded, target_expanded), dim=-1).reshape(-1, 4)
        return torch.tanh(self.cppn(pairs)).reshape(source.shape[0], target.shape[0])

    def forward(self, x: Tensor) -> Tensor:
        """Paint substrate weights and run the substrate network.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, 25)``.

        Returns
        -------
        Tensor
            Substrate output tensor.
        """
        w1 = self._paint(self.input_coords, self.hidden_coords)
        w2 = self._paint(self.hidden_coords, self.output_coords)
        hidden = torch.tanh(x @ w1)
        return hidden @ w2


def build() -> nn.Module:
    """Build a compact HyperNEAT substrate.

    Returns
    -------
    nn.Module
        Configured ``HyperNEATSubstrate`` instance.
    """
    return HyperNEATSubstrate()


def example_input() -> Tensor:
    """Create an example substrate input vector.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 25)``.
    """
    return torch.randn(1, 25)


MENAGERIE_ENTRIES = [
    ("HyperNEAT Substrate+CPPN Weight Painter", "build", "example_input", "2009", "DD")
]
