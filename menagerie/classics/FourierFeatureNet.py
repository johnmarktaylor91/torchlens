"""Fourier Feature Network compact reconstruction.

Paper: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
Domains, 2020.  Coordinates are mapped through random Fourier features before an MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactFourierFeatureNet(nn.Module):
    """Coordinate MLP with fixed Gaussian Fourier feature mapping."""

    def __init__(
        self, in_dim: int = 2, mapping_size: int = 16, hidden: int = 48, out_dim: int = 3
    ) -> None:
        """Initialize Fourier feature network.

        Parameters
        ----------
        in_dim:
            Coordinate dimension.
        mapping_size:
            Number of Fourier frequencies.
        hidden:
            MLP hidden width.
        out_dim:
            Output signal dimension.
        """

        super().__init__()
        self.register_buffer("basis", torch.randn(in_dim, mapping_size) * 8.0)
        self.net = nn.Sequential(
            nn.Linear(mapping_size * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate the Fourier-feature MLP.

        Parameters
        ----------
        coords:
            Coordinate tensor of shape ``(batch, points, in_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted signal values.
        """

        proj = 2.0 * torch.pi * torch.matmul(coords, self.basis)
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(features)


def build_FourierFeatureNet() -> nn.Module:
    """Build compact Fourier Feature Net.

    Returns
    -------
    nn.Module
        Random-init Fourier feature network.
    """

    return CompactFourierFeatureNet()


def example_input() -> torch.Tensor:
    """Create compact coordinate input.

    Returns
    -------
    torch.Tensor
        Coordinate tensor of shape ``(1, 16, 2)``.
    """

    return torch.rand(1, 16, 2) * 2.0 - 1.0


build = build_FourierFeatureNet

MENAGERIE_ENTRIES = [
    ("FourierFeatureNet", "build_FourierFeatureNet", "example_input", "2020", "E5")
]
