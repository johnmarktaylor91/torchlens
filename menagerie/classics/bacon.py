"""BACON: band-limited coordinate networks for multiscale scene representation.

Paper: Lindell et al., "BACON: Band-limited Coordinate Networks for Multiscale
Scene Representation", CVPR 2022.

BACON composes multiplicative sinusoidal filters with analytically controlled
frequency bands and can expose intermediate level-of-detail outputs.  This
compact version keeps the multiplicative filter stack and multi-scale heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaconLayer(nn.Module):
    """Band-limited sinusoidal multiplicative filter layer."""

    def __init__(self, in_dim: int, hidden: int, freq: float) -> None:
        """Initialize coordinate and hidden projections."""

        super().__init__()
        self.coord = nn.Linear(in_dim, hidden)
        self.hidden = nn.Linear(hidden, hidden)
        self.freq = freq

    def forward(self, coords: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Apply a multiplicative sinusoidal filter."""

        band = torch.sin(self.freq * self.coord(coords))
        return band * torch.tanh(self.hidden(state))


class BACON(nn.Module):
    """Compact multi-scale BACON coordinate network."""

    def __init__(self, in_dim: int = 3, hidden: int = 48) -> None:
        """Initialize filter stack and scale heads."""

        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList(
            [BaconLayer(in_dim, hidden, freq) for freq in (1.0, 2.0, 4.0, 8.0)]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(4)])

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Evaluate multiscale coordinate outputs."""

        state = torch.sin(self.input(coords))
        outs = []
        for layer, head in zip(self.layers, self.heads, strict=True):
            state = layer(coords, state)
            outs.append(head(state))
        return torch.cat(outs, dim=-1)


def build() -> nn.Module:
    """Build compact BACON."""

    return BACON()


def example_input() -> torch.Tensor:
    """Return 3D query coordinates."""

    return torch.randn(1, 32, 3)


MENAGERIE_ENTRIES = [
    ("BACON", "build", "example_input", "2022", "implicit"),
]
