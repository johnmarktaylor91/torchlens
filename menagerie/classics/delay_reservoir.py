"""Optical delay reservoir computer, 2012, Paquot and Duport.

Paper: Paquot 2012, "Optoelectronic reservoir computing."
A single nonlinear delayed node is unfolded into virtual time-multiplexed nodes; this
minimal differentiable version omits optical device calibration and trains only a readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OpticalDelayReservoir(nn.Module):
    """Single-node delay reservoir with masked virtual nodes and linear readout."""

    def __init__(self, virtual_nodes: int = 16, steps: int = 100) -> None:
        """Initialize fixed input masks, feedback, and readout.

        Parameters
        ----------
        virtual_nodes
            Number of time-multiplexed states in the delay line.
        steps
            Number of input samples consumed from the time series.
        """
        super().__init__()
        self.virtual_nodes = virtual_nodes
        self.steps = steps
        self.alpha = 0.72
        self.gamma = 0.95
        self.register_buffer("mask", torch.randn(virtual_nodes) * 0.5)
        self.readout = nn.Linear(virtual_nodes, 4)

    def forward(self, series: Tensor) -> Tensor:
        """Run nonlinear delayed feedback over an input time series.

        Parameters
        ----------
        series
            Input sequence with shape ``(batch, 100)``.

        Returns
        -------
        Tensor
            Readout features from the virtual-node reservoir.
        """
        state = series.new_zeros(series.shape[0], self.virtual_nodes)
        for step in range(self.steps):
            delayed = torch.roll(state, shifts=1, dims=1)
            drive = series[:, step : step + 1] * self.mask
            state = torch.tanh(self.alpha * delayed + self.gamma * drive)
        return self.readout(state)


MENAGERIE_ENTRIES = [
    ("Optical Reservoir Computer (single-node delay)", "build", "example_input", "2012", "DA")
]


def build() -> nn.Module:
    """Build a compact optical delay reservoir.

    Returns
    -------
    nn.Module
        Configured reservoir module.
    """
    return OpticalDelayReservoir()


def example_input() -> Tensor:
    """Create an example scalar time series.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 100)``.
    """
    return torch.randn(1, 100)
