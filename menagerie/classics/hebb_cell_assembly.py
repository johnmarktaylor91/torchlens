"""Hebb cell-assembly net, 1949, as reverberating threshold dynamics.

Paper: Hebb 1949, "The Organization of Behavior."
A recurrent threshold net receives Hebbian-potentiated assembly weights, so a
partial cue can ignite a self-sustaining distributed activity pattern.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Hebb cell-assembly net", "build", "example_input", "1949", "CA")]


class CellAssembly(nn.Module):
    """Fixed Hebbian assembly with recurrent ignition."""

    def __init__(self, n_units: int = 10, steps: int = 5, threshold: float = 0.35) -> None:
        """Initialize a small Hebbian weight matrix.

        Parameters
        ----------
        n_units
            Number of binary units.
        steps
            Number of reverberation steps.
        threshold
            Activation threshold.
        """
        super().__init__()
        patterns = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 0, 1, 1],
            ],
            dtype=torch.float32,
        )[:, :n_units]
        weights = patterns.T @ patterns / patterns.shape[0]
        weights.fill_diagonal_(0.0)
        self.register_buffer("weights", weights)
        self.steps = steps
        self.threshold = threshold

    def forward(self, cue: Tensor) -> Tensor:
        """Ignite a cell assembly from a partial binary cue.

        Parameters
        ----------
        cue
            Binary cue with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Settled assembly activation.
        """
        state = cue.clamp(0.0, 1.0)
        for _ in range(self.steps):
            drive = state @ self.weights
            recurrent = torch.sigmoid(16.0 * (drive - self.threshold))
            state = torch.maximum(state, recurrent)
        return state


def build() -> nn.Module:
    """Build a small Hebbian cell assembly.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return CellAssembly()


def example_input() -> Tensor:
    """Create a partial binary cue.

    Returns
    -------
    Tensor
        Example cue with shape ``(1, 10)``.
    """
    return torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
