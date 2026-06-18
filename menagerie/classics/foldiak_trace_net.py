"""Foldiak trace-learning network, 1991, Peter Foldiak.

Paper: "Learning invariance from transformation sequences." Hebbian temporal trace
learning and anti-Hebbian lateral decorrelation are represented by feedforward
rectification, masked lateral competition, and differentiable exponential traces.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Földiák Trace Learning Network", "build", "example_input", "1991", "DA")]


class FoldiakTraceNet(nn.Module):
    """Feature encoder with lateral decorrelation and temporal trace output."""

    def __init__(self, n_input: int = 32, n_units: int = 24, trace_decay: float = 0.8) -> None:
        """Initialize feedforward and lateral weights.

        Parameters
        ----------
        n_input
            Input feature count.
        n_units
            Number of invariant units.
        trace_decay
            Exponential trace retention coefficient.
        """
        super().__init__()
        self.encoder = nn.Linear(n_input, n_units)
        self.lateral = nn.Parameter(torch.randn(n_units, n_units) * 0.02)
        self.trace_decay = trace_decay
        self.n_units = n_units
        self.register_buffer("off_diagonal", 1.0 - torch.eye(n_units))

    def forward(self, x_seq: Tensor) -> tuple[Tensor, Tensor]:
        """Run lateral competition over a transformation sequence.

        Parameters
        ----------
        x_seq
            Input tensor of shape ``(batch, time, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Sequence outputs and final temporal trace.
        """
        trace = x_seq.new_zeros(x_seq.shape[0], self.n_units)
        outputs: list[Tensor] = []
        lateral = 0.5 * (self.lateral + self.lateral.T) * self.off_diagonal
        for step in range(x_seq.shape[1]):
            h = torch.relu(self.encoder(x_seq[:, step]))
            y = torch.relu(h - h @ lateral)
            trace = self.trace_decay * trace + (1.0 - self.trace_decay) * y
            outputs.append(y)
        return torch.stack(outputs, dim=1), trace


def build() -> nn.Module:
    """Build a small Foldiak trace-learning module.

    Returns
    -------
    nn.Module
        Configured ``FoldiakTraceNet`` instance.
    """
    return FoldiakTraceNet()


def example_input() -> Tensor:
    """Return a transformation-sequence example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 50, 32)``.
    """
    return torch.randn(1, 50, 32)
