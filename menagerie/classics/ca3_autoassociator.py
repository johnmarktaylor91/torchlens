"""CA3 hippocampal autoassociator, 1971, Marr, Treves, and Rolls.

Paper: "Simple memory: A theory for archicortex." Sparse recurrent collateral
weights store binary patterns and iterated threshold recall completes partial cues.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("CA3 Hippocampal Autoassociator (Marr-Treves-Rolls)", "build", "example_input", "1971", "DB")
]


class CA3Autoassociator(nn.Module):
    """Fixed recurrent associative memory for pattern completion."""

    def __init__(self, n_units: int = 100, n_patterns: int = 8, steps: int = 4) -> None:
        """Initialize Hebbian recurrent weights from random stored patterns.

        Parameters
        ----------
        n_units
            Number of CA3 recurrent units.
        n_patterns
            Number of random prototype patterns to store.
        steps
            Number of recall iterations.
        """
        super().__init__()
        patterns = torch.sign(torch.randn(n_patterns, n_units))
        weights = patterns.T @ patterns / float(n_units)
        weights = weights * (1.0 - torch.eye(n_units))
        self.register_buffer("weights", weights)
        self.steps = steps

    def forward(self, cue: Tensor) -> Tensor:
        """Recall a completed pattern from a partial cue.

        Parameters
        ----------
        cue
            Cue tensor of shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Recalled bipolar pattern.
        """
        state = torch.tanh(cue)
        for _ in range(self.steps):
            state = torch.tanh(state @ self.weights)
        return state


def build() -> nn.Module:
    """Build a small CA3 autoassociator.

    Returns
    -------
    nn.Module
        Configured ``CA3Autoassociator`` instance.
    """
    return CA3Autoassociator()


def example_input() -> Tensor:
    """Return a partial cue example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 100)``.
    """
    return torch.sign(torch.randn(1, 100))
