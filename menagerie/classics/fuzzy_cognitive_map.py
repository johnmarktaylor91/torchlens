"""Fuzzy Cognitive Map, 1986, Bart Kosko.

Paper: Kosko 1986, "Fuzzy cognitive maps." Signed causal concept weights are
iterated through a squashing nonlinearity to settle concept activations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("Fuzzy Cognitive Map", "build", "example_input", "1986", "CF")]


class FuzzyCognitiveMap(nn.Module):
    """Recurrent fuzzy concept graph with zero self-causality."""

    def __init__(self, n_concepts: int = 6, steps: int = 5) -> None:
        """Initialize signed causal weights.

        Parameters
        ----------
        n_concepts
            Number of concept nodes.
        steps
            Number of recurrent settling steps.
        """
        super().__init__()
        self.steps = steps
        weights = torch.randn(n_concepts, n_concepts) * 0.5
        weights = weights * (1.0 - torch.eye(n_concepts))
        self.weights = nn.Parameter(weights)

    def forward(self, c0: Tensor) -> Tensor:
        """Iterate fuzzy concept activations.

        Parameters
        ----------
        c0
            Initial concept activations of shape ``(batch, n_concepts)``.

        Returns
        -------
        Tensor
            Activation trajectory including the initial state.
        """
        c = c0
        trajectory = [c]
        mask = 1.0 - torch.eye(self.weights.shape[0], device=c0.device, dtype=c0.dtype)
        for _ in range(self.steps):
            c = torch.sigmoid(c @ (self.weights * mask))
            trajectory.append(c)
        return torch.stack(trajectory, dim=1)


def build() -> nn.Module:
    """Build a small fuzzy cognitive map.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return FuzzyCognitiveMap()


def example_input() -> Tensor:
    """Return initial concept activations.

    Returns
    -------
    Tensor
        Example tensor of shape ``(2, 6)``.
    """
    return torch.rand(2, 6)
