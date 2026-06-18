"""Synergetic Neural Network, 1988, Hermann Haken.

Paper: "Synergetic Computers and Cognition."
Prototype and adjoint vectors induce low-dimensional order parameters whose
macroscopic dynamics complete noisy patterns without Hopfield energy descent.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SynergeticNeuralNetwork(nn.Module):
    """Haken synergetic pattern-completion dynamics."""

    def __init__(self, n_features: int = 256, n_prototypes: int = 6, steps: int = 5) -> None:
        """Initialize prototype and adjoint vectors.

        Parameters
        ----------
        n_features:
            Pattern dimensionality.
        n_prototypes:
            Number of stored prototype patterns.
        steps:
            Number of order-parameter updates.
        """
        super().__init__()
        self.steps = steps
        self.lambda_gain = 0.75
        self.saturation = 0.18
        self.cubic = 0.08
        prototypes = torch.randn(n_prototypes, n_features)
        adjoints = torch.linalg.pinv(prototypes.T)
        self.register_buffer("prototypes", prototypes)
        self.register_buffer("adjoints", adjoints)

    def forward(self, cue: Tensor) -> Tensor:
        """Complete a cue through synergetic order-parameter dynamics.

        Parameters
        ----------
        cue:
            Input cue tensor with shape ``(batch, n_features)``.

        Returns
        -------
        Tensor
            Reconstructed pattern.
        """
        order = cue @ self.adjoints.T
        for _ in range(self.steps):
            radius = (order * order).sum(dim=-1, keepdim=True)
            delta = self.lambda_gain * order - self.saturation * radius * order
            delta = delta - self.cubic * order.pow(3)
            order = order + 0.25 * delta
        return torch.tanh(order @ self.prototypes)


def build() -> nn.Module:
    """Build a small synergetic neural network.

    Returns
    -------
    nn.Module
        Configured ``SynergeticNeuralNetwork`` instance.
    """
    return SynergeticNeuralNetwork()


def example_input() -> Tensor:
    """Create a noisy synergetic cue.

    Returns
    -------
    Tensor
        Example cue with shape ``(1, 256)``.
    """
    return torch.randn(1, 256) * 0.5


MENAGERIE_ENTRIES = [("Synergetic Neural Network (Haken)", "build", "example_input", "1988", "MB1")]
