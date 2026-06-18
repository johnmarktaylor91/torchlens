"""SINO starter-iterator neural operator, 2026, Qin et al.

Paper: 2026, "Starter-Iterator Neural Operator."
A starter network initializes a solution field and a shared iterator refines it; this
minimal convolutional operator omits PDE-specific inverse-problem training.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SINO(nn.Module):
    """Convolutional starter plus shared iterative refinement operator."""

    def __init__(self, channels: int = 16, iterations: int = 4) -> None:
        """Initialize starter, iterator, and inverse heads.

        Parameters
        ----------
        channels
            Hidden field channel count.
        iterations
            Number of iterator applications.
        """
        super().__init__()
        self.iterations = iterations
        self.starter = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.iterator = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.forward_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.inverse_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, field: Tensor) -> Tensor:
        """Refine a field with a shared neural-operator iterator.

        Parameters
        ----------
        field
            Input field with shape ``(batch, 1, 64, 64)``.

        Returns
        -------
        Tensor
            Concatenated forward and inverse predictions.
        """
        state = torch.tanh(self.starter(field))
        for _ in range(self.iterations):
            state = state + 0.25 * self.iterator(state)
        return torch.cat((self.forward_head(state), self.inverse_head(state)), dim=1)


MENAGERIE_ENTRIES = [
    ("SINO (Starter-Iterator Neural Operator)", "build", "example_input", "2026", "DA")
]


def build() -> nn.Module:
    """Build a SINO module.

    Returns
    -------
    nn.Module
        Configured SINO module.
    """
    return SINO()


def example_input() -> Tensor:
    """Create a scalar field example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 64, 64)``.
    """
    return torch.randn(1, 1, 64, 64)
