"""Interactive Activation and Competition, 1981, as localist settling dynamics.

Paper: McClelland and Rumelhart 1981, "An Interactive Activation Model of Context Effects in Letter Perception."
Localist pools excite compatible units across pools and inhibit competitors
within a pool under bounded Grossberg-style activation updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("Interactive Activation and Competition (IAC)", "build", "example_input", "1981", "CB")
]


class IACNet(nn.Module):
    """Interactive activation and competition network."""

    def __init__(self, pool_sizes: tuple[int, ...] = (4, 4, 4), steps: int = 6) -> None:
        """Initialize excitatory between-pool and inhibitory within-pool weights.

        Parameters
        ----------
        pool_sizes
            Unit counts for localist pools.
        steps
            Number of settling steps.
        """
        super().__init__()
        n_units = sum(pool_sizes)
        weights = torch.zeros(n_units, n_units)
        start = 0
        for pool_size in pool_sizes:
            end = start + pool_size
            weights[start:end, start:end] = -0.18
            weights[start:end, start:end].fill_diagonal_(0.0)
            start = end
        weights = weights + torch.randn(n_units, n_units).abs() * 0.08
        weights.fill_diagonal_(0.0)
        self.register_buffer("weights", weights)
        self.register_buffer("bias", torch.zeros(n_units))
        self.steps = steps
        self.min_act = -0.2
        self.max_act = 1.0
        self.rest = 0.0
        self.decay = 0.12

    def forward(self, external: Tensor) -> Tensor:
        """Settle activations under external clamp input.

        Parameters
        ----------
        external
            External clamp vector with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Settled bounded activations.
        """
        activation = external.clamp(self.min_act, self.max_act)
        for _ in range(self.steps):
            net = activation @ self.weights + self.bias + external
            positive = (self.max_act - activation) * net
            negative = (activation - self.min_act) * net
            delta = torch.where(net > 0.0, positive, negative)
            activation = activation + 0.1 * (delta - self.decay * (activation - self.rest))
            activation = activation.clamp(self.min_act, self.max_act)
        return activation


def build() -> nn.Module:
    """Build a small IAC network.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return IACNet()


def example_input() -> Tensor:
    """Create an external clamp example.

    Returns
    -------
    Tensor
        Example clamp with shape ``(1, 12)``.
    """
    x = torch.zeros(1, 12)
    x[:, [0, 5, 10]] = 0.8
    return x
