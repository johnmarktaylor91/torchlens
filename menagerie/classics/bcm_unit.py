"""BCM synaptic plasticity unit, 1982, Bienenstock, Cooper, and Munro.

Paper: "Theory for the development of neuron selectivity: orientation specificity
and binocular interaction in visual cortex." The forward pass is a linear
postsynaptic response; BCM's sliding-threshold update is provided as a helper.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("BCM Synaptic Plasticity Unit", "build", "example_input", "1982", "DA")]


class BCMUnit(nn.Module):
    """Linear postsynaptic unit with BCM update substrate."""

    def __init__(self, n_input: int = 64, n_output: int = 8, eta: float = 0.01) -> None:
        """Initialize synaptic weights and sliding threshold.

        Parameters
        ----------
        n_input
            Presynaptic feature count.
        n_output
            Postsynaptic unit count.
        eta
            Learning-rate scale for the helper update.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_output, n_input) * 0.05)
        self.register_buffer("theta", torch.full((n_output,), 0.1))
        self.eta = eta

    def bcm_delta(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the BCM weight-change tensor without mutating parameters.

        Parameters
        ----------
        x
            Presynaptic inputs of shape ``(batch, n_input)``.
        y
            Postsynaptic responses of shape ``(batch, n_output)``.

        Returns
        -------
        Tensor
            Average BCM update with shape ``(n_output, n_input)``.
        """
        plasticity = y * (y - self.theta)
        return self.eta * plasticity.T @ x / x.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        """Compute postsynaptic responses.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        Tensor
            Linear postsynaptic activity.
        """
        return x @ self.weight.T


def build() -> nn.Module:
    """Build a small BCM unit module.

    Returns
    -------
    nn.Module
        Configured ``BCMUnit`` instance.
    """
    return BCMUnit()


def example_input() -> Tensor:
    """Return a presynaptic vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)
