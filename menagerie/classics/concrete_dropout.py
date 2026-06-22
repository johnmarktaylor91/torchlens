"""Concrete Dropout: learned dropout probability via concrete relaxation.

Gal, Hron & Kendall, NeurIPS 2017.  Concrete Dropout learns dropout probabilities
by sampling a continuous Concrete mask and applying the usual inverted-dropout
rescaling, enabling differentiable uncertainty calibration.  This compact module
wraps two linear layers with learned logit dropout probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcreteDropoutLayer(nn.Module):
    """Linear layer with learned Concrete dropout mask."""

    def __init__(self, in_features: int, out_features: int, temperature: float = 0.1) -> None:
        """Initialize a Concrete Dropout layer.

        Parameters
        ----------
        in_features:
            Input feature count.
        out_features:
            Output feature count.
        temperature:
            Concrete relaxation temperature.
        """

        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.p_logit = nn.Parameter(torch.tensor(-2.0))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned Concrete dropout and the wrapped linear projection.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Projected features.
        """

        p = torch.sigmoid(self.p_logit)
        u = torch.full_like(x, 0.37)
        drop = torch.sigmoid(
            (torch.log(p + 1e-7) - torch.log1p(-p + 1e-7) + torch.log(u) - torch.log1p(-u))
            / self.temperature
        )
        retained = 1.0 - drop
        return self.linear(x * retained / (1.0 - p).clamp_min(1e-4))


class ConcreteDropoutNet(nn.Module):
    """Compact MLP using Concrete Dropout wrappers."""

    def __init__(self) -> None:
        """Initialize the compact network."""

        super().__init__()
        self.cd1 = ConcreteDropoutLayer(10, 32)
        self.cd2 = ConcreteDropoutLayer(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the Concrete Dropout MLP.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.cd2(torch.relu(self.cd1(x)))


def build() -> nn.Module:
    """Build a compact Concrete Dropout network.

    Returns
    -------
    nn.Module
        Random-init Concrete Dropout MLP.
    """

    return ConcreteDropoutNet()


def example_input() -> torch.Tensor:
    """Create tabular input.

    Returns
    -------
    torch.Tensor
        Input tensor.
    """

    return torch.randn(2, 10)


MENAGERIE_ENTRIES = [("Concrete Dropout", "build", "example_input", "2017", "BAYES")]
