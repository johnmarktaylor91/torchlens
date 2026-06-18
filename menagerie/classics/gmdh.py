"""GMDH polynomial network (1971), Alexey Ivakhnenko.

Paper: "Polynomial theory of complex systems."
Layers of Kolmogorov-Gabor quadratic partial-polynomial units evaluate selected
input pairs; model selection is frozen here while coefficients remain random.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GMDHLayer(nn.Module):
    """Layer of quadratic partial-polynomial GMDH units."""

    def __init__(self, selected_pairs: Tensor) -> None:
        """Initialize selected input pairs and polynomial coefficients.

        Parameters
        ----------
        selected_pairs
            Long tensor of shape ``(n_units, 2)`` with input feature indices.
        """
        super().__init__()
        self.register_buffer("selected_pairs", selected_pairs.long())
        self.coefficients = nn.Parameter(torch.randn(selected_pairs.shape[0], 6) * 0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate quadratic pairwise units.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, features)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(batch, n_units)``.
        """
        xi = x[:, self.selected_pairs[:, 0]]
        xj = x[:, self.selected_pairs[:, 1]]
        ones = torch.ones_like(xi)
        features = torch.stack((ones, xi, xj, xi * xi, xj * xj, xi * xj), dim=-1)
        return (features * self.coefficients.unsqueeze(0)).sum(dim=-1)


class GMDH(nn.Module):
    """Stacked GMDH partial-polynomial network."""

    def __init__(self) -> None:
        """Initialize a compact two-layer GMDH network."""
        super().__init__()
        self.layer1 = GMDHLayer(torch.tensor([[0, 1], [0, 2], [1, 3], [2, 3]]))
        self.layer2 = GMDHLayer(torch.tensor([[0, 1], [1, 2], [2, 3]]))
        self.readout = nn.Linear(3, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Run frozen pair selection with learned quadratic coefficients.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, 4)``.

        Returns
        -------
        Tensor
            Regression output tensor.
        """
        hidden = torch.tanh(self.layer1(x))
        partials = torch.tanh(self.layer2(hidden))
        return self.readout(partials)


def build() -> nn.Module:
    """Build a small GMDH module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return GMDH()


def example_input() -> Tensor:
    """Return an example tabular input.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(2, 4)
