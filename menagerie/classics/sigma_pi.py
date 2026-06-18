"""Sigma-Pi higher-order unit, 1986, Rumelhart, Hinton, and Williams.

Paper: "Learning Internal Representations by Error Propagation." Sigma-Pi
units sum weighted products of selected input subsets, representing explicit
multiplicative conjunctions rather than only affine sums.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn


class SigmaPiNetwork(nn.Module):
    """Weighted sum of multiplicative input monomials."""

    def __init__(self, n_in: int = 5, n_out: int = 3, order: int = 2) -> None:
        """Initialize a Sigma-Pi network.

        Parameters
        ----------
        n_in:
            Number of input features.
        n_out:
            Number of output units.
        order:
            Multiplicative subset order.
        """
        super().__init__()
        subsets = list(itertools.combinations(range(n_in), order))
        self.register_buffer("subsets", torch.tensor(subsets, dtype=torch.long))
        self.linear = nn.Linear(len(subsets), n_out)

    def monomials(self, x: Tensor) -> Tensor:
        """Compute selected multiplicative monomial features.

        Parameters
        ----------
        x:
            Input tensor with shape ``(B, n_in)``.

        Returns
        -------
        Tensor
            Product features with shape ``(B, n_monomials)``.
        """
        gathered = x[:, self.subsets]
        return gathered.prod(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigma-Pi monomial expansion and linear summation.

        Parameters
        ----------
        x:
            Input tensor with shape ``(B, n_in)``.

        Returns
        -------
        Tensor
            Output activations.
        """
        return self.linear(self.monomials(x))


def build() -> nn.Module:
    """Build a small random-init Sigma-Pi network.

    Returns
    -------
    nn.Module
        A traceable ``SigmaPiNetwork`` instance.
    """
    return SigmaPiNetwork()


def example_input() -> Tensor:
    """Return a continuous input batch.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 5)``.
    """
    return torch.tensor([[0.2, -0.5, 0.7, 1.0, -0.3], [0.8, 0.1, -0.4, 0.6, 0.9]])
