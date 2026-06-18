"""Generalized Regression Neural Network, 1991, Specht, "A General Regression Neural Network".

The network performs Nadaraya-Watson regression: Gaussian exemplar weights average
stored target values to produce a smooth regression estimate.
"""

import torch
from torch import Tensor, nn


class GRNN(nn.Module):
    """Gaussian-kernel exemplar regression network."""

    def __init__(
        self, dim: int = 4, target_dim: int = 2, n_examples: int = 8, sigma: float = 0.8
    ) -> None:
        """Initialize fixed exemplar inputs and targets.

        Parameters
        ----------
        dim:
            Input feature dimension.
        target_dim:
            Regression target dimension.
        n_examples:
            Number of stored exemplars.
        sigma:
            Gaussian kernel width.
        """
        super().__init__()
        self.register_buffer("prototypes", torch.randn(n_examples, dim))
        self.register_buffer("targets", torch.randn(n_examples, target_dim))
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        """Return kernel-regression predictions.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Regression output of shape ``(batch, target_dim)``.
        """
        d2 = torch.cdist(x, self.prototypes).pow(2)
        weights = torch.softmax(-d2 / (2.0 * self.sigma * self.sigma), dim=-1)
        return weights @ self.targets


def build() -> nn.Module:
    """Build a small GRNN regressor.

    Returns
    -------
    nn.Module
        Configured ``GRNN`` instance.
    """
    return GRNN()


def example_input() -> Tensor:
    """Create a feature-vector example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4)``.
    """
    return torch.randn(2, 4)
