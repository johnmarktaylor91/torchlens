"""Probabilistic Neural Network, 1990, Specht, "Probabilistic Neural Networks".

One Gaussian Parzen kernel is placed at each exemplar; class posteriors are sums
of exemplar kernels grouped by class labels.
"""

import torch
from torch import Tensor, nn


class SpechtPNN(nn.Module):
    """Parzen-window probabilistic neural network classifier."""

    def __init__(
        self, dim: int = 4, n_classes: int = 3, n_examples: int = 9, sigma: float = 0.8
    ) -> None:
        """Initialize fixed exemplar prototypes and labels.

        Parameters
        ----------
        dim:
            Feature dimension.
        n_classes:
            Number of classes.
        n_examples:
            Number of exemplar kernels.
        sigma:
            Gaussian kernel width.
        """
        super().__init__()
        labels = torch.arange(n_examples) % n_classes
        self.register_buffer("prototypes", torch.randn(n_examples, dim))
        self.register_buffer("labels", torch.nn.functional.one_hot(labels, n_classes).float())
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        """Return normalized class posterior estimates.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        Tensor
            Posterior-like class probabilities.
        """
        d2 = torch.cdist(x, self.prototypes).pow(2)
        kernels = torch.exp(-d2 / (2.0 * self.sigma * self.sigma))
        scores = kernels @ self.labels
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)


def build() -> nn.Module:
    """Build a small Specht PNN classifier.

    Returns
    -------
    nn.Module
        Configured ``SpechtPNN`` instance.
    """
    return SpechtPNN()


def example_input() -> Tensor:
    """Create a feature-vector example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 4)``.
    """
    return torch.randn(2, 4)
