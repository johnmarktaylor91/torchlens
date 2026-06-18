"""Mark-I / cross-coupled Perceptron (1962), Frank Rosenblatt.

Paper: "Principles of Neurodynamics."
A fixed random sensory-to-association projection feeds threshold association
units, optional cross-coupled association recurrence, and an adaptive response layer.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MarkIPerceptron(nn.Module):
    """Rosenblatt-style S-A-R perceptron with fixed random association layer."""

    def __init__(self, n_sensory: int = 10, n_assoc: int = 12, n_response: int = 3) -> None:
        """Initialize fixed S-to-A projection and trainable A-to-R response layer.

        Parameters
        ----------
        n_sensory
            Number of sensory units.
        n_assoc
            Number of association units.
        n_response
            Number of response units.
        """
        super().__init__()
        sensory_assoc = torch.randint(-1, 2, (n_sensory, n_assoc), dtype=torch.float32)
        lateral = torch.randn(n_assoc, n_assoc) * 0.08
        lateral.fill_diagonal_(0.0)
        self.register_buffer("sensory_assoc", sensory_assoc)
        self.register_buffer("lateral", lateral)
        self.response = nn.Linear(n_assoc, n_response)

    def forward(self, x: Tensor) -> Tensor:
        """Compute thresholded association and response activations.

        Parameters
        ----------
        x
            Sensory input tensor of shape ``(batch, n_sensory)``.

        Returns
        -------
        Tensor
            Binary response activations.
        """
        association = (x @ self.sensory_assoc >= 0.0).to(x.dtype)
        association = (association + association @ self.lateral >= 0.5).to(x.dtype)
        response_sum = self.response(association)
        return (response_sum >= 0.0).to(x.dtype)


def build() -> nn.Module:
    """Build a small Mark-I perceptron module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return MarkIPerceptron()


def example_input() -> Tensor:
    """Return an example sensory pattern.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(2, 10)
