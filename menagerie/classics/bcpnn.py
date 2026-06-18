"""BCPNN, 1989, Lansner and Ekeberg.

Paper: "A one-layer feedback artificial neural network with a Bayesian learning rule."
Activations are interpreted as probabilities, biases encode priors, and weights
encode pointwise mutual information for Bayesian-Hebbian propagation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BCPNN(nn.Module):
    """Bayesian confidence propagation with fixed PMI weights."""

    def __init__(self, n_units: int = 32) -> None:
        """Initialize prior and joint probabilities.

        Parameters
        ----------
        n_units:
            Number of probabilistic units.
        """
        super().__init__()
        probs = torch.rand(n_units) * 0.5 + 0.25
        joint = torch.outer(probs, probs) + torch.rand(n_units, n_units) * 0.015
        joint = joint.clamp_min(1.0e-4)
        bias = torch.log(probs.clamp_min(1.0e-4))
        weights = torch.log(joint / torch.outer(probs, probs).clamp_min(1.0e-4))
        self.register_buffer("bias", bias)
        self.register_buffer("weights", weights)

    def forward(self, x: Tensor) -> Tensor:
        """Propagate probabilistic activity through PMI weights.

        Parameters
        ----------
        x:
            Input evidence tensor with shape ``(batch, n_units)``.

        Returns
        -------
        Tensor
            Posterior-like activation probabilities.
        """
        activity = torch.sigmoid(x)
        logits = self.bias + activity @ self.weights.T
        return torch.sigmoid(logits)


def build() -> nn.Module:
    """Build a small BCPNN module.

    Returns
    -------
    nn.Module
        Configured ``BCPNN`` instance.
    """
    return BCPNN()


def example_input() -> Tensor:
    """Create probabilistic BCPNN input evidence.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 32)``.
    """
    return torch.randn(1, 32) * 0.5


MENAGERIE_ENTRIES = [
    ("BCPNN (Bayesian Confidence Propagation NN)", "build", "example_input", "1989", "MB1")
]
