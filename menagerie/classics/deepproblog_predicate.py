"""DeepProbLog neural predicate, 2018, Manhaeve et al., "DeepProbLog".

Paper: Manhaeve 2018, "DeepProbLog: Neural Probabilistic Logic Programming."
This simplified module maps image-like inputs to digit probabilities and performs a
tiny differentiable weighted model count for an even-digit query.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NeuralPredicate(nn.Module):
    """MLP neural predicate with differentiable proof marginalization."""

    def __init__(self, n_digits: int = 10) -> None:
        """Initialize image predicate network.

        Parameters
        ----------
        n_digits
            Number of digit classes.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 32), nn.ReLU(), nn.Linear(32, n_digits)
        )
        proof_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.register_buffer("even_digit", proof_mask)

    def forward(self, image: Tensor) -> Tensor:
        """Compute neural predicate probabilities and an even-digit proof score.

        Parameters
        ----------
        image
            Image-like tensor of shape ``(batch, 1, 28, 28)``.

        Returns
        -------
        Tensor
            Digit probabilities plus even-query marginal.
        """
        probs = torch.softmax(self.net(image), dim=-1)
        even_score = torch.sum(probs * self.even_digit, dim=-1, keepdim=True)
        return torch.cat((probs, even_score), dim=-1)


MENAGERIE_ENTRIES = [("DeepProbLog neural predicate", "build", "example_input", "2018", "CD")]


def build() -> nn.Module:
    """Build a simplified DeepProbLog neural predicate.

    Returns
    -------
    nn.Module
        Configured predicate module.
    """
    return NeuralPredicate()


def example_input() -> Tensor:
    """Create image-like predicate inputs.

    Returns
    -------
    Tensor
        Example image batch with shape ``(2, 1, 28, 28)``.
    """
    return torch.randn(2, 1, 28, 28)
