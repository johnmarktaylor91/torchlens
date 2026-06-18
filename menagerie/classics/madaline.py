"""MADALINE MRI/MRII (1962), Bernard Widrow and Marcian Hoff.

Paper: "Associative storage and retrieval of digital information in networks of adaptive neurons."
Several sign-thresholded ADALINE units feed a fixed majority/logic combiner,
capturing the pre-backprop many-ADALINE classifier.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MADALINE(nn.Module):
    """Many-ADALINE network with a fixed majority-vote combiner."""

    def __init__(self, n_in: int = 8, n_adalines: int = 5) -> None:
        """Initialize the ADALINE bank.

        Parameters
        ----------
        n_in
            Number of input features.
        n_adalines
            Number of thresholded ADALINE units.
        """
        super().__init__()
        self.adalines = nn.Linear(n_in, n_adalines)
        self.register_buffer("combiner", torch.ones(n_adalines, 1))
        self.register_buffer("majority_threshold", torch.tensor(float(n_adalines // 2 + 1)))

    def forward(self, x: Tensor) -> Tensor:
        """Classify by thresholded ADALINE majority vote.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        Tensor
            Binary output in ``{0, 1}``.
        """
        sums = self.adalines(x)
        votes = (sums >= 0.0).to(x.dtype)
        vote_count = votes @ self.combiner
        return (vote_count >= self.majority_threshold).to(x.dtype)


def build() -> nn.Module:
    """Build a small MADALINE module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    return MADALINE()


def example_input() -> Tensor:
    """Return an example input pattern.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(2, 8)
