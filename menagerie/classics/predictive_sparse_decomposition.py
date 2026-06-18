"""Predictive Sparse Decomposition, 2008, Kavukcuoglu, Ranzato, and LeCun.

Paper: Fast Inference in Sparse Coding Algorithms with Applications to Object Recognition.
Joint model of an overcomplete dictionary and a feed-forward predictor for
sparse codes; the forward pass emits predicted codes and reconstructions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PredictiveSparseDecomposition(nn.Module):
    """Sparse-code predictor plus linear dictionary decoder."""

    def __init__(self, n_input: int = 16, n_code: int = 24, sparsity: float = 0.1) -> None:
        """Initialize PSD parameters.

        Parameters
        ----------
        n_input:
            Input dimensionality.
        n_code:
            Sparse code dimensionality.
        sparsity:
            Soft-threshold amount for shrinkage.
        """
        super().__init__()
        self.encoder = nn.Linear(n_input, n_code)
        self.dictionary = nn.Parameter(torch.randn(n_code, n_input) * 0.08)
        self.sparsity = sparsity

    def shrink(self, activations: Tensor) -> Tensor:
        """Apply soft-threshold shrinkage to predicted codes.

        Parameters
        ----------
        activations:
            Dense encoder activations.

        Returns
        -------
        Tensor
            Sparse predicted codes.
        """
        return torch.sign(activations) * F.relu(torch.abs(activations) - self.sparsity)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict sparse codes and reconstruct inputs.

        Parameters
        ----------
        inputs:
            Input batch of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Sparse codes, reconstruction, and per-example PSD objective terms.
        """
        code = self.shrink(self.encoder(inputs))
        normalized_dictionary = F.normalize(self.dictionary, dim=-1)
        reconstruction = code @ normalized_dictionary
        reconstruction_error = (inputs - reconstruction).pow(2).sum(dim=-1)
        sparse_penalty = code.abs().sum(dim=-1)
        objective = reconstruction_error + self.sparsity * sparse_penalty
        return code, reconstruction, objective


def build() -> nn.Module:
    """Build a small PSD module.

    Returns
    -------
    nn.Module
        PredictiveSparseDecomposition instance.
    """
    return PredictiveSparseDecomposition()


def example_input() -> Tensor:
    """Return a sample input batch.

    Returns
    -------
    Tensor
        Float tensor of shape ``(2, 16)``.
    """
    return torch.randn(2, 16)
