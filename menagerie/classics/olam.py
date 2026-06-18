"""Optimal Linear Associative Memory (1977), Teuvo Kohonen.

Paper: "Associative memory: A system-theoretical approach."
The OLAM variant replaces simple correlation storage with a pseudoinverse-optimal
linear map, giving exact recall for nonorthogonal stored keys when possible.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class OLAM(nn.Module):
    """Pseudoinverse-optimal heteroassociative linear memory."""

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        """Compute and store the optimal linear association matrix.

        Parameters
        ----------
        keys
            Key matrix of shape ``(patterns, key_dim)``.
        values
            Value matrix of shape ``(patterns, value_dim)``.
        """
        super().__init__()
        weights = values.T @ torch.linalg.pinv(keys.T)
        self.register_buffer("weights", weights)

    def forward(self, q: Tensor) -> Tensor:
        """Recall values with the pseudoinverse association.

        Parameters
        ----------
        q
            Query tensor of shape ``(batch, key_dim)``.

        Returns
        -------
        Tensor
            Recalled value tensor.
        """
        return q @ self.weights.T


def build() -> nn.Module:
    """Build a small OLAM module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    keys = torch.tensor(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.1, 1.0, 0.2, 0.0],
            [0.0, 0.2, 1.0, 0.1],
        ]
    )
    values = torch.tensor([[1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    return OLAM(keys, values)


def example_input() -> Tensor:
    """Return an example nonorthogonal cue.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[1.0, 0.2, 0.0, 0.0]])
