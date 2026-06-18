"""Linear Associative Memory / CMM (1972), James Anderson and Teuvo Kohonen.

Papers: Anderson's linear associator and Kohonen's correlation matrix memory.
Stored key-value pairs are superposed by an outer-product memory matrix; recall
is a single linear projection from cue to associated value.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearAssociativeMemory(nn.Module):
    """Outer-product heteroassociative memory."""

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        """Build a Hebbian correlation matrix from key-value pairs.

        Parameters
        ----------
        keys
            Key matrix of shape ``(patterns, key_dim)``.
        values
            Value matrix of shape ``(patterns, value_dim)``.
        """
        super().__init__()
        memory = values.T @ keys
        self.register_buffer("memory", memory / keys.shape[0])

    def forward(self, q: Tensor) -> Tensor:
        """Recall values associated with query keys.

        Parameters
        ----------
        q
            Query tensor of shape ``(batch, key_dim)``.

        Returns
        -------
        Tensor
            Recalled value tensor.
        """
        return q @ self.memory.T


def build() -> nn.Module:
    """Build a small linear associative memory.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    keys = torch.eye(4)
    values = torch.tensor(
        [
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    return LinearAssociativeMemory(keys, values)


def example_input() -> Tensor:
    """Return an example query cue.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]])
