"""Binary CMM / Steinbuch Lernmatrix / Willshaw net (1961/1969).

Binary associations are stored by clipped Hebbian outer products. Recall sums
active key rows and thresholds them, following Steinbuch and Willshaw memory
rules in a traceable forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BinaryCMM(nn.Module):
    """Clipped-Hebbian binary associative memory."""

    def __init__(self, keys: Tensor, values: Tensor) -> None:
        """Initialize the binary clipped association matrix.

        Parameters
        ----------
        keys
            Binary key matrix of shape ``(patterns, key_dim)``.
        values
            Binary value matrix of shape ``(patterns, value_dim)``.
        """
        super().__init__()
        memory = torch.clamp(keys.T @ values, max=1.0)
        self.register_buffer("memory", memory)

    def forward(self, q: Tensor) -> Tensor:
        """Recall a binary value by thresholded memory lookup.

        Parameters
        ----------
        q
            Binary query tensor of shape ``(batch, key_dim)``.

        Returns
        -------
        Tensor
            Binary recalled value tensor.
        """
        scores = q @ self.memory
        threshold = torch.clamp(q.sum(dim=-1, keepdim=True), min=1.0)
        return (scores >= threshold).to(q.dtype)


def build() -> nn.Module:
    """Build a small binary CMM module.

    Returns
    -------
    nn.Module
        Initialized module.
    """
    keys = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    values = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    return BinaryCMM(keys, values)


def example_input() -> Tensor:
    """Return an example binary cue.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.tensor([[1.0, 1.0, 0.0, 0.0]])
