"""Quaternion neural network, 2018, Parcollet et al.

Paper: Parcollet 2018, "Quaternion Recurrent Neural Networks." A minimal
Hamilton-product linear layer shares parameters across real, i, j, and k
components, omitting convolutional and recurrent variants.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QuaternionLinear(nn.Module):
    """Linear layer whose matrix-vector product follows Hamilton algebra."""

    def __init__(self, in_quaternions: int = 4, out_quaternions: int = 3) -> None:
        """Initialize four real-valued component matrices.

        Parameters
        ----------
        in_quaternions
            Number of input quaternion features.
        out_quaternions
            Number of output quaternion features.
        """
        super().__init__()
        scale = 0.1
        self.r_weight = nn.Parameter(torch.randn(out_quaternions, in_quaternions) * scale)
        self.i_weight = nn.Parameter(torch.randn(out_quaternions, in_quaternions) * scale)
        self.j_weight = nn.Parameter(torch.randn(out_quaternions, in_quaternions) * scale)
        self.k_weight = nn.Parameter(torch.randn(out_quaternions, in_quaternions) * scale)
        self.bias = nn.Parameter(torch.zeros(out_quaternions * 4))

    def forward(self, x: Tensor) -> Tensor:
        """Apply a Hamilton-product linear transform.

        Parameters
        ----------
        x
            Real tensor with four quaternion components concatenated,
            shape ``(batch, 4 * in_quaternions)``.

        Returns
        -------
        Tensor
            Concatenated quaternion output tensor.
        """
        xr, xi, xj, xk = x.chunk(4, dim=-1)
        yr = (
            xr @ self.r_weight.T
            - xi @ self.i_weight.T
            - xj @ self.j_weight.T
            - xk @ self.k_weight.T
        )
        yi = (
            xr @ self.i_weight.T
            + xi @ self.r_weight.T
            + xj @ self.k_weight.T
            - xk @ self.j_weight.T
        )
        yj = (
            xr @ self.j_weight.T
            - xi @ self.k_weight.T
            + xj @ self.r_weight.T
            + xk @ self.i_weight.T
        )
        yk = (
            xr @ self.k_weight.T
            + xi @ self.j_weight.T
            - xj @ self.i_weight.T
            + xk @ self.r_weight.T
        )
        return torch.tanh(torch.cat((yr, yi, yj, yk), dim=-1) + self.bias)


class QuaternionNetwork(nn.Module):
    """Two-layer quaternion feed-forward network."""

    def __init__(self) -> None:
        """Initialize a small quaternion MLP."""
        super().__init__()
        self.hidden = QuaternionLinear(4, 5)
        self.output = QuaternionLinear(5, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Compute quaternion-valued features.

        Parameters
        ----------
        x
            Concatenated quaternion tensor with shape ``(batch, 16)``.

        Returns
        -------
        Tensor
            Concatenated quaternion output tensor with shape ``(batch, 12)``.
        """
        return self.output(self.hidden(x))


def build() -> nn.Module:
    """Build a small quaternion neural network.

    Returns
    -------
    nn.Module
        Configured ``QuaternionNetwork`` instance.
    """
    return QuaternionNetwork()


def example_input() -> Tensor:
    """Create a quaternion feature example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)


MENAGERIE_ENTRIES = [("Quaternion neural network", "build", "example_input", "2018", "CH-D")]
