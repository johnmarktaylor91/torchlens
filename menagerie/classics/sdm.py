"""Sparse Distributed Memory, 1988, Kanerva, "Sparse Distributed Memory".

High-dimensional addresses activate many random hard locations by similarity; reads
sum their counters and threshold the result as content-addressable recall.
"""

import torch
from torch import Tensor, nn


class SparseDistributedMemory(nn.Module):
    """Kanerva-style SDM with differentiable soft location activation."""

    def __init__(self, dim: int = 16, n_locations: int = 32, tau: float = 2.0) -> None:
        """Initialize random hard locations and counter contents.

        Parameters
        ----------
        dim:
            Address and data dimensionality.
        n_locations:
            Number of hard locations.
        tau:
            Softmax temperature for traceable soft addressing.
        """
        super().__init__()
        addresses = torch.sign(torch.randn(n_locations, dim))
        counters = torch.randn(n_locations, dim) * 0.1
        self.register_buffer("addresses", addresses)
        self.register_buffer("counters", counters)
        self.tau = tau

    def forward(self, address: Tensor) -> Tensor:
        """Read memory at a batch of addresses with soft hard-location activation.

        Parameters
        ----------
        address:
            Address tensor of shape ``(batch, dim)`` with bipolar values preferred.

        Returns
        -------
        Tensor
            Recalled bipolar-like data tensor.
        """
        scores = address @ self.addresses.T
        weights = torch.softmax(scores / self.tau, dim=-1)
        summed = weights @ self.counters
        return torch.tanh(summed)


def build() -> nn.Module:
    """Build a small sparse distributed memory.

    Returns
    -------
    nn.Module
        Configured ``SparseDistributedMemory`` instance.
    """
    return SparseDistributedMemory()


def example_input() -> Tensor:
    """Create a bipolar address example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 16)``.
    """
    return torch.sign(torch.randn(2, 16))
