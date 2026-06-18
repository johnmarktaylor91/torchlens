"""Semantic Pointer Architecture, 2012.

Eliasmith's SPA builds cognitive representations from high-dimensional semantic
pointers using vector-symbolic binding, cleanup memory, and NEF-style population codes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SemanticPointerArchitecture(nn.Module):
    """Vector-symbolic binding and cleanup memory module."""

    def __init__(self, d: int = 16, n_symbols: int = 6) -> None:
        """Initialize semantic pointers and a linear cleanup memory.

        Parameters
        ----------
        d
            Pointer dimensionality.
        n_symbols
            Number of stored cleanup vectors.
        """
        super().__init__()
        pointers = torch.randn(n_symbols, d)
        pointers = pointers / pointers.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("pointers", pointers)
        self.population = nn.Linear(d, d)

    def _bind_with_first_pointer(self, x: Tensor) -> Tensor:
        """Bind inputs with the first pointer by circular convolution.

        Parameters
        ----------
        x
            Input semantic pointers.

        Returns
        -------
        Tensor
            Bound pointers.
        """
        parts: list[Tensor] = []
        key = self.pointers[0]
        for shift in range(x.shape[-1]):
            parts.append(x[:, shift].unsqueeze(-1) * torch.roll(key, shifts=shift).unsqueeze(0))
        return torch.stack(parts, dim=0).sum(dim=0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Bind input pointers and clean them up against memory.

        Parameters
        ----------
        x
            Semantic pointer tensor of shape ``(batch, d)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Bound pointer, cleanup similarity, and cleaned vector.
        """
        encoded = torch.tanh(self.population(x))
        bound = self._bind_with_first_pointer(encoded)
        similarity = bound @ self.pointers.t()
        weights = torch.softmax(8.0 * similarity, dim=-1)
        cleaned = weights @ self.pointers
        return bound, similarity, cleaned


def build() -> nn.Module:
    """Build a small Semantic Pointer Architecture module.

    Returns
    -------
    nn.Module
        Random SPA module.
    """
    return SemanticPointerArchitecture()


def example_input() -> Tensor:
    """Return a float32 semantic pointer.

    Returns
    -------
    Tensor
        Input of shape ``(2, 16)``.
    """
    x = torch.randn(2, 16, dtype=torch.float32)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
