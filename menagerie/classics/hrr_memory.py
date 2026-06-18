"""HRR holographic memory, 1995, Plate, "Holographic Reduced Representations".

Paper: Plate 1995, "Holographic Reduced Representations."
Role and filler vectors are bound by circular convolution, bundled by superposition,
unbound by circular correlation, and softly cleaned up against a random codebook.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class HRRMemory(nn.Module):
    """Plate-style holographic memory with circular convolution binding."""

    def __init__(self, dim: int = 16, n_items: int = 12) -> None:
        """Initialize cleanup codebook.

        Parameters
        ----------
        dim
            Hypervector dimensionality.
        n_items
            Number of cleanup memory vectors.
        """
        super().__init__()
        codebook = torch.randn(n_items, dim)
        self.register_buffer("codebook", torch.nn.functional.normalize(codebook, dim=-1))

    def _bind(self, left: Tensor, right: Tensor) -> Tensor:
        """Bind two HRR vectors by circular convolution.

        Parameters
        ----------
        left
            Left hypervector tensor.
        right
            Right hypervector tensor.

        Returns
        -------
        Tensor
            Circular convolution result.
        """
        return torch.fft.ifft(
            torch.fft.fft(left, dim=-1) * torch.fft.fft(right, dim=-1), dim=-1
        ).real

    def _unbind(self, bound: Tensor, role: Tensor) -> Tensor:
        """Unbind an HRR vector by circular correlation.

        Parameters
        ----------
        bound
            Bound memory tensor.
        role
            Role hypervector tensor.

        Returns
        -------
        Tensor
            Approximate filler vector.
        """
        return torch.fft.ifft(
            torch.fft.fft(bound, dim=-1) * torch.conj(torch.fft.fft(role, dim=-1)), dim=-1
        ).real

    def forward(self, pairs: Tensor) -> Tensor:
        """Store role/filler pairs and retrieve the first filler.

        Parameters
        ----------
        pairs
            Tensor of shape ``(batch, pairs, 2, dim)`` containing role and filler vectors.

        Returns
        -------
        Tensor
            Soft cleanup retrieval for the first role.
        """
        roles = pairs[:, :, 0]
        fillers = pairs[:, :, 1]
        memory = self._bind(roles, fillers).sum(dim=1)
        retrieved = self._unbind(memory, roles[:, 0])
        normed = torch.nn.functional.normalize(retrieved, dim=-1)
        weights = torch.softmax(normed @ self.codebook.T * 4.0, dim=-1)
        return weights @ self.codebook


MENAGERIE_ENTRIES = [
    ("HRR holographic memory system (Plate)", "build", "example_input", "1995", "CE")
]


def build() -> nn.Module:
    """Build a small HRR memory.

    Returns
    -------
    nn.Module
        Configured HRR module.
    """
    return HRRMemory()


def example_input() -> Tensor:
    """Create role/filler pair examples.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 3, 2, 16)``.
    """
    return torch.randn(2, 3, 2, 16)
