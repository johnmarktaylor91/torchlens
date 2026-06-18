"""Lee-Seung NMF parts-based network, 1999, Lee and Seung.

Paper: "Learning the parts of objects by non-negative matrix factorization." Positive
basis and coefficient parameters reconstruct a nonnegative input vector through
additive parts; multiplicative learning rules are outside the forward pass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [
    ("NMF Parts-Based Network (Lee-Seung)", "build", "example_input", "1999", "DA")
]


class NMFNet(nn.Module):
    """Nonnegative linear reconstruction module."""

    def __init__(self, n_input: int = 256, n_parts: int = 32) -> None:
        """Initialize unconstrained basis and coefficient parameters.

        Parameters
        ----------
        n_input
            Reconstructed vector dimensionality.
        n_parts
            Number of additive parts.
        """
        super().__init__()
        self.basis_raw = nn.Parameter(torch.randn(n_input, n_parts) * 0.05)
        self.coef_raw = nn.Parameter(torch.randn(n_parts) * 0.05)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct nonnegative inputs from parts.

        Parameters
        ----------
        x
            Nonnegative input tensor of shape ``(batch, n_input)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Reconstruction and per-example squared reconstruction error.
        """
        basis = torch.relu(self.basis_raw)
        coef = torch.relu(self.coef_raw).unsqueeze(0).expand(x.shape[0], -1)
        recon = coef @ basis.T
        error = (x - recon).pow(2).sum(dim=-1)
        return recon, error


def build() -> nn.Module:
    """Build a small NMF module.

    Returns
    -------
    nn.Module
        Configured ``NMFNet`` instance.
    """
    return NMFNet()


def example_input() -> Tensor:
    """Return a nonnegative vector example.

    Returns
    -------
    Tensor
        Example tensor with shape ``(1, 256)``.
    """
    return torch.rand(1, 256)
