"""MPS/TTN tensor-network classifier, 2016, Stoudenmire and Schwab.

Paper: Stoudenmire 2016, "Supervised learning with tensor networks."
Inputs are lifted to local trigonometric features and contracted through a small
matrix-product-state tensor train; tree tensor variants and DMRG training are omitted.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class MPSTensorNetworkClassifier(nn.Module):
    """Traceable matrix-product-state classifier over compressed image features."""

    def __init__(self, n_sites: int = 16, bond_dim: int = 4, n_classes: int = 10) -> None:
        """Initialize tensor-train cores.

        Parameters
        ----------
        n_sites
            Number of pooled feature sites.
        bond_dim
            Internal MPS bond dimension.
        n_classes
            Number of output classes.
        """
        super().__init__()
        self.n_sites = n_sites
        self.left = nn.Parameter(torch.randn(2, bond_dim) * 0.15)
        self.cores = nn.Parameter(torch.randn(n_sites - 2, bond_dim, 2, bond_dim) * 0.15)
        self.right = nn.Parameter(torch.randn(bond_dim, 2, n_classes) * 0.15)

    def forward(self, x: Tensor) -> Tensor:
        """Classify flattened inputs by contracting local feature maps with an MPS.

        Parameters
        ----------
        x
            Flattened input tensor with shape ``(batch, 784)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        pooled = x.reshape(x.shape[0], self.n_sites, -1).mean(dim=-1).sigmoid()
        phi = torch.stack(
            (torch.cos(math.pi * 0.5 * pooled), torch.sin(math.pi * 0.5 * pooled)), dim=-1
        )
        state = torch.einsum("bp,pd->bd", phi[:, 0], self.left)
        for site in range(self.n_sites - 2):
            state = torch.einsum("bd,dpq,bp->bq", state, self.cores[site], phi[:, site + 1])
        return torch.einsum("bd,dp c,bp->bc", state, self.right, phi[:, -1])


MENAGERIE_ENTRIES = [("MPS/TTN Tensor-Network Classifier", "build", "example_input", "2016", "DA")]


def build() -> nn.Module:
    """Build a compact MPS classifier.

    Returns
    -------
    nn.Module
        Configured tensor-network classifier.
    """
    return MPSTensorNetworkClassifier()


def example_input() -> Tensor:
    """Create a flattened image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 784)``.
    """
    return torch.rand(1, 784)
