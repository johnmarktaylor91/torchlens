"""Tensor Product Representation, 1990, Smolensky, "Tensor Product Variable Binding".

Paper: Smolensky 1990, "Tensor Product Variable Binding and the Representation of Symbolic Structures."
Fillers are bound to roles by outer products and superposed; unbinding contracts the
tensor with precomputed dual-role vectors from the role pseudoinverse.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TensorProductRepresentation(nn.Module):
    """Outer-product role/filler binding and dual-role unbinding."""

    def __init__(self, n_roles: int = 4, filler_dim: int = 5, role_dim: int = 4) -> None:
        """Initialize role vectors and their duals.

        Parameters
        ----------
        n_roles
            Number of role vectors.
        filler_dim
            Filler vector width.
        role_dim
            Role vector width.
        """
        super().__init__()
        roles = torch.randn(n_roles, role_dim)
        self.roles = nn.Parameter(roles)
        self.register_buffer("dual_roles", torch.linalg.pinv(roles).T)
        self.filler_dim = filler_dim

    def forward(self, fillers: Tensor) -> Tensor:
        """Bind fillers to roles and unbind them with dual roles.

        Parameters
        ----------
        fillers
            Filler tensor of shape ``(batch, slots, filler_dim)``.

        Returns
        -------
        Tensor
            Recovered fillers with shape ``(batch, slots, filler_dim)``.
        """
        role_ids = torch.arange(fillers.shape[1], dtype=torch.long, device=fillers.device)
        roles = self.roles[role_ids]
        bound = torch.einsum("bsf,sr->bsfr", fillers, roles)
        structure = bound.sum(dim=1)
        duals = self.dual_roles[role_ids]
        return torch.einsum("bfr,sr->bsf", structure, duals)


MENAGERIE_ENTRIES = [
    ("Tensor Product Representation (TPR)", "build", "example_input", "1990", "CE")
]


def build() -> nn.Module:
    """Build a small TPR module.

    Returns
    -------
    nn.Module
        Configured TPR module.
    """
    return TensorProductRepresentation()


def example_input() -> Tensor:
    """Create filler examples.

    Returns
    -------
    Tensor
        Fillers with shape ``(2, 3, 5)``.
    """
    return torch.randn(2, 3, 5)
