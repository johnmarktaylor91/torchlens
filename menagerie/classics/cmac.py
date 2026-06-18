"""CMAC cerebellar controller, 1975, James S. Albus.

Paper: "A New Approach to Manipulator Control: The Cerebellar Model
Articulation Controller." CMAC tile-codes continuous inputs into overlapping
associative memory addresses whose table entries are summed for control output.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CMAC(nn.Module):
    """Tile-coded associative table lookup controller."""

    def __init__(
        self,
        n_in: int = 3,
        n_tilings: int = 4,
        table_size: int = 64,
        n_out: int = 2,
    ) -> None:
        """Initialize the CMAC table and tiling offsets.

        Parameters
        ----------
        n_in:
            Continuous input dimensionality.
        n_tilings:
            Number of overlapping tilings.
        table_size:
            Hash table size for active addresses.
        n_out:
            Output dimensionality.
        """
        super().__init__()
        self.n_in = n_in
        self.n_tilings = n_tilings
        self.table_size = table_size
        offsets = torch.arange(n_tilings, dtype=torch.float32).unsqueeze(1) / float(n_tilings)
        scales = torch.arange(1, n_in + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("offsets", offsets * scales)
        self.register_buffer("hash_weights", torch.arange(1, n_in + 1, dtype=torch.long))
        self.table = nn.Embedding(table_size, n_out)

    def addresses(self, x: Tensor) -> Tensor:
        """Compute hashed tile addresses for each tiling.

        Parameters
        ----------
        x:
            Continuous input tensor with shape ``(B, n_in)``.

        Returns
        -------
        Tensor
            Integer addresses with shape ``(B, n_tilings)``.
        """
        tiles = torch.floor(x[:, None, :] * 4.0 + self.offsets[None, :, :]).to(torch.long)
        hashed = (tiles * self.hash_weights[None, None, :]).sum(dim=-1)
        hashed = hashed + torch.arange(self.n_tilings, device=x.device).unsqueeze(0) * 17
        return torch.remainder(hashed, self.table_size)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Sum active CMAC table entries for a control output.

        Parameters
        ----------
        x:
            Continuous input tensor with shape ``(B, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output vector and active tile addresses.
        """
        active = self.addresses(x)
        values = self.table(active)
        return values.sum(dim=1), active.to(x.dtype)


def build() -> nn.Module:
    """Build a small random-init CMAC controller.

    Returns
    -------
    nn.Module
        A traceable ``CMAC`` instance.
    """
    return CMAC()


def example_input() -> Tensor:
    """Return bounded continuous CMAC inputs.

    Returns
    -------
    Tensor
        Example tensor with shape ``(2, 3)``.
    """
    return torch.tensor([[0.15, 0.45, 0.75], [0.82, 0.25, 0.33]], dtype=torch.float32)
