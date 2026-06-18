"""VSA record encoder, 1994/2009, Gayler and Kanerva, "Vector Symbolic Architectures".

Paper: Gayler 1998, "Multiplicative Binding, Representation Operators & Analogy."
Bipolar key vectors bind to values by elementwise product, sequence roles are formed
by permutation, and bundled records are decoded by unbinding plus soft cleanup.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VSARecord(nn.Module):
    """Bind, permute, bundle, and decode bipolar record vectors."""

    def __init__(self, n_slots: int = 4, dim: int = 16) -> None:
        """Initialize random bipolar keys.

        Parameters
        ----------
        n_slots
            Number of record slots.
        dim
            Hypervector dimensionality.
        """
        super().__init__()
        self.register_buffer("keys", torch.sign(torch.randn(n_slots, dim)))

    def forward(self, values: Tensor) -> Tensor:
        """Encode a record and decode every slot.

        Parameters
        ----------
        values
            Bipolar value vectors of shape ``(batch, slots, dim)``.

        Returns
        -------
        Tensor
            Decoded value estimates with shape ``(batch, slots, dim)``.
        """
        bound: list[Tensor] = []
        for slot in range(self.keys.shape[0]):
            permuted = torch.roll(values[:, slot], shifts=slot, dims=-1)
            bound.append(permuted * self.keys[slot])
        record = torch.tanh(torch.stack(bound, dim=1).sum(dim=1))
        decoded: list[Tensor] = []
        for slot in range(self.keys.shape[0]):
            unbound = record * self.keys[slot]
            decoded.append(torch.roll(unbound, shifts=-slot, dims=-1))
        return torch.stack(decoded, dim=1)


MENAGERIE_ENTRIES = [
    ("VSA record encoder (bind+bundle+permute)", "build", "example_input", "1994/2009", "CE")
]


def build() -> nn.Module:
    """Build a small VSA record encoder.

    Returns
    -------
    nn.Module
        Configured VSA module.
    """
    return VSARecord()


def example_input() -> Tensor:
    """Create bipolar value vectors.

    Returns
    -------
    Tensor
        Example values with shape ``(2, 4, 16)``.
    """
    return torch.sign(torch.randn(2, 4, 16))
