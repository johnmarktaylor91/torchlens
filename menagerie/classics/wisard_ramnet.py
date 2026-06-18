"""WiSARD / n-tuple RAMnet, 1959/1981, as a weightless discriminator.

Paper: Bledsoe and Browning 1959; Aleksander, Thomas, and Bowden 1981, "WISARD."
Binarized retina bits are grouped into random n-tuples whose addresses index
class-specific RAM tables; class evidence is the sum of addressed table values.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

MENAGERIE_ENTRIES = [("WiSARD / n-tuple RAMnet", "build", "example_input", "1959/1981", "CA")]


class WiSARDRAMNet(nn.Module):
    """Weightless RAM discriminator using fixed n-tuple addresses."""

    def __init__(
        self, retina_dim: int = 16, n_classes: int = 3, n_rams: int = 5, tuple_size: int = 3
    ) -> None:
        """Initialize tuple indices and class RAM tables.

        Parameters
        ----------
        retina_dim
            Number of retina bits.
        n_classes
            Number of discriminators.
        n_rams
            Number of RAMs per discriminator.
        tuple_size
            Number of retina bits per address.
        """
        super().__init__()
        indices = torch.arange(n_rams * tuple_size).reshape(n_rams, tuple_size) % retina_dim
        powers = (2 ** torch.arange(tuple_size)).float()
        tables = torch.rand(n_classes, n_rams, 2**tuple_size)
        self.register_buffer("indices", indices)
        self.register_buffer("powers", powers)
        self.register_buffer("tables", tables)
        self.n_classes = n_classes
        self.n_rams = n_rams

    def forward(self, retina: Tensor) -> Tensor:
        """Score a binarized retina against class RAM discriminators.

        Parameters
        ----------
        retina
            Binarized retina with shape ``(batch, retina_dim)``.

        Returns
        -------
        Tensor
            Class evidence scores.
        """
        bits = retina[:, self.indices].round().clamp(0.0, 1.0)
        addresses = (bits * self.powers).sum(dim=-1).long()
        expanded = addresses.unsqueeze(1).expand(-1, self.n_classes, -1).unsqueeze(-1)
        table_batch = self.tables.unsqueeze(0).expand(retina.shape[0], -1, -1, -1)
        return torch.gather(table_batch, dim=-1, index=expanded).squeeze(-1).sum(dim=-1)


def build() -> nn.Module:
    """Build a small WiSARD RAMnet.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return WiSARDRAMNet()


def example_input() -> Tensor:
    """Create a binarized retina example.

    Returns
    -------
    Tensor
        Example retina with shape ``(2, 16)``.
    """
    return (torch.rand(2, 16) > 0.5).float()
