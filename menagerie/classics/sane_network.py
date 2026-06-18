"""SANE neuroevolution network, 1996, Moriarty and Miikkulainen.

Paper: "Efficient reinforcement learning through symbiotic evolution."
Slot-specific hidden neurons are assembled into a cooperative network; population
fitness assignment and evolutionary selection are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SANENeuroevolutionNetwork(nn.Module):
    """Symbiotic hidden-neuron assembly with selectable slots."""

    def __init__(self, n_in: int = 8, n_slots: int = 8, n_out: int = 3) -> None:
        """Initialize neuron slot input and output weights.

        Parameters
        ----------
        n_in
            Number of input features.
        n_slots
            Number of available hidden neuron slots.
        n_out
            Number of output channels.
        """
        super().__init__()
        self.slot_wx = nn.Parameter(torch.randn(n_slots, n_in) * 0.2)
        self.slot_wy = nn.Parameter(torch.randn(n_slots, n_out) * 0.2)
        self.out_bias = nn.Parameter(torch.zeros(n_out))
        self.register_buffer("selected_slots", torch.tensor([0, 2, 4, 6], dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        """Assemble selected neuron slots and compute network output.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, 8)``.

        Returns
        -------
        Tensor
            Output tensor from the assembled neuron set.
        """
        wx = self.slot_wx.index_select(0, self.selected_slots)
        wy = self.slot_wy.index_select(0, self.selected_slots)
        hidden = torch.tanh(x @ wx.T)
        return hidden @ wy + self.out_bias


def build() -> nn.Module:
    """Build a small SANE network substrate.

    Returns
    -------
    nn.Module
        Configured ``SANENeuroevolutionNetwork`` instance.
    """
    return SANENeuroevolutionNetwork()


def example_input() -> Tensor:
    """Create an example input.

    Returns
    -------
    Tensor
        State tensor with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


MENAGERIE_ENTRIES = [("SANE Neuroevolution Network", "build", "example_input", "1996", "DD")]
