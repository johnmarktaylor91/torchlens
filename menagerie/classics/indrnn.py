"""Independently Recurrent Neural Network, 2018, Li et al.

Paper: "Independently Recurrent Neural Network (IndRNN)." Each hidden neuron
has its own scalar recurrent weight, giving a diagonal recurrence suitable for
deep stacked recurrent networks.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class IndRNN(nn.Module):
    """Stacked IndRNN with per-neuron scalar recurrence."""

    def __init__(self, input_size: int = 128, hidden_size: int = 48, n_layers: int = 2) -> None:
        """Initialize input projections and diagonal recurrent weights.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            Hidden size per layer.
        n_layers:
            Number of stacked recurrent layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [nn.Linear(input_size if i == 0 else hidden_size, hidden_size) for i in range(n_layers)]
        )
        self.recurrent = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_size).uniform_(0.0, 0.9)) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run stacked independent recurrent layers.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        seq = x
        for layer_index, layer in enumerate(self.layers):
            hidden = x.new_zeros(x.shape[0], self.hidden_size)
            outputs: list[Tensor] = []
            recurrent = torch.clamp(self.recurrent[layer_index], -0.99, 0.99)
            projected = layer(seq)
            for step in range(x.shape[1]):
                hidden = torch.relu(projected[:, step] + recurrent * hidden)
                outputs.append(hidden)
            seq = torch.stack(outputs, dim=1)
        return seq


def build() -> nn.Module:
    """Build a compact IndRNN.

    Returns
    -------
    nn.Module
        Random-initialized IndRNN.
    """
    return IndRNN()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 64, 128)``.
    """
    return torch.randn(1, 64, 128)


MENAGERIE_ENTRIES = [
    ("IndRNN Independently Recurrent Neural Network", "build", "example_input", "2018", "DE")
]
