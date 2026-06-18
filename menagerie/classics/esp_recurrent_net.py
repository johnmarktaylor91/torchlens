"""ESP recurrent neuroevolution net, 1999, Gomez and Miikkulainen.

Paper: "Solving non-Markovian control tasks with neuroevolution."
Each recurrent neuron corresponds to an independently evolved subpopulation member;
the module implements the assembled recurrent phenotype only.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ESPRecurrentNeuroevolutionNet(nn.Module):
    """Complete recurrent phenotype assembled from evolved neuron weights."""

    def __init__(self, n_in: int = 8, hidden_size: int = 10, n_out: int = 3) -> None:
        """Initialize recurrent phenotype parameters.

        Parameters
        ----------
        n_in
            Number of features per time step.
        hidden_size
            Number of recurrent neurons.
        n_out
            Number of output channels.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.wx = nn.Parameter(torch.randn(hidden_size, n_in) * 0.2)
        self.wh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.wy = nn.Parameter(torch.randn(n_out, hidden_size) * 0.2)

    def forward(self, x_seq: Tensor) -> Tensor:
        """Run the recurrent phenotype over a batch-first sequence.

        Parameters
        ----------
        x_seq
            Input sequence tensor of shape ``(batch, time, 8)``.

        Returns
        -------
        Tensor
            Output sequence with shape ``(batch, time, n_out)``.
        """
        batch = x_seq.shape[0]
        hidden = x_seq.new_zeros(batch, self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x_seq.shape[1]):
            hidden = torch.tanh(x_seq[:, step] @ self.wx.T + hidden @ self.wh.T)
            outputs.append(hidden @ self.wy.T)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small ESP recurrent phenotype.

    Returns
    -------
    nn.Module
        Configured ``ESPRecurrentNeuroevolutionNet`` instance.
    """
    return ESPRecurrentNeuroevolutionNet()


def example_input() -> Tensor:
    """Create an example control sequence.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16, 8)``.
    """
    return torch.randn(1, 16, 8)


MENAGERIE_ENTRIES = [("ESP Recurrent Neuroevolution Net", "build", "example_input", "1999", "DD")]
