"""Leabra, 2000.

O'Reilly, "Biologically plausible error-driven learning using local activation
differences." Point-neuron rates settle under bidirectional weights and k-WTA inhibition.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LeabraNet(nn.Module):
    """Small bidirectional settling network with k-WTA inhibition."""

    def __init__(self, n_in: int = 10, n_hidden: int = 8, n_out: int = 4, steps: int = 5) -> None:
        """Initialize bidirectional rate layers.

        Parameters
        ----------
        n_in
            Number of input units.
        n_hidden
            Number of hidden units.
        n_out
            Number of output units.
        steps
            Number of settling cycles.
        """
        super().__init__()
        self.steps = steps
        self.k_hidden = max(1, n_hidden // 3)
        self.k_out = max(1, n_out // 2)
        self.in_to_hid = nn.Linear(n_in, n_hidden, bias=False)
        self.hid_to_out = nn.Linear(n_hidden, n_out, bias=False)
        self.out_feedback = nn.Linear(n_out, n_hidden, bias=False)
        self.hid_feedback = nn.Linear(n_hidden, n_in, bias=False)

    def _kwta(self, x: Tensor, k: int) -> Tensor:
        """Apply k-winner-take-all thresholding.

        Parameters
        ----------
        x
            Pre-activation tensor.
        k
            Number of winners to retain per example.

        Returns
        -------
        Tensor
            Bounded activation with non-winners suppressed.
        """
        threshold = torch.topk(x, k=k, dim=-1).values[..., -1:]
        return torch.sigmoid(4.0 * (x - threshold))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Settle hidden and output layers from an input pattern.

        Parameters
        ----------
        x
            Input tensor of shape ``(batch, n_in)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Output and hidden rates after settling.
        """
        hidden = self._kwta(self.in_to_hid(x), self.k_hidden)
        out = self._kwta(self.hid_to_out(hidden), self.k_out)
        for _ in range(self.steps):
            hidden_drive = self.in_to_hid(x) + self.out_feedback(out)
            hidden = 0.7 * hidden + 0.3 * self._kwta(hidden_drive, self.k_hidden)
            out_drive = self.hid_to_out(hidden)
            out = 0.7 * out + 0.3 * self._kwta(out_drive, self.k_out)
        return out, hidden


def build() -> nn.Module:
    """Build a small random Leabra-style settling network.

    Returns
    -------
    nn.Module
        Random-initialized Leabra module.
    """
    return LeabraNet()


def example_input() -> Tensor:
    """Return a float32 example input.

    Returns
    -------
    Tensor
        Example input of shape ``(2, 10)``.
    """
    return torch.randn(2, 10, dtype=torch.float32)
