"""Simple Recurrent Unit, 2017, Lei and Zhang.

Paper: "Training RNNs as Fast as CNNs." Matrix multiplications are parallel
over time; recurrence is restricted to cheap elementwise state updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleRecurrentUnit(nn.Module):
    """Single-layer SRU sequence module."""

    def __init__(self, input_size: int = 128, hidden_size: int = 48) -> None:
        """Initialize parallel projections and diagonal recurrent gates.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            State size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.skip = nn.Linear(input_size, hidden_size, bias=False)
        self.v_f = nn.Parameter(torch.zeros(hidden_size))
        self.v_r = nn.Parameter(torch.zeros(hidden_size))
        self.bias_f = nn.Parameter(torch.zeros(hidden_size))
        self.bias_r = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """Run the SRU pointwise recurrence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Output sequence with shape ``(batch, time, hidden_size)``.
        """
        candidate, forget_raw, reset_raw = self.proj(x).chunk(3, dim=-1)
        residual = self.skip(x)
        cell = x.new_zeros(x.shape[0], self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            forget = torch.sigmoid(forget_raw[:, step] + self.v_f * cell + self.bias_f)
            cell = forget * cell + (1.0 - forget) * candidate[:, step]
            reset = torch.sigmoid(reset_raw[:, step] + self.v_r * cell + self.bias_r)
            outputs.append(reset * torch.tanh(cell) + (1.0 - reset) * residual[:, step])
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact SRU.

    Returns
    -------
    nn.Module
        Random-initialized SRU.
    """
    return SimpleRecurrentUnit()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 64, 128)``.
    """
    return torch.randn(1, 64, 128)


MENAGERIE_ENTRIES = [("SRU Simple Recurrent Unit", "build", "example_input", "2017", "DE")]
