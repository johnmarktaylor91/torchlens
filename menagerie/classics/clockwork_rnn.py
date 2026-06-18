"""Clockwork RNN, 2014, Jan Koutnik et al.

Paper: A Clockwork RNN.
Hidden units are partitioned into modules with distinct update periods and a
block-triangular recurrence so fast modules can read slower modules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClockworkRNN(nn.Module):
    """Multi-timescale recurrent network with clocked hidden groups."""

    def __init__(
        self,
        input_size: int = 6,
        group_size: int = 4,
        periods: tuple[int, ...] = (1, 2, 4),
        output_size: int = 3,
    ) -> None:
        """Initialize the clockwork recurrence.

        Parameters
        ----------
        input_size:
            Per-step input size.
        group_size:
            Hidden units per clock module.
        periods:
            Update period for each hidden group.
        output_size:
            Output feature size.
        """
        super().__init__()
        self.periods = periods
        self.group_size = group_size
        self.hidden_size = group_size * len(periods)
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.recurrent = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.output = nn.Linear(self.hidden_size, output_size)
        mask = torch.zeros(self.hidden_size, self.hidden_size)
        for row_group in range(len(periods)):
            row_slice = slice(row_group * group_size, (row_group + 1) * group_size)
            for col_group in range(row_group, len(periods)):
                col_slice = slice(col_group * group_size, (col_group + 1) * group_size)
                mask[row_slice, col_slice] = 1.0
        self.register_buffer("recurrent_mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        """Run the clockwork recurrence over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(B, T, input_size)``.

        Returns
        -------
        Tensor
            Output sequence with shape ``(B, T, output_size)``.
        """
        batch, steps, _ = x.shape
        hidden = x.new_zeros(batch, self.hidden_size)
        outputs: list[Tensor] = []
        weight = self.recurrent * self.recurrent_mask
        for step in range(steps):
            proposal = torch.tanh(self.input_proj(x[:, step]) + hidden @ weight.T + self.bias)
            parts: list[Tensor] = []
            for group_index, period in enumerate(self.periods):
                start = group_index * self.group_size
                end = start + self.group_size
                part = proposal[:, start:end] if step % period == 0 else hidden[:, start:end]
                parts.append(part)
            hidden = torch.cat(parts, dim=1)
            outputs.append(self.output(hidden))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small Clockwork RNN.

    Returns
    -------
    nn.Module
        Random-initialized Clockwork RNN.
    """
    return ClockworkRNN()


def example_input() -> Tensor:
    """Return a traceable sequence batch.

    Returns
    -------
    Tensor
        Float sequence tensor with shape ``(1, 6, 6)``.
    """
    return torch.randn(1, 6, 6)
