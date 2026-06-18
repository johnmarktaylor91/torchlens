"""Differentiable neural stack, 2015, Grefenstette et al., "Learning to transduce with unbounded memory".

Paper: Grefenstette 2015, "Learning to transduce with unbounded memory."
This small module implements the core soft stack mechanism: an LSTM controller emits
push vectors plus push and pop strengths, and a differentiable stack read summarizes
the current top contents. Queue and deque variants from the paper are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentiableNeuralStack(nn.Module):
    """LSTM-controlled differentiable stack with soft push/pop/read operations."""

    def __init__(self, input_size: int = 4, hidden_size: int = 6, stack_width: int = 5) -> None:
        """Initialize controller and stack action heads.

        Parameters
        ----------
        input_size
            Number of features in each sequence input.
        hidden_size
            Controller hidden size.
        stack_width
            Width of stored stack vectors.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.stack_width = stack_width
        self.controller = nn.LSTMCell(input_size + stack_width, hidden_size)
        self.push_value = nn.Linear(hidden_size, stack_width)
        self.strengths = nn.Linear(hidden_size, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Run a traceable soft stack over a time-major sequence.

        Parameters
        ----------
        x
            Input sequence of shape ``(time, batch, input_size)``.

        Returns
        -------
        Tensor
            Stack read vectors for each step, shape ``(time, batch, stack_width)``.
        """
        time, batch, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        values = x.new_zeros(batch, time, self.stack_width)
        strengths = x.new_zeros(batch, time)
        read = x.new_zeros(batch, self.stack_width)
        reads: list[Tensor] = []
        for step in range(time):
            h, c = self.controller(torch.cat((x[step], read), dim=-1), (h, c))
            action = torch.sigmoid(self.strengths(h))
            push_strength = action[:, 0]
            pop_strength = action[:, 1].unsqueeze(-1)
            values = torch.cat(
                (
                    values[:, :step],
                    torch.tanh(self.push_value(h)).unsqueeze(1),
                    values[:, step + 1 :],
                ),
                dim=1,
            )
            decayed = torch.relu(strengths[:, :step] - pop_strength)
            strengths = torch.cat(
                (decayed, push_strength.unsqueeze(-1), strengths[:, step + 1 :]), dim=1
            )
            weights = torch.softmax(strengths * 5.0, dim=-1)
            read = torch.sum(weights.unsqueeze(-1) * values, dim=1)
            reads.append(read)
        return torch.stack(reads, dim=0)


MENAGERIE_ENTRIES = [
    ("Differentiable Neural Stack / Queue / Deque", "build", "example_input", "2015", "CD")
]


def build() -> nn.Module:
    """Build a small differentiable neural stack.

    Returns
    -------
    nn.Module
        Configured stack module.
    """
    return DifferentiableNeuralStack()


def example_input() -> Tensor:
    """Create a time-major sequence input.

    Returns
    -------
    Tensor
        Example input with shape ``(4, 2, 4)``.
    """
    return torch.randn(4, 2, 4)
