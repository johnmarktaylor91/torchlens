"""Stack/Queue-Augmented RNN, 2015, Armand Joulin and Tomas Mikolov.

Paper: Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets.
A recurrent controller emits continuous push and pop strengths for a
differentiable external stack whose read vector feeds the next step.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StackAugmentedRNN(nn.Module):
    """RNN controller with a differentiable continuous stack."""

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 16,
        stack_width: int = 6,
        output_size: int = 4,
    ) -> None:
        """Initialize controller and stack action heads.

        Parameters
        ----------
        input_size:
            Per-step input size.
        hidden_size:
            Controller hidden size.
        stack_width:
            Width of pushed value vectors.
        output_size:
            Per-step output size.
        """
        super().__init__()
        self.stack_width = stack_width
        self.controller = nn.GRUCell(input_size + stack_width, hidden_size)
        self.action = nn.Linear(hidden_size, 2)
        self.value = nn.Linear(hidden_size, stack_width)
        self.output = nn.Linear(hidden_size + stack_width, output_size)

    def _pop(self, strengths: Tensor, pop_strength: Tensor) -> Tensor:
        """Remove continuous mass from the stack top downward.

        Parameters
        ----------
        strengths:
            Stack strengths ``(B, depth)``.
        pop_strength:
            Requested pop mass ``(B, 1)``.

        Returns
        -------
        Tensor
            Updated strengths.
        """
        remaining = pop_strength
        updated: list[Tensor] = []
        for index in range(strengths.shape[1] - 1, -1, -1):
            current = strengths[:, index : index + 1]
            removed = torch.minimum(current, remaining)
            updated.append(current - removed)
            remaining = remaining - removed
        return torch.cat(list(reversed(updated)), dim=1)

    def _read(self, values: Tensor, strengths: Tensor) -> Tensor:
        """Read the top unit mass from the stack.

        Parameters
        ----------
        values:
            Stack values ``(B, depth, width)``.
        strengths:
            Stack strengths ``(B, depth)``.

        Returns
        -------
        Tensor
            Weighted read vector.
        """
        remaining = strengths.new_ones(strengths.shape[0], 1)
        weights: list[Tensor] = []
        for index in range(strengths.shape[1] - 1, -1, -1):
            current = torch.minimum(strengths[:, index : index + 1], remaining)
            weights.append(current)
            remaining = remaining - current
        read_weights = torch.cat(list(reversed(weights)), dim=1)
        return (values * read_weights.unsqueeze(2)).sum(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run controller and differentiable stack over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(B, T, input_size)``.

        Returns
        -------
        Tensor
            Output sequence.
        """
        batch, steps, _ = x.shape
        hidden = x.new_zeros(batch, self.controller.hidden_size)
        values = x.new_zeros(batch, 0, self.stack_width)
        strengths = x.new_zeros(batch, 0)
        read = x.new_zeros(batch, self.stack_width)
        outputs: list[Tensor] = []
        for step in range(steps):
            hidden = self.controller(torch.cat([x[:, step], read], dim=1), hidden)
            push_strength, pop_strength = torch.sigmoid(self.action(hidden)).chunk(2, dim=1)
            if strengths.shape[1] > 0:
                strengths = self._pop(strengths, pop_strength)
            push_value = torch.tanh(self.value(hidden)).unsqueeze(1)
            values = torch.cat([values, push_value], dim=1)
            strengths = torch.cat([strengths, push_strength], dim=1)
            read = self._read(values, strengths)
            outputs.append(self.output(torch.cat([hidden, read], dim=1)))
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact stack-augmented RNN.

    Returns
    -------
    nn.Module
        Random-initialized StackAugmentedRNN.
    """
    return StackAugmentedRNN()


def example_input() -> Tensor:
    """Return a traceable sequence batch.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 5, 5)``.
    """
    return torch.randn(1, 5, 5)
