"""Differentiable Forth interpreter, 2017, Riedel et al., "Differentiable Forth".

Paper: Riedel 2017, "Differentiable Forth: Training Neural Program Interpreters."
This simplified interpreter applies trainable soft instruction slots over a small
stack machine with push, pop, add, dup, and swap-like tensor updates.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentiableForth(nn.Module):
    """Soft differentiable stack-machine interpreter."""

    def __init__(self, depth: int = 5, width: int = 4, n_slots: int = 4) -> None:
        """Initialize trainable instruction slots.

        Parameters
        ----------
        depth
            Data stack depth.
        width
            Stack value width.
        n_slots
            Number of program slots.
        """
        super().__init__()
        self.depth = depth
        self.width = width
        self.instruction_logits = nn.Parameter(torch.randn(n_slots, 5) * 0.1)
        self.push_value = nn.Parameter(torch.randn(width) * 0.1)

    def _ops(self, stack: Tensor) -> Tensor:
        """Construct candidate instruction results.

        Parameters
        ----------
        stack
            Current stack of shape ``(batch, depth, width)``.

        Returns
        -------
        Tensor
            Candidate next stacks for five instructions.
        """
        push = torch.cat(
            (self.push_value.view(1, 1, -1).expand(stack.shape[0], 1, -1), stack[:, :-1]), dim=1
        )
        pop = torch.cat((stack[:, 1:], torch.zeros_like(stack[:, :1])), dim=1)
        add_top = stack[:, :1] + stack[:, 1:2]
        add = torch.cat((add_top, stack[:, 2:], torch.zeros_like(stack[:, :1])), dim=1)
        dup = torch.cat((stack[:, :1], stack[:, :-1]), dim=1)
        swap = torch.cat((stack[:, 1:2], stack[:, :1], stack[:, 2:]), dim=1)
        return torch.stack((push, pop, add, dup, swap), dim=-1)

    def forward(self, stack: Tensor) -> Tensor:
        """Run soft program slots over a stack.

        Parameters
        ----------
        stack
            Initial stack tensor of shape ``(batch, depth, width)``.

        Returns
        -------
        Tensor
            Final stack tensor.
        """
        state = stack
        for slot in range(self.instruction_logits.shape[0]):
            weights = torch.softmax(self.instruction_logits[slot], dim=-1)
            state = torch.sum(self._ops(state) * weights.view(1, 1, 1, -1), dim=-1)
        return state


MENAGERIE_ENTRIES = [
    ("Differentiable Forth interpreter (d4)", "build", "example_input", "2017", "CD")
]


def build() -> nn.Module:
    """Build a simplified differentiable Forth interpreter.

    Returns
    -------
    nn.Module
        Configured interpreter module.
    """
    return DifferentiableForth()


def example_input() -> Tensor:
    """Create a stack input.

    Returns
    -------
    Tensor
        Example stack with shape ``(2, 5, 4)``.
    """
    return torch.randn(2, 5, 4)
