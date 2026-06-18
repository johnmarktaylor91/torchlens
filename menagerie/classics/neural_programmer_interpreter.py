"""Neural Programmer-Interpreter, 2016, Reed and de Freitas, "Neural Programmer-Interpreters".

Paper: Reed 2016, "Neural Programmer-Interpreters."
This module captures the differentiable core step: an LSTM consumes state, current
program embedding, and arguments, then emits next-program attention, arguments, and stop probability.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class NPICore(nn.Module):
    """Single-step NPI controller core with differentiable program attention."""

    def __init__(
        self, state_size: int = 6, n_programs: int = 5, n_args: int = 3, hidden_size: int = 8
    ) -> None:
        """Initialize embeddings, recurrent core, and decoder heads.

        Parameters
        ----------
        state_size
            Encoded environment state width.
        n_programs
            Number of program memory slots.
        n_args
            Number of scalar arguments.
        hidden_size
            LSTM hidden size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.core = nn.LSTMCell(state_size + hidden_size + n_args, hidden_size)
        self.program_memory = nn.Parameter(torch.randn(n_programs, hidden_size) * 0.2)
        self.program_project = nn.Linear(n_programs, hidden_size)
        self.arg_head = nn.Linear(hidden_size, n_args)
        self.stop_head = nn.Linear(hidden_size, 1)

    def forward(self, packed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one NPI controller step.

        Parameters
        ----------
        packed
            Tensor of shape ``(batch, state_size + n_programs + n_args)`` with state,
            soft current-program logits, and current arguments.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Next-program probabilities, argument predictions, and stop probability.
        """
        state_size = self.core.input_size - self.hidden_size - self.arg_head.out_features
        n_programs = self.program_memory.shape[0]
        state = packed[:, :state_size]
        program_logits = packed[:, state_size : state_size + n_programs]
        args = packed[:, state_size + n_programs :]
        batch = state.shape[0]
        h0 = state.new_zeros(batch, self.hidden_size)
        c0 = state.new_zeros(batch, self.hidden_size)
        program = self.program_project(torch.softmax(program_logits, dim=-1))
        h, _ = self.core(torch.cat((state, program, args), dim=-1), (h0, c0))
        program_logits = h @ self.program_memory.T
        return (
            torch.softmax(program_logits, dim=-1),
            self.arg_head(h),
            torch.sigmoid(self.stop_head(h)),
        )


MENAGERIE_ENTRIES = [
    ("Neural Programmer-Interpreter (NPI)", "build", "example_input", "2016", "CD")
]


def build() -> nn.Module:
    """Build a compact NPI core.

    Returns
    -------
    nn.Module
        Configured NPI module.
    """
    return NPICore()


def example_input() -> Tensor:
    """Create packed state, program-logit, and argument examples.

    Returns
    -------
    Tensor
        Example packed NPI inputs with shape ``(2, 14)``.
    """
    return torch.randn(2, 14)
