"""JANET, 2017, van der Westhuizen and Lasenby.

Paper: "The Unreasonable Effectiveness of the Forget Gate." The recurrent cell
keeps only a forget gate with a coupled update path, simplifying the LSTM while
preserving gated memory.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class JANET(nn.Module):
    """Forget-gate-only recurrent sequence model."""

    def __init__(self, input_size: int = 64, hidden_size: int = 48) -> None:
        """Initialize forget and candidate projections.

        Parameters
        ----------
        input_size:
            Per-step feature size.
        hidden_size:
            Hidden state size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.forget = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        nn.init.constant_(self.forget.bias, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Run JANET recurrence over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        hidden = x.new_zeros(x.shape[0], self.hidden_size)
        cell = x.new_zeros(x.shape[0], self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            joined = torch.cat((x[:, step], hidden), dim=-1)
            forget = torch.sigmoid(self.forget(joined))
            update = torch.tanh(self.candidate(joined))
            cell = forget * cell + (1.0 - forget) * update
            hidden = torch.tanh(cell)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact JANET module.

    Returns
    -------
    nn.Module
        Random-initialized JANET.
    """
    return JANET()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 32, 64)``.
    """
    return torch.randn(1, 32, 64)


MENAGERIE_ENTRIES = [("JANET Forget-Gate-Only Cell", "build", "example_input", "2017", "DE")]
