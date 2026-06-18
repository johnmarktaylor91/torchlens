"""Original LSTM, 1997, Hochreiter and Schmidhuber, "Long Short-Term Memory".

Implements the constant-error carousel with input and output gates, no forget gate,
and a unit self-loop on cell state as in the original LSTM formulation.
"""

import torch
from torch import Tensor, nn


class OriginalLSTM1997(nn.Module):
    """Small original no-forget-gate LSTM sequence module."""

    def __init__(self, input_size: int = 4, hidden_size: int = 5) -> None:
        """Initialize gated input, cell-input, and output transformations.

        Parameters
        ----------
        input_size:
            Number of features per time step.
        hidden_size:
            Number of memory cells.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_input = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Run the no-forget LSTM over a batch-first sequence.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden states for all time steps, shape ``(batch, time, hidden_size)``.
        """
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            z = torch.cat((x[:, step], h), dim=-1)
            i = torch.sigmoid(self.input_gate(z))
            g = torch.tanh(self.cell_input(z))
            c = c + i * g
            o = torch.sigmoid(self.output_gate(torch.cat((x[:, step], h), dim=-1)))
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small random-init original LSTM module.

    Returns
    -------
    nn.Module
        Configured ``OriginalLSTM1997`` instance.
    """
    return OriginalLSTM1997()


def example_input() -> Tensor:
    """Create a batch-first float sequence example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 3, 4)``.
    """
    return torch.randn(2, 3, 4)
