"""Peephole LSTM, 2000, Gers and Schmidhuber, "Recurrent Nets that Time and Count".

LSTM gates receive diagonal connections from the cell state, allowing gate decisions
to inspect the memory value directly for precise timing behavior.
"""

import torch
from torch import Tensor, nn


class PeepholeLSTM(nn.Module):
    """Batch-first peephole LSTM with diagonal cell-to-gate parameters."""

    def __init__(self, input_size: int = 4, hidden_size: int = 5) -> None:
        """Initialize affine gates and peephole weights.

        Parameters
        ----------
        input_size:
            Number of input features per time step.
        hidden_size:
            Number of memory cells.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.w_ci = nn.Parameter(torch.randn(hidden_size) * 0.05)
        self.w_cf = nn.Parameter(torch.randn(hidden_size) * 0.05)
        self.w_co = nn.Parameter(torch.randn(hidden_size) * 0.05)

    def forward(self, x: Tensor) -> Tensor:
        """Run the peephole LSTM recurrence.

        Parameters
        ----------
        x:
            Input sequence of shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Hidden states of shape ``(batch, time, hidden_size)``.
        """
        batch = x.shape[0]
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            i_raw, f_raw, g_raw, o_raw = self.gates(torch.cat((x[:, step], h), dim=-1)).chunk(4, -1)
            i = torch.sigmoid(i_raw + self.w_ci * c)
            f = torch.sigmoid(f_raw + self.w_cf * c)
            g = torch.tanh(g_raw)
            c = f * c + i * g
            o = torch.sigmoid(o_raw + self.w_co * c)
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a small random-init peephole LSTM.

    Returns
    -------
    nn.Module
        Configured ``PeepholeLSTM`` instance.
    """
    return PeepholeLSTM()


def example_input() -> Tensor:
    """Create a batch-first float sequence example.

    Returns
    -------
    Tensor
        Example input with shape ``(2, 3, 4)``.
    """
    return torch.randn(2, 3, 4)
