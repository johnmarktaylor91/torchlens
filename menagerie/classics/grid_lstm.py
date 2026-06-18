"""Grid LSTM, 2015, Kalchbrenner, Danihelka, and Graves.

Paper: "Grid Long Short-Term Memory." LSTM memory flows along multiple grid
dimensions; this minimal priority-dimension model updates a depth dimension
inside each time step.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GridLSTM(nn.Module):
    """Two-dimensional depth-by-time Grid LSTM."""

    def __init__(self, input_size: int = 32, hidden_size: int = 32, depth: int = 3) -> None:
        """Initialize per-dimension LSTM transforms.

        Parameters
        ----------
        input_size:
            Per-step input size.
        hidden_size:
            Hidden size for both grid dimensions.
        depth:
            Number of depth transitions per time step.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.depth_gate = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.time_gate = nn.Linear(2 * hidden_size, 4 * hidden_size)

    def _lstm_dim(self, gates: Tensor, memory: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one LSTM dimension update.

        Parameters
        ----------
        gates:
            Pre-activation gate tensor.
        memory:
            Previous memory tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated hidden and memory tensors.
        """
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=-1)
        memory = torch.sigmoid(f_gate) * memory + torch.sigmoid(i_gate) * torch.tanh(g_gate)
        hidden = torch.sigmoid(o_gate) * torch.tanh(memory)
        return hidden, memory

    def forward(self, x: Tensor) -> Tensor:
        """Run priority-depth Grid LSTM over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, input_size)``.

        Returns
        -------
        Tensor
            Time-dimension hidden sequence with shape ``(batch, time, hidden_size)``.
        """
        batch = x.shape[0]
        time_hidden = x.new_zeros(batch, self.hidden_size)
        time_memory = x.new_zeros(batch, self.hidden_size)
        outputs: list[Tensor] = []
        for step in range(x.shape[1]):
            depth_hidden = torch.tanh(self.input_proj(x[:, step]))
            depth_memory = x.new_zeros(batch, self.hidden_size)
            for _ in range(self.depth):
                joined = torch.cat((depth_hidden, time_hidden), dim=-1)
                depth_hidden, depth_memory = self._lstm_dim(self.depth_gate(joined), depth_memory)
            joined = torch.cat((depth_hidden, time_hidden), dim=-1)
            time_hidden, time_memory = self._lstm_dim(self.time_gate(joined), time_memory)
            outputs.append(time_hidden)
        return torch.stack(outputs, dim=1)


def build() -> nn.Module:
    """Build a compact Grid LSTM.

    Returns
    -------
    nn.Module
        Random-initialized Grid LSTM.
    """
    return GridLSTM()


def example_input() -> Tensor:
    """Return an example sequence.

    Returns
    -------
    Tensor
        Float tensor with shape ``(1, 32, 32)``.
    """
    return torch.randn(1, 32, 32)


MENAGERIE_ENTRIES = [("Grid LSTM", "build", "example_input", "2015", "DE")]
