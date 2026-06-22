"""Tiny Legendre Memory Unit recurrent model.

Paper: Voelker et al. 2019, "Legendre Memory Units: Continuous-Time
Representation in Recurrent Neural Networks."
"""

from __future__ import annotations

import torch
from torch import nn


class LMUCell(nn.Module):
    """Legendre delay-line memory cell with learned encoders."""

    def __init__(self, input_size: int = 2, hidden_size: int = 3, order: int = 4) -> None:
        """Initialize the LMU cell.

        Parameters
        ----------
        input_size:
            Number of input features per time step.
        hidden_size:
            Hidden state width.
        order:
            Legendre memory order.
        """

        super().__init__()
        q = torch.arange(order, dtype=torch.float32)
        row = (2 * q + 1).view(order, 1)
        i = q.view(order, 1)
        j = q.view(1, order)
        sign = torch.where(torch.remainder(i - j + 1, 2) == 0, 1.0, -1.0)
        a = torch.where(i < j, -1.0, sign) * row / 4.0
        b = ((-1.0) ** q * (2 * q + 1)).view(order, 1) / 4.0
        self.register_buffer("a_matrix", a)
        self.register_buffer("b_vector", b)
        self.hidden_size = hidden_size
        self.order = order
        self.input_encoder = nn.Linear(input_size, 1)
        self.memory_encoder = nn.Linear(hidden_size * order, 1)
        self.hidden = nn.Linear(input_size + hidden_size * order, hidden_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one LMU step.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_size)``.
        memory:
            Legendre memory tensor of shape ``(batch, hidden_size, order)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Hidden state and updated memory.
        """

        batch = x.shape[0]
        drive = self.input_encoder(x) + self.memory_encoder(memory.reshape(batch, -1))
        next_memory = memory @ self.a_matrix.T.unsqueeze(0)
        next_memory = next_memory + drive.unsqueeze(-1) * self.b_vector.T.unsqueeze(0)
        hidden_input = torch.cat([x, next_memory.reshape(batch, -1)], dim=-1)
        return torch.tanh(self.hidden(hidden_input)), next_memory


class TinyLMU(nn.Module):
    """Small sequence classifier built from an LMU cell."""

    def __init__(self, input_size: int = 2, hidden_size: int = 3, order: int = 4) -> None:
        """Initialize the compact LMU model.

        Parameters
        ----------
        input_size:
            Number of input features per time step.
        hidden_size:
            Hidden state width.
        order:
            Legendre memory order.
        """

        super().__init__()
        self.cell = LMUCell(input_size, hidden_size, order)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LMU over a short sequence.

        Parameters
        ----------
        x:
            Input sequence of shape ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Sequence-level logits.
        """

        batch = x.shape[0]
        memory = x.new_zeros(batch, self.cell.hidden_size, self.cell.order)
        hidden = x.new_zeros(batch, self.cell.hidden_size)
        for step in range(x.shape[1]):
            hidden, memory = self.cell(x[:, step, :], memory)
        return self.head(hidden)


def build() -> nn.Module:
    """Build a tiny LMU model.

    Returns
    -------
    nn.Module
        Random-initialized LMU.
    """

    return TinyLMU().eval()


def example_input() -> torch.Tensor:
    """Create a small LMU sequence input.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(1, 5, 2)``.
    """

    return torch.randn(1, 5, 2)


MENAGERIE_ENTRIES = [("nengodl_lmu", "build", "example_input", "2019", "DB")]
