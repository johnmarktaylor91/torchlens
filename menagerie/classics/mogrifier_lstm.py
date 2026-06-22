"""Mogrifier LSTM.

Paper: "Mogrifier LSTM", Melis, Dyer, and Blunsom, ICLR 2020.

The Mogrifier alternates multiplicative conditioning between the current input
and previous hidden state before the ordinary LSTM gates are evaluated. The
extra gates are initialized around identity by using ``2 * sigmoid(...)``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MogrifierLSTMCell(nn.Module):
    """LSTM cell with alternating input/hidden mogrification."""

    def __init__(self, input_size: int = 32, hidden_size: int = 32, rounds: int = 4) -> None:
        """Initialize mogrifier projections and LSTM gates.

        Parameters
        ----------
        input_size:
            Width of each input vector.
        hidden_size:
            Width of the recurrent hidden state.
        rounds:
            Number of alternating mogrifier updates.
        """
        super().__init__()
        self.rounds = rounds
        q_count = (rounds + 1) // 2
        r_count = rounds // 2
        self.q_layers = nn.ModuleList([nn.Linear(hidden_size, input_size) for _ in range(q_count)])
        self.r_layers = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(r_count)])
        self.input_gates = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Advance one recurrent step.

        Parameters
        ----------
        x:
            Current input with shape ``(batch, input_size)``.
        state:
            Previous ``(hidden, cell)`` tensors.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(hidden, cell)`` tensors.
        """
        hidden, cell = state
        x_mod = x
        h_mod = hidden
        q_index = 0
        r_index = 0
        for step in range(self.rounds):
            if step % 2 == 0:
                x_mod = 2.0 * torch.sigmoid(self.q_layers[q_index](h_mod)) * x_mod
                q_index += 1
            else:
                h_mod = 2.0 * torch.sigmoid(self.r_layers[r_index](x_mod)) * h_mod
                r_index += 1
        gates = self.input_gates(x_mod) + self.hidden_gates(h_mod)
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class MogrifierLSTMModel(nn.Module):
    """Compact token language model backed by a Mogrifier LSTM."""

    def __init__(self, vocab_size: int = 96, width: int = 32) -> None:
        """Initialize embedding, recurrent cell, and classifier.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        width:
            Embedding and hidden width.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.cell = MogrifierLSTMCell(width, width)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, ids: Tensor) -> Tensor:
        """Compute token logits from ids.

        Parameters
        ----------
        ids:
            Long token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Logits with shape ``(batch, time, vocab_size)``.
        """
        emb = self.embed(ids)
        hidden = emb.new_zeros(emb.shape[0], emb.shape[2])
        cell = emb.new_zeros(emb.shape[0], emb.shape[2])
        outputs = []
        for step in range(emb.shape[1]):
            hidden, cell = self.cell(emb[:, step], (hidden, cell))
            outputs.append(hidden)
        return self.head(self.norm(torch.stack(outputs, dim=1)))


def build() -> nn.Module:
    """Build a compact Mogrifier LSTM.

    Returns
    -------
    nn.Module
        Random-initialized Mogrifier LSTM model.
    """
    return MogrifierLSTMModel()


def example_input() -> Tensor:
    """Return example token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 12)``.
    """
    return torch.randint(0, 96, (1, 12), dtype=torch.long)


MENAGERIE_ENTRIES = [("Mogrifier LSTM", "build", "example_input", "2020", "DE")]
