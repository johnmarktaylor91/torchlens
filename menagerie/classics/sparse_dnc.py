"""DNC with Sparse Access, 2016, Rae et al., "Scaling Memory-Augmented Neural Networks".

Paper: Rae 2016, "Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes."
This simplified module keeps the differentiable sparse-access core: a controller
emits read and write keys, memory is addressed by soft top-k-like cosine attention,
and slots are updated by gated interpolation. Temporal links are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseDNC(nn.Module):
    """Small sparse-access DNC-style memory reader/writer."""

    def __init__(
        self, input_size: int = 4, hidden_size: int = 8, memory_slots: int = 10, word_size: int = 6
    ) -> None:
        """Initialize controller and memory access heads.

        Parameters
        ----------
        input_size
            Input feature width.
        hidden_size
            Controller hidden size.
        memory_slots
            Number of external memory locations.
        word_size
            Memory word width.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.word_size = word_size
        self.controller = nn.LSTMCell(input_size + word_size, hidden_size)
        self.read_key = nn.Linear(hidden_size, word_size)
        self.write_key = nn.Linear(hidden_size, word_size)
        self.write_value = nn.Linear(hidden_size, word_size)
        self.write_gate = nn.Linear(hidden_size, 1)

    def _attention(self, key: Tensor, memory: Tensor) -> Tensor:
        """Compute sharp cosine memory attention.

        Parameters
        ----------
        key
            Query key of shape ``(batch, word_size)``.
        memory
            Memory tensor of shape ``(batch, slots, word_size)``.

        Returns
        -------
        Tensor
            Attention weights over memory slots.
        """
        key_norm = torch.nn.functional.normalize(key, dim=-1)
        mem_norm = torch.nn.functional.normalize(memory, dim=-1)
        return torch.softmax(torch.sum(mem_norm * key_norm[:, None, :], dim=-1) * 8.0, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Run sparse memory access over a time-major sequence.

        Parameters
        ----------
        x
            Input sequence of shape ``(time, batch, input_size)``.

        Returns
        -------
        Tensor
            Read vectors for each time step.
        """
        time, batch, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)
        read = x.new_zeros(batch, self.word_size)
        memory = x.new_zeros(batch, self.memory_slots, self.word_size)
        outputs: list[Tensor] = []
        for step in range(time):
            h, c = self.controller(torch.cat((x[step], read), dim=-1), (h, c))
            write_weights = self._attention(self.write_key(h), memory)
            gate = torch.sigmoid(self.write_gate(h)).unsqueeze(-1)
            candidate = torch.tanh(self.write_value(h)).unsqueeze(1)
            memory = (
                memory * (1.0 - gate * write_weights.unsqueeze(-1))
                + gate * write_weights.unsqueeze(-1) * candidate
            )
            read_weights = self._attention(self.read_key(h), memory)
            read = torch.sum(read_weights.unsqueeze(-1) * memory, dim=1)
            outputs.append(read)
        return torch.stack(outputs, dim=0)


MENAGERIE_ENTRIES = [("DNC with Sparse Access", "build", "example_input", "2016", "CD")]


def build() -> nn.Module:
    """Build a compact sparse DNC.

    Returns
    -------
    nn.Module
        Configured sparse DNC module.
    """
    return SparseDNC()


def example_input() -> Tensor:
    """Create a time-major input sequence.

    Returns
    -------
    Tensor
        Example input with shape ``(4, 2, 4)``.
    """
    return torch.randn(4, 2, 4)
