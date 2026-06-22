"""Differentiable Neural Computer (DNC).

Graves et al., Nature 2016.  A DNC combines a recurrent controller with an
external memory matrix, differentiable content addressing, erase/add writes, and
read vectors fed back to the controller.  This compact random-init version keeps
the controller-interface-memory loop with one write head and one read head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactDNC(nn.Module):
    """Compact DNC with content addressing."""

    def __init__(
        self, input_dim: int = 5, hidden: int = 24, cells: int = 8, width: int = 6
    ) -> None:
        """Initialize DNC controller and interface heads.

        Parameters
        ----------
        input_dim:
            Input feature size.
        hidden:
            Controller hidden size.
        cells:
            Number of memory cells.
        width:
            Memory cell width.
        """

        super().__init__()
        self.cells = cells
        self.width = width
        self.controller = nn.GRUCell(input_dim + width, hidden)
        self.interface = nn.Linear(hidden, 4 * width + 3)
        self.output = nn.Linear(hidden + width, 3)

    def _content_weights(
        self, memory: torch.Tensor, key: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine-similarity content weights.

        Parameters
        ----------
        memory:
            Memory matrix.
        key:
            Addressing key.
        beta:
            Address sharpness.

        Returns
        -------
        torch.Tensor
            Content weights over cells.
        """

        mem = nn.functional.normalize(memory, dim=-1)
        key_n = nn.functional.normalize(key, dim=-1)
        return torch.softmax(beta * torch.matmul(mem, key_n.unsqueeze(-1)).squeeze(-1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the DNC over a short sequence.

        Parameters
        ----------
        x:
            Sequence input ``(batch, time, input_dim)``.

        Returns
        -------
        torch.Tensor
            Output logits for each time step.
        """

        batch = x.shape[0]
        h = x.new_zeros(batch, self.controller.hidden_size)
        memory = x.new_zeros(batch, self.cells, self.width)
        read = x.new_zeros(batch, self.width)
        outs = []
        for step in range(x.shape[1]):
            h = self.controller(torch.cat([x[:, step], read], dim=-1), h)
            interface = self.interface(h)
            read_key, write_key, erase, add = torch.split(
                interface[:, : 4 * self.width], self.width, dim=-1
            )
            gates = torch.sigmoid(interface[:, 4 * self.width :])
            beta_r = 1.0 + gates[:, 0:1] * 5.0
            beta_w = 1.0 + gates[:, 1:2] * 5.0
            write_gate = gates[:, 2:3]
            ww = self._content_weights(memory, write_key, beta_w) * write_gate
            memory = memory * (1.0 - ww.unsqueeze(-1) * torch.sigmoid(erase).unsqueeze(1))
            memory = memory + ww.unsqueeze(-1) * torch.tanh(add).unsqueeze(1)
            rw = self._content_weights(memory, read_key, beta_r)
            read = torch.matmul(rw.unsqueeze(1), memory).squeeze(1)
            outs.append(self.output(torch.cat([h, read], dim=-1)))
        return torch.stack(outs, dim=1)


def build() -> nn.Module:
    """Build compact DNC.

    Returns
    -------
    nn.Module
        Random-init DNC reconstruction.
    """

    return CompactDNC()


def example_input() -> torch.Tensor:
    """Create a short sequence input.

    Returns
    -------
    torch.Tensor
        Sequence tensor.
    """

    return torch.randn(1, 5, 5)


MENAGERIE_ENTRIES = [("DNC", "build", "example_input", "2016", "MEM")]
