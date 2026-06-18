"""Differentiable machines, 2015-2017, NRAM, Neural Enquirer, and Neural Map.

Paper: Kurach 2015, "Neural Random-Access Machines"; Yin 2016, "Neural Enquirer";
Parisotto 2017, "Neural Map." These modules implement the core differentiable forward
substrates; hard program search, table supervision, RL policy training, and symbolic execution
are omitted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class NRAM(nn.Module):
    """Soft-register neural random-access machine with differentiable read/write."""

    def __init__(self, tape_size: int = 16, width: int = 8, steps: int = 4) -> None:
        """Initialize controller and memory-operation heads.

        Parameters
        ----------
        tape_size
            Number of memory cells.
        width
            Width of each memory value.
        steps
            Number of fuzzy execution steps.
        """
        super().__init__()
        self.tape_size = tape_size
        self.width = width
        self.steps = steps
        self.encoder = nn.Linear(tape_size, width)
        self.controller = nn.GRUCell(width * 2, width)
        self.addr_head = nn.Linear(width, tape_size)
        self.write_head = nn.Linear(width, width)
        self.out = nn.Linear(width * 2, width)

    def forward(self, tape: Tensor) -> Tensor:
        """Execute soft read/write steps over an external memory tape.

        Parameters
        ----------
        tape
            Memory seed tensor with shape ``(batch, 16)``.

        Returns
        -------
        Tensor
            Controller and readout features.
        """
        batch = tape.shape[0]
        memory = tape.unsqueeze(-1).repeat(1, 1, self.width)
        h = self.encoder(tape)
        read = tape.new_zeros(batch, self.width)
        for _ in range(self.steps):
            h = self.controller(torch.cat((h, read), dim=-1), h)
            address = torch.softmax(self.addr_head(h), dim=-1)
            read = torch.sum(address.unsqueeze(-1) * memory, dim=1)
            write = torch.tanh(self.write_head(h)).unsqueeze(1)
            memory = memory * (1.0 - address.unsqueeze(-1)) + write * address.unsqueeze(-1)
        return self.out(torch.cat((h, read), dim=-1))


class NeuralEnquirer(nn.Module):
    """Stacked differentiable table executor with query-conditioned row annotations."""

    def __init__(self, n_cols: int = 6, query_dim: int = 12, hidden_size: int = 16) -> None:
        """Initialize query encoder and executor layers.

        Parameters
        ----------
        n_cols
            Number of numeric table columns.
        query_dim
            Query feature width.
        hidden_size
            Row annotation width.
        """
        super().__init__()
        self.n_cols = n_cols
        self.query_dim = query_dim
        self.query_encoder = nn.Linear(query_dim, hidden_size)
        self.cell_encoder = nn.Linear(n_cols, hidden_size)
        self.executor = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(3)])
        self.answer = nn.Linear(hidden_size, n_cols)

    def forward(self, packed: Tensor) -> Tensor:
        """Execute differentiable table reasoning from packed table/query input.

        Parameters
        ----------
        packed
            Tensor with shape ``(batch, 11, 12)``; first 10 rows encode table cells in the
            first 6 columns and the last row encodes the query.

        Returns
        -------
        Tensor
            Soft answer over table columns.
        """
        table = packed[:, :10, : self.n_cols]
        query = packed[:, 10, : self.query_dim]
        q = torch.tanh(self.query_encoder(query))
        annotation = torch.tanh(self.cell_encoder(table))
        for layer in self.executor:
            q_rows = q.unsqueeze(1).expand_as(annotation)
            score = torch.sigmoid(layer(torch.cat((annotation, q_rows), dim=-1)))
            annotation = annotation + score
        pooled = torch.sum(
            torch.softmax(annotation.mean(dim=-1), dim=-1).unsqueeze(-1) * annotation, dim=1
        )
        return self.answer(pooled)


class NeuralMap(nn.Module):
    """Spatial memory map with pose-conditioned local read and differentiable write."""

    def __init__(self, channels: int = 32, height: int = 16, width: int = 16) -> None:
        """Initialize read and write transforms.

        Parameters
        ----------
        channels
            Number of map channels.
        height
            Map height.
        width
            Map width.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.global_read = nn.Linear(channels, channels)
        self.write = nn.Linear(channels + 2, channels)
        self.policy = nn.Linear(channels * 2, 6)

    def forward(self, packed: Tensor) -> Tensor:
        """Read and update a spatial memory map from a packed map plus pose.

        Parameters
        ----------
        packed
            Tensor with shape ``(batch, 34, 16, 16)``; first 32 channels are the map and
            the last 2 channels broadcast normalized pose coordinates.

        Returns
        -------
        Tensor
            Policy feature logits.
        """
        memory = packed[:, :32]
        pose_map = packed[:, 32:, 0, 0]
        batch = memory.shape[0]
        flat = memory.flatten(2).transpose(1, 2)
        attention = torch.softmax(flat.mean(dim=-1), dim=-1)
        global_context = torch.sum(attention.unsqueeze(-1) * flat, dim=1)
        coords_y = torch.linspace(-1.0, 1.0, self.height, device=packed.device, dtype=packed.dtype)
        coords_x = torch.linspace(-1.0, 1.0, self.width, device=packed.device, dtype=packed.dtype)
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        dist = (xx.unsqueeze(0) - pose_map[:, 0:1, None]).pow(2)
        dist = dist + (yy.unsqueeze(0) - pose_map[:, 1:2, None]).pow(2)
        local_weight = torch.softmax(-20.0 * dist.reshape(batch, -1), dim=-1)
        local = torch.sum(local_weight.unsqueeze(-1) * flat, dim=1)
        write_value = torch.tanh(self.write(torch.cat((local, pose_map), dim=-1)))
        updated = global_context + self.global_read(write_value)
        return self.policy(torch.cat((updated, local), dim=-1))


MENAGERIE_ENTRIES = [
    ("NRAM (Neural Random-Access Machine)", "build_nram", "example_input_nram", "2015", "DA"),
    ("Neural Enquirer", "build_neural_enquirer", "example_input_neural_enquirer", "2016", "DA"),
    ("Neural Map", "build_neural_map", "example_input_neural_map", "2017", "DA"),
]


def build_nram() -> nn.Module:
    """Build an NRAM module.

    Returns
    -------
    nn.Module
        Configured NRAM module.
    """
    return NRAM()


def example_input_nram() -> Tensor:
    """Create an NRAM tape example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 16)``.
    """
    return torch.randn(1, 16)


def build_neural_enquirer() -> nn.Module:
    """Build a Neural Enquirer module.

    Returns
    -------
    nn.Module
        Configured Neural Enquirer.
    """
    return NeuralEnquirer()


def example_input_neural_enquirer() -> Tensor:
    """Create packed table/query input.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 11, 12)``.
    """
    return torch.randn(1, 11, 12)


def build_neural_map() -> nn.Module:
    """Build a Neural Map module.

    Returns
    -------
    nn.Module
        Configured Neural Map.
    """
    return NeuralMap()


def example_input_neural_map() -> Tensor:
    """Create packed map and pose input.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 34, 16, 16)``.
    """
    memory = torch.randn(1, 32, 16, 16)
    pose = torch.zeros(1, 2, 16, 16)
    pose[:, 0] = 0.25
    pose[:, 1] = -0.25
    return torch.cat((memory, pose), dim=1)
