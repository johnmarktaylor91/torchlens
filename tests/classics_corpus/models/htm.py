"""Hierarchical Temporal Memory, 2004, Hawkins and George.

Paper: Hierarchical Temporal Memory / Cortical Learning Algorithm.
Sparse distributed representation model with a k-WTA spatial pooler and a
temporal memory that predicts next-step cell activations from distal segments.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialPooler(nn.Module):
    """HTM spatial pooler with fixed permanences and k-WTA columns."""

    def __init__(self, n_inputs: int = 32, n_columns: int = 12, active_columns: int = 3) -> None:
        """Initialize spatial pooler buffers.

        Parameters
        ----------
        n_inputs:
            Number of binary input bits.
        n_columns:
            Number of cortical columns.
        active_columns:
            Number of columns selected by k-WTA.
        """
        super().__init__()
        self.active_columns = active_columns
        self.register_buffer("permanence", torch.rand(n_columns, n_inputs))
        self.register_buffer("boost", torch.ones(n_columns))

    def forward(self, inputs: Tensor) -> Tensor:
        """Convert binary input SDRs to sparse column SDRs.

        Parameters
        ----------
        inputs:
            Binary input tensor of shape ``(batch, n_inputs)``.

        Returns
        -------
        Tensor
            Float column SDR of shape ``(batch, n_columns)``.
        """
        connected = (self.permanence > 0.5).to(dtype=self.permanence.dtype)
        overlap = inputs.to(dtype=connected.dtype) @ connected.T
        boosted = overlap * self.boost
        winners = torch.topk(boosted, k=self.active_columns, dim=-1).indices
        sdr = torch.zeros_like(boosted)
        return sdr.scatter(-1, winners, 1.0)


class TemporalMemory(nn.Module):
    """HTM temporal memory with distal-segment predictions."""

    def __init__(
        self, n_columns: int = 12, cells_per_column: int = 4, activation_threshold: float = 1.5
    ) -> None:
        """Initialize temporal memory buffers.

        Parameters
        ----------
        n_columns:
            Number of spatial-pooler columns.
        cells_per_column:
            Number of cells per column.
        activation_threshold:
            Segment activation threshold.
        """
        super().__init__()
        self.n_columns = n_columns
        self.cells_per_column = cells_per_column
        self.activation_threshold = activation_threshold
        n_cells = n_columns * cells_per_column
        self.register_buffer("distal", torch.rand(n_cells, n_cells) * 0.4)

    def forward(self, column_sdr: Tensor) -> tuple[Tensor, Tensor]:
        """Compute bursting active cells and predictive cells.

        Parameters
        ----------
        column_sdr:
            Column SDR of shape ``(batch, n_columns)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Active cells and predictive cells of shape ``(batch, n_cells)``.
        """
        batch = column_sdr.shape[0]
        burst = column_sdr.unsqueeze(-1).expand(batch, self.n_columns, self.cells_per_column)
        active_cells = burst.reshape(batch, self.n_columns * self.cells_per_column)
        connected = (self.distal > 0.3).to(dtype=active_cells.dtype)
        segment_drive = active_cells @ connected.T
        predictive_cells = (segment_drive > self.activation_threshold).to(dtype=active_cells.dtype)
        return active_cells, predictive_cells


class HierarchicalTemporalMemory(nn.Module):
    """Small HTM/CLA spatial-pooler plus temporal-memory stack."""

    def __init__(self) -> None:
        """Initialize the HTM stack."""
        super().__init__()
        self.spatial_pooler = SpatialPooler()
        self.temporal_memory = TemporalMemory()

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run spatial pooling and temporal-memory prediction.

        Parameters
        ----------
        inputs:
            Binary SDR tensor of shape ``(batch, n_inputs)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Column SDR, active cells, and predictive cells.
        """
        columns = self.spatial_pooler(inputs)
        active_cells, predictive_cells = self.temporal_memory(columns)
        return columns, active_cells, predictive_cells


def build() -> nn.Module:
    """Build a small HTM stack.

    Returns
    -------
    nn.Module
        HierarchicalTemporalMemory instance.
    """
    return HierarchicalTemporalMemory()


def example_input() -> Tensor:
    """Return a sample binary SDR batch.

    Returns
    -------
    Tensor
        Boolean tensor of shape ``(2, 32)``.
    """
    return torch.rand(2, 32) > 0.75
