"""Tests for buffer source and sink accessor aliases."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class BufferWriteModel(nn.Module):
    """Model that reads and writes a registered buffer."""

    def __init__(self) -> None:
        """Initialize the registered buffer."""

        super().__init__()
        self.register_buffer("state", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read from and write to the registered buffer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Computed output using the mutated buffer.
        """

        y = self.state + x
        self.state.copy_(y)
        return self.state * x


def test_buffer_source_and_sink_accessor_counts() -> None:
    """Trace and Op buffer source/sink accessors reflect read/write Ops."""

    trace = tl.trace(BufferWriteModel(), torch.ones(2), save_arg_values=True)
    buffer_sources = [
        op.label for op in trace.layer_list if op.is_buffer and op.buffer_write_kind is None
    ]
    buffer_sinks = [
        op.label for op in trace.layer_list if op.is_buffer and op.buffer_write_kind is not None
    ]

    assert trace.num_buffer_source_ops == len(buffer_sources)
    assert trace.num_buffer_sink_ops == len(buffer_sinks)
    assert buffer_sources
    assert buffer_sinks
    assert any(op.buffer_source_ops for op in trace.compute_ops)
    assert any(op.buffer_sink_ops for op in trace.compute_ops)
