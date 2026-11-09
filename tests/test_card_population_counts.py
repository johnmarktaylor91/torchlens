"""The trace card counts one kind of record per clause.

The repr line says how many ops ran and how many layers they form ("69 ops in 61 layers
(up to 2 passes)" when a layer repeats), and the saved row counts op outputs, then buffers,
inputs and outputs, so no count mixes populations.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

import torchlens as tl


class _Loop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(3):
            x = torch.tanh(self.lin(x))
        return self.bn(x)


def _card(model: nn.Module) -> tuple[object, str]:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model.eval(), torch.randn(2, 4))
    return trace, str(trace)


def test_repr_line_counts_ops_in_layers_with_passes() -> None:
    """Seven computing ops in three layers, the loop's two layers running three passes."""

    _trace, text = _card(_Loop())
    first = text.splitlines()[0]
    assert ", 7 ops in 3 layers (up to 3 passes)" in first, first


def test_saved_row_counts_op_outputs_then_other_records() -> None:
    """The saved row counts op outputs, then buffers, inputs and outputs."""

    trace, text = _card(_Loop())
    row = next(line for line in text.splitlines() if line.strip().startswith("saved"))
    assert row.strip().startswith(f"saved     7 of {trace._num_ops} op outputs, "), row
    assert row.endswith("; 2 buffers, 1 input, 1 output"), row
