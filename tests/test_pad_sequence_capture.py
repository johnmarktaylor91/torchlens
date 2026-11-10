"""``torch.nn.utils.rnn.pad_sequence`` captures as one op and validates.

The Python helper delegates to the C builtin ``torch._C._nn.pad_sequence``, a
namespace the overridable crawl never reads, so its ``full`` / ``select`` /
``copy_`` kernel dispatched with no owning op: ``bfs_completeness`` failed and
the padded tensor reached its consumer with no parents. The builtin is now a
decorated roster row, so the call records once with every sequence as a parent,
however the caller bound the helper.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_sequence

import torchlens as tl


class PadPerSequence(nn.Module):
    """Split one projection into ragged sequences and pad them (the list2padseq idiom)."""

    def __init__(self, batch_first: bool) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        seqs = [h[:3], h[3:5], h[5:6]]
        padded = pad_sequence(seqs, batch_first=self.batch_first, padding_value=-1.0)
        return padded.tanh().sum(1 if self.batch_first else 0)


class PackedLSTM(nn.Module):
    """``pack_sequence`` pads through the same builtin before packing for an LSTM."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.lstm = nn.LSTM(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        _, (hidden, _) = self.lstm(pack_sequence([h[:3], h[3:5], h[5:6]]))
        return hidden[0]


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Run seeded forward validation."""

    torch.manual_seed(0)
    return bool(tl.validate(model, x, scope="forward"))


@pytest.mark.parametrize("batch_first", [True, False])
def test_pad_sequence_is_one_op_with_every_sequence_as_parent(batch_first: bool) -> None:
    """The padded tensor has one owning op whose parents are the three slices."""

    model = PadPerSequence(batch_first)
    x = torch.randn(6, 4)
    trace = tl.trace(model, x)
    ops = [op for op in trace._layer_list if op.func_name == "pad_sequence"]
    assert len(ops) == 1
    assert len(ops[0].parents) == 3
    assert torch.equal(trace[trace.output_layers[0]].out, model(x))
    assert _validate(model, x)


def test_pack_sequence_into_lstm_validates() -> None:
    """``pack_sequence`` (pad, then pack) feeding an LSTM passes forward validation."""

    model = PackedLSTM()
    x = torch.randn(6, 4)
    # Ops are views over their trace's rows; the private op list does not keep
    # the trace, so hold it while reading (the public doors keep it themselves).
    trace = tl.trace(model, x)
    names = [op.func_name for op in trace._layer_list]
    assert "pad_sequence" in names
    assert _validate(model, x)
