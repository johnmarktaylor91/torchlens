"""A completed multi-pass Recording cooks into the multi-pass Trace ``tl.trace`` gives.

A ``Recorder`` called three times on one module records three passes. Cooking
folds the later passes into the first pass's label space, so the result groups
the repeated layers by call site exactly as an exhaustive capture of a module
that calls the same cell three times: the passes are numbered in call order,
``layer:k`` lookups resolve, and each pass keeps its own output layer. A halted
multi-pass Recording still refuses with the typed code, because a halt
frontier is defined for one forward pass.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import CaptureNotActiveError

_CODE = "recording_multipass_not_convertible"


class _Cell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _ThreeCalls(nn.Module):
    """The exhaustive counterpart of three Recorder passes over one ``_Cell``."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = _Cell()

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[Any, ...]:
        return self.cell(a), self.cell(b), self.cell(c)


def _inputs(count: int) -> list[torch.Tensor]:
    torch.manual_seed(1)
    return [torch.rand(2, 4) for _ in range(count)]


def _record_passes(model: nn.Module, inputs: list[torch.Tensor]) -> Any:
    with tl.recording.Recorder(model.eval(), save=lambda ctx: ctx.kind == "op") as recorder:
        for x in inputs:
            recorder(x)
    return recorder.recording


def _passes(trace: Any) -> list[tuple[str, int, int]]:
    """Execution-ordered (layer type, pass number, pass count) per non-input op."""

    return [(op.type, op.pass_number, op.num_passes) for op in trace.ops if op.type != "input"]


def _op(trace: Any, layer_type: str, pass_number: int) -> Any:
    (op,) = [o for o in trace.ops if o.type == layer_type and o.pass_number == pass_number]
    return op


def test_three_pass_recording_cooks_like_trace() -> None:
    """Pass numbering, pass counts and per-pass values agree with an exhaustive capture."""

    inputs = _inputs(3)
    torch.manual_seed(0)
    exhaustive = tl.trace(_ThreeCalls().eval(), tuple(inputs))
    torch.manual_seed(0)
    recording = _record_passes(_Cell(), inputs)
    assert recording.num_passes == 3

    cooked = recording.to_trace()

    assert _passes(cooked) == _passes(exhaustive)
    assert _passes(cooked) == [
        ("linear", 1, 3),
        ("tanh", 1, 3),
        ("linear", 2, 3),
        ("tanh", 2, 3),
        ("linear", 3, 3),
        ("tanh", 3, 3),
    ]
    for layer_type in ("linear", "tanh"):
        labels = sorted(op.layer_label for op in cooked.ops if op.type == layer_type)
        bases = {label.rsplit(":", 1)[0] for label in labels}
        assert len(bases) == 1, labels
        assert [label.rsplit(":", 1)[1] for label in labels] == ["1", "2", "3"]
        for pass_number in (1, 2, 3):
            cooked_op = _op(cooked, layer_type, pass_number)
            assert cooked[cooked_op.layer_label] is cooked_op
            assert torch.equal(cooked_op.out, _op(exhaustive, layer_type, pass_number).out)


def test_three_pass_cooked_values_come_from_their_own_pass() -> None:
    """Pass ``k`` of the cooked Trace holds the activations the Recorder saved in pass ``k``."""

    inputs = _inputs(3)
    recording = _record_passes(_Cell(), inputs)
    cooked = recording.to_trace()

    by_pass = {}
    for record in recording.records:
        if record.ctx.layer_type == "linear":
            by_pass[record.ctx.pass_number] = record.ram_payload
    assert sorted(by_pass) == [1, 2, 3]
    for pass_number, payload in by_pass.items():
        assert torch.equal(_op(cooked, "linear", pass_number).out, payload)
    assert not torch.equal(by_pass[1], by_pass[2])


def test_three_pass_recording_keeps_one_output_layer_per_pass() -> None:
    """The cooked Trace ends in three output layers, one per recorded pass, in pass order."""

    inputs = _inputs(3)
    torch.manual_seed(0)
    exhaustive = tl.trace(_ThreeCalls().eval(), tuple(inputs))
    torch.manual_seed(0)
    cooked = _record_passes(_Cell(), inputs).to_trace()

    assert len(cooked.output_layers) == len(exhaustive.output_layers) == 3
    assert [cooked[label].pass_number for label in cooked.output_layers] == [1, 2, 3]
    assert [cooked[label].type for label in cooked.output_layers] == ["tanh"] * 3
    assert len(cooked.input_layers) == len(exhaustive.input_layers) == 3


def test_two_pass_recording_cooks_two_passes() -> None:
    """Two passes fold the same way: every op carries pass 1 or 2 of 2."""

    cooked = _record_passes(_Cell(), _inputs(2)).to_trace()

    assert _passes(cooked) == [("linear", 1, 2), ("tanh", 1, 2), ("linear", 2, 2), ("tanh", 2, 2)]


def test_halted_multi_pass_recording_keeps_the_typed_refusal() -> None:
    """A halt frontier is defined for one pass, so a halted multi-pass Recording refuses."""

    inputs = _inputs(2)
    with tl.recording.Recorder(
        _Cell().eval(),
        save=lambda ctx: ctx.kind == "op",
        stop_after=lambda ctx: ctx.kind == "op" and ctx.pass_number == 2,
    ) as recorder:
        for x in inputs:
            recorder(x)
    recording = recorder.recording
    assert recording.num_passes == 2
    assert recording.is_stopped

    with pytest.raises(CaptureNotActiveError) as excinfo:
        recording.to_trace()
    assert excinfo.value.fields["code"] == _CODE
    assert "halted" in str(excinfo.value)
