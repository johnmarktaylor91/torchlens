"""``Trace.input_ops`` / ``output_ops`` hold Ops when a boundary Layer has several passes.

Recurrence grouping folds input rows that feed the same loop body into one
multi-pass input Layer (``for x in (a, b): h = h + lin(x)``). Torch lists such a
Layer in ``Trace.input_layers`` once per pass under its bare layer label, which
resolves to the Layer, not an Op; the accessors used to wrap those Layers and
refused on the multi-pass one (``Layer 'input_1' has 2 ops``), which crashed
``build_agent_json`` and the folded ``torchlens_overview`` on the classics
NeoDTI and ADL-MVDR entries. The oracles below are the tensors the test passed
in and the model's own eager outputs, never another TorchLens read.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from support.structural_grouper import legacy_structural_grouper  # noqa: F401
from torch import nn

import torchlens as tl
from tests.classics_corpus._loader import build_entry, corpus_entries
from torchlens._data_classes._trace_accessors import boundary_label_ops
from torchlens._data_classes.op import Op
from torchlens.agent import call_tool
from torchlens.report._agent_json import build_agent_json

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")


class _LoopOverInputs(nn.Module):
    """Two inputs consumed by one loop body, so their input rows share a Layer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(1, 4)
        for x in (a, b):
            h = h + self.lin(x)
        return h


class _TwoInputs(nn.Module):
    """Single-pass control: two inputs through separate code."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.lin(a) + torch.relu(b)


class _LoopOutputs(nn.Module):
    """Returns one loop-body result per input, so output rows may group too."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(torch.relu(self.lin(x)) for x in (a, b))


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(1, 4), torch.randn(1, 4)


def _loop_trace() -> tuple[Any, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    model = _LoopOverInputs()
    inputs = _inputs()
    return tl.trace(model, inputs), inputs


def test_loop_inputs_fold_into_one_two_pass_layer() -> None:
    """Precondition: the fixture really produces a two-pass input Layer."""

    trace, _ = _loop_trace()
    assert trace.input_layers == ["input_1", "input_1"]
    assert trace["input_1"].num_passes == 2


def test_input_ops_lists_each_pass_as_an_op_in_argument_order() -> None:
    trace, (a, b) = _loop_trace()

    ops = list(trace.input_ops)
    assert len(ops) == 2 == trace.num_input_ops
    assert all(isinstance(op, Op) for op in ops)
    assert [op.pass_index for op in ops] == [1, 2]
    assert torch.equal(ops[0].out, a)
    assert torch.equal(ops[1].out, b)
    assert trace.input_ops.keys() == [op.label for op in ops]


def test_num_input_layers_counts_the_multi_pass_layer_once() -> None:
    trace, _ = _loop_trace()

    assert trace.num_input_layers == 1
    assert trace.num_input_ops == 2


def test_single_pass_inputs_are_ops_too() -> None:
    torch.manual_seed(0)
    a, b = _inputs()
    trace = tl.trace(_TwoInputs(), (a, b))

    ops = list(trace.input_ops)
    assert all(isinstance(op, Op) for op in ops)
    assert [torch.equal(op.out, x) for op, x in zip(ops, (a, b), strict=True)] == [True, True]
    assert trace.num_input_layers == 2 == trace.num_input_ops


def test_output_ops_hold_ops_matching_the_eager_outputs() -> None:
    torch.manual_seed(0)
    model = _LoopOutputs()
    a, b = _inputs()
    with torch.no_grad():
        expected = model(a, b)
    trace = tl.trace(model, (a, b))

    ops = list(trace.output_ops)
    assert all(isinstance(op, Op) for op in ops)
    assert len(ops) == len(expected) == trace.num_output_ops
    for op, value in zip(ops, expected, strict=True):
        assert torch.equal(op.out, value)
    assert trace.num_output_layers == len(dict.fromkeys(trace.output_layers))


def test_boundary_label_ops_expands_an_under_listed_layer() -> None:
    """A multi-pass Layer listed fewer times than it has passes yields every pass once."""

    trace, (a, b) = _loop_trace()

    ops = boundary_label_ops(["input_1"], trace.__getitem__)
    assert [op.pass_index for op in ops] == [1, 2]
    assert torch.equal(ops[0].out, a) and torch.equal(ops[1].out, b)


def test_agent_json_and_overview_succeed_on_a_two_pass_input(tmp_path: Path) -> None:
    trace, _ = _loop_trace()
    expected = [f"{trace['input_1'].layer_label}:{index}" for index in (1, 2)]

    assert build_agent_json(trace)["inputs"] == expected
    assert trace.to_agent_json()["inputs"] == expected

    path = tmp_path / "loop_inputs.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    assert [op.label for op in loaded.input_ops] == expected
    envelope = call_tool("torchlens_overview", {"path": str(path)})
    assert envelope["data"]["capture"]["capture_status"] == "complete"


_CORPUS_MULTIPASS_INPUT_IDS = ("NeoDTI", "ADL-MVDR")


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(entry, id=entry.id)
        for entry in corpus_entries()
        if entry.id in _CORPUS_MULTIPASS_INPUT_IDS
    ],
)
def test_classics_multipass_input_entries_build_agent_json(entry: Any) -> None:
    """The two classics entries whose input Layer has two passes dump every input."""

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model, inputs = build_entry(entry)
        trace = tl.trace(model, inputs)
    tensors = [value for value in inputs if isinstance(value, torch.Tensor)]

    assert any(trace[label].num_passes > 1 for label in trace.input_layers)
    dumped = build_agent_json(trace)["inputs"]
    assert len(dumped) == len(tensors) == trace.num_input_ops
    for label, value in zip(dumped, tensors, strict=True):
        assert torch.equal(trace[label].out, value)


def test_corpus_selection_found_both_entries() -> None:
    found = {entry.id for entry in corpus_entries() if entry.id in _CORPUS_MULTIPASS_INPUT_IDS}
    assert found == set(_CORPUS_MULTIPASS_INPUT_IDS)
