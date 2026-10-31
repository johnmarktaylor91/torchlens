"""Readers that resolve one object per ``input_layers`` label stay correct on a folded input.

Two inputs consumed by one loop body can fold into one multi-pass input Layer,
which torch lists in ``input_layers`` once per pass under one bare label. Three
readers resolve each listing to one object per label: the structure-only
discharge preflight (``trace[label]``, whose shape and dtype mirror pass 1) and
the live input-contract checks (``layer_dict_all_keys.get(label)``, the last
pass). They stay correct because an input's equivalence class is its shape and
dtype (``backends/torch/sources.py``), so only inputs of identical shape and
dtype fold and every pass of a folded input Layer shares both. These tests pin
that premise and the readers' verdicts on both sides of it. The oracles are the
tensors the test passed in and the model's own eager outputs, never another
TorchLens read.

One positional reader was wrong regardless of geometry: the fast live run's
input refresh looked each listing up in ``layer_dict_all_keys``, which maps a
folded input's bare label to its last pass, so a saved earlier pass kept its
capture-time tensor after ``trace.run(inputs=..., fast=True)``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from support.structural_grouper import legacy_structural_grouper  # noqa: F401
from torch import nn

import torchlens as tl
from torchlens._fast_live_steer import fast_live_input_admission
from torchlens._runnable import RunnableErrorCode
from torchlens._runnable_execution import _first_failed_live_input_check
from torchlens.capture.structure_only import StructureOnlyCapabilityError, _input_geometry
from torchlens.options import CaptureOptions
from torchlens.types import ContractCheck

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")


class _LoopOverInputs(nn.Module):
    """Two inputs of any shape or dtype consumed by one loop body."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = torch.zeros(1, 4)
        for x in (a, b):
            h = h + self.lin(x.to(torch.float32)).sum(0, keepdim=True)
        return h


class _TwoInputs(nn.Module):
    """Single-pass control: two inputs through separate code."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.lin(a) + self.lin(b) * 2


def _inputs(kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    if kind == "shape":
        return torch.randn(1, 4), torch.randn(3, 4)
    if kind == "dtype":
        return torch.randn(1, 4), torch.randn(1, 4, dtype=torch.float64)
    return torch.randn(1, 4), torch.randn(1, 4)


def _trace(kind: str, **kwargs: Any) -> tuple[Any, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    inputs = _inputs(kind)
    return tl.trace(_LoopOverInputs(), inputs, **kwargs), inputs


def _geometry(*tensors: torch.Tensor) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple((tuple(t.shape), str(t.dtype)) for t in tensors)


@pytest.mark.parametrize("kind", ["shape", "dtype"])
def test_inputs_of_different_shape_or_dtype_never_share_a_layer(kind: str) -> None:
    """Premise: a shape or dtype difference keeps the two inputs as separate Layers."""

    trace, _ = _trace(kind)
    assert trace.input_layers == ["input_1", "input_2"]
    assert [len(trace[label].ops) for label in trace.input_layers] == [1, 1]


def test_identical_inputs_fold_and_every_pass_matches_its_input() -> None:
    """Premise: identical inputs fold, and each pass carries its own input's geometry."""

    trace, (a, b) = _trace("same")
    assert trace.input_layers == ["input_1", "input_1"]
    passes = sorted(trace["input_1"].ops.values(), key=lambda op: op.pass_index)
    assert [torch.equal(op.out, t) for op, t in zip(passes, (a, b), strict=True)] == [True, True]
    assert _geometry(*(op.out for op in passes)) == _geometry(a, b)


@pytest.mark.parametrize("kind", ["shape", "dtype", "same"])
def test_structure_only_input_geometry_reports_each_input(kind: str) -> None:
    """The preflight geometry is the passed tensors' (shape, dtype), in order."""

    trace, (a, b) = _trace(kind)
    assert _input_geometry(trace) == _geometry(a, b)


def test_discharge_refuses_when_only_the_second_input_differs() -> None:
    """A second-input geometry difference against a folded twin refuses typed."""

    a, b = _inputs("same")
    torch.manual_seed(0)
    structure = tl.trace(_LoopOverInputs(), (a, b), capture=CaptureOptions(structure_only=True))
    assert structure.input_layers == ["input_1", "input_1"]
    torch.manual_seed(0)
    real = tl.trace(_LoopOverInputs(), (a, torch.randn(5, 4)))
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        structure.discharge_against(real)
    assert excinfo.value.fields["reason"] == "input_geometry"


@pytest.mark.parametrize("kind", ["shape", "dtype", "same"])
def test_live_input_checks_admit_the_captured_inputs(kind: str) -> None:
    """The captured inputs themselves pass both live input-contract checks."""

    trace, inputs = _trace(kind)
    assert _first_failed_live_input_check(trace, list(inputs), {}) is None
    assert fast_live_input_admission(trace, list(inputs), {}, allow_shape_change=False) == (
        None,
        False,
    )


@pytest.mark.parametrize("position", [0, 1])
def test_folded_live_checks_refuse_a_wrong_input_in_either_position(position: int) -> None:
    """A wrong first or second input fails the shape check and the dtype admission."""

    trace, inputs = _trace("same")
    wrong_shape = list(inputs)
    wrong_shape[position] = torch.randn(2, 4)
    failed = _first_failed_live_input_check(trace, wrong_shape, {})
    assert isinstance(failed, ContractCheck)
    assert failed.diagnostic is not None
    assert failed.diagnostic.code is RunnableErrorCode.INPUT_SHAPE_MISMATCH
    wrong_dtype = list(inputs)
    wrong_dtype[position] = inputs[position].to(torch.float64)
    admitted, _ = fast_live_input_admission(trace, wrong_dtype, {}, allow_shape_change=True)
    assert isinstance(admitted, ContractCheck)
    assert admitted.diagnostic is not None
    assert admitted.diagnostic.code is RunnableErrorCode.INPUT_DTYPE_MISMATCH


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("model_cls", [_LoopOverInputs, _TwoInputs])
def test_run_refreshes_each_saved_input_pass_with_its_own_input(
    model_cls: type[nn.Module], fast: bool
) -> None:
    """After ``trace.run``, a saved input op holds the new tensor for its position."""

    torch.manual_seed(0)
    model = model_cls().eval()
    a, b = torch.randn(1, 4), torch.randn(1, 4)
    new_a, new_b = torch.randn(1, 4), torch.randn(1, 4)
    trace = tl.trace(model, (a, b), save=tl.label("input_1") | tl.in_module("lin"))
    first_input = next(iter(trace.input_ops.values()))
    assert first_input.has_saved_activation
    assert torch.equal(first_input.out, a)
    result = trace.run(inputs=[new_a, new_b], fast=fast)
    with torch.no_grad():
        assert torch.equal(result.output, model(new_a, new_b))
    refreshed = result.trace if result.trace is not None else trace
    assert torch.equal(refreshed[first_input.label].out, new_a)
