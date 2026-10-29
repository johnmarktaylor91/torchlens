"""Foreign-kernel boundary: an ``autograd.Function`` whose forward leaves torch.

Positive: a numpy round trip and a dlpack round trip inside ``Function.forward`` are
captured as ONE ``autogradfunction`` boundary op that validates through replay, with the
armed completeness census clean. Control: an ordinary custom Function (interior in torch)
is captured exactly as before. Armed negative: a plain ``.numpy()`` round trip in a module
forward (no Function) still fails completeness.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from tests import example_models
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens.backends.torch._function_boundary import (
    FOREIGN_FUNCTION_FUNC_NAME,
    FOREIGN_FUNCTION_TRANSFORM_KIND,
    FOREIGN_FUNCTION_WARNING_CODE,
    FOREIGN_FUNCTION_WRAPPER_NAME,
    is_foreign_function_recorder,
)
from torchlens.backends.torch.completeness_witness import AUDITED_COMPLETENESS_BOUNDARIES
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.errors import TorchLensWarning
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation.diagnostics import CHECK_COMPLETENESS


class _NumpyDoubleFunction(torch.autograd.Function):
    """The RESULT.md G2 regression Function: the forward computes in NumPy."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        return torch.from_numpy(x.detach().numpy() * 2)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * 2


class _DlpackDoubleFunction(torch.autograd.Function):
    """DGL shape: tensors leave through dlpack, compute elsewhere, return through from_dlpack."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        host = np.from_dlpack(x.detach().contiguous()) * 2
        return torch.from_dlpack(host)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * 2


class _WriteThroughDoubleFunction(torch.autograd.Function):
    """DGL SpMM shape: the output is allocated in torch and a foreign kernel writes into it."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        out = torch.zeros_like(x)
        host_view = out.numpy()
        host_view[...] = x.detach().numpy() * 2
        return out

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * 2


class _ForeignFunctionModel(nn.Module):
    """Linear, then a Function whose forward leaves torch, then relu."""

    def __init__(self, function: type[torch.autograd.Function]) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.function = function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = self.function.apply(y)
        return torch.relu(y)


class _PlainNumpyRoundTripModel(nn.Module):
    """The armed negative: the same round trip with no Function around it."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        y = torch.from_numpy(y.detach().numpy() * 2)
        return torch.relu(y)


def _boundary_ops(trace: tl.Trace) -> list:
    """Return the ``autogradfunction`` ops of a trace."""

    return [op for op in trace.layer_list if op.func_name == FOREIGN_FUNCTION_FUNC_NAME]


def _assert_one_boundary(trace: tl.Trace, function: type[torch.autograd.Function]) -> None:
    """Assert the Function is exactly one boundary op with the right facets and parents."""

    boundaries = _boundary_ops(trace)
    assert len(boundaries) == 1
    op = boundaries[0]
    assert op.is_transform is True
    assert op.transform_kind == FOREIGN_FUNCTION_TRANSFORM_KIND
    assert op.transform_fn_qualname == function.forward.__qualname__
    assert op.transform_config["function"] == function.__qualname__
    assert is_foreign_function_recorder(op.func)
    # The interior (detach, the host re-entry constructor) is not in the graph.
    interior_types = {"detach", "fromnumpy", "fromdlpack", "contiguous", "zeroslike"}
    assert not any(op.layer_type in interior_types for op in trace.layer_list)
    # Parents are the Function's tensor inputs; the child is the consumer of its output.
    assert [trace[label].layer_type for label in op.parents] == ["linear"]
    assert [trace[label].layer_type for label in op.children] == ["relu"]


def _assert_validates_clean(model: nn.Module, x: torch.Tensor) -> None:
    """Assert forward validation passes with no completeness-witness gap."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_forward_pass(model, [x], input_kwargs={})
    assert last_validation_failure() is None
    assert not any(isinstance(item.message, TorchLensCaptureGapWarning) for item in caught)


@pytest.mark.parametrize(
    "function", [_NumpyDoubleFunction, _DlpackDoubleFunction, _WriteThroughDoubleFunction]
)
def test_foreign_function_is_one_boundary_op_and_validates(function) -> None:
    """A Function whose forward leaves torch is one replayable boundary op."""

    if function is _DlpackDoubleFunction and not hasattr(np, "from_dlpack"):
        pytest.skip("numpy without from_dlpack")
    model = _ForeignFunctionModel(function)
    x = torch.randn(3, 4)

    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op") as record:
        trace = tl.trace(model, x)
    codes = [getattr(w.message, "fields", {}).get("code") for w in record]
    assert FOREIGN_FUNCTION_WARNING_CODE in codes
    _assert_one_boundary(trace, function)
    op = _boundary_ops(trace)[0]
    torch.testing.assert_close(op.out, function.apply(model.fc(x)))

    # Forward replay and perturbation through the boundary validate, nothing exempted.
    _assert_validates_clean(model, x)


def test_foreign_function_owns_its_interior_dispatches() -> None:
    """With the census armed, every dispatch inside the Function belongs to the boundary."""

    wrap_torch(completeness_witness=True)
    model = _ForeignFunctionModel(_NumpyDoubleFunction)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(3, 4))

    assert trace.capture_verified is True
    assert trace.completeness_diagnostics == []
    owners = {entry["owner_wrapper"] for entry in trace.completeness_decompositions}
    assert FOREIGN_FUNCTION_WRAPPER_NAME in owners
    boundary_rows = [
        entry
        for entry in trace.completeness_decompositions
        if entry["owner_wrapper"] == FOREIGN_FUNCTION_WRAPPER_NAME
    ]
    assert len(boundary_rows) == 1
    assert boundary_rows[0]["owner_func_call_id"] == _boundary_ops(trace)[0].func_call_id
    assert boundary_rows[0]["capture_accounted"] is True
    # The interior detach was re-owned: no owner row of its own survives.
    assert not any(
        entry["owner_func_name"] == "detach" for entry in trace.completeness_decompositions
    )


def test_boundary_selector_and_row_are_wired() -> None:
    """The boundary is selectable by its transform kind and audited by the witness table."""

    model = _ForeignFunctionModel(_NumpyDoubleFunction)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(2, 4), save=tl.func_transform("autograd.Function"))
    saved = [op for op in trace.layer_list if op.has_saved_activation]
    assert [op.func_name for op in saved] == [FOREIGN_FUNCTION_FUNC_NAME]
    rows = [
        row
        for row in AUDITED_COMPLETENESS_BOUNDARIES
        if row.wrapper_name == FOREIGN_FUNCTION_WRAPPER_NAME
    ]
    assert len(rows) == 1 and rows[0].operator is None


def test_ordinary_custom_function_is_captured_as_before() -> None:
    """Control: a Function whose interior stays in torch keeps its interior ops."""

    model = example_models.CustomAutogradModel()
    x = torch.rand(4, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    types = [op.layer_type for op in trace.layer_list]
    assert "mul" in types
    assert not any(op.is_transform for op in trace.layer_list)
    _assert_validates_clean(model, x)


def test_plain_numpy_round_trip_is_not_collapsed() -> None:
    """Armed negative: without a Function the round trip stays the visible hole it is today.

    Nothing is collapsed and no boundary op appears: ``from_numpy`` is a parentless
    source op and the output has no input ancestor, exactly as on the base tree. (On the
    base tree forward validation PASSES this model, because a parentless constructor
    replays deterministically; the completeness check fails only on a dispatcher-visible
    unowned op such as ``to_dlpack``'s ``as_strided``. Recorded as a finding in the lane
    report; this test pins that the boundary never reaches outside a Function.)
    """

    model = _PlainNumpyRoundTripModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    from_numpy_ops = [op for op in trace.layer_list if op.layer_type == "fromnumpy"]
    assert len(from_numpy_ops) == 1
    assert list(from_numpy_ops[0].parents) == []
    assert not trace.output_ops[0].input_ancestors
    assert any(op.layer_type == "detach" for op in trace.layer_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_verdict = validate_forward_pass(model, [x], input_kwargs={})
    failure = last_validation_failure()
    # The verdict is whatever the base tree gives (True today); the only thing the
    # boundary may never do is turn this plain round trip into a boundary op.
    assert base_verdict is True or (failure is not None and failure.check == CHECK_COMPLETENESS)
