"""Foreign-kernel boundary: an ``autograd.Function`` whose forward leaves torch.

Positive: a numpy round trip and a dlpack round trip inside ``Function.forward`` are
captured as ONE ``autogradfunction`` boundary op that validates through replay, with the
armed completeness census clean; so is a known foreign kernel (the FlashAttention 4 shape:
a ``torch.empty`` output filled below the dispatcher, matched by its interface module). Control: an ordinary custom Function (interior in torch)
is captured exactly as before. Armed negative: a plain ``.numpy()`` round trip in a module
forward (no Function) is never collapsed into a Function boundary.
"""

from __future__ import annotations

import array
import ctypes
import warnings

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from tests import example_models
from torchlens._errors import CaptureGapWarning
from torchlens._record_type import record_type
from torchlens._utils import _torch_compat as tc
from torchlens.backends.torch._foreign_data import (
    FLASH_ATTENTION_4_EXIT,
    FLASH_ATTENTION_4_INTERFACE_MODULE,
    known_foreign_kernel_exits,
)
from torchlens.backends.torch._function_boundary import (
    FOREIGN_FUNCTION_FUNC_NAME,
    FOREIGN_FUNCTION_TRANSFORM_KIND,
    FOREIGN_FUNCTION_WRAPPER_NAME,
    apply_slot_function,
    is_foreign_function_recorder,
)
from torchlens.backends.torch.completeness_witness import AUDITED_COMPLETENESS_BOUNDARIES
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.errors import TorchLensWarning
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation._diagnostics import CHECK_REPLAY


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


class _SameObjectDoubleFunction(torch.autograd.Function):
    """DGL zero-copy shape: ``x.contiguous()`` hands back the input object before the export."""

    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        x = x.contiguous()
        return torch.from_numpy(x.detach().numpy() * 2)

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

    return [op for op in trace._layer_list if op.func_name == FOREIGN_FUNCTION_FUNC_NAME]


def _assert_one_boundary(trace: tl.Trace, function: type[torch.autograd.Function]) -> None:
    """Assert the Function is exactly one boundary op with the right facets and parents."""

    boundaries = _boundary_ops(trace)
    assert len(boundaries) == 1
    op = boundaries[0]
    assert op.call.is_func_transform is True
    assert op.call.func_transform_kind == FOREIGN_FUNCTION_TRANSFORM_KIND
    assert op.call.func_transform_fn_qualname == function.forward.__qualname__
    assert op.call.func_transform_config["function"] == function.__qualname__
    assert is_foreign_function_recorder(op.func)
    # The interior (detach, the host re-entry constructor) is not in the graph.
    interior_types = {"detach", "fromnumpy", "fromdlpack", "contiguous", "zeroslike"}
    assert not any(op.type in interior_types for op in trace._layer_list)
    # Parents are the Function's tensor inputs; the child is the consumer of its output.
    assert [record_type(trace[label]) for label in op.parents] == ["linear"]
    assert [record_type(trace[label]) for label in op.children] == ["relu"]


def _assert_validates_clean(model: nn.Module, x: torch.Tensor) -> None:
    """Assert forward validation passes with no completeness-witness gap."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_forward_pass(model, [x], input_kwargs={})
    assert last_validation_failure() is None
    assert not any(isinstance(item.message, CaptureGapWarning) for item in caught)


@pytest.mark.parametrize(
    "function",
    [
        _NumpyDoubleFunction,
        _DlpackDoubleFunction,
        _SameObjectDoubleFunction,
        _WriteThroughDoubleFunction,
    ],
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
    assert "foreign_kernel_boundary" in codes  # literal: the error-code coverage scan reads it
    _assert_one_boundary(trace, function)
    op = _boundary_ops(trace)[0]
    torch.testing.assert_close(op.out, function.apply(model.fc(x)))

    # Forward replay and perturbation through the boundary validate, nothing exempted.
    _assert_validates_clean(model, x)


def test_foreign_write_reaches_the_boundary_payload_with_eager_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With eager payload clones the boundary op saves the bytes the foreign kernel wrote.

    The interior ``zeros_like`` saved a copy of the same tensor object before the numpy write,
    which leaves the version counter alone; the boundary op must not reuse that copy.
    """

    from torchlens.backends.torch import _wrapper_session

    monkeypatch.setattr(_wrapper_session, "_COW_ENABLED", False)
    model = _ForeignFunctionModel(_WriteThroughDoubleFunction)
    x = torch.randn(3, 4)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, x)
    expected = _WriteThroughDoubleFunction.apply(model.fc(x))
    torch.testing.assert_close(_boundary_ops(trace)[0].out, expected)


def test_foreign_function_owns_its_interior_dispatches() -> None:
    """With the census armed, every dispatch inside the Function belongs to the boundary."""

    wrap_torch(completeness_witness=True)
    model = _ForeignFunctionModel(_NumpyDoubleFunction)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(3, 4))

    assert trace.provenance.verified is True
    assert trace.completeness_diagnostics == []
    owners = {entry["owner_wrapper"] for entry in trace.health.completeness_decompositions}
    assert FOREIGN_FUNCTION_WRAPPER_NAME in owners
    boundary_rows = [
        entry
        for entry in trace.health.completeness_decompositions
        if entry["owner_wrapper"] == FOREIGN_FUNCTION_WRAPPER_NAME
    ]
    assert len(boundary_rows) == 1
    assert boundary_rows[0]["owner_func_call_id"] == _boundary_ops(trace)[0].call.call_id
    assert boundary_rows[0]["capture_accounted"] is True
    # The interior detach was re-owned: no owner row of its own survives.
    assert not any(
        entry["owner_func_name"] == "detach" for entry in trace.health.completeness_decompositions
    )


def test_boundary_selector_and_row_are_wired() -> None:
    """The boundary is selectable by its transform kind and audited by the witness table."""

    model = _ForeignFunctionModel(_NumpyDoubleFunction)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(2, 4), save=tl.func_transform("autograd.Function"))
    saved = [op for op in trace._layer_list if op.has_saved_activation]
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
    types = [op.type for op in trace._layer_list]
    assert "mul" in types
    assert not any(op.call.is_func_transform for op in trace._layer_list)
    _assert_validates_clean(model, x)


class _ReturnInputFunction(torch.autograd.Function):
    """Identity Function whose forward hands back its input object, like a backward hook's."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged."""

        return x

    @staticmethod
    def backward(ctx: object, grad: torch.Tensor) -> torch.Tensor:
        """Pass the gradient through."""

        return grad


class _ReturnInputModel(nn.Module):
    """Doubles, applies the identity Function, and returns the result or its relu."""

    def __init__(self, function_output_is_model_output: bool) -> None:
        """Store whether the Function's output is returned as is."""

        super().__init__()
        self.function_output_is_model_output = function_output_is_model_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double, apply the identity Function, then maybe relu."""

        out = _ReturnInputFunction.apply(x * 2)
        return out if self.function_output_is_model_output else torch.relu(out)


@pytest.mark.parametrize("function_output_is_model_output", [True, False])
def test_function_returning_its_input_records_the_view_torch_makes(
    function_output_is_model_output: bool,
) -> None:
    """Torch hands back ``input.view_as(input)``; the trace records it with the input as parent."""

    model = _ReturnInputModel(function_output_is_model_output)
    x = torch.rand(3, requires_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(model, x)
    (view,) = [op for op in trace._layer_list if op.func_name == "view_as"]
    assert {trace[label].func_name for label in view.parents} == {"__mul__"}
    assert _boundary_ops(trace) == []
    _assert_validates_clean(model, x)


def test_plain_numpy_round_trip_is_not_collapsed() -> None:
    """Armed negative: without a Function nothing is collapsed into an ``autogradfunction`` op.

    The plain round trip is the foreign data round trip's territory
    (``tests/test_foreign_data_roundtrip.py``): the host constructor is one ``foreigndata``
    op whose parent is the departed tensor, the ``detach`` stays visible, and the host
    arithmetic in the gap makes validation fail with a typed reason instead of the former
    silent constant replay.
    """

    model = _PlainNumpyRoundTripModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    assert not any(op.type == "fromnumpy" for op in trace._layer_list)
    round_trips = [op for op in trace._layer_list if op.func_name == "foreigndata"]
    assert len(round_trips) == 1
    assert [record_type(trace[label]) for label in round_trips[0].parents] == ["detach"]
    assert trace.output_ops[0].input_ancestors
    assert any(op.type == "detach" for op in trace._layer_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, [x], input_kwargs={}) is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == CHECK_REPLAY
    assert failure.extra["reason"] == "foreign_data_roundtrip_unreplayable"


class _ForeignHandle:
    """A DGL-style opaque handle: hashing it fails like ``dgl._ffi.object.ObjectBase``.

    DGL's ``ObjectBase.__hash__`` calls ``_api_internal._raw_ptr``, which the shipped
    wheel does not register, so any ``hash()`` of a graph index raises AttributeError.
    """

    def __hash__(self) -> int:
        raise AttributeError("module 'dgl._api_internal' has no attribute '_raw_ptr'")


class _HandleDoubleFunction(torch.autograd.Function):
    """DGL SpMM shape: ``apply(x, handle)`` where the handle is not a tensor."""

    @staticmethod
    def forward(ctx, x, handle):  # type: ignore[override]
        assert isinstance(handle, _ForeignHandle)
        return torch.from_numpy(x.detach().numpy() * 2)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output * 2, None


class _HandleFunctionModel(nn.Module):
    """Linear, then a Function applied with an unhashable non-tensor argument."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.handle = _ForeignHandle()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _HandleDoubleFunction.apply(self.fc(x), self.handle)
        return torch.relu(y)


def test_foreign_function_with_unhashable_handle_argument() -> None:
    """A non-tensor ``apply`` argument is never hashed: the boundary records and replays it."""

    model = _HandleFunctionModel()
    x = torch.randn(3, 4)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, x)
    _assert_one_boundary(trace, _HandleDoubleFunction)
    _assert_validates_clean(model, x)


# --- Known foreign kernels: the FlashAttention 4 shape ------------------------------------


class _StandInKernelFunction(torch.autograd.Function):
    """FlashAttention 4 shape: a fresh ``torch.empty`` output filled below the dispatcher.

    The class reports the FlashAttention 4 interface module as its ``__module__``, so the
    known-foreign-kernel table matches it. The fill writes the bytes through the output's
    data pointer: no torch op, no numpy or dlpack export, nothing the dynamic detector or the
    dispatch census can see, exactly what a CuTe kernel reached through TVM FFI looks like.
    """

    __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE

    @staticmethod
    def forward(ctx, q, k, v):  # type: ignore[override]
        out = torch.empty(q.shape, dtype=torch.float32)
        values = array.array(
            "f",
            [
                a + b * c
                for a, b, c in zip(
                    q.detach().flatten().tolist(),
                    k.detach().flatten().tolist(),
                    v.detach().flatten().tolist(),
                    strict=True,
                )
            ],
        )
        ctypes.memmove(out.data_ptr(), values.buffer_info()[0], len(values) * values.itemsize)
        return out

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output, grad_output, grad_output


class _StandInAttentionModel(nn.Module):
    """Three projections feeding the stand-in kernel, then a consumer."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _StandInKernelFunction.apply(self.q_proj(x), self.k_proj(x), self.v_proj(x))
        return torch.relu(y)


def _stand_in_direct(model: _StandInAttentionModel, x: torch.Tensor) -> torch.Tensor:
    """Return the stand-in kernel's output for ``x`` outside capture."""

    return _StandInKernelFunction.apply(model.q_proj(x), model.k_proj(x), model.v_proj(x))


def test_known_foreign_kernel_function_is_one_boundary_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """A FlashAttention 4 Function is one replayable boundary op with its known exit."""

    monkeypatch.setattr(tc, "HAS_FLASH_ATTENTION_4_INTERFACE", True)
    model = _StandInAttentionModel()
    x = torch.randn(3, 4)

    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op") as record:
        trace = tl.trace(model, x)
    codes = [getattr(w.message, "fields", {}).get("code") for w in record]
    assert "foreign_kernel_boundary" in codes
    boundaries = _boundary_ops(trace)
    assert len(boundaries) == 1
    op = boundaries[0]
    assert op.call.is_func_transform is True
    assert op.call.func_transform_kind == FOREIGN_FUNCTION_TRANSFORM_KIND
    assert op.call.func_transform_config["function"] == _StandInKernelFunction.__qualname__
    assert op.call.func_transform_config["function_module"] == FLASH_ATTENTION_4_INTERFACE_MODULE
    assert op.call.func_transform_config["exits"] == [FLASH_ATTENTION_4_EXIT]
    assert is_foreign_function_recorder(op.func)
    # The allocation and the detaches inside the kernel wrapper are not in the graph.
    assert not any(op.type in {"empty", "detach", "flatten"} for op in trace._layer_list)
    assert sorted(record_type(trace[label]) for label in op.parents) == ["linear"] * 3
    assert [record_type(trace[label]) for label in op.children] == ["relu"]
    torch.testing.assert_close(op.out, _stand_in_direct(model, x))

    # Forward replay and the dispatch census through the boundary validate, nothing exempted.
    _assert_validates_clean(model, x)


def test_known_foreign_kernel_owns_its_interior_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the census armed, the allocation and detaches inside the call belong to the op."""

    monkeypatch.setattr(tc, "HAS_FLASH_ATTENTION_4_INTERFACE", True)
    wrap_torch(completeness_witness=True)
    model = _StandInAttentionModel()
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(3, 4))

    assert trace.provenance.verified is True
    assert trace.completeness_diagnostics == []
    boundary_rows = [
        entry
        for entry in trace.health.completeness_decompositions
        if entry["owner_wrapper"] == FOREIGN_FUNCTION_WRAPPER_NAME
    ]
    assert len(boundary_rows) == 1
    assert boundary_rows[0]["owner_func_call_id"] == _boundary_ops(trace)[0].call.call_id
    assert boundary_rows[0]["capture_accounted"] is True
    assert not any(
        entry["owner_func_name"] in {"empty", "detach"}
        for entry in trace.health.completeness_decompositions
    )


def test_known_foreign_kernel_needs_its_capability_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: with the flag off the same Function is ordinary, and the tripwire still fires.

    Nothing is collapsed, the ``empty`` allocation stands in for the attention, and forward
    validation fails because replay cannot reproduce a fill it never saw: the boundary closes
    the gap, it does not hide the census.
    """

    monkeypatch.setattr(tc, "HAS_FLASH_ATTENTION_4_INTERFACE", False)
    assert known_foreign_kernel_exits(_StandInKernelFunction) == ()
    model = _StandInAttentionModel()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    assert _boundary_ops(trace) == []
    assert any(op.type == "empty" for op in trace._layer_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, [x], input_kwargs={}) is False
    assert last_validation_failure() is not None


def test_known_foreign_kernel_table_matches_the_interface_module_only() -> None:
    """Only Functions of the FlashAttention 4 interface module (or its submodules) match."""

    class _Elsewhere(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            return x

    class _Submodule(torch.autograd.Function):
        __module__ = FLASH_ATTENTION_4_INTERFACE_MODULE + ".sm90"

        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            return x

    assert known_foreign_kernel_exits(_Elsewhere) == ()
    assert known_foreign_kernel_exits(_Submodule) == (
        (FLASH_ATTENTION_4_EXIT,) if tc.HAS_FLASH_ATTENTION_4_INTERFACE else ()
    )
    assert tc._probe_flash_attention_4_interface() is tc.HAS_FLASH_ATTENTION_4_INTERFACE  # noqa: SLF001


def test_apply_slot_shapes_both_wrap() -> None:
    """Both shapes of ``Function.__dict__["apply"]`` yield a ``(cls, *args)`` callable."""

    def plain(cls: type, value: int) -> tuple[type, int]:
        return cls, value

    assert apply_slot_function(classmethod(plain)) is plain
    # A C classmethod descriptor (torch 2.14's slot shape) binds to the given class.
    descriptor = dict.__dict__["fromkeys"]
    call = apply_slot_function(descriptor)
    assert call is not None
    assert call(dict, "ab", 0) == {"a": 0, "b": 0}
    assert apply_slot_function(staticmethod(plain)) is None
    assert apply_slot_function(None) is None


def test_apply_hook_is_installed_on_this_torch() -> None:
    """The running torch's ``apply`` slot is wrapped, under its own name, while capture is on."""

    assert tc.HAS_FUNCTION_APPLY_SLOT is True
    wrap_torch()
    installed = torch.autograd.Function.__dict__["apply"]
    assert isinstance(installed, classmethod)
    assert installed.__func__.__name__ == "apply"
    model = _ForeignFunctionModel(_NumpyDoubleFunction)
    with pytest.warns(TorchLensWarning, match="foreign-kernel boundary op"):
        trace = tl.trace(model, torch.randn(2, 4))
    assert len(_boundary_ops(trace)) == 1
