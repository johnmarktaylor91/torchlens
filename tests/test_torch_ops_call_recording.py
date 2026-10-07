"""Direct ``torch.ops.*`` calls inside forward are recorded with sources and validate.

A raw ``torch.ops.aten`` call, a ``torch.library.custom_op`` and an operator a C++
extension registered with ``TORCH_LIBRARY`` all reach the dispatcher through the
``torch._ops`` call classes, past every namespace wrapper. TorchLens now records such a
call as an ordinary op (parents from its tensor arguments, the operator object as the
replay callable), so the forward graph keeps the edge and validation replays the
operator. Recording is not an exemption: a source-less tensor handed to a recorded
operator still fails ``source_provenance``.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.validation import last_validation_failure
from torchlens.validation._source_provenance import source_provenance_gaps

_GLOBAL_TABLE = torch.randn(4)
_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward"))


def _assert_recorded_and_valid(model: nn.Module, func_name: str) -> None:
    x = torch.randn(3, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x)
    assert func_name in [op.func_name for op in trace.ops]
    assert not [w for w in caught if "no graph/source provenance" in str(w.message)]
    recorded = next(op for op in trace.ops if op.func_name == func_name)
    assert recorded.parents, recorded
    assert not recorded.unattributed_tensor_args
    assert _validate(model, x), last_validation_failure()


class _RawAtenTimesOne(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.tanh(self.fc(x)) * 1.0


class _RawAtenOverloadReturn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.sigmoid.default(self.fc(x))


class _Wrap(nn.Module):
    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.child(x) + 1.0


@pytest.mark.smoke
def test_raw_aten_packet_call_is_recorded_and_validates() -> None:
    """``torch.ops.aten.tanh(x) * 1.0`` used to fail ``bfs_completeness``."""

    _assert_recorded_and_valid(_RawAtenTimesOne(), "tanh")


def test_raw_aten_overload_module_return_is_recorded_and_validates() -> None:
    """A submodule returning ``torch.ops.aten.sigmoid.default(...)`` is no longer adopted."""

    _assert_recorded_and_valid(_Wrap(_RawAtenOverloadReturn()), "sigmoid")


if _HAS_CUSTOM_OP:

    @torch.library.custom_op("tltest_r10::scaled_tanh", mutates_args=())
    def _scaled_tanh(x: torch.Tensor, scale: float) -> torch.Tensor:
        # Unpatchable C calls only, like a kernel written in C++: a Python body built
        # from wrapped torch functions would be captured op by op anyway.
        vf = torch._C._VariableFunctions
        return vf.mul(vf.tanh(x), scale)


class _CustomOpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _scaled_tanh(self.fc(x), 2.0) + x


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_library_custom_op_is_recorded_and_validates() -> None:
    _assert_recorded_and_valid(_CustomOpModel(), "scaled_tanh")


_CPP_SOURCE = """
#include <torch/library.h>
#include <ATen/ATen.h>

at::Tensor tl_r10_cube(const at::Tensor& x) { return x * x * x; }

TORCH_LIBRARY(tltest_r10_ext, m) { m.def("cube(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(tltest_r10_ext, CPU, m) { m.impl("cube", &tl_r10_cube); }
"""


def _cpp_toolchain_available() -> bool:
    return shutil.which("c++") is not None and shutil.which("ninja") is not None


class _CppExtModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.tltest_r10_ext.cube(self.fc(x)) - x


@pytest.mark.slow
@pytest.mark.skipif(not _cpp_toolchain_available(), reason="needs a C++ compiler and ninja")
def test_cpp_extension_torch_library_op_is_recorded_and_validates(tmp_path: Path) -> None:
    """A ``load_inline`` extension op registered with ``TORCH_LIBRARY`` is recorded."""

    from torch.utils.cpp_extension import load_inline

    load_inline(
        name="tltest_r10_ext",
        cpp_sources=[_CPP_SOURCE],
        is_python_module=False,
        build_directory=str(tmp_path),
    )
    _assert_recorded_and_valid(_CppExtModel(), "cube")


class _GlobalIntoRawAten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mul.Tensor(x, _GLOBAL_TABLE) + 1.0


class _ClosureIntoCustomOp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        table = torch.randn(4)
        self.forward = lambda x: torch.ops.aten.add.Tensor(x, table) * 2.0  # type: ignore[method-assign]


@pytest.mark.parametrize(
    "build", [_GlobalIntoRawAten, _ClosureIntoCustomOp], ids=["global", "closure"]
)
def test_source_less_argument_to_a_recorded_op_still_fails(build: type[nn.Module]) -> None:
    """Recording the operator records its arguments' gaps too; nothing is exempted."""

    assert not _validate(build(), torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == "source_provenance", failure
    assert failure.extra["reasons"] == ["unattributed_tensor_args"], failure


def test_recorders_are_wrapped_epoch_scoped() -> None:
    """``unwrap_torch`` restores the pristine ``torch._ops`` call classes; ``wrap_torch`` re-arms."""

    marker = "__tl_torch_ops_recorder__"
    unwrap_torch()
    try:
        for cls in (torch._ops.OpOverloadPacket, torch._ops.OpOverload):
            assert not getattr(cls.__dict__["__call__"], marker, False)
        assert torch.allclose(torch.ops.aten.tanh(torch.zeros(2)), torch.zeros(2))
    finally:
        wrap_torch()
    for cls in (torch._ops.OpOverloadPacket, torch._ops.OpOverload):
        assert getattr(cls.__dict__["__call__"], marker, False)


# --- Boundary review fixes: operators that mutate an argument and return nothing ----

if _HAS_CUSTOM_OP:
    _VF_MUL_ = torch._C.TensorBase.mul_

    @torch.library.custom_op("tltest_r11::scale_", mutates_args=("x",))
    def _scale_(x: torch.Tensor, scale: float) -> None:
        _VF_MUL_(x, scale)  # an unpatchable C write, like a C++ kernel

    @torch.library.custom_op("tltest_r11::doubled", mutates_args=("x",))
    def _doubled(x: torch.Tensor) -> None:
        _VF_MUL_(x, 2.0)

    @torch.library.custom_op("tltest_r11::write_into", mutates_args=("dst",))
    def _write_into(src: torch.Tensor, dst: torch.Tensor) -> None:
        torch._C.TensorBase.copy_(dst, src)

    @torch.library.custom_op("tltest_r15::peek_scale_", mutates_args=("x",))
    def _peek_scale_(x: torch.Tensor, scale: float) -> None:
        # A wrapped metadata read that logs nothing, like torch 2.7's Python
        # ``check_aliasing_constraint`` calling ``untyped_storage()`` in every kernel.
        x.untyped_storage()
        _VF_MUL_(x, scale)

    @torch.library.custom_op("tltest_r15::body_mul_", mutates_args=("x",))
    def _body_mul_(x: torch.Tensor) -> None:
        x.mul_(2.0)  # a wrapped in-place op: the body is captured op by op


class _MutatingCustomOpModel(nn.Module):
    def __init__(self, op_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.op_name = op_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x).clone()
        if self.op_name == "scale_":
            _scale_(y, 2.0)
        elif self.op_name == "doubled":
            _doubled(y)
        elif self.op_name == "peek_scale_":
            _peek_scale_(y, 2.0)
        elif self.op_name == "body_mul_":
            _body_mul_(y)
        elif self.op_name == "foreach":
            torch.ops.aten._foreach_mul_.Scalar([y], 2.0)
        else:
            _write_into(x * 3.0, y)
        return y + 1


def _unrecorded_mutation_rows(trace: Any) -> list[dict[str, object]]:
    return [
        row
        for row in trace.annotations.get("capture_advisories", [])
        if row["kind"] == "unrecorded_operator_mutation"
    ]


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
@pytest.mark.parametrize("op_name", ["scale_", "doubled"])
def test_receiver_mutating_custom_op_is_recorded_in_place(op_name: str) -> None:
    """``mutates_args`` returning None used to vanish: the add read the doubled value."""

    model = _MutatingCustomOpModel(op_name)
    _assert_recorded_and_valid(model, op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    mutation = next(op for op in trace.ops if op.func_name == op_name)
    add = next(op for op in trace.ops if op.func_name == "__add__")
    mutation_layer = mutation.label.split(":")[0]
    assert add.parents == (mutation_layer,), (mutation_layer, add.parents)
    assert not _unrecorded_mutation_rows(trace)


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_receiver_op_reading_metadata_inside_is_still_recorded_in_place() -> None:
    """A wrapped call that logs nothing inside the operator must not drop the write.

    Torch 2.7's custom-op backend kernel reads ``untyped_storage()`` (a wrapped method)
    after every call; that nested call cleared the bottom-level barcode, so the in-place
    record of the already-labelled receiver was skipped and the add read the clone.
    """

    model = _MutatingCustomOpModel("peek_scale_")
    _assert_recorded_and_valid(model, "peek_scale_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    mutation = next(op for op in trace.ops if op.func_name == "peek_scale_")
    add = next(op for op in trace.ops if op.func_name == "__add__")
    assert add.parents == (mutation.label.split(":")[0],), add.parents
    assert not _unrecorded_mutation_rows(trace)


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_receiver_op_whose_body_logs_ops_keeps_the_body_record() -> None:
    """Narrowness: a body whose own wrapped ops were logged stays the record, once."""

    model = _MutatingCustomOpModel("body_mul_")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    names = [op.func_name for op in trace.ops]
    assert "mul_" in names, names
    assert "body_mul_" not in names, names
    body_op = next(op for op in trace.ops if op.func_name == "mul_")
    add = next(op for op in trace.ops if op.func_name == "__add__")
    assert add.parents == (body_op.label.split(":")[0],), add.parents
    assert _validate(model, torch.randn(3, 4)), last_validation_failure()


@pytest.mark.smoke_cells("test_unrecordable_mutating_operator_fails_validation[write_into]")
@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
@pytest.mark.parametrize("op_name", ["write_into", "foreach"])
def test_unrecordable_mutating_operator_fails_validation(op_name: str) -> None:
    """A write to a non-first or list argument is disclosed, never a silent pass."""

    model = _MutatingCustomOpModel(op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    rows = _unrecorded_mutation_rows(trace)
    expected = "_foreach_mul_" if op_name == "foreach" else op_name
    assert [row for row in rows if expected in str(row["message"])], rows
    gap_reasons = {gap[0] for gap in source_provenance_gaps(trace)}
    assert "unrecorded_operator_mutation" in gap_reasons, gap_reasons
    assert not _validate(model, torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None
    if op_name == "write_into":  # the foreach packet fails bfs_completeness first
        assert failure.check == "source_provenance", failure
        assert "unrecorded_operator_mutation" in failure.extra["reasons"], failure


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_unreadable_schema_none_return_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None-returning operator whose schema cannot be read is disclosed."""

    from torchlens.backends.torch import _torch_ops_calls

    monkeypatch.setattr(_torch_ops_calls, "_DECORATED_BY_OP", {})
    # A fabricated unreadable schema is not a real call shape: keep it out of the
    # ArgSpec usage audit (test_arg_positions), which would see ``scale_`` counted but
    # never extracted (nothing is logged for a disclosed None return).
    monkeypatch.setattr(_state, "_collect_usage_stats", False)
    monkeypatch.setattr(_torch_ops_calls, "_overload_schemas", lambda op: None)
    model = _MutatingCustomOpModel("scale_")
    assert not _validate(model, torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == "source_provenance", failure
    assert "unrecorded_operator_mutation" in failure.extra["reasons"], failure


# --- Re-review fixes: a write that no return aliases --------------------------------

if _HAS_CUSTOM_OP:

    @torch.library.custom_op("tltest_r13::mut_ret", mutates_args=("x",))
    def _mut_ret(x: torch.Tensor) -> torch.Tensor:
        _VF_MUL_(x, 2.0)  # state update, then a fresh output (KV-cache, running stats)
        return torch._C._VariableFunctions.add(x, 0.0)

    @torch.library.custom_op("tltest_r13::unknown_ret", mutates_args="unknown")
    def _unknown_ret(x: torch.Tensor) -> torch.Tensor:
        _VF_MUL_(x, 2.0)
        return torch._C._VariableFunctions.add(x, 1.0)

    @torch.library.custom_op("tltest_r13::buf_ret", mutates_args=("buf",))
    def _buf_ret(x: torch.Tensor, buf: torch.Tensor) -> torch.Tensor:
        torch._C.TensorBase.add_(buf, 1.0)
        return torch._C._VariableFunctions.mul(x, 2.0)

    _R13_LIB = torch.library.Library("tltest_r13", "FRAGMENT")
    _R13_LIB.define("inplace(Tensor(a!) x) -> Tensor(a!)")

    def _r13_inplace(x: torch.Tensor) -> torch.Tensor:
        _VF_MUL_(x, 2.0)
        return x

    _R13_LIB.impl("inplace", _r13_inplace, "CPU")


class _UnreturnedWriteModel(nn.Module):
    def __init__(self, op_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.op_name = op_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x).clone()
        if self.op_name == "mut_ret":
            return y + _mut_ret(y)
        if self.op_name == "unknown_ret":
            return y + _unknown_ret(y)
        buf = self.fc(x).clone()
        return _buf_ret(y, buf) + buf


class _AliasedWriteModel(nn.Module):
    def __init__(self, op_name: str) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.op_name = op_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x).clone()
        if self.op_name == "add_":
            torch.ops.aten.add_.Tensor(y, 1.0)
            return y + 1
        if self.op_name == "inplace":
            torch.ops.tltest_r13.inplace(y)
            return y + 1
        out = torch.empty(3, 4)
        torch.ops.aten.tanh.out(y.detach(), out=out)
        return out + y


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
@pytest.mark.parametrize("op_name", ["mut_ret", "unknown_ret", "buf_ret"])
def test_write_no_return_aliases_fails_validation(op_name: str) -> None:
    """A write beside a fresh return used to validate True with a pre-mutation graph."""

    model = _UnreturnedWriteModel(op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    rows = _unrecorded_mutation_rows(trace)
    assert [row for row in rows if op_name in str(row["message"])], rows
    assert not _validate(model, torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == "source_provenance", failure
    assert "unrecorded_operator_mutation" in failure.extra["reasons"], failure


@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
@pytest.mark.parametrize("op_name", ["add_", "out", "inplace"])
def test_write_a_return_aliases_is_still_recorded(op_name: str) -> None:
    """In-place and ``out=`` operators return the tensor they write: recorded, valid."""

    model = _AliasedWriteModel(op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    assert not _unrecorded_mutation_rows(trace)
    func_name = "tanh" if op_name == "out" else op_name
    writer = next(op for op in trace.ops if op.func_name == func_name)
    add = next(op for op in trace.ops if op.func_name == "__add__")
    assert writer.label.split(":")[0] in add.parents, (writer.label, add.parents)
    assert _validate(model, torch.randn(3, 4)), last_validation_failure()
