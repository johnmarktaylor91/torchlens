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
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.validation import last_validation_failure

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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
@pytest.mark.parametrize("op_name", ["write_into", "foreach"])
def test_unrecordable_mutating_operator_fails_validation(op_name: str) -> None:
    """A write to a non-first or list argument is disclosed, never a silent pass."""

    model = _MutatingCustomOpModel(op_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.randn(3, 4))
    assert _unrecorded_mutation_rows(trace)
    assert not _validate(model, torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None


@pytest.mark.smoke
@pytest.mark.skipif(not _HAS_CUSTOM_OP, reason="torch.library.custom_op needs torch>=2.4")
def test_unreadable_schema_none_return_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None-returning operator whose schema cannot be read is disclosed."""

    from torchlens.backends.torch import _torch_ops_calls

    monkeypatch.setattr(_torch_ops_calls, "_DECORATED_BY_OP", {})
    monkeypatch.setattr(_torch_ops_calls, "_overload_schemas", lambda op: None)
    model = _MutatingCustomOpModel("scale_")
    assert not _validate(model, torch.randn(3, 4))
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check == "source_provenance", failure
    assert "unrecorded_operator_mutation" in failure.extra["reasons"], failure
