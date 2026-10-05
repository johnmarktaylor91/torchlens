"""Rescue recovery oracle vs ``TorchFunctionMode`` dunder respelling.

The rescue re-run arms a ``TorchFunctionMode``; with a mode active torch
dispatches C-level operator dunders as their named method, so the mode-free
primary logs ``__radd__`` / ``__truediv__`` / ``__iadd__`` while the rescued
trace logs ``add`` / ``div`` / ``add_``. The two-sided op-count oracle read
those as LOST ops and refused a rescue that had recovered the escaped op,
returning the broken primary as ``escape_rescue_unrecovered``. These tests pin
the canonicalization (``rescue._canonical_op_name``) against torch itself, end
to end through ``tl.trace``, and keep the oracle red for a genuine loss.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.rescue import (
    _MODE_RESPELLED_DUNDERS,
    _canonical_op_name,
    capture_with_rescue,
)
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

_FLOAT_BINARY = [
    "add",
    "sub",
    "mul",
    "truediv",
    "div",
    "floordiv",
    "mod",
    "pow",
    "radd",
    "rsub",
    "rmul",
    "rtruediv",
    "rdiv",
    "rfloordiv",
    "rmod",
    "rpow",
    "iadd",
    "isub",
    "imul",
    "itruediv",
    "idiv",
    "ifloordiv",
    "imod",
    "ipow",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
]
_BOOL_BINARY = ["and", "or", "xor", "rand", "ror", "rxor", "iand", "ior", "ixor"]
_INT_BINARY = ["lshift", "rshift", "rlshift", "rrshift", "ilshift", "irshift"]
_UNARY = ["neg", "abs", "pos"]


class _RecordingMode(TorchFunctionMode):
    """Record the name of every function torch hands the mode."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.names.append(str(getattr(func, "__name__", "")))
        return func(*args, **(kwargs or {}))


def _dunder_call(stem: str) -> Callable[[], Any]:
    """Return a thunk invoking ``Tensor.__<stem>__`` with fitting operands."""

    name = f"__{stem}__"
    args: tuple[Any, ...]
    if stem in _BOOL_BINARY:
        receiver, args = torch.tensor([True, False]), (True,)
    elif stem in _INT_BINARY:
        receiver, args = torch.tensor([1, 2]), (1,)
    elif stem in _UNARY:
        receiver, args = torch.tensor([1.5, -2.5]), ()
    else:
        receiver, args = torch.tensor([1.5, 2.5]), (2.0,)
    bound = getattr(receiver, name)
    return lambda: bound(*args)


@pytest.mark.parametrize("stem", _FLOAT_BINARY + _BOOL_BINARY + _INT_BINARY + _UNARY)
def test_canonical_name_matches_what_torch_hands_a_mode(stem: str) -> None:
    """Every operator dunder canonicalizes to what the rescue's mode logs.

    Derived from torch, not asserted from the table: whatever the running
    torch passes to a ``TorchFunctionMode`` for the dunder must land on the
    dunder's own canonical name.
    """

    if not hasattr(torch.Tensor, f"__{stem}__"):
        pytest.skip(f"torch has no Tensor.__{stem}__")
    call = _dunder_call(stem)
    with _RecordingMode() as mode:
        call()
    seen = [name for name in mode.names if name not in {"untyped_storage", "__get__"}]
    assert seen, f"__{stem}__ never reached the mode"
    assert {_canonical_op_name(name) for name in seen} == {_canonical_op_name(f"__{stem}__")}


def test_respelling_rows_keep_operation_identity() -> None:
    """Rows never merge reflected-noncommutative or in-place ops into others."""

    assert _canonical_op_name("__rsub__") == "rsub" != _canonical_op_name("sub")
    assert _canonical_op_name("__rpow__") == "rpow" != _canonical_op_name("pow")
    for dunder, spelling in _MODE_RESPELLED_DUNDERS.items():
        names_inplace_method = spelling.endswith("_") and not spelling.startswith("__")
        assert dunder.startswith("__i") == names_inplace_method, dunder


def _stub_trace(op_names: list[str], *, signal: bool = False) -> Any:
    """Build a minimal trace stub for direct driver tests."""

    return types.SimpleNamespace(
        escape_diagnostics=[],
        _had_unattributed_tensor_args=signal,
        ops=[types.SimpleNamespace(func_name=name) for name in op_names],
        modules=[],
        capture_verification_reason=None,
    )


def test_respelled_dunders_do_not_refuse_a_recovery() -> None:
    """Primary ``__radd__``/``__iadd__`` vs rescued ``add``/``add_`` is no loss."""

    primary = _stub_trace(["__radd__", "__rmul__", "__truediv__", "__iadd__", "relu"], signal=True)
    rescued = _stub_trace(["add", "mul", "div", "add_", "relu", "relu"])
    traces = iter([primary, rescued])

    result = capture_with_rescue(lambda: next(traces))

    assert result is rescued
    assert result.capture_verification_reason == "mode_rescue_rerun"
    assert result.rescue_rerun["recovered_ops"] == ("relu",)
    assert result.rescue_rerun["lost_ops"] == ()


@pytest.mark.parametrize(
    ("primary_ops", "rescued_ops", "lost"),
    [
        (["__radd__", "__truediv__", "relu", "tanh"], ["add", "div", "relu", "relu"], ("tanh",)),
        (["__rsub__", "relu"], ["sub", "relu", "relu"], ("rsub",)),
    ],
)
def test_genuine_loss_still_refuses_the_rescue(
    primary_ops: list[str], rescued_ops: list[str], lost: tuple[str, ...]
) -> None:
    """A rescued trace that really misses an op keeps the primary."""

    primary = _stub_trace(primary_ops, signal=True)
    traces = iter([primary, _stub_trace(rescued_ops)])

    result = capture_with_rescue(lambda: next(traces))

    assert result is primary
    assert result.capture_verification_reason == "escape_rescue_unrecovered"
    assert result.rescue_rerun["recovered"] is False
    assert result.rescue_rerun["lost_ops"] == lost


@pytest.fixture()
def raw_cos() -> Iterator[Any]:
    """A pristine pre-wrap ``torch.cos`` reference, rewrapping afterwards."""

    unwrap_torch()
    raw = torch.cos
    assert not is_decorated_function(raw)
    try:
        yield raw
    finally:
        wrap_torch()


def _arith(y: torch.Tensor) -> torch.Tensor:
    return (1 + y) + (2 * y) + (y / 3) + (3 - y) + (2**y) + (3 / y)


def _inplace(y: torch.Tensor) -> torch.Tensor:
    z = y.clone()
    z += 1
    z -= 0.5
    z *= 2
    z /= 2
    z //= 1
    z %= 5
    return z + y % 2


def _bitwise(y: torch.Tensor) -> torch.Tensor:
    b = y > 0.5
    return (True & b) | (False | b) | (False ^ b)


@pytest.mark.parametrize("body", [_arith, _inplace, _bitwise], ids=["arith", "inplace", "bitwise"])
def test_stale_ref_rescue_with_operators_returns_the_rescued_trace(
    raw_cos: Any, body: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    """End to end: a stale pre-wrap ``cos`` feeding operator dunders is recovered."""

    wrap_torch()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return body(raw_cos(v))

    trace = tl.trace(Model(), torch.tensor([0.25, 0.5, 0.75]))

    info = trace.rescue_rerun
    assert info is not None and info["recovered"] is True, info
    assert info["lost_ops"] == ()
    assert trace.capture_verification_reason == "mode_rescue_rerun"
    assert "cos" in [op.func_name for op in trace.ops]
