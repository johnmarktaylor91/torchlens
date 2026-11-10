"""Admission rules of the wrapper fast lane (``backends/torch/_call_plan.py``).

The fast lane carries only out-of-place, single-output, non-fused calls of
C-level torch callables with plain tensor arguments, outside transform, compile
and autocast regions. Each test here names one rule: the call it describes must
run the wrapper's general body (no fast commit), and the capture must record it
exactly as it does with the fast lane switched off.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import _call_plan, wrappers
from torchlens.backends.torch._fused_norm_writes import _FUSED_MUTATOR_NAMES
from torchlens.backends.torch.buffer_writes import (
    BufferWriteTracker,
    _could_mutate,
    peek_buffer_write_tracker,
)
from torchlens.intervention._runtime_probes import _is_inplace_style_func_name


class _Forward(nn.Module):
    """A module whose forward is one supplied function."""

    def __init__(self, body: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _fast_names(
    monkeypatch: pytest.MonkeyPatch, body: Callable[[torch.Tensor], torch.Tensor] | nn.Module
) -> tuple[list[str], Any]:
    """Trace ``body`` and return the names the fast lane committed, with the trace."""

    names: list[str] = []
    real_commit = _call_plan.commit_fast_call

    def spy(emit: Any, trace: Any, call: tuple[Any, ...], *rest: Any) -> Any:
        names.append(call[1])
        return real_commit(emit, trace, call, *rest)

    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "commit_fast_call", spy)
        model = body if isinstance(body, nn.Module) else _Forward(body)
        trace = tl.trace(model, torch.ones(2, 3))
    return names, trace


def _general_trace(
    monkeypatch: pytest.MonkeyPatch, body: Callable[[torch.Tensor], torch.Tensor]
) -> Any:
    """Trace ``body`` with every call refused by the admission (the general body only)."""

    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "admit_call", lambda *_: None)
        return tl.trace(_Forward(body), torch.ones(2, 3))


def _op_rows(trace: Any) -> list[tuple[Any, ...]]:
    """Return the per-op facts a fast and a general capture of one forward must share."""

    return [
        (
            op.label,
            op.func_name,
            tuple(op.shape) if op.shape is not None else None,
            str(op.dtype),
            tuple(op.parents),
            op.is_inplace,
        )
        for op in trace.ops
    ]


def _assert_same_capture(
    monkeypatch: pytest.MonkeyPatch, body: Callable[[torch.Tensor], torch.Tensor], trace: Any
) -> None:
    """Assert ``trace`` records the same ops as a capture with the fast lane off."""

    assert _op_rows(trace) == _op_rows(_general_trace(monkeypatch, body))


def test_plain_out_of_place_calls_take_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The admitted class: C-level, out-of-place, one new plain tensor out."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + 1).to(torch.float64)

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__add__", "relu", "to"]
    _assert_same_capture(monkeypatch, body, trace)


def test_positional_inplace_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """``F.relu(x, True)``: a positional ``inplace=True`` never reaches the fast lane."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x * 2, True)

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__mul__"]
    assert any(op.is_inplace for op in trace.ops)
    _assert_same_capture(monkeypatch, body, trace)


def test_fused_write_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fused running-statistic writer (``batch_norm`` in training) gets no plan."""

    class Norm(nn.Module):
        """A training-mode batch norm over the module's own running statistics."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.zeros(3))
            self.register_buffer("var", torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.batch_norm(x, None, None, self.mean, self.var, True, 0.1, 1e-5, False)

    names, trace = _fast_names(monkeypatch, Norm())
    assert names == []
    assert any(op.func_name == "batch_norm" for op in trace.ops)
    for name in _FUSED_MUTATOR_NAMES:
        func = getattr(torch, name, None)
        assert func is None or _call_plan.build_call_plan(func, name, None) is None


def test_to_with_the_same_dtype_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """``x.to(x.dtype)`` returns its input: the post-call check refuses the output."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return (x + 1).to(torch.float32)

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__add__"]
    _assert_same_capture(monkeypatch, body, trace)


def test_contiguous_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """``contiguous`` on a contiguous tensor returns its input: refused after the call."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return (x + 1).contiguous()

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__add__"]
    _assert_same_capture(monkeypatch, body, trace)


def test_autocast_region_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call inside an enabled autocast region is refused by the per-call guard."""

    def body(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = x * 3
        return y - 1

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__sub__"]
    _assert_same_capture(monkeypatch, body, trace)


def test_transform_begun_inside_the_forward_runs_the_general_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A functorch transform entered mid-forward is caught before the admission."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return torch.func.vmap(lambda row: row * 2)(x) + 1

    names, trace = _fast_names(monkeypatch, body)
    assert "__mul__" not in names
    _assert_same_capture(monkeypatch, body, trace)


def test_compile_region_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """While Dynamo traces the frame the compile guard runs and the fast lane does not."""

    compiling = [False]

    def body(x: torch.Tensor) -> torch.Tensor:
        shifted = x + 1
        compiling[0] = True
        try:
            hidden = torch.relu(shifted)
        finally:
            compiling[0] = False
        return hidden * 2

    monkeypatch.setattr(wrappers, "_is_inside_dynamo_compilation", lambda: compiling[0])
    with pytest.warns(UserWarning, match="torch.compile"):
        names, trace = _fast_names(monkeypatch, body)
    assert "__add__" in names
    assert "relu" not in names
    assert all(op.func_name != "relu" for op in trace.ops)


def test_tensor_subclass_argument_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tensor subclass argument is not a plain tensor: the admission refuses it."""

    class Tagged(torch.Tensor):
        """A plain tensor subclass."""

    def body(x: torch.Tensor) -> torch.Tensor:
        return torch.neg(x.as_subclass(Tagged)).as_subclass(torch.Tensor) * 2

    names, trace = _fast_names(monkeypatch, body)
    assert "neg" not in names
    _assert_same_capture(monkeypatch, body, trace)


def test_no_planned_name_is_in_the_inplace_family() -> None:
    """Every planned ``torch.Tensor`` callable is out-of-place by name and by schema.

    The plan's ``name_could_mutate`` equals the buffer tracker's own verdict, so
    the fast lane clears the storage metadata cache exactly when today's
    pre-call snapshot does.
    """

    planned = 0
    for name in dir(torch.Tensor):
        func = getattr(torch.Tensor, name, None)
        func = _state._decorated_to_orig.get(id(func), func)
        plan = _call_plan.build_call_plan(func, name, None) if callable(func) else None
        if plan is None:
            continue
        planned += 1
        assert not _is_inplace_style_func_name(name), name
        assert name not in _FUSED_MUTATOR_NAMES, name
        assert plan.name_could_mutate == _could_mutate(name), name
    assert planned > 50


def test_buffer_tracker_shape_the_fast_lane_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracker the fast lane inspects is the exact class with the three tables it reads."""

    _, trace = _fast_names(monkeypatch, lambda x: x + 1)
    tracker = peek_buffer_write_tracker(trace)
    if tracker is not None:
        assert type(tracker) is BufferWriteTracker
    for field in ("address_to_tensor", "storage_key_to_addresses", "_pending_lazy_buffer_ids"):
        assert field in BufferWriteTracker.__init__.__code__.co_names
