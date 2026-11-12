"""Session facts of the torch wrapper (``backends/torch/_wrapper_session.py``).

The wrapper reads the facts that cannot change inside one capture session once,
at the session's first logged call, and keys them by the capture context's
fingerprint and the trace. These tests pin the soundness side of that cache:

* a session fact changed between two captures is seen by the next capture
  (container structure, the observability-session reader, autocast);
* facts that can change inside a forward stay per-call reads (the observability
  session, the grad mode carried in the autocast snapshot);
* equal autocast snapshots are one object inside a trace and never shared by two
  traces.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _op_markers, _wrapper_session, wrappers


class _StackModel(nn.Module):
    """Consume a list at an op boundary."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Stack two tensors, then add one."""

        return torch.add(torch.stack([left, right]), 1)


class _FlagModel(nn.Module):
    """Two ops with a hook between them that runs a callback mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        self.between: Any = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add, run the callback, multiply."""

        y = torch.add(x, 1)
        if self.between is not None:
            self.between()
        return torch.mul(y, 2)


class _NoGradInside(nn.Module):
    """One op under ``torch.no_grad`` between two grad-mode ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add, then multiply under no_grad, then subtract."""

        y = torch.add(x, 1)
        with torch.no_grad():
            z = torch.mul(y, 2)
        return torch.sub(z, 3)


def _ops(trace: Any, name: str) -> list[Any]:
    return [op for op in trace._layer_list if op.func_name == name]


def _stack_containers(structure: bool) -> Any:
    trace = tl.trace(
        _StackModel(),
        (torch.tensor([1.0]), torch.tensor([2.0])),
        capture=tl.options.CaptureOptions(capture_container_structure=structure),
    )
    return next(op for op in trace.ops if op.func_name == "stack").input_containers


def test_container_structure_change_between_captures_is_seen() -> None:
    """Switching container structure on, then off, takes effect on the next capture."""

    assert _stack_containers(False) == ()
    assert _stack_containers(True) != ()
    assert _stack_containers(False) == ()


def _spy_markers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    pushed: list[str] = []
    original = wrappers._push_op_markers

    def spy(trace: Any, func_name: str, func_call_id: int) -> tuple[bool, Any]:
        pushed.append(func_name)
        return original(trace, func_name, func_call_id)

    monkeypatch.setattr(wrappers, "_push_op_markers", spy)
    return pushed


def test_session_reader_change_between_captures_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A different observability-session reader is used from the next capture on."""

    pushed = _spy_markers(monkeypatch)
    model, x = _FlagModel(), torch.ones(2)
    monkeypatch.setattr(_op_markers, "_active_session_fn", lambda: None)
    tl.trace(model, x)
    assert pushed == []
    session = SimpleNamespace(profiler=None)
    monkeypatch.setattr(_op_markers, "_active_session_fn", lambda: session)
    tl.trace(model, x)
    assert "add" in pushed and "mul" in pushed
    pushed.clear()
    monkeypatch.setattr(_op_markers, "_active_session_fn", lambda: None)
    tl.trace(model, x)
    assert pushed == []


def test_session_opened_mid_forward_is_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """The observability session itself is read on every call, not once per capture."""

    pushed = _spy_markers(monkeypatch)
    active: list[Any] = [None]
    monkeypatch.setattr(_op_markers, "_active_session_fn", lambda: active[0])
    model = _FlagModel()

    def open_session() -> None:
        active[0] = SimpleNamespace(profiler=None)

    model.between = open_session
    tl.trace(model, torch.ones(2))
    assert "mul" in pushed
    assert "add" not in pushed


def test_autocast_change_between_captures_is_recorded() -> None:
    """Autocast entered for one capture shows in that capture only."""

    model, x = _FlagModel(), torch.ones(2)

    def cpu_enabled(trace: Any) -> list[bool]:
        states = [op.call.autocast_state for op in _ops(trace, "add") + _ops(trace, "mul")]
        return [state.get("cpu", {}).get("enabled", False) for state in states]

    assert not any(cpu_enabled(tl.trace(model, x)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert any(cpu_enabled(tl.trace(model, x)))
    assert not any(cpu_enabled(tl.trace(model, x)))


def test_grad_mode_inside_forward_is_recorded_per_op() -> None:
    """``torch.no_grad`` inside the forward marks only the op it wraps."""

    trace = tl.trace(_NoGradInside(), torch.ones(2, requires_grad=True))

    def grad_enabled(name: str) -> bool:
        (op,) = _ops(trace, name)
        return op.call.autocast_state["__execution__"]["grad_enabled"]

    assert grad_enabled("add") is True
    assert grad_enabled("mul") is False
    assert grad_enabled("sub") is True


def test_equal_autocast_snapshots_are_one_object_per_trace() -> None:
    """Ops with equal autocast state share a snapshot; two traces share none."""

    model, x = _FlagModel(), torch.ones(2)
    first, second = tl.trace(model, x), tl.trace(model, x)
    first_states = [op.call.autocast_state for op in _ops(first, "add") + _ops(first, "mul")]
    second_states = [op.call.autocast_state for op in _ops(second, "add") + _ops(second, "mul")]
    assert len(first_states) == 2
    assert len({id(state) for state in first_states}) == 1
    assert first_states[0] == second_states[0]
    assert not {id(state) for state in first_states} & {id(state) for state in second_states}


def test_session_facts_are_read_again_for_a_new_session() -> None:
    """A new fingerprint or trace reads the facts afresh; no fingerprint keeps nothing."""

    trace = SimpleNamespace(_capture_container_structure=False, _options_emit_nvtx=False)
    other = SimpleNamespace(_capture_container_structure=True, _options_emit_nvtx=False)
    reads: list[int] = []

    def reader() -> Any:
        reads.append(1)
        return lambda: None

    saved = _wrapper_session.FACTS
    try:
        key = object()
        facts = _wrapper_session.session_facts(trace, key, reader)
        assert _wrapper_session.session_facts(trace, key, reader) is facts
        assert len(reads) == 1
        trace._capture_container_structure = True
        renewed = _wrapper_session.session_facts(trace, object(), reader)
        assert renewed is not facts and renewed.container_structure is True
        moved = _wrapper_session.session_facts(other, renewed.fingerprint, reader)
        assert moved is not renewed and moved.trace_id == id(other)
        kept = _wrapper_session.FACTS
        loose = _wrapper_session.session_facts(trace, None, reader)
        assert loose is not kept and _wrapper_session.FACTS is kept
        assert len(reads) == 4
    finally:
        _wrapper_session.FACTS = saved
