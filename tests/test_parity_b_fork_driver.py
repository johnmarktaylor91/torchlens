"""Backend-neutral finished-trace replay walk (``intervention/_backend_replay.py``).

The walk is driven here by a NumPy adapter over hand-built replay steps, with no
framework and no ``Trace``: it covers the propagation rules every real adapter
inherits (no-op edits reproduce payloads exactly, repeated parents stay distinct
occurrences, a multi-output op runs once, a changed guard or layout refuses, and
nothing is written when a step refuses).
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from torchlens.intervention._backend_replay import (
    ReplayStep,
    ReplayWalk,
    propagate,
    verify_backend_replay,
)
from torchlens.intervention.errors import HookValueError, ReplayPreconditionError


class _NumpyAdapter:
    """Replays the ops of :func:`_graph` with NumPy and counts each op's calls."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def can_replay(self, trace: Any, op: Any) -> str | None:
        return None

    def forward(self, trace: Any, op: Any, parent_values: Sequence[Any]) -> tuple[Any, ...]:
        self.calls[op.name] = self.calls.get(op.name, 0) + 1
        return tuple(op.fn(*parent_values))

    def to_host(self, value: Any) -> Any:
        return np.asarray(value)

    def zeros_like(self, value: Any) -> Any:
        return np.zeros_like(value)

    def close(self, expected: Any, actual: Any) -> bool:
        return bool(np.array_equal(np.asarray(expected), np.asarray(actual)))


def _op(name: str, fn: Any) -> SimpleNamespace:
    return SimpleNamespace(name=name, fn=fn)


def _graph() -> tuple[list[ReplayStep], dict[str, np.ndarray]]:
    """x -> y = 2x -> sq = y*y (repeated parent) -> (lo, hi) = split(sq) -> out = lo + hi;
    flag = sum(x) > 0 is a guard."""

    steps = [
        ReplayStep(_op("x", None), ("x",), (), "source", False),
        ReplayStep(_op("y", lambda a: (a * 2.0,)), ("y",), ("x",), "op", False),
        ReplayStep(_op("sq", lambda a, b: (a * b,)), ("sq",), ("y", "y"), "op", False),
        ReplayStep(_op("split", lambda a: (a[:2], a[2:])), ("lo", "hi"), ("sq",), "op", False),
        ReplayStep(_op("out", lambda a, b: (a + b,)), ("out",), ("lo", "hi"), "op", False),
        ReplayStep(_op("flag", lambda a: (np.sum(a) > 0,)), ("flag",), ("x",), "op", True),
    ]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x * 2.0
    sq = y * y
    payloads = {
        "x": x,
        "y": y,
        "sq": sq,
        "lo": sq[:2],
        "hi": sq[2:],
        "out": sq[:2] + sq[2:],
        "flag": np.sum(x) > 0,
    }
    return steps, payloads


def _walk(
    adapter: _NumpyAdapter, payloads: dict[str, np.ndarray], edits: dict[str, Any]
) -> ReplayWalk:
    def edit(step: ReplayStep, slot: str, value: Any) -> tuple[Any, list[Any]]:
        if slot not in edits:
            return value, []
        return edits[slot](value), [f"edit:{slot}"]

    return ReplayWalk(
        trace=None,
        adapter=adapter,
        payload=payloads.__getitem__,
        edit=edit,
        edited_slots=frozenset(edits),
    )


def test_noop_edit_reproduces_every_payload_exactly() -> None:
    steps, payloads = _graph()
    adapter = _NumpyAdapter()
    walk = propagate(_walk(adapter, payloads, {"y": lambda v: v}), steps)
    for slot, value in walk.overlay.items():
        assert np.array_equal(value, payloads[slot]), slot
    assert walk.records == {"y": ["edit:y"]}


def test_edit_propagates_through_repeated_parent_and_multi_output() -> None:
    steps, payloads = _graph()
    adapter = _NumpyAdapter()
    walk = propagate(_walk(adapter, payloads, {"y": lambda v: v + 1.0}), steps)
    y = payloads["y"] + 1.0
    # Both occurrences of the repeated parent read the edited value: (y+1)^2, not (y+1)*y.
    assert np.array_equal(walk.overlay["sq"], y * y)
    assert np.array_equal(walk.overlay["out"], (y * y)[:2] + (y * y)[2:])
    # The multi-output op ran once and rebuilt both slots.
    assert adapter.calls["split"] == 1
    assert set(walk.overlay) == {"y", "sq", "lo", "hi", "out"}
    # Ops outside the cone never ran.
    assert "flag" not in adapter.calls


def test_edit_on_one_output_slot_keeps_its_sibling() -> None:
    steps, payloads = _graph()
    adapter = _NumpyAdapter()
    walk = propagate(_walk(adapter, payloads, {"hi": lambda v: v * 0.0}), steps)
    assert np.array_equal(walk.overlay["lo"], payloads["lo"])
    assert np.array_equal(walk.overlay["out"], payloads["lo"])
    assert "sq" not in walk.overlay


def test_changed_guard_refuses_and_writes_nothing_outside_the_walk() -> None:
    steps, payloads = _graph()
    adapter = _NumpyAdapter()
    walk = _walk(adapter, payloads, {"x": lambda v: -v})
    with pytest.raises(ReplayPreconditionError) as caught:
        propagate(walk, steps)
    assert caught.value.fields["code"] == "backend_replay_guard_changed"
    assert np.array_equal(payloads["x"], np.array([1.0, 2.0, 3.0, 4.0]))


def test_unchanged_guard_replays() -> None:
    steps, payloads = _graph()
    walk = propagate(_walk(_NumpyAdapter(), payloads, {"x": lambda v: v * 3.0}), steps)
    assert bool(walk.overlay["flag"]) is True


def test_edit_with_another_shape_fails_the_payload_gate() -> None:
    steps, payloads = _graph()
    walk = _walk(_NumpyAdapter(), payloads, {"y": lambda v: v[:3]})
    with pytest.raises(HookValueError) as caught:
        propagate(walk, steps)
    assert caught.value.fields["code"] == "intervention_replacement_invalid"


def test_edit_with_another_dtype_fails_the_payload_gate() -> None:
    steps, payloads = _graph()
    walk = _walk(_NumpyAdapter(), payloads, {"y": lambda v: v.astype(np.float32)})
    with pytest.raises(HookValueError) as caught:
        propagate(walk, steps)
    assert caught.value.fields["code"] == "intervention_replacement_invalid"


def test_wrong_output_arity_refuses() -> None:
    steps, payloads = _graph()
    steps[3] = ReplayStep(_op("split", lambda a: (a,)), ("lo", "hi"), ("sq",), "op", False)
    walk = _walk(_NumpyAdapter(), payloads, {"y": lambda v: v + 1.0})
    with pytest.raises(ReplayPreconditionError) as caught:
        propagate(walk, steps)
    assert caught.value.fields["code"] == "backend_replay_op_unreplayable"


def test_preserved_origin_value_stands_as_given() -> None:
    steps, payloads = _graph()
    new_y = payloads["y"] * 10.0
    walk = _walk(_NumpyAdapter(), payloads, {})
    walk.origins = {"y": new_y}
    propagate(walk, steps)
    assert np.array_equal(walk.overlay["y"], new_y)
    assert np.array_equal(walk.overlay["sq"], new_y * new_y)


def test_downstream_layout_change_refuses() -> None:
    """An edit that sends a downstream op down a shape-dependent branch refuses."""

    steps, payloads = _graph()
    steps[2] = ReplayStep(
        _op("sq", lambda a, b: (np.repeat(a * b, 2) if a[0] > 2.5 else a * b,)),
        ("sq",),
        ("y", "y"),
        "op",
        False,
    )
    walk = _walk(_NumpyAdapter(), payloads, {"y": lambda v: v + 1.0})
    with pytest.raises(ReplayPreconditionError) as caught:
        propagate(walk, steps)
    assert caught.value.fields["code"] == "backend_replay_layout_changed"
    assert "sq" not in walk.overlay


def test_verify_refuses_a_trace_without_a_replay_adapter() -> None:
    """A PyTorch trace has no backend replay adapter, so the verifier refuses it."""

    with pytest.raises(ReplayPreconditionError) as caught:
        verify_backend_replay(SimpleNamespace(_provenance_backend="torch"))
    assert caught.value.fields["code"] == "backend_replay_no_adapter"
