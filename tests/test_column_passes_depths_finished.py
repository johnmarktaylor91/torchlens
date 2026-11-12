"""Steps 4 (layer depths) and 17 (tracing finished) over raw ``Op`` rows.

Two contracts per step. The semantic contract: the column body and the facade
body leave identical stored results (every distance, reached marker and lineage
set for step 4; every finished flag and bound template display key for step
17) on models with several inputs and outputs, a buffer, a skip connection, a
loop, a tensor created inside ``forward`` and keyword and nested call
arguments. The counter contract: the column body really runs on an ordinary
trace and makes far fewer Python calls than the facade body it replaces. Step
17 runs on the sealed store, so its column body is the sealed-store body.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _finished_marks, _layer_depths, finalization
from torchlens._utils._record_group_table import op_attr

_DEPTH_FIELDS = (
    "min_distance_from_input",
    "max_distance_from_input",
    "min_distance_to_output",
    "max_distance_to_output",
    "has_input_ancestor",
    "has_output_descendant",
)
_LINEAGE_FIELDS = ("input_ancestors", "output_descendants")


class _Branchy(nn.Module):
    """Two inputs, two outputs, a buffer, a skip connection and a short loop."""

    def __init__(self, steps: int = 3) -> None:
        """Build the shared layer and the buffer."""

        super().__init__()
        self.steps = steps
        self.fc = nn.Linear(4, 4)
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the loop, join both inputs and return two tensors."""

        a = torch.relu(self.fc(x))
        for _ in range(self.steps):
            a = torch.tanh(self.fc(a))
        b = y + self.offset
        c = torch.cat([a, b], dim=1)
        d = c.sum(dim=1, keepdim=True)
        fresh = torch.ones(4) * 3
        return x * 2 + d + fresh, c


class _Chain(nn.Module):
    """A plain chain of additions: every op has one flat positional template."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one forty times."""

        for _ in range(40):
            x = x + 1
        return x


_MODELS: dict[str, Callable[[], tuple[nn.Module, tuple[torch.Tensor, ...]]]] = {
    "branchy": lambda: (_Branchy(), (torch.ones(2, 4), torch.ones(2, 4))),
    "long_loop": lambda: (_Branchy(steps=40), (torch.ones(2, 4), torch.ones(2, 4))),
    "chain": lambda: (_Chain(), (torch.ones(2, 4),)),
}


def _depth_snapshot(trace: Any) -> dict[str, Any]:
    """Return what step 4 stores on every raw op."""

    snapshot = {}
    for label, op in trace._raw_graph_ws.raw_layer_dict.items():
        state: dict[str, Any] = {name: getattr(op, op_attr(name)) for name in _DEPTH_FIELDS}
        for name in _LINEAGE_FIELDS:
            state[name] = sorted(getattr(op, op_attr(name)))
        snapshot[label] = state
    return snapshot


def _template_keys(value: Any, found: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Collect ``(parent label, bound display key)`` of every parent reference."""

    kind = type(value).__name__
    if kind == "ParentRef":
        found.append((value.parent_label, getattr(value, "_display_key", None)))
    elif kind == "CapturedArgTemplate":
        _template_keys(value.args, found)
        _template_keys(value.kwargs, found)
    elif isinstance(value, (tuple, list)):
        for member in value:
            _template_keys(member, found)
    elif isinstance(value, dict):
        for member in value.values():
            _template_keys(member, found)
    return found


def _finished_snapshot(trace: Any) -> dict[str, Any]:
    """Return what step 17 stores: the flags and every bound template key."""

    snapshot = {}
    for label, op in trace._layer_dict_main_keys.items():
        snapshot[label] = {
            "finished": op._tracing_finished,
            "args": _template_keys(op._call_args_template, []),
            "kwargs": _template_keys(op._call_kwargs_template, []),
        }
    return {"ops": snapshot, "trace": trace._tracing_finished}


def _python_calls(fn: Callable[[], Any]) -> int:
    """Return the number of Python and C call events ``fn`` makes."""

    counts = [0]

    def profiler(_frame: Any, event: str, _arg: Any) -> None:
        """Count every call event."""

        if event in ("call", "c_call"):
            counts[0] += 1

    sys.setprofile(profiler)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return counts[0]


_STEPS: dict[str, tuple[Any, str, Any, str, Callable[[Any], Any], Callable[[Any], dict[str, Any]]]]
_STEPS = {
    # step -> (hooked module, hooked name, patched module, lookup name, facade stub, snapshot)
    "depths": (
        _layer_depths,
        "_mark_layer_depths",
        _layer_depths,
        "_column_lookup",
        lambda _trace: None,
        _depth_snapshot,
    ),
    "finished": (
        finalization,
        "_set_tracing_finished",
        _finished_marks,
        "_mark_sealed_rows",
        lambda *_args: False,
        _finished_snapshot,
    ),
}


def _trace_through_step(
    step: str, model_name: str, monkeypatch: pytest.MonkeyPatch, *, facade: bool
) -> dict[str, Any]:
    """Trace one model, hooking one step; return its snapshot and call count."""

    home, name, lookup_home, lookup_name, stub, snapshot = _STEPS[step]
    if facade:
        monkeypatch.setattr(lookup_home, lookup_name, stub)
    real_step = getattr(home, name)
    seen: dict[str, Any] = {}

    def hooked(trace_arg: Any) -> None:
        """Run the real step counted and snapshot its result."""

        seen["calls"] = _python_calls(lambda: real_step(trace_arg))
        seen["snapshot"] = snapshot(trace_arg)
        seen["ops"] = len(seen["snapshot"].get("ops", seen["snapshot"]))

    monkeypatch.setattr(home, name, hooked)
    torch.manual_seed(0)
    model, inputs = _MODELS[model_name]()
    tl.trace(model, inputs)
    monkeypatch.undo()
    return seen


@pytest.fixture
def isolated_patch() -> Iterator[pytest.MonkeyPatch]:
    """Yield a private monkeypatch context that is always undone."""

    patch = pytest.MonkeyPatch()
    try:
        yield patch
    finally:
        patch.undo()


@pytest.mark.parametrize("step", sorted(_STEPS))
@pytest.mark.parametrize("model_name", sorted(_MODELS))
def test_column_body_matches_facade_body(
    step: str, model_name: str, isolated_patch: pytest.MonkeyPatch
) -> None:
    """Column and facade bodies of a step leave identical stored results."""

    column = _trace_through_step(step, model_name, isolated_patch, facade=False)
    facade = _trace_through_step(step, model_name, isolated_patch, facade=True)
    assert column["snapshot"] == facade["snapshot"]
    assert column["snapshot"]


def test_depth_snapshot_covers_reached_and_unreached_ops(
    isolated_patch: pytest.MonkeyPatch,
) -> None:
    """The comparison model has skip paths and an op no input reaches."""

    snapshot = _trace_through_step("depths", "branchy", isolated_patch, facade=False)["snapshot"]
    spans = [
        (state["min_distance_from_input"], state["max_distance_from_input"])
        for state in snapshot.values()
    ]
    assert any(low is None for low, _high in spans)
    assert any(low is not None and low < high for low, high in spans)
    assert all(
        state["has_input_ancestor"]
        for state in snapshot.values()
        if state["min_distance_from_input"] is not None
    )


def test_finished_snapshot_binds_single_and_multi_pass_keys(
    isolated_patch: pytest.MonkeyPatch,
) -> None:
    """Parent references are bound; a looped layer's key carries its pass."""

    seen = _trace_through_step("finished", "branchy", isolated_patch, facade=False)
    snapshot = seen["snapshot"]
    keys = [
        key for state in snapshot["ops"].values() for _label, key in state["args"] + state["kwargs"]
    ]
    assert snapshot["trace"] is True
    assert all(state["finished"] is True for state in snapshot["ops"].values())
    bound = [key for key in keys if isinstance(key, str)]
    assert any(":" in key for key in bound)
    assert any(":" not in key for key in bound)


@pytest.mark.parametrize("step", sorted(_STEPS))
def test_column_body_makes_few_calls_per_op(step: str, isolated_patch: pytest.MonkeyPatch) -> None:
    """On a plain chain the column body costs a few calls per op, the facade many."""

    column = _trace_through_step(step, "chain", isolated_patch, facade=False)
    facade = _trace_through_step(step, "chain", isolated_patch, facade=True)
    assert column["ops"] == facade["ops"] >= 40
    # Measured per op on the 1,000-op chain (py3.10): depths 166 facade, finished 77.
    assert column["calls"] * 4 < facade["calls"], (column["calls"], facade["calls"])
