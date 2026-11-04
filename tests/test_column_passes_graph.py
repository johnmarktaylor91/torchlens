"""Column-view bodies of postprocess step 3 (orphan flood) and step 5 (branch attribution).

Two contracts. The counter contract: the per-op loops of
``_remove_orphan_nodes`` and ``_mark_conditional_branches`` make fewer than half
the Python-level calls through the ``OpColumns`` view that they make through the
``Op`` attribute facade on the same model. The semantic contract: both bodies
leave the same stored result, op field by op field and in the trace-level
ledgers the passes write.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _column_passes_graph

#: Module functions holding the per-op loops of steps 3 and 5.
_STEP3_LOOPS = ("flood_orphan_reachability",)
_STEP5_LOOPS = (
    "reset_and_flood_if_edges",
    "assign_conditional_branch_stacks",
    "attribute_forward_edges",
    "materialize_derived_views",
)

#: Op fields the two passes write.
_OP_FIELDS = (
    "is_internal_sink",
    "is_terminal_bool",
    "is_orphan",
    "conditional_entry_children",
    "is_in_conditional_body",
    "conditional_branch_stack",
    "conditional_branch_depth",
    "conditional_arm_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
)

#: Trace-level ledgers the passes write.
_TRACE_FIELDS = (
    "internal_sink_ops",
    "internally_terminated_bool_ops",
    "_graph_conditional_branch_edges",
    "_graph_conditional_arm_entry_edges",
    "_graph_conditional_edge_call_indices",
)


class _Branchy(nn.Module):
    """Conditional arms, an elif, internal sinks and a disconnected orphan chain."""

    def __init__(self) -> None:
        """Build two small linear stages."""

        super().__init__()
        torch.manual_seed(0)
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a branchy forward pass with sinks and an orphan."""

        unused = torch.zeros(2) + 1  # reaches neither input nor output: pruned
        del unused
        hidden = self.first(x)
        _sink = hidden * 3  # internal sink: no children, not an output
        for _ in range(3):
            if hidden.sum() > 100:
                hidden = hidden + 1
            elif hidden.mean() > -100:
                hidden = torch.relu(self.second(hidden))
            else:
                hidden = hidden - 1
        return hidden * 2


def _python_calls(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[int, Any]:
    """Return the Python-level call events ``fn`` makes (itself excluded) and its result."""

    calls = 0

    def profiler(frame: Any, event: str, arg: Any) -> None:
        """Count Python function entries."""

        nonlocal calls
        if event == "call":
            calls += 1

    sys.setprofile(profiler)
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.setprofile(None)
    return calls - 1, result


def _counted_trace(monkeypatch: pytest.MonkeyPatch, *, facade: bool) -> tuple[dict[str, int], Any]:
    """Trace ``_Branchy`` counting the calls each per-op loop makes.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture; undone when the test ends.
    facade:
        Force the facade bodies by making the column lookup decline.

    Returns
    -------
    tuple[dict[str, int], Any]
        Calls per loop function name, and the finished trace.
    """

    counts: dict[str, int] = {}
    with monkeypatch.context() as patch:
        if facade:
            patch.setattr(_column_passes_graph, "_graph_lookup", lambda trace: None)
        for name in _STEP3_LOOPS + _STEP5_LOOPS:
            real = getattr(_column_passes_graph, name)

            def counting(*args: Any, _real: Any = real, _name: str = name, **kwargs: Any) -> Any:
                """Count the calls one loop function makes."""

                calls, result = _python_calls(_real, *args, **kwargs)
                counts[_name] = counts.get(_name, 0) + calls
                return result

            patch.setattr(_column_passes_graph, name, counting)
        trace = tl.trace(_Branchy().eval(), torch.ones(2, 4))
    return counts, trace


@pytest.fixture
def both_traces(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Any, dict, dict]]:
    """Yield the column and facade traces of ``_Branchy`` with their loop call counts."""

    column_counts, column_trace = _counted_trace(monkeypatch, facade=False)
    facade_counts, facade_trace = _counted_trace(monkeypatch, facade=True)
    yield column_trace, facade_trace, column_counts, facade_counts


@pytest.mark.parametrize("loops", [_STEP3_LOOPS, _STEP5_LOOPS], ids=["step3", "step5"])
def test_column_loops_make_under_half_the_facade_calls(
    both_traces: tuple[Any, Any, dict, dict], loops: tuple[str, ...]
) -> None:
    """Counter: each pass's per-op loops cost under half the facade's Python calls."""

    _column_trace, _facade_trace, column_counts, facade_counts = both_traces
    assert set(loops) <= set(column_counts), "every loop function must run"
    column = sum(column_counts[name] for name in loops)
    facade = sum(facade_counts[name] for name in loops)
    assert facade > 0
    assert column * 2 < facade, f"column {column} calls against facade {facade}"


def test_column_and_facade_bodies_store_the_same_result(
    both_traces: tuple[Any, Any, dict, dict],
) -> None:
    """Both bodies leave identical op fields and trace ledgers."""

    column_trace, facade_trace, _column_counts, _facade_counts = both_traces
    column_ops = list(column_trace.ops)
    facade_ops = list(facade_trace.ops)
    assert [op.label for op in column_ops] == [op.label for op in facade_ops]
    assert any(op.conditional_branch_stack for op in column_ops), "step 5 must attribute"
    assert any(op.is_internal_sink for op in column_ops), "step 3 must log a sink"
    for column_op, facade_op in zip(column_ops, facade_ops, strict=True):
        for name in _OP_FIELDS:
            left = getattr(column_op, name, None)
            right = getattr(facade_op, name, None)
            assert left == right, f"{column_op.label}.{name}: {left!r} != {right!r}"
    for name in _TRACE_FIELDS:
        left = getattr(column_trace, name)
        right = getattr(facade_trace, name)
        assert left == right, f"trace.{name}: {left!r} != {right!r}"


def test_gained_branch_entries_keep_the_unshared_suffix() -> None:
    """The edge helper returns the child's stack beyond the shared prefix."""

    parent = [(0, "then"), (1, "else")]
    child = [(0, "then"), (2, "then"), (3, "elif_1")]
    assert _column_passes_graph.get_gained_branch_entries(parent, child) == [
        (2, "then"),
        (3, "elif_1"),
    ]
    assert _column_passes_graph.get_gained_branch_entries(child, child) == []
