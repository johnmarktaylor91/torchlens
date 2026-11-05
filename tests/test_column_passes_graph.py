"""Column-view bodies of postprocess step 3 (orphan flood) and step 5 (branch attribution).

Two contracts. The counter contract: per op added to a model, the per-op loops
of ``_remove_orphan_nodes`` and ``_mark_conditional_branches`` make fewer than
half the Python-level calls through the ``OpColumns`` view that they make
through the ``Op`` attribute facade (the slope between two model sizes, so the
fixed cost of binding the accessors once per pass is excluded). The semantic
contract: both bodies leave the same stored result, op field by op field right
after each pass and in the finished trace, and in the trace-level ledgers the
passes write.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _column_passes_graph, control_flow, graph_traversal
from torchlens._utils._record_group_table import record_member

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


#: Loop iterations of the small and the large ``_Branchy`` model.
_SMALL, _LARGE = 4, 16


class _Branchy(nn.Module):
    """Conditional arms, an elif, internal sinks and a disconnected orphan chain."""

    def __init__(self, iterations: int) -> None:
        """Build two small linear stages looped ``iterations`` times."""

        super().__init__()
        self.iterations = iterations
        torch.manual_seed(0)
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a branchy forward pass with sinks and an orphan."""

        unused = torch.zeros(2) + 1  # reaches neither input nor output: pruned
        del unused
        hidden = self.first(x)
        _sink = hidden * 3  # internal sink: no children, not an output
        for _ in range(self.iterations):
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


def _snapshot(trace: Any) -> dict[str, str]:
    """Return ``repr`` of every pass-written field of every raw op, keyed by label."""

    return {
        label: repr({name: record_member(op, name, None) for name in _OP_FIELDS})
        for label, op in trace._raw_graph_ws.raw_layer_dict.items()
    }


def _counted_trace(
    monkeypatch: pytest.MonkeyPatch, *, facade: bool, iterations: int
) -> tuple[dict[str, int], Any, dict[str, dict[str, str]]]:
    """Trace ``_Branchy`` counting the calls each per-op loop makes.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture; undone when the test ends.
    facade:
        Force the facade bodies by making the column lookup decline.
    iterations:
        Loop iterations of the traced model.

    Returns
    -------
    tuple[dict[str, int], Any, dict[str, dict[str, str]]]
        Calls per loop function name, the finished trace, and the raw-op
        snapshots taken right after step 3 and right after step 5.
    """

    counts: dict[str, int] = {}
    snapshots: dict[str, dict[str, str]] = {}
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
        for module, name in (
            (graph_traversal, "_remove_orphan_nodes"),
            (control_flow, "_mark_conditional_branches"),
        ):
            real_pass = getattr(module, name)

            def snapshotting(trace_arg: Any, _real: Any = real_pass, _name: str = name) -> None:
                """Run one pass, then snapshot the raw ops it wrote."""

                _real(trace_arg)
                snapshots[_name] = _snapshot(trace_arg)

            patch.setattr(module, name, snapshotting)
        trace = tl.trace(_Branchy(iterations).eval(), torch.ones(2, 4))
    return counts, trace, snapshots


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[tuple[bool, int], Any]]:
    """Yield ``(facade, iterations) -> (counts, trace, snapshots)`` for both sizes and bodies."""

    yield {
        (facade, iterations): _counted_trace(monkeypatch, facade=facade, iterations=iterations)
        for facade in (False, True)
        for iterations in (_SMALL, _LARGE)
    }


@pytest.mark.parametrize("loops", [_STEP3_LOOPS, _STEP5_LOOPS], ids=["step3", "step5"])
def test_column_loops_make_under_half_the_facade_calls_per_op(
    runs: dict[tuple[bool, int], Any], loops: tuple[str, ...]
) -> None:
    """Counter: per added op, each pass's loops cost under half the facade's Python calls."""

    def slope(facade: bool) -> int:
        """Return the calls the large model adds over the small one."""

        small, large = runs[(facade, _SMALL)][0], runs[(facade, _LARGE)][0]
        assert set(loops) <= set(small) and set(loops) <= set(large), "every loop must run"
        return sum(large[name] - small[name] for name in loops)

    added_ops = len(runs[(False, _LARGE)][1].ops) - len(runs[(False, _SMALL)][1].ops)
    column, facade = slope(False), slope(True)
    assert added_ops > 0 and facade > 0
    assert column * 2 < facade, (
        f"column +{column} calls against facade +{facade} for +{added_ops} ops"
    )


def test_column_and_facade_bodies_store_the_same_result(
    runs: dict[tuple[bool, int], Any],
) -> None:
    """Both bodies leave identical op fields after each pass and in the finished trace."""

    _column_counts, column_trace, column_snaps = runs[(False, _LARGE)]
    _facade_counts, facade_trace, facade_snaps = runs[(True, _LARGE)]
    assert set(column_snaps) == {"_remove_orphan_nodes", "_mark_conditional_branches"}
    for name, snapshot in column_snaps.items():
        assert snapshot == facade_snaps[name], f"raw ops differ after {name}"
    assert any(
        "'elif_" in value for value in column_snaps["_mark_conditional_branches"].values()
    ), "step 5 must attribute an elif arm"
    column_ops = list(column_trace.ops)
    facade_ops = list(facade_trace.ops)
    assert [op.label for op in column_ops] == [op.label for op in facade_ops]
    assert any(op.control_flow.branch_stack for op in column_ops), "step 5 must attribute"
    assert any(op.is_internal_sink for op in column_ops), "step 3 must log a sink"
    for column_op, facade_op in zip(column_ops, facade_ops, strict=True):
        for name in _OP_FIELDS:
            left = record_member(column_op, name, None)
            right = record_member(facade_op, name, None)
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
