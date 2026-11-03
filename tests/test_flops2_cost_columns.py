"""Forward, backward and total FLOPs columns on the cost tree.

The forward column is unchanged by the backward; the backward column sums to
the executed count exactly under every view (unattributed calls and nodes with
no forward op are their own rows); the total adds the backward to the
forward's multiply-accumulate work, one convention, never to the
elementwise-inclusive forward column. A trace whose backward was not run or is
unknown shows that state, never 0.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from tests.test_flops1_backward_dispatch import ZOO_BUILDERS
from tests.test_flops2_backward_attribution import SEGMENT_FORWARD, _checkpointed_trace
from torchlens.report import backward_status, cost_tree


def _observed(case: str) -> tuple[Any, Any]:
    """Capture a zoo case, read its tree before the backward, then run the backward."""

    model, inputs = ZOO_BUILDERS[case]()
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    before = cost_tree(trace, fold_repeats=False)
    out = trace[trace.output_layers[0]].out
    trace.log_backward(out.sum())
    return trace, before


def _assert_columns_conserve(tree: Any, executed: int) -> None:
    rows = tree.rows
    assert tree.total_backward_flops == executed
    assert sum(row.self_backward_flops for row in rows) == executed
    assert sum(row.self_forward_mac_flops for row in rows) == tree.total_forward_mac_flops
    for row in rows:
        assert (
            row.self_forward_backward_flops == row.self_forward_mac_flops + row.self_backward_flops
        )
        assert row.subtree_forward_backward_flops == (
            row.subtree_forward_mac_flops + row.subtree_backward_flops
        )
    roots = [row for row in rows if row.parent_id is None]
    assert sum(row.subtree_backward_flops for row in roots) == executed


@pytest.mark.parametrize(
    "case",
    ["eager_attention_h2", "rnn_unrolled"],
)
def test_backward_column_conserves_the_executed_count(case: str) -> None:
    """Self backward cells sum to the executed count; the forward column is unchanged."""

    trace, before = _observed(case)
    try:
        executed = backward_status(trace).executed_flops
        assert executed is not None
        tree = cost_tree(trace)
        _assert_columns_conserve(tree, executed)
        # Unfolded, every forward cell is what it was before the backward ran.
        forward_cells = [
            (row.row_id, row.self_flops, row.subtree_flops)
            for row in cost_tree(trace, fold_repeats=False).rows
            if not row.row_id.startswith("::backward::")
        ]
        assert forward_cells == [(r.row_id, r.self_flops, r.subtree_flops) for r in before.rows]
        for view in (cost_tree(trace, k=1), cost_tree(trace, max_depth=0)):
            assert sum(row.self_backward_flops for row in view.rows) == executed
        assert "executed backward:" in str(tree) and "fwd MAC+bwd" in str(tree)
    finally:
        trace.cleanup()


def test_states_without_a_counted_backward_are_never_zero() -> None:
    """Not run and unknown backwards leave the cells None and say which."""

    model, inputs = ZOO_BUILDERS["partial_requires_grad"]()
    with torch.no_grad():
        off = tl.trace(model, inputs)
    unobserved = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    try:
        for trace, state, words in (
            (off, "grad_disabled", "not run"),
            (unobserved, "grad_enabled_unobserved", "unknown"),
        ):
            tree = cost_tree(trace)
            assert tree.backward_state == state and tree.total_backward_flops is None
            assert all(row.self_backward_flops is None for row in tree.rows)
            assert all(row.self_forward_backward_flops is None for row in tree.rows)
            assert f"executed backward: {words}" in str(tree)
            assert "fwd MAC+bwd" not in str(tree)
    finally:
        off.cleanup()
        unobserved.cleanup()


def test_checkpoint_node_work_has_its_own_row() -> None:
    """A reentrant checkpoint node has no forward op: its work is a remainder row."""

    trace = _checkpointed_trace("reentrant")
    try:
        status = backward_status(trace)
        tree = cost_tree(trace)
        _assert_columns_conserve(tree, status.executed_flops)
        (no_op,) = [row for row in tree.rows if row.row_id == "::backward::no_op"]
        assert no_op.self_backward_flops >= SEGMENT_FORWARD
        assert no_op.self_flops == 0 and no_op.self_forward_mac_flops == 0
    finally:
        trace.cleanup()
