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
    assert tree.backward_total == executed
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
    ["partial_requires_grad", "frozen_backbone", "eager_attention_h2", "rnn_unrolled"],
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
        for view in (cost_tree(trace, top_k=1), cost_tree(trace, max_depth=0)):
            assert sum(row.self_backward_flops for row in view.rows) == executed
        assert "executed backward:" in str(tree) and "fwd MAC+bwd" in str(tree)
    finally:
        trace.cleanup()


def test_forward_mac_column_matches_true_macs() -> None:
    """The MAC column is 2 x true MACs of the forward (the backward's convention)."""

    trace, _before = _observed("partial_requires_grad")
    try:
        # Three Linear layers on a (2, 16) batch: 2 * 2 * (16*32 + 32*32 + 32*4).
        assert cost_tree(trace).total_forward_mac_flops == 2 * 2 * (16 * 32 + 32 * 32 + 32 * 4)
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
            assert tree.backward_state == state and tree.backward_total is None
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


def test_unattributed_backward_work_is_a_row_not_spread() -> None:
    """Counted work under no recorded node gets its own row (zero here: the seed is a copy)."""

    model, inputs = ZOO_BUILDERS["frozen_conv_backbone"]()
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    try:
        out = trace[trace.output_layers[0]].out
        trace.log_backward(out.sum())
        status = backward_status(trace)
        assert status.unattributed_flops == 0
        tree = cost_tree(trace)
        assert not [row for row in tree.rows if row.row_id == "::backward::unattributed"]
        _assert_columns_conserve(tree, status.executed_flops)
    finally:
        trace.cleanup()


def test_siblings_whose_backward_differs_never_fold() -> None:
    """Strict folding: (xN) never hides a sibling whose backward cost differs."""

    model, inputs = ZOO_BUILDERS["rnn_unrolled"]()
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    try:
        out = trace[trace.output_layers[0]].out
        trace.log_backward(out.sum())
        tree = cost_tree(trace)
        for row in tree.rows:
            if row.folded > 1:
                assert row.self_backward_flops % row.folded == 0
        # The first step's state is a constant zero tensor, so its w_hh backward has
        # no input gradient: that call differs and must not hide under a fold. Exact
        # conservation through folded rows proves no differing sibling was folded.
        _assert_columns_conserve(tree, backward_status(trace).executed_flops)
    finally:
        trace.cleanup()
