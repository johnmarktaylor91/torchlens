"""Regression gate: rolled-view edge labels must never collide with anything.

Renders the same 16-model rolled inspection set used to tune the edge-label
placement constants, lays each graph out with ``dot -Tjson``, and runs the
exact per-label geometry audit (``tests/support/label_geometry.py``).  Any
hard violation -- a label penetrating another label, a node outline, an edge
spline, an arrowhead, or a cluster border by more than the calibrated
threshold -- fails the gate with a self-explaining message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

import test_loop_module_rolling as demos
from support.label_geometry import audit_gv_source

# The 16 rolled demo configs the placement sweep was calibrated against:
# the 12 committed fixtures plus 4 extra inspection models.
DEMO_CONFIGS: list[tuple[str, type[nn.Module], dict[str, object]]] = [
    ("inside_outside_relu", demos.InsideOutsideRelu, {"vis_call_depth": 1}),
    ("inside_outside_relu_separable", demos.InsideOutsideReluSeparable, {"vis_call_depth": 1}),
    ("inside_outside_block_collapsed", demos.InsideOutsideBlock, {"vis_call_depth": 1}),
    ("inside_outside_block_expanded", demos.InsideOutsideBlock, {"vis_call_depth": 1000}),
    ("deep_loop_body", demos.DeepLoopBody, {"vis_call_depth": 1000}),
    ("rnn_cell", demos.TanhRNNCellLoop, {"vis_call_depth": 1000}),
    ("repeated_block_stack_collapsed", demos.RepeatedBlockStack, {"vis_call_depth": 1}),
    ("repeated_block_stack_expanded", demos.RepeatedBlockStack, {"vis_call_depth": 1000}),
    ("two_distinct_loops", demos.TwoDistinctLoops, {"vis_call_depth": 1}),
    ("buffer_loop", demos.BufferRewriteLoops, {"show_buffer_layers": "always"}),
    ("nested_loop", demos.NestedLoopBlock, {"vis_call_depth": 1}),
    ("parallel_fanout", demos.ParallelFanout, {"vis_call_depth": 1}),
    ("collapsed_block_recurrence", demos.CollapsedBlockRecurrence, {"vis_call_depth": 1000}),
    ("inside_outside_loop", demos.InsideOutsideLoop, {"vis_call_depth": 1}),
    ("parallel_siblings_loop", demos.ParallelSiblingsLoop, {"vis_call_depth": 1000}),
    ("shared_two_site_recurrences", demos.SharedTwoSiteRecurrences, {"vis_call_depth": 1}),
]


@pytest.mark.parametrize(
    ("name", "model_cls", "kwargs"),
    DEMO_CONFIGS,
    ids=[config[0] for config in DEMO_CONFIGS],
)
def test_rolled_edge_labels_have_zero_hard_violations(
    name: str, model_cls: type[nn.Module], kwargs: dict[str, object], tmp_path: Path
) -> None:
    """Every rolled demo graph lays out with zero hard label-geometry violations."""

    trace = demos._trace(model_cls())
    gv_source = trace.draw(
        vis_mode="rolled",
        vis_save_only=True,
        vis_fileformat="dot",
        vis_outpath=str(tmp_path / name),
        **kwargs,
    )
    result = audit_gv_source(gv_source)
    assert result.hard_violation_count == 0, result.describe_violations(name)


_NEGATIVE_CONTROL_GV = """
digraph negative_control {
\trankdir=BT
\ta [shape=box]
\tb [shape=box]
\ta -> b [headlabel="overlapping head label" labeldistance=0]
}
"""


def test_audit_negative_control_flags_label_node_overlap() -> None:
    """The audit itself catches a deliberately overlapping head label.

    ``labeldistance=0`` centers the head label on the edge's head point, so it
    must penetrate node ``b`` -- if the audit reports this graph clean, the
    gate has gone blind and the positive tests above prove nothing.
    """

    result = audit_gv_source(_NEGATIVE_CONTROL_GV)
    assert result.hard_violation_count >= 1
    violation_types = {violation["type"] for violation in result.violations}
    assert "label-node" in violation_types
    # Failure messages must be self-explaining: graph, edge, label, type, depth.
    description = result.describe_violations("negative_control")
    assert "label-node" in description
    assert "a->b" in description
    assert "overlapping head label" in description
    assert "penetration" in description
