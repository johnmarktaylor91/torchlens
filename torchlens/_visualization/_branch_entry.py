"""``IF`` / ``ELIF`` labels on the edges that enter a condition test.

Released TorchLens labels the edge entering a conditional's test subgraph
with the test's kind, in addition to the THEN / ELIF n / ELSE marks on the
arm edges and the ``IF:`` prefix on the test node itself.  The reader then
sees where the branch starts, not only where its test sits.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = ["conditional_branch_edge_kind", "edge_is_conditional_branch"]

# Bound the forward walk that maps a branch-entry edge to its condition bool, so a
# malformed or huge conditional subgraph can never turn edge labelling into a hot loop.
_BRANCH_KIND_SEARCH_LIMIT = 256


def _bool_kinds(trace: Trace) -> dict[str, str]:
    """Map every condition bool's base label to ``"IF"`` or ``"ELIF"``."""

    bool_kind: dict[str, str] = {}
    for event in trace.graph.conditional_records or ():
        for bool_label in getattr(event, "bool_layers", ()) or ():
            bool_layer = trace.layers[bool_label]
            bool_ops = tuple(getattr(bool_layer, "ops", {}).values())
            context_kinds = (
                {record_member(op, "conditional_context_kind", None) for op in bool_ops}
                if bool_ops
                else {record_member(bool_layer, "conditional_context_kind", None)}
            )
            bool_kind[bool_label.split(":", 1)[0]] = (
                "ELIF" if context_kinds == {"elif_test"} else "IF"
            )
    return bool_kind


def conditional_branch_edge_kind(child_node: Any, trace: Trace) -> str:
    """Return the branch-test kind (``IF`` or ``ELIF``) for a branch-entry edge.

    ``conditional_branch_edges`` records the edges that enter each condition
    test but carries no kind.  Map the edge's child to the nearest downstream
    branch bool via a bounded forward walk, then use that bool's AST-derived
    ``conditional_context_kind``.  One source-level ``if`` evaluated
    repeatedly in a loop is ``IF`` on every evaluation; a static ``elif`` test
    is classified ``elif_test`` and renders as ``ELIF``.

    Parameters
    ----------
    child_node:
        Destination node of the branch-entry edge (the condition-subgraph entry).
    trace:
        Owning trace with conditional metadata.

    Returns
    -------
    str
        ``"IF"`` or ``"ELIF"`` (``"IF"`` when no bool is reachable).
    """

    bool_kind = _bool_kinds(trace)
    if not bool_kind:
        return "IF"
    start = child_node.layer_label.split(":", 1)[0]
    seen: set[str] = {start}
    frontier = deque([start])
    steps = 0
    while frontier and steps < _BRANCH_KIND_SEARCH_LIMIT:
        steps += 1
        current = frontier.popleft()
        if current in bool_kind:
            return bool_kind[current]
        # ``label in trace`` disagrees with ``trace[label]`` for some intermediate
        # condition ops, so resolve by indexing and treat a miss as a dead end.
        try:
            node = trace[current]
        except (KeyError, ValueError):
            continue
        for child_label in getattr(node, "children", ()):
            base = child_label.split(":", 1)[0]
            if base not in seen:
                seen.add(base)
                frontier.append(base)
    return "IF"


def edge_is_conditional_branch(
    parent_node: Any, child_node: Any, trace: Trace, vis_mode: str
) -> bool:
    """Return True when an edge enters a condition test (a branch-entry edge).

    Parameters
    ----------
    parent_node, child_node:
        Source and destination of the edge.
    trace:
        Owning trace with conditional metadata.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``; a rolled edge matches on base labels.
    """

    branch_edges = trace.graph.conditional_branch_edges or ()
    edge_key = (parent_node.layer_label, child_node.layer_label)
    if vis_mode == "unrolled":
        return edge_key in branch_edges
    if vis_mode == "rolled":
        return any(
            (branch_parent.split(":")[0], branch_child.split(":")[0]) == edge_key
            for branch_parent, branch_child in branch_edges
        )
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")
