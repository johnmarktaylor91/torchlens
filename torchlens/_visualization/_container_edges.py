"""Edge labels that name a container slot (dict key, list index, dataclass field).

With ``show_containers`` on, an edge from a container leaf carries the slot the
value occupies. These helpers pick that text for one edge and decide which
model-input and model-output leaves already get it from their container node
(``show_containers="nodes"``), so the label is drawn once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._data_classes.op import Op
from .._ir.container_registry import ContainerRecord, Role
from ._render_common import BaseGraphNode, GraphNode, _unwrap_focus_node
from ._render_leaf import _base_node_for_metadata, _container_edge_label
from .request import ShowContainersLiteral

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = [
    "_occurrence_matches_op",
    "_op_is_model_input_container_leaf",
    "_op_is_model_output_container_leaf",
    "container_edge_text",
]


def container_edge_text(
    parent_node: GraphNode,
    child_base: BaseGraphNode | None,
    show_containers: ShowContainersLiteral,
) -> str | None:
    """Return the plain container-slot text for one edge, or ``None``.

    The child's slot wins; in ``show_containers="nodes"`` mode an edge whose
    parent is a container leaf not already labelled by its container node falls
    back to the parent's slot.

    Parameters
    ----------
    parent_node:
        Rendered tail of the edge.
    child_base:
        Metadata node of the edge's head, or ``None`` when it has none.
    show_containers:
        The draw's ``show_containers`` setting.

    Returns
    -------
    str | None
        Plain slot text, or ``None`` when the edge carries no container label.
    """

    if not show_containers:
        return None
    text = _container_edge_label(child_base)
    if (
        text is None
        and show_containers == "nodes"
        and not _op_is_model_input_container_leaf(parent_node)
        and not _op_is_model_output_container_leaf(parent_node)
    ):
        text = _container_edge_label(_base_node_for_metadata(parent_node))
    return text


def _op_is_model_input_container_leaf(node: GraphNode) -> bool:
    """Return whether ``node`` is a model-input container leaf.

    Parameters
    ----------
    node:
        Candidate render node.

    Returns
    -------
    bool
        ``True`` when the node is an input leaf that already receives a
        container-node source edge in ``show_containers="nodes"`` mode.
    """
    op = _unwrap_focus_node(node)
    if not isinstance(op, Op):
        return False
    trace = getattr(op, "source_trace", None)
    if trace is None:
        return False
    for record in getattr(trace, "_containers", {}).values():
        if not isinstance(record, ContainerRecord):
            continue
        for snapshot in record.snapshots:
            if snapshot.role != Role.MODEL_INPUT:
                continue
            for occurrence in snapshot.leaf_occurrences:
                if _occurrence_matches_op(trace, occurrence.producer_op_label, op):
                    return True
    return False


def _op_is_model_output_container_leaf(node: GraphNode) -> bool:
    """Return whether ``node`` is a final model-output container leaf.

    Parameters
    ----------
    node:
        Candidate render node.

    Returns
    -------
    bool
        ``True`` when a sink container-node edge should carry its key label.
    """

    if not isinstance(_unwrap_focus_node(node), Op):
        return False
    trace = getattr(node, "source_trace", None)
    if trace is None:
        return False
    return getattr(node, "layer_label", None) in set(getattr(trace, "output_layers", ()) or ())


def _occurrence_matches_op(trace: Trace, label: str | None, op: Op) -> bool:
    """Return whether a container occurrence producer label resolves to ``op``.

    Parameters
    ----------
    trace:
        Trace containing ``op``.
    label:
        Final or raw producer label from a container occurrence.
    op:
        Candidate operation.

    Returns
    -------
    bool
        ``True`` if the label identifies ``op``.
    """

    if label is None:
        return False
    if label in trace.ops and trace.ops[label] is op:
        return True
    return (
        getattr(op, "layer_label_raw", None) == label
        or getattr(op, "_layer_label_raw", None) == label
    )
