"""Aggregate-``Layer`` conditional metadata merge and rebuild for Step 15.5."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _append_unique_child_label(
    child_labels: list[str], seen_labels: set[str], child_label: str
) -> None:
    """Append a child label if it has not been seen yet.

    Parameters
    ----------
    child_labels:
        Ordered list being accumulated.
    seen_labels:
        Membership set mirroring ``child_labels`` (the bare list scan made
        each aggregate projection O(k^2) in its child count -- R52).
    child_label:
        Child label to append.
    """
    if child_label not in seen_labels:
        seen_labels.add(child_label)
        child_labels.append(child_label)


def _strip_pass_suffix(layer_label: str) -> str:
    """Strip the ``:call_index`` suffix from a layer label.

    Parameters
    ----------
    layer_label:
        Pass-qualified or pass-free layer label.

    Returns
    -------
    str
        Layer label without any pass suffix.
    """
    return layer_label.split(":", 1)[0]


def _merge_layer_log_conditional_fields(
    layer_log: Layer,
    pass_log: Op,
) -> None:
    """Merge one pass's conditional metadata into an aggregate ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being updated.
    pass_log:
        Per-pass layer entry contributing conditional metadata.
    """
    if pass_log.control_flow.is_in_conditional_body:
        layer_log._control_flow_is_in_conditional_body = True

    stack_signature = tuple(pass_log.control_flow.branch_stack)
    if stack_signature not in layer_log.control_flow.branch_stack_ops:
        layer_log.control_flow.role_stacks.append(list(pass_log.control_flow.branch_stack))
        layer_log.control_flow.branch_stack_ops[stack_signature] = []

    signature_ops = layer_log.control_flow.branch_stack_ops[stack_signature]
    if pass_log.pass_number not in signature_ops:
        signature_ops.append(pass_log.pass_number)
        signature_ops.sort()

    for conditional_id, branch_children in pass_log.control_flow.arm_children.items():
        merged_branch_children = layer_log.control_flow.arm_children.setdefault(
            conditional_id,
            {},
        )
        for branch_kind, child_labels in branch_children.items():
            merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
            merged_seen = set(merged_child_labels)
            for child_label in child_labels:
                _append_unique_child_label(
                    merged_child_labels,
                    merged_seen,
                    _strip_pass_suffix(child_label),
                )


def _rebuild_layer_log_conditional_views(layer_log: Layer) -> None:
    """Recompute aggregate conditional compatibility views for one ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry whose derived conditional fields are refreshed.
    """
    layer_log._control_flow_is_in_conditional_body = any(
        len(branch_stack) > 0 for branch_stack in layer_log.control_flow.role_stacks
    )

    conditional_entry_children: list[str] = []
    entry_seen: set[str] = set()
    for _, pass_log in sorted(layer_log.ops.items()):
        for child_label in pass_log.control_flow.entry_children:
            _append_unique_child_label(
                conditional_entry_children,
                entry_seen,
                _strip_pass_suffix(child_label),
            )
    layer_log._control_flow_entry_children = conditional_entry_children

    conditional_then_children: list[str] = []
    then_seen: set[str] = set()
    conditional_elif_children: dict[int, list[str]] = {}
    elif_seen_by_index: dict[int, set[str]] = {}
    conditional_else_children: list[str] = []
    else_seen: set[str] = set()
    for branch_children in layer_log.control_flow.arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique_child_label(conditional_then_children, then_seen, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            elif_children = conditional_elif_children.setdefault(elif_index, [])
            elif_seen = elif_seen_by_index.setdefault(elif_index, set())
            for child_label in child_labels:
                _append_unique_child_label(elif_children, elif_seen, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique_child_label(conditional_else_children, else_seen, child_label)

    layer_log._control_flow_then_children = conditional_then_children
    layer_log._control_flow_elif_children = conditional_elif_children
    layer_log._control_flow_else_children = conditional_else_children


def _rebuild_conditional_edge_call_indices(self: Trace) -> None:
    """Recompute rolled conditional edge-pass metadata from arm-entry edges.

    Parameters
    ----------
    self:
        Model log being finalized.
    """
    if self._graph_conditional_edge_call_indices:
        self._graph_conditional_edge_call_indices = {
            edge_key: sorted(dict.fromkeys(call_indexs))
            for edge_key, call_indexs in self._graph_conditional_edge_call_indices.items()
        }
        return

    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for (conditional_id, branch_kind), edge_list in self._graph_conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            parent_no_pass = _strip_pass_suffix(parent_label)
            child_no_pass = _strip_pass_suffix(child_label)
            child_call_index = _get_label_call_index(child_label)
            edge_key = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if child_call_index not in conditional_edge_call_indices[edge_key]:
                conditional_edge_call_indices[edge_key].append(child_call_index)

    self._graph_conditional_edge_call_indices = {
        edge_key: sorted(call_indexs)
        for edge_key, call_indexs in conditional_edge_call_indices.items()
    }


def _get_label_call_index(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:call_index`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` for pass-free labels.
    """
    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1
