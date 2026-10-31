"""Conditional labels, events, and branch-edge invariants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .invariants import (
        _expected_layer_log_child_union,
        _fail_conditional_invariant,
        _is_prefix_stack,
        _strip_pass_suffix,
    )

__all__ = (
    "_check_conditional_child_labels_resolve",
    "_check_conditional_bool_classification",
    "_check_conditional_event_references",
    "_check_conditional_branch_stack_monotonicity",
    "_check_conditional_elif_key_contiguity",
    "_check_conditional_bool_event_backrefs",
    "_check_conditional_layer_aggregate_views",
    "_check_conditional_rolled_edge_call_indices",
    "_check_conditional_transient_bool_keys_removed",
    "_check_conditional_arm_child_pass_union",
    "_check_conditional_branch_entry_edges",
)


def _check_conditional_child_labels_resolve(
    ml: Trace,
    name: str,
    valid_child_labels: set[str],
) -> None:
    """Check conditional child labels resolve to known layers.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 3: every child label in conditional child views exists in the log.
    for layer in ml.layer_list:
        for field_name, child_labels in (
            ("conditional_entry_children", layer.conditional_entry_children),
            ("conditional_then_children", layer.conditional_then_children),
            ("conditional_else_children", layer.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.{field_name} contains missing child label "
                        f"{child_label!r}",
                    )
        for elif_index, child_labels in layer.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for layer_log in ml.layer_logs.values():
        for field_name, child_labels in (
            ("conditional_entry_children", layer_log.conditional_entry_children),
            ("conditional_then_children", layer_log.conditional_then_children),
            ("conditional_else_children", layer_log.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.{field_name} contains missing child "
                        f"label {child_label!r}",
                    )
        for elif_index, child_labels in layer_log.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for parent_label, child_label in ml._graph_conditional_branch_edges:
        if child_label not in valid_child_labels:
            _fail_conditional_invariant(
                name,
                3,
                f"Trace.graph.conditional_branch_edges contains missing child label {child_label!r} "
                f"for parent {parent_label!r}",
            )
    for (conditional_id, branch_kind), edge_list in ml._graph_conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if child_label not in valid_child_labels:
                _fail_conditional_invariant(
                    name,
                    3,
                    f"Trace.graph.conditional_arm_entry_edges contains missing child label "
                    f"{child_label!r} for edge {(conditional_id, branch_kind, parent_label)}",
                )


def _check_conditional_bool_classification(
    ml: Trace,
    name: str,
    branch_context_kinds: set[str],
    wrapped_context_kinds: set[str],
) -> None:
    """Check conditional bool classification fields are mutually consistent.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 4: bool classification fields are mutually consistent.
    for layer in ml.layer_list:
        expected_is_branch = layer.conditional_context_kind in branch_context_kinds
        if layer.is_terminal_conditional_bool != expected_is_branch:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool={layer.is_terminal_conditional_bool} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )
        if layer.is_terminal_conditional_bool and layer.terminal_conditional_id is None:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool=True but terminal_conditional_id is None",
            )
        if layer.conditional_context_kind is not None and not layer.is_terminal_bool:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_context_kind={layer.conditional_context_kind!r} but "
                f"is_terminal_bool=False",
            )
        if (
            layer.conditional_wrapper_kind is not None
            and layer.conditional_context_kind not in wrapped_context_kinds
        ):
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_wrapper_kind={layer.conditional_wrapper_kind!r} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )


def _check_conditional_event_references(
    ml: Trace,
    name: str,
    event_id_set: set[int],
) -> None:
    """Check referenced conditional ids resolve to events.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 5: every referenced cond_id corresponds to a ConditionalEvent.
    referenced_cond_ids: set[int] = set()
    for layer in ml.layer_list:
        referenced_cond_ids.update(
            conditional_id for conditional_id, _ in layer.conditional_branch_stack
        )
        if layer.terminal_conditional_id is not None:
            referenced_cond_ids.add(layer.terminal_conditional_id)
        referenced_cond_ids.update(layer.conditional_arm_children)
    for layer_log in ml.layer_logs.values():
        referenced_cond_ids.update(
            conditional_id
            for branch_stack in layer_log.conditional_role_stacks
            for conditional_id, _ in branch_stack
        )
        referenced_cond_ids.update(layer_log.conditional_arm_children)
    referenced_cond_ids.update(
        conditional_id for conditional_id, _ in ml._graph_conditional_arm_entry_edges
    )

    for conditional_id in sorted(referenced_cond_ids):
        if conditional_id not in event_id_set:
            _fail_conditional_invariant(
                name,
                5,
                f"Referenced cond_id {conditional_id} has no matching ConditionalEvent.id "
                f"in Trace.graph.conditional_records",
            )


def _check_conditional_branch_stack_monotonicity(
    ml: Trace,
    name: str,
) -> None:
    """Check parent-child conditional stacks are monotone by prefix.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 6: parent->child stacks are monotone by prefix relation.
    for parent_op in ml.layer_list:
        for child_label in parent_op.children:
            child_layer = ml[child_label]
            if parent_op.pass_index != child_layer.pass_index:
                continue
            if parent_op.conditional_branch_stack == child_layer.conditional_branch_stack:
                continue
            if _is_prefix_stack(
                parent_op.conditional_branch_stack, child_layer.conditional_branch_stack
            ):
                continue
            if _is_prefix_stack(
                child_layer.conditional_branch_stack, parent_op.conditional_branch_stack
            ):
                continue
            _fail_conditional_invariant(
                name,
                6,
                f"Edge ({parent_op.layer_label!r}, {child_label!r}) has non-prefix "
                f"conditional stacks parent={parent_op.conditional_branch_stack} "
                f"child={child_layer.conditional_branch_stack}",
            )


def _check_conditional_elif_key_contiguity(
    ml: Trace,
    name: str,
) -> None:
    """Check elif branch keys are contiguous on conditional events.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 7: elif keys are contiguous on ConditionalEvent.
    for event in ml._graph_conditional_records:
        for field_name, mapping in (
            ("branch_ranges", event.branch_ranges),
            ("branch_test_spans", event.branch_test_spans),
        ):
            elif_indices = sorted(
                int(key.split("_", 1)[1]) for key in mapping if key.startswith("elif_")
            )
            if elif_indices != list(range(1, len(elif_indices) + 1)):
                _fail_conditional_invariant(
                    name,
                    7,
                    f"ConditionalEvent id={event.id} {field_name} has non-contiguous elif keys "
                    f"{elif_indices}",
                )


def _check_conditional_bool_event_backrefs(
    ml: Trace,
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check conditional event bool-layer backreferences.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 8: ConditionalEvent.bool_layers back-reference to the event id.
    for event in ml._graph_conditional_records:
        for bool_label in event.bool_layers:
            if bool_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers contains missing label "
                    f"{bool_label!r}",
                )
            try:
                bool_layer = ml.ops[bool_label]
            except (KeyError, ValueError, TypeError):
                bool_layer = ml[bool_label]
            bool_layer_ops = getattr(bool_layer, "ops", None)
            bool_layer_values = getattr(bool_layer_ops, "values", None)
            if isinstance(bool_layer_ops, Mapping):
                bool_ops = list(bool_layer_ops.values())
            elif callable(bool_layer_values):
                bool_ops = list(bool_layer_values())
            else:
                bool_ops = [bool_layer]
            mismatched_bool_ops = [
                op for op in bool_ops if getattr(op, "terminal_conditional_id", None) != event.id
            ]
            if mismatched_bool_ops:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers includes {bool_label!r} but "
                    f"{bool_label}.terminal_conditional_id="
                    f"{getattr(mismatched_bool_ops[0], 'terminal_conditional_id', None)}",
                )


def _check_conditional_layer_aggregate_views(
    ml: Trace,
    name: str,
) -> None:
    """Check layer conditional aggregate views match pass-level data.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 9: Layer conditional aggregate views match pass-level data.
    for layer_log in ml.layer_logs.values():
        expected_stack_order: list[list[tuple[int, str]]] = []
        expected_stack_ops: dict[tuple[tuple[int, str], ...], list[int]] = {}
        for call_index, pass_log in sorted(layer_log.ops.items()):
            stack_signature = tuple(pass_log.conditional_branch_stack)
            if stack_signature not in expected_stack_ops:
                expected_stack_order.append(list(pass_log.conditional_branch_stack))
                expected_stack_ops[stack_signature] = []
            expected_stack_ops[stack_signature].append(call_index)

        if [list(stack) for stack in layer_log.conditional_role_stacks] != expected_stack_order:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_role_stacks="
                f"{layer_log.conditional_role_stacks} != expected {expected_stack_order}",
            )
        if layer_log.conditional_branch_stack_ops != expected_stack_ops:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_branch_stack_ops="
                f"{layer_log.conditional_branch_stack_ops} != expected "
                f"{expected_stack_ops}",
            )


def _check_conditional_rolled_edge_call_indices(
    ml: Trace,
    name: str,
) -> None:
    """Check rolled conditional edge call indices.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 10: rolled conditional_edge_call_indices reference known
    # layer-level arm-entry edges. Exact pass lists live only in
    # conditional_edge_call_indices after the label remap.
    actual_arm_edges: set[tuple[str, str, int, str]] = set()
    for (conditional_id, branch_kind), edge_list in ml._graph_conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            actual_arm_edges.add(
                (
                    _strip_pass_suffix(parent_label),
                    _strip_pass_suffix(child_label),
                    conditional_id,
                    branch_kind,
                )
            )

    for edge_key, call_indexs in ml._graph_conditional_edge_call_indices.items():
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        if call_indexs != sorted(call_indexs) or len(call_indexs) != len(set(call_indexs)):
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_edge_call_indices[{edge_key}] has unsorted or duplicate pass list "
                f"{call_indexs}",
            )
        for call_index in call_indexs:
            if call_index < 1:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes invalid pass {call_index}",
                )
            actual_edge = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if actual_edge not in actual_arm_edges:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes pass metadata but "
                    f"conditional_arm_entry_edges has no matching layer edge",
                )

    for actual_edge in sorted(actual_arm_edges):
        parent_no_pass, child_no_pass, conditional_id, branch_kind = actual_edge
        edge_key = (parent_no_pass, child_no_pass, conditional_id, branch_kind)
        if edge_key not in ml._graph_conditional_edge_call_indices:
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_arm_entry_edges implies rolled edge {actual_edge} but "
                f"conditional_edge_call_indices[{edge_key}]={ml._graph_conditional_edge_call_indices.get(edge_key)}",
            )


def _check_conditional_transient_bool_keys_removed(
    ml: Trace,
    name: str,
) -> None:
    """Check transient bool conditional keys were removed.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 11: no transient _bool_conditional_key remains after step 5c.
    for layer in ml.layer_list:
        if hasattr(layer, "_bool_conditional_key"):
            _fail_conditional_invariant(
                name,
                11,
                f"{layer.layer_label} still has transient _bool_conditional_key attribute",
            )


def _check_conditional_arm_child_pass_union(
    ml: Trace,
    name: str,
) -> None:
    """Check layer conditional arm children are exact pass unions.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 12: Layer conditional_arm_children is the exact pass union.
    for layer_log in ml.layer_logs.values():
        expected_children_by_cond = _expected_layer_log_child_union(layer_log)
        if layer_log.conditional_arm_children != expected_children_by_cond:
            _fail_conditional_invariant(
                name,
                12,
                f"Layer {layer_log.layer_label}.conditional_arm_children="
                f"{layer_log.conditional_arm_children} != expected pass union "
                f"{expected_children_by_cond}",
            )


def _check_conditional_branch_entry_edges(
    ml: Trace,
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check legacy branch edges match conditional entry children.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 13: legacy IF-view conditional_branch_edges ↔ start-children.
    for parent_label, bool_label in ml._graph_conditional_branch_edges:
        if parent_label not in layer_label_set:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges references missing parent layer {parent_label!r}",
            )
        parent_layer = ml.layer_logs[parent_label]
        if bool_label not in parent_layer.conditional_entry_children:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges includes ({parent_label!r}, {bool_label!r}) but "
                f"{parent_label}.conditional_entry_children={parent_layer.conditional_entry_children}",
            )

    for parent_layer in ml.layer_logs.values():
        for bool_label in parent_layer.conditional_entry_children:
            if (parent_layer.layer_label, bool_label) not in ml._graph_conditional_branch_edges:
                _fail_conditional_invariant(
                    name,
                    13,
                    f"{parent_layer.layer_label}.conditional_entry_children includes "
                    f"{bool_label!r} but conditional_branch_edges={ml._graph_conditional_branch_edges}",
                )
