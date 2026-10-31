"""Recurrence and foundational conditional invariants."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _check_conditional_arm_child_pass_union,
        _check_conditional_arm_edges_match_graph,
        _check_conditional_bool_classification,
        _check_conditional_bool_event_backrefs,
        _check_conditional_branch_entry_edges,
        _check_conditional_branch_membership_records,
        _check_conditional_branch_stack_monotonicity,
        _check_conditional_child_labels_resolve,
        _check_conditional_elif_key_contiguity,
        _check_conditional_event_references,
        _check_conditional_layer_aggregate_views,
        _check_conditional_public_accessor_summary,
        _check_conditional_rolled_edge_call_indices,
        _check_conditional_transient_bool_keys_removed,
    )

__all__ = (
    "_dtype_values_match",
    "_check_recurrence_invariants",
    "_check_branching_invariants",
    "_fail_conditional_invariant",
    "_strip_pass_suffix",
    "_append_unique",
    "_is_prefix_stack",
    "_expected_layer_pass_child_views",
    "_expected_layer_log_child_views",
    "_expected_layer_log_child_union",
    "_valid_conditional_child_labels",
    "_check_conditional_invariants",
    "_check_conditional_arm_entry_child_symmetry",
    "_check_conditional_derived_child_views",
)


def _dtype_values_match(left: object, right: object) -> bool:
    """Return whether two dtype representations are equivalent.

    Parameters
    ----------
    left:
        Recorded dtype value.
    right:
        Payload dtype value.

    Returns
    -------
    bool
        ``True`` when exact or normalized string forms agree.
    """

    if left == right:
        return True
    left_str = str(left).replace("torch.", "")
    right_str = str(right).replace("torch.", "")
    return left_str == right_str


def _check_recurrence_invariants(ml: Trace) -> None:
    """Check E: recurrence / loop invariants.

    Validates:
    - is_recurrent == True iff any layer has >1 pass.
    - max_layer_op_count matches the maximum pass count.
    - layer_num_calls keys are valid no-pass labels.
    - Layer.ops dict keys are contiguous {1..N}.
    """
    name = "recurrence_invariants"

    any_recurrent = any(v > 1 for v in ml.layer_num_calls.values())
    if ml.graph.is_recurrent != any_recurrent:
        raise MetadataInvariantError(
            name,
            f"is_recurrent={ml.graph.is_recurrent} but any layer has >1 pass = {any_recurrent}",
        )

    if ml.graph.is_recurrent:
        expected_max = max(ml.layer_num_calls.values())
        if ml.graph.max_ops_per_layer != expected_max:
            raise MetadataInvariantError(
                name,
                f"max_layer_op_count={ml.graph.max_ops_per_layer} != "
                f"max(layer_num_calls)={expected_max}",
            )

    # Per-layer pass consistency: layer_num_calls is keyed by no-pass labels.
    # Validate that each key exists in layer_labels, and that the
    # recorded count matches the actual Layer.num_passes.
    no_call_labels = set(ml.layer_labels)
    for label_key, num_calls in ml.layer_num_calls.items():
        if label_key not in no_call_labels:
            raise MetadataInvariantError(
                name,
                f"layer_num_calls key '{label_key}' not in layer_labels",
            )
        if label_key in ml.layer_logs:
            actual = ml.layer_logs[label_key].num_passes
            if num_calls != actual:
                raise MetadataInvariantError(
                    name,
                    f"layer_num_calls['{label_key}']={num_calls} != Layer.num_passes={actual}",
                )

    # For top-level (no-pass) layer_logs, verify pass dict consistency
    for no_call_label, ll in ml.layer_logs.items():
        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{no_call_label}' ops keys={actual_keys} != expected {expected_keys}",
            )


def _check_branching_invariants(ml: Trace) -> None:
    """Check F: is_branching matches whether any layer has >1 child."""
    name = "branching_invariants"
    any_branching = any(len(lpl.children) > 1 for lpl in ml.layer_list)
    if ml.graph.is_branching != any_branching:
        raise MetadataInvariantError(
            name,
            f"is_branching={ml.graph.is_branching} but any layer has >1 child = {any_branching}",
        )


def _fail_conditional_invariant(check_name: str, number: int, message: str) -> None:
    """Raise a numbered conditional metadata invariant failure.

    Parameters
    ----------
    check_name:
        ``MetadataInvariantError.check_name`` value for this check family.
    number:
        Conditional invariant number within this check family.
    message:
        Human-readable failure details.
    """

    raise MetadataInvariantError(check_name, f"Invariant {number}: {message}")


def _strip_pass_suffix(layer_label: str) -> str:
    """Return a layer label without any trailing ``:call_index`` suffix.

    Parameters
    ----------
    layer_label:
        Layer label that may include a pass suffix.

    Returns
    -------
    str
        Pass-stripped label.
    """

    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return label_parts[0]
    return layer_label


def _append_unique(values: list[str], value: str) -> None:
    """Append ``value`` to ``values`` only if not already present.

    Parameters
    ----------
    values:
        Ordered list being built.
    value:
        Candidate value to append.
    """

    if value not in values:
        values.append(value)


def _is_prefix_stack(prefix: list[tuple[int, str]], full: list[tuple[int, str]]) -> bool:
    """Return whether one conditional branch stack prefixes another.

    Parameters
    ----------
    prefix:
        Candidate prefix stack.
    full:
        Candidate full stack.

    Returns
    -------
    bool
        ``True`` if ``prefix`` matches the first ``len(prefix)`` entries of
        ``full``.
    """

    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def _expected_layer_pass_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project pass-level child views from the primary conditional structure.

    Parameters
    ----------
    conditional_arm_children:
        Primary ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Op``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views.
    """

    then_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("then", [])
        }
    )
    elif_children: dict[int, list[str]] = {}
    grouped_elif_children: dict[int, set[str]] = defaultdict(set)
    for branch_children in conditional_arm_children.values():
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            grouped_elif_children[elif_index].update(child_labels)
    for elif_index, elif_label_set in sorted(grouped_elif_children.items()):
        elif_children[elif_index] = sorted(elif_label_set)

    else_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("else", [])
        }
    )
    return then_children, elif_children, else_children


def _expected_layer_log_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project aggregate child views from a ``Layer`` primary structure.

    Parameters
    ----------
    conditional_arm_children:
        Aggregate ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Layer``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views preserving first-seen order.
    """

    then_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    else_children: list[str] = []
    for branch_children in conditional_arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique(then_children, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            expected_children = elif_children.setdefault(elif_index, [])
            for child_label in child_labels:
                _append_unique(expected_children, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique(else_children, child_label)
    return then_children, elif_children, else_children


def _expected_layer_log_child_union(
    layer_log: Layer,
) -> dict[int, dict[str, list[str]]]:
    """Build the expected aggregate ``conditional_arm_children`` for a ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being validated.

    Returns
    -------
    dict[int, dict[str, list[str]]]
        Pass-stripped union of every pass-level child list.
    """

    expected_children_by_cond: dict[int, dict[str, list[str]]] = {}
    for call_index, pass_log in sorted(layer_log.ops.items()):
        for conditional_id, branch_children in pass_log.conditional_arm_children.items():
            merged_branch_children = expected_children_by_cond.setdefault(conditional_id, {})
            for branch_kind, child_labels in branch_children.items():
                merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
                for child_label in child_labels:
                    _append_unique(merged_child_labels, _strip_pass_suffix(child_label))
    return expected_children_by_cond


def _valid_conditional_child_labels(ml: Trace) -> set[str]:
    """Return the set of valid labels for conditional child references.

    Parameters
    ----------
    ml:
        Model log being validated.

    Returns
    -------
    set[str]
        Union of pass-level labels and aggregate ``Layer`` keys.
    """

    return set(ml.layer_labels) | set(ml.layer_logs)


def _check_conditional_invariants(ml: Trace) -> None:
    """Check F2 conditional metadata invariants.

    Parameters
    ----------
    ml:
        Model log being validated.
    """

    name = "conditional_invariants"
    layer_label_set = set(ml.layer_labels)
    valid_child_labels = _valid_conditional_child_labels(ml)
    event_id_set = {event.id for event in ml.conditional_records}
    branch_context_kinds = {"if_test", "elif_test", "ifexp"}
    wrapped_context_kinds = branch_context_kinds | {"bool_cast"}

    _check_conditional_arm_entry_child_symmetry(ml, name, layer_label_set)
    _check_conditional_derived_child_views(ml, name)
    _check_conditional_child_labels_resolve(ml, name, valid_child_labels)
    _check_conditional_bool_classification(
        ml,
        name,
        branch_context_kinds,
        wrapped_context_kinds,
    )
    _check_conditional_event_references(ml, name, event_id_set)
    _check_conditional_branch_stack_monotonicity(ml, name)
    _check_conditional_elif_key_contiguity(ml, name)
    _check_conditional_bool_event_backrefs(ml, name, layer_label_set)
    _check_conditional_layer_aggregate_views(ml, name)
    _check_conditional_rolled_edge_call_indices(ml, name)
    _check_conditional_transient_bool_keys_removed(ml, name)
    _check_conditional_arm_child_pass_union(ml, name)
    _check_conditional_branch_entry_edges(ml, name, layer_label_set)
    _check_conditional_arm_edges_match_graph(ml, name)
    _check_conditional_branch_membership_records(ml, name)
    _check_conditional_public_accessor_summary(ml, name)


def _check_conditional_arm_entry_child_symmetry(
    ml: Trace,
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check conditional arm-entry edge and child-map symmetry.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 1: conditional_arm_entry_edges ↔ conditional_arm_children.
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if parent_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] references "
                    f"missing parent layer {parent_label!r}",
                )
            parent_layer = ml.layer_logs[parent_label]
            branch_children = parent_layer.conditional_arm_children.get(conditional_id, {}).get(
                branch_kind, []
            )
            if child_label not in branch_children:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] includes edge "
                    f"({parent_label!r}, {child_label!r}) but "
                    f"{parent_label}.conditional_arm_children[{conditional_id}][{branch_kind!r}]="
                    f"{branch_children}",
                )

    for parent_layer in ml.layer_logs.values():
        for conditional_id, branch_map in parent_layer.conditional_arm_children.items():
            for branch_kind, child_labels in branch_map.items():
                model_edges = ml.conditional_arm_entry_edges.get((conditional_id, branch_kind), [])
                for child_label in child_labels:
                    if (parent_layer.layer_label, child_label) not in model_edges:
                        _fail_conditional_invariant(
                            name,
                            1,
                            f"{parent_layer.layer_label}.conditional_arm_children"
                            f"[{conditional_id}][{branch_kind!r}] includes {child_label!r} "
                            f"but conditional_arm_entry_edges[{(conditional_id, branch_kind)}]={model_edges}",
                        )


def _check_conditional_derived_child_views(
    ml: Trace,
    name: str,
) -> None:
    """Check derived conditional child views match primary structures.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 2: per-layer derived views are exact projections of the primary structures.
    for layer in ml.layer_list:
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_pass_child_views(layer.conditional_arm_children)
        )
        if list(layer.conditional_then_children) != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_then_children={layer.conditional_then_children} != "
                f"expected projection {expected_then_children}",
            )
        if layer.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_elif_children={layer.conditional_elif_children} != "
                f"expected projection {expected_elif_children}",
            )
        if list(layer.conditional_else_children) != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_else_children={layer.conditional_else_children} != "
                f"expected projection {expected_else_children}",
            )

    for layer_log in ml.layer_logs.values():
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_log_child_views(layer_log.conditional_arm_children)
        )
        if list(layer_log.conditional_then_children) != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_then_children="
                f"{layer_log.conditional_then_children} != expected projection "
                f"{expected_then_children}",
            )
        if layer_log.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_elif_children="
                f"{layer_log.conditional_elif_children} != expected projection "
                f"{expected_elif_children}",
            )
        if list(layer_log.conditional_else_children) != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_else_children="
                f"{layer_log.conditional_else_children} != expected projection "
                f"{expected_else_children}",
            )
