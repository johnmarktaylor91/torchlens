"""Conditional child-label renaming for ``labeling._replace_layer_names_for_layer_entry``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._utils._record_group_table import record_member


def _rename_elif_children(
    conditional_elif_children: dict[int, list[str]],
    mapping: dict[str, str],
) -> dict[int, list[str]]:
    """Rename labels inside ``conditional_elif_children``.

    Args:
        conditional_elif_children: Mapping from elif index to child labels.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        A new dict with all child labels rewritten.
    """
    return {
        elif_ix: [mapping[layer_label] for layer_label in child_labels]
        for elif_ix, child_labels in conditional_elif_children.items()
    }


def _rename_children_by_cond(
    conditional_arm_children: dict[int, dict[str, list[str]]],
    mapping: dict[str, str],
) -> dict[int, dict[str, list[str]]]:
    """Rename labels inside ``conditional_arm_children``.

    Args:
        conditional_arm_children: ``cond_id -> branch_kind -> child labels``.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        A new dict with every nested child label rewritten.
    """
    return {
        cond_id: {
            branch_kind: [mapping[layer_label] for layer_label in child_labels]
            for branch_kind, child_labels in branch_children.items()
        }
        for cond_id, branch_children in conditional_arm_children.items()
    }


def _rename_conditional_child_views(
    layer_entry: Any,
    layer_mapping: dict[str, str],
    set_entry_field: Callable[[str, Any], None],
) -> None:
    """Rename the then/elif/else conditional child views of one Op on their own.

    Used when the Op carries no ``conditional_arm_children`` to re-project the
    views from; each view that is set is renamed to final layer labels.

    Args:
        layer_entry: Op (or lightweight test double) being renamed.
        layer_mapping: Raw-to-final parent-layer label mapping.
        set_entry_field: Writer that sets one renamed field on ``layer_entry``.
    """
    elif_children = record_member(layer_entry, "conditional_elif_children", None)
    if elif_children:
        set_entry_field(
            "conditional_elif_children",
            _rename_elif_children(elif_children, layer_mapping),
        )
    conditional_then_children_value = record_member(layer_entry, "conditional_then_children", None)
    if conditional_then_children_value:
        set_entry_field(
            "conditional_then_children",
            [layer_mapping[layer_label] for layer_label in conditional_then_children_value],
        )
    conditional_else_children_value = record_member(layer_entry, "conditional_else_children", None)
    if conditional_else_children_value:
        set_entry_field(
            "conditional_else_children",
            [layer_mapping[layer_label] for layer_label in conditional_else_children_value],
        )
