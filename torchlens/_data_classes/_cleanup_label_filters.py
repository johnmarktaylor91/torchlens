"""Removed-label substitution filters for conditional metadata during cleanup."""

from __future__ import annotations


def _map_removed_label(
    label: str, labels_to_remove: set[str], replacements: dict[str, str] | None
) -> str | None:
    """Return the surviving label for one reference, or ``None`` to drop it.

    Args:
        label: Referenced label.
        labels_to_remove: Labels removed in this pass.
        replacements: Optional removed-label -> survivor substitutions (merge
            removals repoint references; plain removals drop them).

    Returns:
        ``label`` itself when it survives, the substituted survivor when a
        replacement is known, else ``None``.
    """
    if label not in labels_to_remove:
        return label
    if replacements is not None:
        return replacements.get(label)
    return None


def _filter_conditional_arm_children(
    conditional_arm_children: dict[int, dict[str, list[str]]],
    labels_to_remove: set[str],
    replacements: dict[str, str] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Drop or substitute removed labels in ``conditional_arm_children``.

    Args:
        conditional_arm_children: ``cond_id -> branch_kind -> child labels``.
        labels_to_remove: Labels that should be removed.
        replacements: Optional removed-label -> survivor substitutions.

    Returns:
        A new nested dict with removed labels substituted or pruned, empty
        containers pruned, and substituted duplicates deduplicated in order.
    """
    filtered_children_by_cond: dict[int, dict[str, list[str]]] = {}
    for cond_id, branch_children in conditional_arm_children.items():
        filtered_branch_children: dict[str, list[str]] = {}
        for branch_kind, child_labels in branch_children.items():
            kept_children: list[str] = []
            kept_seen: set[str] = set()
            for child_label in child_labels:
                mapped = _map_removed_label(child_label, labels_to_remove, replacements)
                if mapped is not None and mapped not in kept_seen:
                    kept_seen.add(mapped)
                    kept_children.append(mapped)
            if kept_children:
                filtered_branch_children[branch_kind] = kept_children
        if filtered_branch_children:
            filtered_children_by_cond[cond_id] = filtered_branch_children
    return filtered_children_by_cond


def _filter_conditional_arm_entry_edges(
    conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]],
    labels_to_remove: set[str],
    replacements: dict[str, str] | None = None,
) -> dict[tuple[int, str], list[tuple[str, str]]]:
    """Drop or substitute removed labels in ``conditional_arm_entry_edges``.

    Args:
        conditional_arm_entry_edges: ``(cond_id, branch_kind) -> [(parent, child)]``.
        labels_to_remove: Labels that should be removed.
        replacements: Optional removed-label -> survivor substitutions.

    Returns:
        A new dict with empty edge lists pruned and substituted duplicate
        edges deduplicated in order.
    """
    filtered_arm_edges: dict[tuple[int, str], list[tuple[str, str]]] = {}
    for key, edge_list in conditional_arm_entry_edges.items():
        filtered_edges: list[tuple[str, str]] = []
        filtered_seen: set[tuple[str, str]] = set()
        for parent, child in edge_list:
            mapped_parent = _map_removed_label(parent, labels_to_remove, replacements)
            mapped_child = _map_removed_label(child, labels_to_remove, replacements)
            if mapped_parent is None or mapped_child is None:
                continue
            mapped_edge = (mapped_parent, mapped_child)
            if mapped_edge not in filtered_seen:
                filtered_seen.add(mapped_edge)
                filtered_edges.append(mapped_edge)
        if filtered_edges:
            filtered_arm_edges[key] = filtered_edges
    return filtered_arm_edges


def _filter_conditional_edge_call_indices(
    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]],
    labels_to_remove_no_pass: set[str],
    replacements_no_pass: dict[str, str] | None = None,
) -> dict[tuple[str, str, int, str], list[int]]:
    """Drop or substitute removed labels in ``conditional_edge_call_indices`` keys.

    Args:
        conditional_edge_call_indices: ``(parent_no_pass, child_no_pass, cond_id, branch_kind) -> pass list``.
        labels_to_remove_no_pass: Pass-stripped labels that should be removed.
        replacements_no_pass: Optional pass-stripped removed-label -> survivor
            substitutions. A substituted key colliding with an existing key
            merges the two pass lists (sorted union).

    Returns:
        A new dict with removed-key entries substituted or pruned.
    """
    filtered_indices: dict[tuple[str, str, int, str], list[int]] = {}
    for key, call_indexs in conditional_edge_call_indices.items():
        mapped_parent = _map_removed_label(key[0], labels_to_remove_no_pass, replacements_no_pass)
        mapped_child = _map_removed_label(key[1], labels_to_remove_no_pass, replacements_no_pass)
        if mapped_parent is None or mapped_child is None:
            continue
        mapped_key = (mapped_parent, mapped_child, key[2], key[3])
        existing = filtered_indices.get(mapped_key)
        if existing is None:
            filtered_indices[mapped_key] = list(call_indexs)
        else:
            filtered_indices[mapped_key] = sorted(set(existing) | set(call_indexs))
    return filtered_indices
