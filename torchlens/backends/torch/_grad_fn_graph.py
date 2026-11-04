"""Backward graph relations of ``GradFn`` records: parents, children, siblings, co-parents.

``_sync_grad_fn_graph_relations`` derives every relation from scratch.
``_extend_grad_fn_graph_relations`` updates them in place when the projection
fold appends records discovered after every existing one, touching only the
records whose relations the appended records change; both share the sibling
and co-parent rules, so the two agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._data_classes.grad_fn import GradFn


def _sibling_labels(
    label: str, parents: Sequence[str], children_of: Callable[[str], Sequence[str]]
) -> list[str]:
    """Return the other children of a record's parents, first-seen order, no repeats."""

    sibling_labels: list[str] = []
    sibling_membership: set[str] = set()
    for parent_label in parents:
        for sibling_label in children_of(parent_label):
            if sibling_label != label and sibling_label not in sibling_membership:
                sibling_labels.append(sibling_label)
                sibling_membership.add(sibling_label)
    return sibling_labels


def _co_parent_labels(
    label: str, children: Sequence[str], parents_of: Callable[[str], Sequence[str]]
) -> list[str]:
    """Return the other parents of a record's children, first-seen order, no repeats."""

    co_parent_labels: list[str] = []
    co_parent_membership: set[str] = set()
    for child_label in children:
        for co_parent_label in parents_of(child_label):
            if co_parent_label != label and co_parent_label not in co_parent_membership:
                co_parent_labels.append(co_parent_label)
                co_parent_membership.add(co_parent_label)
    return co_parent_labels


def _sync_grad_fn_graph_relations(trace: Any) -> None:
    """Populate backward-oriented GradFn graph relation labels.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` should be synchronized.
    """

    id_to_label = {
        grad_fn_object_id: grad_fn.label
        for grad_fn_object_id, grad_fn in trace._grad_fn_logs.items()
    }
    child_map: dict[str, list[str]] = {}
    parent_map: dict[str, list[str]] = {
        grad_fn.label: [] for grad_fn in trace._grad_fn_logs.values()
    }
    parent_membership: dict[str, set[str]] = {
        grad_fn.label: set() for grad_fn in trace._grad_fn_logs.values()
    }

    for grad_fn in trace._grad_fn_logs.values():
        children = [
            id_to_label[next_grad_fn_id]
            for next_grad_fn_id in grad_fn.next_grad_fn_ids
            if next_grad_fn_id in id_to_label
        ]
        child_map[grad_fn.label] = children
        for child_label in children:
            if grad_fn.label not in parent_membership[child_label]:
                parent_map[child_label].append(grad_fn.label)
                parent_membership[child_label].add(grad_fn.label)

    def children_of(label: str) -> list[str]:
        return child_map.get(label, [])

    def parents_of(label: str) -> list[str]:
        return parent_map.get(label, [])

    for grad_fn in trace._grad_fn_logs.values():
        grad_fn.children = child_map[grad_fn.label]
        grad_fn.parents = parent_map[grad_fn.label]
        grad_fn.siblings = _sibling_labels(grad_fn.label, grad_fn.parents, children_of)
        grad_fn.co_parents = _co_parent_labels(grad_fn.label, grad_fn.children, parents_of)


def _extend_grad_fn_graph_relations(
    grad_fn_logs: Mapping[int, GradFn],
    new_ids: Sequence[int],
    label_to_id: Mapping[str, int],
) -> None:
    """Update relations for records appended to ``grad_fn_logs``, equal to a scratch sync.

    The appended records come after every existing record and no existing
    record lists one of them as a child, so each existing record keeps its
    children, an existing child gains the appended parents at the end of its
    parent list, and only the siblings of those children and the co-parents
    of their parents can change.

    Parameters
    ----------
    grad_fn_logs:
        Every record, the appended ones last.
    new_ids:
        Object ids of the appended records, in order.
    label_to_id:
        Label to object id for every record, the appended ones included.
    """

    def record(label: str) -> GradFn:
        return grad_fn_logs[label_to_id[label]]

    # Child id -> the appended parents it gained, in order.
    touched_child_ids: dict[int, list[str]] = {}
    for object_id in new_ids:
        grad_fn_logs[object_id].parents = []
    for object_id in new_ids:
        grad_fn_record = grad_fn_logs[object_id]
        child_ids = [
            child_id for child_id in grad_fn_record.next_grad_fn_ids if child_id in grad_fn_logs
        ]
        grad_fn_record.children = [grad_fn_logs[child_id].label for child_id in child_ids]
        for child_id in child_ids:
            child = grad_fn_logs[child_id]
            if grad_fn_record.label not in child.parents:
                child.parents = [*child.parents, grad_fn_record.label]
                touched_child_ids.setdefault(child_id, []).append(grad_fn_record.label)

    def children_of(label: str) -> Sequence[str]:
        return record(label).children

    def parents_of(label: str) -> Sequence[str]:
        return record(label).parents

    new_id_set = set(new_ids)
    co_parent_ids: dict[int, None] = dict.fromkeys(new_ids)
    for child_id in [*new_ids, *touched_child_ids]:
        child = grad_fn_logs[child_id]
        child.siblings = _sibling_labels(child.label, child.parents, children_of)
        if child_id in touched_child_ids:
            co_parent_ids.update(dict.fromkeys(label_to_id[label] for label in child.parents))
    for object_id in co_parent_ids:
        grad_fn_record = grad_fn_logs[object_id]
        sole_children = set(grad_fn_record.children)
        if object_id not in new_id_set and len(sole_children) == 1:
            # One distinct child: the scratch rule reads exactly that child's
            # parents, so its appended parents land at the end.
            (child_label,) = sole_children
            appended = [
                label
                for label in touched_child_ids.get(label_to_id[child_label], [])
                if label != grad_fn_record.label and label not in grad_fn_record.co_parents
            ]
            if appended:
                grad_fn_record.co_parents = [*grad_fn_record.co_parents, *appended]
            continue
        grad_fn_record.co_parents = _co_parent_labels(
            grad_fn_record.label, grad_fn_record.children, parents_of
        )
