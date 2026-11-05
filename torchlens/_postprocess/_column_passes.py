"""Postprocess pass bodies that read and write ``Op`` fields through column views.

Each pass here keeps its facade body as the fallback and runs the column body
whenever every raw op shares one row store and raw-label lookups resolve
through the raw layer dict (no live capture-event index), which is what
``Trace.__getitem__`` does during the pass. Both bodies visit the same rows in
the same order and perform the same reads and writes, so the stored result is
identical (checked by per-pass store fingerprints over the corpus).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._data_classes._op_columns import column_rows_by_label

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _column_lookup(trace: Trace) -> Any:
    """Return ``(columns, rows_by_label)`` when the column body applies, else ``None``."""

    if trace.__dict__.get("capture_events") is not None:
        return None
    return column_rows_by_label(trace._raw_graph_ws.raw_layer_dict)


def _row_or_refuse(trace: Trace, rows: dict[str, int], label: str) -> int:
    """Return ``label``'s row, raising the facade's unknown-label refusal otherwise."""

    row = rows.get(label)
    if row is None:
        trace[label]  # raises the typed op_lookup_not_found refusal
        raise KeyError(label)
    return row


def find_output_ancestors(trace: Trace) -> None:
    """Step 2: Mark every node that is an ancestor of an output node.

    Walks the capture DAG in reverse topological order so every child's complete
    descendant set is finalized before it is propagated to a parent. This keeps
    ancestry metadata complete even when optional Step 4 distance computation is
    disabled.
    """

    labels = trace._raw_graph_ws.raw_layer_labels_list
    lookup = _column_lookup(trace)
    if lookup is None:
        _find_output_ancestors_facade(trace, labels)
        return
    columns, rows = lookup
    children = columns.getter("children")
    has_output_descendant = columns.getter("has_output_descendant")
    mark_output_descendant = columns.setter("has_output_descendant")
    output_descendants = columns.getter("output_descendants")
    for node_label in reversed(labels):
        row = _row_or_refuse(trace, rows, node_label)
        for child_label in children(row):
            child_row = _row_or_refuse(trace, rows, child_label)
            if has_output_descendant(child_row):
                mark_output_descendant(row, True)
                output_descendants(row).update(output_descendants(child_row))


def _find_output_ancestors_facade(trace: Trace, labels: list[str]) -> None:
    """Facade body of :func:`find_output_ancestors` for non-column traces."""

    for node_label in reversed(labels):
        node = trace[node_label]
        for child_node_label in node.children:
            child = trace[child_node_label]
            if child.has_output_descendant:
                node.has_output_descendant = True
                node.output_descendants.update(child.output_descendants)
