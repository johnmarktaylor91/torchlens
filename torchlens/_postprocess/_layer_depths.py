"""Step 4 of graph traversal: optional layer-depth marking.

``_mark_layer_depths`` runs a forward flood from the input nodes and a backward
flood from the output nodes, recording on every op the min/max hop counts from
each side. It is the last of the four graph-traversal steps whose first three
live in ``graph_traversal``; it depends on nothing but the trace's raw graph
workspace.

Each flood has two bodies with the same stored result. The column body works on
the raw rows of a plain building ``OpRowStore`` and runs only after a read-only
scan has shown that every cell it will touch holds exactly what the ``Op``
facade would hand back unchanged (or a default-omitted cell whose answer is
known), so it visits the same nodes in the same order and performs the same
reads and writes without a call per field access. Any other trace keeps the
facade body.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from .._trace_core.cell_states import LazyCell
from .._trace_core.op_store import OpRowStore
from .._utils._record_group_table import op_attr
from ._column_passes import _column_lookup

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

#: Flood mode -> stored fields: min distance, max distance, reached marker,
#: lineage set, and the relation the flood follows.
_MODE_FIELDS: dict[str, tuple[str, str, str, str, str]] = {
    "input": (
        "min_distance_from_input",
        "max_distance_from_input",
        "has_input_ancestor",
        "input_ancestors",
        "children",
    ),
    "output": (
        "min_distance_to_output",
        "max_distance_to_output",
        "has_output_descendant",
        "output_descendants",
        "parents",
    ),
}


def _mark_layer_depths(self: Trace) -> None:
    """Step 4: Compute min/max hop distances from inputs and outputs.

    Runs two unidirectional floods: forward from inputs (following children)
    and backward from outputs (following parents). Each flood records
    min_distance_from_{input,output} and max_distance_from_{input,output} on
    every reachable node.

    This step is CONDITIONAL on ``self._options_mark_layer_depths`` — it is
    skipped when the user doesn't need distance metadata.
    """
    lookup = _column_lookup(self)
    for mode in ("input", "output"):
        if lookup is None or not _flood_columns(self, mode, lookup[0], lookup[1]):
            _flood_graph_from_input_or_output_nodes(self, mode)


def _flood_columns(self: Trace, mode: str, columns: Any, rows: dict[str, int]) -> bool:
    """Run one flood on the raw rows; return False when the facade body must run.

    Nothing is written unless the whole flood can run here: the scan refuses
    first, so a False return leaves the trace exactly as it found it.
    """

    store = columns.store
    if store.__class__ is not OpRowStore or store._rows is None:
        return False
    fids = _bound_fids(columns, _MODE_FIELDS[mode])
    if fids is None:
        return False
    labels = self._raw_graph_ws.raw_layer_labels_list
    if mode == "input":
        starts = self.input_layers[:]
        order: Sequence[str] = labels
    else:
        starts = self.output_layers[:]
        order = labels[::-1]
    raw_rows = store._rows
    try:
        plain = _value_cells_are_plain(raw_rows, rows, fids) and _graph_cells_are_plain(
            raw_rows, rows, starts, order, fids[4]
        )
    except TypeError:  # an unhashable label: the facade body raises its own error
        plain = False
    if plain:
        _flood_plain_rows(store, rows, starts, order, fids)
    return plain


def _bound_fids(columns: Any, names: tuple[str, str, str, str, str]) -> tuple[int, ...] | None:
    """Return the field ids of one mode, or ``None`` when the facade must serve a field.

    Binding the column accessors is the check: it refuses a field whose reads
    or writes the facade intercepts and a field served by a descriptor the
    column view does not model.
    """

    try:
        for name in names[:2]:
            columns.getter(name)
            columns.setter(name)
        columns.setter(names[2])
        columns.getter(names[3])
        columns.getter(names[4])
    except ValueError:
        return None
    fid_by_name = columns.store.layout.fid_by_name
    return tuple([fid_by_name[name] for name in names])


def _value_cells_are_plain(
    raw_rows: list[Any], rows: dict[str, int], fids: tuple[int, ...]
) -> bool:
    """Return whether every distance is unset and every lineage cell reads back unchanged.

    A distance cell must hold ``None`` or the default-omitted marker of a
    ``None`` default, so the flood starts from nothing and every minimum it
    writes stays at or below its maximum; a lineage cell must hold a ``set`` or
    the default-omitted marker whose factory is ``set``. Classes are compared
    exactly, so a preset, pooled, encoded or derived cell sends the flood to
    the facade.
    """

    lineage_fid = fids[3]
    for row in rows.values():
        cells = raw_rows[row]
        for fid in fids[:2]:
            value = cells[fid]
            if value is not None and (
                value.__class__ is not LazyCell or value.make or value.source is not None
            ):
                return False
        value = cells[lineage_fid]
        if value.__class__ is not set and (
            value.__class__ is not LazyCell or value.source is not set
        ):
            return False
    return True


def _graph_cells_are_plain(
    raw_rows: list[Any],
    rows: dict[str, int],
    starts: Sequence[str],
    order: Sequence[str],
    forward_fid: int,
) -> bool:
    """Return whether every label the flood resolves is a raw op of this store.

    A forward-relation cell must hold a ``list`` or ``tuple`` of known labels,
    or the default-omitted marker of an empty default.
    """

    # Complexity: a read-only scan with cell tests inlined on purpose; a helper
    # call per op here would cost what the column body saves.
    if not rows.keys() >= set(starts) or not rows.keys() >= set(order):
        return False
    for label in order:
        targets = raw_rows[rows[label]][forward_fid]
        kind = targets.__class__
        if kind is LazyCell:
            if targets.source is not list and (targets.make or targets.source != ()):
                return False
        elif kind is not list and kind is not tuple:
            return False
        else:
            for target in targets:
                if target not in rows:
                    return False
    return True


def _seed_plain_rows(
    store: Any, rows: dict[str, int], starts: Sequence[str], fids: tuple[int, ...]
) -> None:
    """Give each start node distance zero, the reached marker and its own label."""

    min_fid, max_fid, marker_fid, lineage_fid = fids[:4]
    raw_rows = store._rows
    cell_get = store.cell_get
    for label in starts:
        row = rows[label]
        cells = raw_rows[row]
        low = cells[min_fid]
        if low.__class__ is not int or low > 0:
            cells[min_fid] = 0
        high = cells[max_fid]
        if high.__class__ is not int or high < 0:
            cells[max_fid] = 0
        cells[marker_fid] = True
        lineage = cells[lineage_fid]
        if lineage.__class__ is not set:
            lineage = cell_get(row, lineage_fid)
        lineage.add(label)


def _flood_plain_rows(
    store: Any,
    rows: dict[str, int],
    starts: Sequence[str],
    order: Sequence[str],
    fids: tuple[int, ...],
) -> None:
    """Column body of one flood over rows the two scans accepted.

    Mirrors :func:`_flood_graph_from_input_or_output_nodes` read for read and
    write for write. A cell that is not an ``int`` is an unset distance; a
    default-omitted lineage or relation cell is answered by the store exactly
    where the facade would read it. A tie is written as the facade body writes
    it, so equal distances keep sharing one ``int`` object.
    """

    # Complexity: one hot loop with field access inlined on purpose; a call per
    # access is the cost this body removes, so it is not split into helpers.

    min_fid, max_fid, marker_fid, lineage_fid, forward_fid = fids
    raw_rows = store._rows
    cell_get = store.cell_get
    _seed_plain_rows(store, rows, starts, fids)
    for label in order:
        row = rows[label]
        cells = raw_rows[row]
        low = cells[min_fid]
        high = cells[max_fid]
        if low.__class__ is not int or high.__class__ is not int:
            continue
        lineage = cells[lineage_fid]
        if lineage.__class__ is not set:
            lineage = cell_get(row, lineage_fid)
        targets = cells[forward_fid]
        if targets.__class__ is LazyCell:
            targets = cell_get(row, forward_fid)
        low += 1
        high += 1
        if high == low:
            low = high
        for target_label in targets:
            target_row = rows[target_label]
            target = raw_rows[target_row]
            current = target[min_fid]
            if current.__class__ is not int or low <= current:
                target[min_fid] = low
            current = target[max_fid]
            if current.__class__ is not int or high >= current:
                target[max_fid] = high
            target[marker_fid] = True
            reached = target[lineage_fid]
            if reached.__class__ is not set:
                reached = cell_get(target_row, lineage_fid)
            reached |= lineage


def _flood_graph_from_input_or_output_nodes(self: Trace, mode: str) -> None:
    """Flood the graph from input or output nodes, tracking min/max distance.

    Traverses unidirectionally from starting nodes (input or output), recording
    each node's min and max hop count from the start. Also marks each node's
    ancestry (input_ancestors or output_descendants).

    Unlike the bidirectional flood in Step 3, this flood is UNIDIRECTIONAL:
    from inputs it follows children (forward), from outputs it follows
    parents (backward). This ensures hop counts reflect actual data-flow
    distance, not arbitrary graph traversal paths.

    Nodes are processed once in topological order: forward from inputs, or
    backward from outputs. Each visited node propagates finalized distance and
    lineage state to its data-flow successors.

    Args:
        mode: 'input' to flood forward from inputs, 'output' to flood backward from outputs.
    """
    traversal_order: Iterable[str]
    if mode == "input":
        starting_nodes = self.input_layers[:]
        min_field = op_attr("min_distance_from_input")
        max_field = op_attr("max_distance_from_input")
        marker_field = op_attr("has_input_ancestor")
        layer_logging_field = "input_ancestors"
        forward_field = "children"
        traversal_order = self._raw_graph_ws.raw_layer_labels_list
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = op_attr("min_distance_to_output")
        max_field = op_attr("max_distance_to_output")
        marker_field = op_attr("has_output_descendant")
        layer_logging_field = "output_descendants"
        forward_field = "parents"
        traversal_order = reversed(self._raw_graph_ws.raw_layer_labels_list)
    else:
        raise ValueError("Mode must be either 'input' or 'output'")

    for starting_node_label in starting_nodes:
        starting_node = self[starting_node_label]
        _update_node_distance_vals(starting_node, min_field, max_field, 0)
        setattr(starting_node, marker_field, True)
        getattr(starting_node, layer_logging_field).add(starting_node_label)

    for current_node_label in traversal_order:
        current_node = self[current_node_label]
        current_min = getattr(current_node, min_field)
        current_max = getattr(current_node, max_field)
        if current_min is None or current_max is None:
            continue

        current_lineage = getattr(current_node, layer_logging_field)
        for next_node_label in getattr(current_node, forward_field):
            next_node = self[next_node_label]
            _update_node_distance_vals(next_node, min_field, max_field, current_min + 1)
            _update_node_distance_vals(next_node, min_field, max_field, current_max + 1)
            setattr(next_node, marker_field, True)
            getattr(next_node, layer_logging_field).update(current_lineage)


def _update_node_distance_vals(
    current_node: Op,
    min_field: str,
    max_field: str,
    nodes_since_start: int,
) -> None:
    """Update a node's min/max distance fields if the current hop count is a new extreme."""
    if getattr(current_node, min_field) is None:
        setattr(current_node, min_field, nodes_since_start)
    else:
        setattr(
            current_node,
            min_field,
            min(nodes_since_start, getattr(current_node, min_field)),
        )

    if getattr(current_node, max_field) is None:
        setattr(current_node, max_field, nodes_since_start)
    else:
        setattr(
            current_node,
            max_field,
            max(nodes_since_start, getattr(current_node, max_field)),
        )
