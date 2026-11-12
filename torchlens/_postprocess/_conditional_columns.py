"""Step 5 as a column algorithm over the building op store's integer rows.

Conditional marking reads a handful of op fields and writes a dozen. Through
the ``Op`` facade, or through one ``OpColumns`` getter call per cell, that
costs well over a hundred counted calls per op. This module reads each field
step 5 needs once, as a row-indexed column, and writes results straight into
the row lists, so the per-op and per-edge loops make no call at all; the
per-distinct-call-site work (AST classification, branch-stack translation) is
memoized on the identity of the shared frame objects.

The column bodies apply when the raw ops share one plain building
``OpRowStore`` (no dataflow edges, so every setter writes the raw cell) and
every field step 5 touches is a plain stored column. A cell whose class is not
a plain builtin (a lazy, pooled or grouped cell, an unset cell, a sentinel) is
read through its ``OpColumns`` getter, which answers exactly as the facade.
Every other trace keeps the facade bodies in ``_column_passes_graph`` and
``control_flow``; both leave identical stores (``tests/test_column_passes_graph.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._trace_core.op_store import OpRowStore
from ._branch_attribution import _code_context_position_key
from ._column_passes import _row_or_refuse

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = (
    "StepColumns",
    "assign_branch_stacks",
    "attribute_edges",
    "bind",
    "derive_arm_views",
    "flood_if_edges",
    "reset_if_state",
)

#: Classes every column getter returns unchanged (no slow cell is one of them).
_PLAIN = frozenset({list, dict, tuple, str, int, float, bool, type(None)})
#: Fields step 5 reads and never writes, read once as columns.
_READS = (
    "is_orphan",
    "is_scalar_bool",
    "code_context",
    "children",
    "parents",
    "has_output_descendant",
    "_label_raw",
    "pass_number",
)
#: Fields step 5 writes straight into the rows (``OpColumns.setter``'s fast path).
_WRITES = (
    "is_terminal_bool",
    "conditional_context_kind",
    "conditional_wrapper_kind",
    "is_terminal_conditional_bool",
    "terminal_conditional_id",
    "conditional_entry_children",
    # ``Op.is_in_conditional_body`` is a property whose setter stores this slot.
    "_is_in_conditional_body",
    "conditional_branch_stack",
    "conditional_branch_depth",
    "conditional_arm_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
)


class StepColumns:
    """One trace's step-5 columns: row lists, field ids and the read columns.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    columns:
        ``OpColumns`` over the trace's building store.
    rows:
        Raw label -> store row.
    """

    __slots__ = ("cells", "col", "columns", "fid", "kept", "kept_raw", "rows")

    def __init__(self, trace: Trace, columns: Any, rows: dict[str, int]) -> None:
        """Read every column step 5 reads; a getter refusal propagates to :func:`bind`."""

        store = columns.store
        self.columns = columns
        self.cells: list[list[Any]] = store._rows
        self.rows = rows
        fid_by_name = store.layout.fid_by_name
        self.fid = {name: fid_by_name[name] for name in (*_READS, *_WRITES)}
        for name in _WRITES:
            columns.setter(name)  # raises for a guarded or unknown field
        self.col = {name: self._read(name) for name in _READS}
        orphan = self.col["is_orphan"]
        #: Non-orphan rows in ``Trace.__iter__`` order (the raw layer dict's).
        self.kept = [row for row in rows.values() if not orphan[row]]
        #: Non-orphan rows of the raw label list, in its order.
        raw_rows = [rows[label] for label in trace._raw_graph_ws.raw_layer_labels_list]
        self.kept_raw = [row for row in raw_rows if not orphan[row]]

    def _read(self, name: str) -> list[Any]:
        """Return field ``name`` per store row (``None`` for rows outside the trace)."""

        cells = self.cells
        column: list[Any] = [None] * len(cells)
        if name != "is_orphan":
            # The bulk read tests each cell class once, not each cell, and reads
            # every slow or unset cell through the getter (an unset cell raises).
            row_list = list(self.rows.values())
            for row, value in zip(row_list, self.columns.column(name, row_list), strict=True):
                column[row] = value
            return column
        get = self.columns.getter(name, False)
        fid = self.fid[name]
        plain = _PLAIN
        for row in self.rows.values():
            value = cells[row][fid]
            column[row] = value if value.__class__ in plain else get(row)
        return column

    def read(self, row: int, name: str) -> Any:
        """Return one cell of a written field exactly as the facade reads it."""

        return self.columns.getter(name)(row)


def bind(trace: Trace, lookup: Any) -> StepColumns | None:
    """Return the trace's step-5 columns, or ``None`` when the facade bodies apply.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    lookup:
        ``_column_passes_graph._graph_lookup(trace)``: ``(columns, rows)`` or ``None``.

    Returns
    -------
    StepColumns | None
        ``None`` for a trace whose store is not one plain building row store, a
        field the column view does not model, a raw label with no row, or a cell
        whose getter refuses; the facade then reads and refuses as it always has.
    """

    if lookup is None:
        return None
    columns, rows = lookup
    store = columns.store
    if store.__class__ is not OpRowStore or store._rows is None or store.dataflow_edges is not None:
        return None
    try:
        return StepColumns(trace, columns, rows)
    except (AttributeError, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Terminal bools: seeding, the terminal list and the classification results
# ---------------------------------------------------------------------------


def seed_proven_bools(trace: Trace, plan: StepColumns, proven: frozenset[str]) -> None:
    """Column body of ``control_flow._seed_proven_bool_consumers``."""

    terminated = trace.internally_terminated_bool_ops
    seen = set(terminated)
    rows = plan.rows
    scalar = plan.col["is_scalar_bool"]
    orphan = plan.col["is_orphan"]
    cells = plan.cells
    fid = plan.fid["is_terminal_bool"]
    for label in trace._raw_graph_ws.raw_layer_labels_list:
        if label not in proven:
            continue
        row = rows[label]
        if not scalar[row] or orphan[row]:
            continue
        if label not in seen:
            seen.add(label)
            terminated += [label]
        cells[row][fid] = True


def terminal_bools(trace: Trace, plan: StepColumns) -> list[tuple[str, Any]]:
    """Return ``(raw label, captured frames)`` per terminal scalar bool, in raw order."""

    terminal = set(trace.internally_terminated_bool_ops)
    rows = plan.rows
    scalar = plan.col["is_scalar_bool"]
    orphan = plan.col["is_orphan"]
    frames = plan.col["code_context"]
    bools: list[tuple[str, Any]] = []
    for label in trace._raw_graph_ws.raw_layer_labels_list:
        if label in terminal:
            row = rows[label]
            if scalar[row] and not orphan[row]:
                bools += [(label, frames[row])]
    return bools


def store_bool_results(
    plan: StepColumns,
    labels: list[str],
    outputs: list[tuple[str, str | None, bool]],
    first_ids: dict[str, int | None],
) -> None:
    """Write each terminal bool's classification and its first linked event id."""

    rows = plan.rows
    cells = plan.cells
    fid = plan.fid
    kind_fid = fid["conditional_context_kind"]
    wrapper_fid = fid["conditional_wrapper_kind"]
    conditional_fid = fid["is_terminal_conditional_bool"]
    id_fid = fid["terminal_conditional_id"]
    for label, (kind, wrapper_kind, is_conditional) in zip(labels, outputs, strict=True):
        cell = cells[rows[label]]
        cell[kind_fid] = kind
        cell[wrapper_fid] = wrapper_kind
        cell[conditional_fid] = is_conditional
    for label, first_id in first_ids.items():
        cells[rows[label]][id_fid] = first_id


# ---------------------------------------------------------------------------
# Phase 5d: reset and backward IF flood
# ---------------------------------------------------------------------------


def reset_if_state(plan: StepColumns) -> None:
    """Clear every non-orphan op's IF entry children and conditional-body flag."""

    cells = plan.cells
    entry_fid = plan.fid["conditional_entry_children"]
    body_fid = plan.fid["_is_in_conditional_body"]
    for row in plan.kept:
        cell = cells[row]
        cell[entry_fid] = []
        cell[body_fid] = False


def flood_if_edges(trace: Trace, plan: StepColumns, branch_bool_labels: list[str]) -> None:
    """Backward-flood IF edges from the branch bools over the parent column.

    Cognitive complexity is over 15 on purpose (hot path, once per edge): the
    row lookups and cell stores are inlined so the loop makes no call.
    """

    edges = trace._graph_conditional_branch_edges
    rows = plan.rows
    cells = plan.cells
    parents = plan.col["parents"]
    has_output_descendant = plan.col["has_output_descendant"]
    entry_fid = plan.fid["conditional_entry_children"]
    body_fid = plan.fid["_is_in_conditional_body"]
    nodes_seen: dict[str, None] = {}
    node_stack = branch_bool_labels[:]
    while node_stack:
        node_label = node_stack[-1]
        del node_stack[-1]
        try:
            row = rows[node_label]
        except KeyError:
            row = _row_or_refuse(trace, rows, node_label)
        if node_label in nodes_seen:
            continue
        for parent_label in parents[row]:
            try:
                parent_row = rows[parent_label]
            except KeyError:
                parent_row = _row_or_refuse(trace, rows, parent_label)
            parent_cell = cells[parent_row]
            if has_output_descendant[parent_row]:
                entry_children = parent_cell[entry_fid]
                if entry_children.__class__ is list:
                    entry_children += [node_label]
                else:
                    plan.read(parent_row, "conditional_entry_children").append(node_label)
                parent_cell[body_fid] = False
                nodes_seen[parent_label] = None
                edges += [(parent_label, node_label)]
            elif parent_label not in nodes_seen:
                parent_cell[body_fid] = True
                node_stack += [parent_label]
        nodes_seen[node_label] = None


# ---------------------------------------------------------------------------
# Phase 5e: branch stacks and forward arm edges
# ---------------------------------------------------------------------------


def assign_branch_stacks(
    plan: StepColumns,
    events_by_key: dict[Any, Any],
    translate: Callable[[Any, dict[Any, Any]], list[tuple[int, str]]],
) -> None:
    """Give every non-orphan op its own branch stack, its depth and empty arm children.

    An op's frames are a fresh list of ``FuncCallLocation`` objects shared by
    every op of its call site, so a stack is first memoized on the identity of
    those objects (pinned for the pass) and then, as in the facade body, on the
    six frame fields attribution reads: each distinct stack is translated once.
    """

    cells = plan.cells
    frames_by_row = plan.col["code_context"]
    stack_fid = plan.fid["conditional_branch_stack"]
    depth_fid = plan.fid["conditional_branch_depth"]
    arms_fid = plan.fid["conditional_arm_children"]
    by_identity: dict[tuple[int, ...], tuple[list[tuple[int, str]], int]] = {}
    by_position: dict[Any, tuple[list[tuple[int, str]], int]] = {}
    pinned: list[Any] = []
    for row in plan.kept_raw:
        frames = frames_by_row[row]
        identity = tuple(map(id, frames)) if frames.__class__ is list else None
        if identity is not None and identity in by_identity:
            stack, depth = by_identity[identity]
        else:
            position = _code_context_position_key(frames)
            if position in by_position:
                stack, depth = by_position[position]
            else:
                stack = list(translate(frames, events_by_key))
                depth = len(stack)
                by_position[position] = (stack, depth)
            if identity is not None:
                pinned += [frames]
                by_identity[identity] = (stack, depth)
        cell = cells[row]
        cell[stack_fid] = stack[:]
        cell[depth_fid] = depth
        cell[arms_fid] = {}


def attribute_edges(trace: Trace, plan: StepColumns) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Attribute every forward edge entering a conditional arm; return the two maps.

    Records the entered arm's child on the parent and returns ``(entry edges,
    call indices)`` with the keys in the facade body's insertion order.

    Notes
    -----
    Cognitive complexity is over 15 on purpose (once per gained entry): the map
    inserts are inlined so recording an entry makes no call.
    """

    entry_edges: dict[tuple[int, str], list[tuple[str, str]]] = {}
    call_indices: dict[tuple[str, str, int, str], list[int]] = {}
    cells = plan.cells
    label_raw, pass_number = plan.col["_label_raw"], plan.col["pass_number"]
    arms_fid = plan.fid["conditional_arm_children"]
    base_label: dict[int, str] = {}
    for parent_row, child_row, child_label, gained in _gained_edges(trace, plan):
        arms = cells[parent_row][arms_fid]
        parent_raw = label_raw[parent_row]
        child_raw = label_raw[child_row]
        if parent_row not in base_label:
            base_label[parent_row] = parent_raw.split(":", 1)[0]
        if child_row not in base_label:
            base_label[child_row] = child_raw.split(":", 1)[0]
        for conditional_id, branch_kind in gained:
            if conditional_id not in arms:
                arms[conditional_id] = {}
            by_kind = arms[conditional_id]
            if branch_kind in by_kind:
                by_kind[branch_kind] += [child_label]
            else:
                by_kind[branch_kind] = [child_label]
            edge_key = (conditional_id, branch_kind)
            if edge_key in entry_edges:
                entry_edges[edge_key] += [(parent_raw, child_raw)]
            else:
                entry_edges[edge_key] = [(parent_raw, child_raw)]
            call_key = (base_label[parent_row], base_label[child_row], conditional_id, branch_kind)
            if call_key in call_indices:
                call_indices[call_key] += [pass_number[child_row]]
            else:
                call_indices[call_key] = [pass_number[child_row]]
    return entry_edges, call_indices


def _gained_edges(trace: Trace, plan: StepColumns) -> list[tuple[int, int, str, Any]]:
    """Return ``(parent row, child row, child label, gained entries)`` per arm-entering edge.

    Edges between non-orphan ops in raw parent order and child order; the
    gained entries are the child's stack beyond the shared prefix with the
    parent's (``_column_passes_graph.get_gained_branch_entries``). Equal stacks
    and an empty child stack gain nothing, so they skip without the scan.
    """

    found: list[tuple[int, int, str, Any]] = []
    rows = plan.rows
    cells = plan.cells
    children, orphan = plan.col["children"], plan.col["is_orphan"]
    stack_fid = plan.fid["conditional_branch_stack"]
    for parent_row in plan.kept_raw:
        parent_stack = cells[parent_row][stack_fid]
        for child_label in children[parent_row]:
            try:
                child_row = rows[child_label]
            except KeyError:
                child_row = _row_or_refuse(trace, rows, child_label)
            if orphan[child_row]:
                continue
            child_stack = cells[child_row][stack_fid]
            if child_stack.__class__ is not list:  # a row outside the raw label list
                child_stack = plan.read(child_row, "conditional_branch_stack")
            if not child_stack or child_stack == parent_stack:
                continue
            shared = 0
            for parent_entry, child_entry in zip(parent_stack, child_stack, strict=False):
                if parent_entry != child_entry:
                    break
                shared += 1
            gained = child_stack[shared:]
            if gained:
                found += [(parent_row, child_row, child_label, gained)]
    return found


# ---------------------------------------------------------------------------
# Phase 5f: derived per-arm child views
# ---------------------------------------------------------------------------


def derive_arm_views(plan: StepColumns) -> None:
    """Write each non-orphan op's sorted THEN, ELIF and ELSE children."""

    cells = plan.cells
    arms_fid = plan.fid["conditional_arm_children"]
    then_fid = plan.fid["conditional_then_children"]
    elif_fid = plan.fid["conditional_elif_children"]
    else_fid = plan.fid["conditional_else_children"]
    for row in plan.kept_raw:
        cell = cells[row]
        arm_children = cell[arms_fid]
        if arm_children.__class__ is not dict:
            arm_children = plan.read(row, "conditional_arm_children")
        if not arm_children:  # what the three builders return for no arms, fresh per op
            cell[then_fid] = []
            cell[elif_fid] = {}
            cell[else_fid] = []
            continue
        cell[then_fid], cell[elif_fid], cell[else_fid] = _arm_views(arm_children)


def _arm_views(
    arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Return the THEN, ELIF (by index) and ELSE views of one op's arm children.

    The values ``_column_passes_graph``'s ``_then_children``, ``_elif_children``
    and ``_else_children`` return, built in one walk over the arms.
    """

    then_children: list[str] = []
    else_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    for branch_children in arm_children.values():
        for branch_kind, child_labels in branch_children.items():
            if branch_kind == "then":
                then_children += child_labels
            elif branch_kind == "else":
                else_children += child_labels
            elif branch_kind.startswith("elif_"):
                elif_index = int(branch_kind.split("_", 1)[1])
                if elif_index in elif_children:
                    elif_children[elif_index] += child_labels
                else:
                    elif_children[elif_index] = list(child_labels)
    elif_view = (
        {index: sorted(set(labels)) for index, labels in sorted(elif_children.items())}
        if elif_children
        else {}
    )
    return sorted(set(then_children)), elif_view, sorted(set(else_children))
