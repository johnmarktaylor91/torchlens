"""Step 17 body: mark every final op finished.

The step sets ``_tracing_finished`` on each op of the trace's main lookup. Two
bodies give the same stored result: the column body writes the flag into the
raw row of a plain building ``OpRowStore``; any other trace takes the facade
body. The template display keys this step used to bind are bound by the replay
template family when a template is first read
(``_replay_template_cells.resolve_template``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._data_classes._op_columns import column_rows_by_label
from .._trace_core.op_store import OpRowStore

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = ("mark_ops_finished",)

#: The stored field the column body writes.
_FLAG = "_tracing_finished"


def mark_ops_finished(trace: Trace) -> None:
    """Mark the trace's final ops finished."""

    ops = trace._layer_dict_main_keys
    plan = _column_plan(ops, trace._layer_dict_all_keys)
    if plan is None:
        for layer_label in ops:
            ops[layer_label]._tracing_finished = True
        return
    raw_rows, rows, flag_fid = plan
    for label in dict.keys(ops):
        raw_rows[rows[label]][flag_fid] = True


def _column_plan(ops: dict[str, Any], records: Any) -> tuple[list[Any], dict[str, int], int] | None:
    """Return ``(raw rows, label -> row, flag field id)`` when the column body applies.

    It applies when every op is a plain ``Op`` of one building row store, the
    all-keys lookup resolves each main label to that same op, and the column
    view serves the flag under its public name.
    """

    lookup = column_rows_by_label(ops) if isinstance(records, dict) else None
    if lookup is None:
        return None
    columns, rows = lookup
    store = columns.store
    if store.__class__ is not OpRowStore or store._rows is None:
        return None
    if not _same_records(ops, records):
        return None
    try:
        columns.setter(_FLAG)
    except ValueError:
        return None
    return store._rows, rows, store.layout.fid_by_name[_FLAG]


def _same_records(ops: Any, records: Any) -> bool:
    """Return whether ``records`` holds each main label's op, without reading records.

    Two row-key label maps of one owner compare their row keys; any other pair
    compares the records by identity, as the facade walk reads them.
    """

    owner = getattr(ops, "_owner", None)
    if owner is None or getattr(records, "_owner", None) is not owner:
        for label, op in ops.items():  # noqa: SIM110 - a generator costs a call per op
            if label not in records or records[label] is not op:
                return False
        return True
    # Row keys are ints: the item views compare as sets, in C, with no call per op.
    return dict.items(ops) <= dict.items(records)
