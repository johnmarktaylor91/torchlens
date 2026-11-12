"""Step 17 body: mark every final op finished.

The step sets ``_tracing_finished`` on each op of the trace's main lookup. It
runs after ``seal_op_store``, and two bodies give the same stored result: the
sealed body writes the flag for every main row at once, into the sealed store's
packed or object column (or in place into the rows of a sealed row-major
store); any other trace takes the facade body. The template display keys this step used to bind are bound by the replay
template family when a template is first read
(``_replay_template_cells.resolve_template``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._trace_core.op_store import OpRowStore, _SealedRowMajorOpRowStore

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = ("mark_ops_finished",)

#: The stored field the sealed body writes.
_FLAG = "_tracing_finished"


def mark_ops_finished(trace: Trace) -> None:
    """Mark the trace's final ops finished."""

    ops = trace._layer_dict_main_keys
    records = trace._layer_dict_all_keys
    if _mark_sealed_rows(ops, records, trace):
        return
    for layer_label in ops:
        ops[layer_label]._tracing_finished = True


def _mark_sealed_rows(ops: Any, records: Any, trace: Trace) -> bool:
    """Write the flag for every main row of a sealed store at once; return whether it did.

    It applies when both lookups are row-key maps of one owner bound to the
    trace's sealed Op store, every main key is a store row, and no fork has
    viewed the store. Every other case returns ``False`` for the other bodies.
    """

    owner = getattr(ops, "_owner", None)
    if owner is None or not ops or getattr(records, "_owner", None) is not owner:
        return False
    core = trace.__dict__.get("_trace_core")
    store = core.ops if core is not None else None
    if store is None or owner._store is not store or not store._sealed:
        return False
    if store._cow_shared or store._mutable_keys is not None:
        return False
    keys = dict.values(ops)
    if min(keys) < 0 or not _same_records(ops, records):
        return False
    fid = store.layout.fid_by_name[_FLAG]
    rows = store._rows
    if rows is not None:
        if store.__class__ is not _SealedRowMajorOpRowStore:
            return False
        for row in keys:
            rows[row][fid] = True
        return True
    if store.__class__ is not OpRowStore:
        return False
    return _write_sealed_column(store, fid, keys)


def _write_sealed_column(store: Any, fid: int, keys: Any) -> bool:
    """Write ``True`` into one columnar store's flag column at ``keys``; return whether it did.

    A packed column must be boolean, and no overlay entry may shadow a cell.
    """

    if store._overlay:
        return False
    column = store._columns[fid]
    index = np.fromiter(keys, dtype=np.intp, count=len(keys))
    if column.packed:
        if column.values.dtype != np.bool_:
            return False
        column.values[index] = True
    else:
        values = column.values
        for row in keys:
            values[row] = True
    if column.present is not None:
        np.frombuffer(column.present, dtype=np.uint8)[index] = 1
    return True


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
