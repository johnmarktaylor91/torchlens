"""Step 16's row scans: buffer version labels, gradient-record owners, module inputs.

Step 16 runs ``seal_scan`` when it seals the module facts, so the deferred
module build reads plain labels; ``rebuild_buffer_accessor`` runs
``seal_buffer_versions`` itself when it is given none. While the ops live in
one plain building row store the scan reads the raw cells with no call per
op (see ``_data_classes/_op_column_scan.py``); otherwise it reads the facade.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .._data_classes._op_column_scan import _building_rows, ops_with
from .._data_classes._op_columns import OpColumns
from .._trace_core.cell_states import STATE_DEFAULT, LazyCell
from .._trace_core.op_store import OpRowStore, PooledCell
from .._trace_core.record_refs import name_owner_by_row
from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Stored op fields the scan reads, by store layout name (``address`` is ``buffer_address``).
_FIELDS = ("input_to_module_calls", "layer_label", "is_buffer", "address", "label")


def seal_buffer_versions(trace: Trace) -> dict[str, list[str]]:
    """Name gradient-record owners and return each buffer's version op labels.

    Parameters
    ----------
    trace:
        Trace whose params and gradient-record or buffer ops are scanned.

    Returns
    -------
    dict[str, list[str]]
        Buffer address to the labels of its version ops, in op order.
    """

    return seal_scan(trace)[1]


def seal_scan(trace: Trace) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Name gradient-record owners; return module inputs and buffer versions.

    Parameters
    ----------
    trace:
        Trace whose params and ops are scanned.

    Returns
    -------
    tuple
        Call label to the layer labels of the ops that feed it, and buffer
        address to the labels of its version ops, both in op order.
    """

    trace_ref = weakref.ref(trace)
    for param in _maybe_grad_owners(trace.__dict__.get("_param_logs")):
        name_owner_by_row(trace_ref, param)
    scan = _building_rows(trace)
    readers = None
    if scan is not None and not [key for key in scan[2] if key < 0]:
        columns = OpColumns(scan[1])
        try:
            readers = [columns.getter(name, None) for name in _FIELDS]
        except (KeyError, ValueError):  # a field the facade transforms: read the facade
            readers = None
    if scan is None or readers is None:
        return _facade_scan(trace, trace_ref)
    owner, store, keys = scan
    rows = store._rows
    fid_by_name = store.layout.fid_by_name
    for op in owner.records(_candidates(rows, keys, fid_by_name["_grad_records"])):
        name_owner_by_row(trace_ref, op)
    calls_of, layer_of, flag_of, address_of, label_of = readers
    layer_fid, flag_fid, address_fid, label_fid = (fid_by_name[name] for name in _FIELDS[1:])
    inputs: dict[str, list[str]] = defaultdict(list)
    for key in _candidates(rows, keys, fid_by_name[_FIELDS[0]]):
        layer = rows[key][layer_fid]
        if layer.__class__ is not str:
            layer = layer_of(key)
        for call_label in calls_of(key) or ():
            inputs[call_label].append(layer)
    versions: dict[str, list[str]] = {}
    for key in _candidates(rows, keys, flag_fid):
        row = rows[key]
        if row[flag_fid] is not True and not flag_of(key):
            continue
        address = row[address_fid]
        if address.__class__ is not str:
            address = address_of(key)
        if address is not None:
            label = row[label_fid]
            versions.setdefault(address, []).append(
                label if label.__class__ is str else label_of(key)
            )
    return inputs, versions


def _candidates(rows: Any, keys: list[int], fid: int) -> list[int]:
    """Return the row keys whose raw cell ``fid`` is not provably empty."""

    # Operators only: this comprehension makes no call per op.
    return [
        key
        for key in keys
        if not (
            (cell := rows[key][fid]) is None
            or cell is False
            or (cell.__class__ in (list, tuple) and not cell)
            or (cell.__class__ is PooledCell and not cell.prototype)
            or (
                cell.__class__ is LazyCell
                and cell.state == STATE_DEFAULT
                and (cell.source in (list, dict, tuple, set) if cell.make else not cell.source)
            )
        )
    ]


def _maybe_grad_owners(params: Any) -> list[Any]:
    """Return the params that may hold gradient records (all of them unless provably not)."""

    records = list(getattr(params, "_list", None) or params or ())
    if not records:
        return records
    try:
        store = records[0].__dict__[CORE_KEY]
        fid = store.layout.fid_by_name["_grad_records"]
        if store.__class__ is not OpRowStore or store._rows is None:
            return records
        if [record for record in records if record.__dict__[CORE_KEY] is not store]:
            return records
        keys = [record.__dict__[ROW_KEY] for record in records]
    except (AttributeError, KeyError):
        return records
    hits = set(_candidates(store._rows, keys, fid))
    return [record for record in records if record.__dict__[ROW_KEY] in hits]


def _facade_scan(trace: Trace, trace_ref: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The scan read through the facade, for any store the raw path does not cover."""

    inputs: dict[str, list[str]] = defaultdict(list)
    for op in ops_with(trace, "input_to_module_calls"):  # only ops that feed a call
        for call_label in op._module_context_input_to_module_calls:
            inputs[call_label].append(op.layer_label)
    versions: dict[str, list[str]] = {}
    # Only an op with gradient records or a buffer flag has work here.
    for entry in ops_with(trace, "_grad_records", "is_buffer"):
        name_owner_by_row(trace_ref, entry)
        if record_member(entry, "is_buffer", False):
            address = entry.buffer_address
            if address is not None:
                versions.setdefault(address, []).append(entry.label)
    return inputs, versions
