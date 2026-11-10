"""Call-free scans of the ``Op`` row store for passes that act on few ops.

Some finalization passes walk every op to find the handful that carry a value
(ops that feed a module call, buffer ops, ops with gradient records). Through
the facade that costs several Python-level calls per op even when the answer
is "nothing here". The helpers in this module look at the raw cells instead,
without a call per op, and hand the caller only the ops it has to visit.

Semantics stay the facade's by construction:

- A raw cell is skipped only when it is provably an empty value (``None``,
  ``False``, an empty plain ``list`` or ``tuple``, or a default marker whose
  default is empty). Every other cell (a pooled cell, a group view, a derived
  marker, an unset cell, a truthy value) selects its op, and the caller then
  reads that op through the facade exactly as before.
- The raw path runs only while the trace's ops live in one plain building
  ``OpRowStore``. For every other store shape (sealed, columnar, fork views,
  a loaded trace) the helpers return what the facade loop used.
"""

from __future__ import annotations

from typing import Any

from .._trace_core.cell_states import STATE_DEFAULT, LazyCell
from .._trace_core.op_store import OpRowStore
from ._op_columns import OpColumns
from ._op_row_containers import OpSequence, _list_iter

__all__ = ("call_outputs", "op_labels", "ops_with")

#: Plain container classes whose empty instance means "no value".
_PLAIN_CONTAINERS = (list, tuple)

#: Default factories that build an empty container.
_EMPTY_FACTORIES = (list, dict, tuple, set)


def _building_rows(trace: Any) -> tuple[Any, Any, list[int]] | None:
    """Return ``(resolver, store, keys)`` when the raw cells may be scanned.

    Parameters
    ----------
    trace:
        Trace whose ordered op list is scanned.

    Returns
    -------
    tuple[Any, Any, list[int]] | None
        The op resolver, the building row store and the op list's row keys in
        order; ``None`` when the ops are not held as row keys of one plain
        building ``OpRowStore`` (the caller then keeps the facade loop).
    """

    ops = trace._layer_list
    core = trace.__dict__.get("_trace_core")
    if ops.__class__ is not OpSequence or core is None:
        return None
    owner = ops._owner
    store = owner._store
    if store is not core.ops or store.__class__ is not OpRowStore or store._rows is None:
        return None
    return owner, store, list(_list_iter(ops))


def ops_with(trace: Any, *names: str) -> Any:
    """Return the ops that may hold a value in any of the stored fields ``names``.

    Parameters
    ----------
    trace:
        Trace whose ordered op list is filtered.
    *names:
        Stored ``Op`` field names (store layout keys).

    Returns
    -------
    Any
        The ops, in trace order, whose raw cell for at least one of ``names``
        is not provably empty: a superset of the ops whose facade read is
        truthy. The whole op list when the raw cells cannot be scanned.
    """

    scan = _building_rows(trace)
    if scan is None:
        return trace._layer_list
    owner, store, keys = scan
    rows = store._rows
    fid_by_name = store.layout.fid_by_name
    hits: set[int] = set()
    for name in names:
        fid = fid_by_name[name]
        # Operators only: this comprehension makes no call per op.
        hits.update(
            [
                key
                for key in keys
                if key < 0
                or not (
                    (cell := rows[key][fid]) is None
                    or cell is False
                    or (cell.__class__ in _PLAIN_CONTAINERS and not cell)
                    or (
                        cell.__class__ is LazyCell
                        and cell.state == STATE_DEFAULT
                        and (cell.source in _EMPTY_FACTORIES if cell.make else not cell.source)
                    )
                )
            ]
        )
    if not hits:
        return ()
    return list(owner.records([key for key in keys if key in hits]))


def op_labels(trace: Any) -> list[Any]:
    """Return ``op.label`` for every op of ``trace``, in trace order.

    Parameters
    ----------
    trace:
        Trace whose op labels are listed.

    Returns
    -------
    list[Any]
        The labels exactly as ``[op.label for op in trace._layer_list]`` gives
        them. Raw cells serve the list only when every one is a plain ``str``,
        which the facade returns unchanged.
    """

    scan = _building_rows(trace)
    if scan is not None:
        _owner, store, keys = scan
        try:
            OpColumns(store).getter("label")  # refuses a field the facade transforms
        except ValueError:
            scan = None
        else:
            fid = store.layout.fid_by_name["label"]
            rows = store._rows
            labels = [rows[key][fid] if key >= 0 else None for key in keys]
            if not [label for label in labels if label.__class__ is not str]:
                return labels
    return [op.label for op in trace._layer_list]


def call_outputs(trace: Any) -> dict[str, dict[str, str]]:
    """Return each module call's output ops as ``{op label: layer label}``.

    Parameters
    ----------
    trace:
        Trace whose ops are read, in trace order.

    Returns
    -------
    dict[str, dict[str, str]]
        Keyed by call label: the ops whose ``is_module_output`` flag is set and
        whose ``output_of_module_calls`` names the call.
    """

    by_call: dict[str, dict[str, str]] = {}
    scan = _building_rows(trace)
    readers = None
    if scan is not None and not [key for key in scan[2] if key < 0]:
        columns = OpColumns(scan[1])
        names = ("is_module_output", "output_of_module_calls", "label", "layer_label")
        try:
            readers = [columns.getter(name) for name in names]
        except ValueError:  # a field the facade transforms: read the facade
            readers = None
    if scan is None or readers is None:
        for op in trace._layer_list:
            if op._module_context_is_module_output:
                for call_label in op._module_context_output_of_module_calls:
                    by_call.setdefault(call_label, {})[op.label] = op.layer_label
        return by_call
    _owner, store, keys = scan
    rows = store._rows
    fid_by_name = store.layout.fid_by_name
    flag_fid, label_fid, layer_fid = (fid_by_name[name] for name in (names[0], *names[2:]))
    flag_of, calls_of, label_of, layer_label_of = readers
    for key in keys:
        row = rows[key]
        # A plain ``True``, ``False`` or ``str`` cell is what the facade returns.
        flag = row[flag_fid]
        if flag is False or not (flag is True or flag_of(key)):
            continue
        label = row[label_fid]
        if label.__class__ is not str:
            label = label_of(key)
        layer_label = row[layer_fid]
        if layer_label.__class__ is not str:
            layer_label = layer_label_of(key)
        for call_label in calls_of(key):
            if call_label not in by_call:
                by_call[call_label] = {}
            by_call[call_label][label] = layer_label
    return by_call
