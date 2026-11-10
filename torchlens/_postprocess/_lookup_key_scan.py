"""Raw row scan behind step 11: final op order, labels and lookup keys.

Step 11 walks the ops in raw order, skips orphans, gives each op its lookup
keys and registers them on the trace. Through the ``Op`` facade that costs
well over a hundred counted calls per op. This scan reads and writes the row
lists of the building op store directly and makes a handful of calls per op;
the containers it fills are the same objects, with the same keys in the same
order and the same values, as the facade loop leaves.

An op the scan cannot describe from plain cells (a buffer, a model input or
output, a module-call entry that is not a ``(name, pass)`` pair or a
``"name:pass"`` label, any cell in another shape) goes through the facade
loop's own per-op function, in its place in the order. A trace whose ops are
not plain building rows of one store keeps the facade loop for the whole step.
"""

from __future__ import annotations

from itertools import repeat
from typing import Any

from .._constants import _RAW_LABEL_SUFFIX
from .._data_classes._op_columns import OpColumns
from .._data_classes._op_row_containers import op_rows, restart_op_holders
from .._errors import AmbiguousOpLookupError
from .._trace_core.cell_states import LazyCell
from .._trace_core.op_store import OpRowStore
from .._trace_core.view_tables import PinnedViews

__all__ = ("scan_lookup_keys",)

_slot = object.__getattribute__
_SUFFIX_LEN = len(_RAW_LABEL_SUFFIX)
#: Fields the scan reads, in the order ``_Scan.fids`` unpacks them.
_READS = (
    "_label_raw",
    "label",
    "layer_label",
    "short_layer_label",
    "short_label",
    "raw_index",
    "num_passes",
    "is_orphan",
    "has_saved_activation",
    "is_buffer",
    "is_input",
    "is_output",
    "fx_qualpath",
    "fx_call_index",
)
#: Module-call fields the exhaustive relabel rewrites, in ``_relabel_module_calls`` order.
_MODULE_LISTS = ("output_of_module_calls", "input_to_module_calls", "modules", "module_call_stack")
#: Fields the scan writes straight into the row.
_WRITES = ("lookup_keys", "ordinal_index", "module", *_MODULE_LISTS)
_RELABEL_MODES = frozenset({"exhaustive", "predicate"})
_STR = frozenset({str})
_NO_ROLES = (False, False, False)


def _building_store(trace: Any) -> Any:
    """Return the trace's plain building row store, or ``None``."""

    core = trace.__dict__.get("_trace_core")
    if core is None:
        return None
    store = core.ops
    if store.__class__ is not OpRowStore or store._rows is None or store.dataflow_edges is not None:
        return None
    return store


def _checked_columns(store: Any) -> bool:
    """Return whether every field the scan touches is a plain stored column."""

    columns = OpColumns(store)
    try:
        for name in _READS:
            columns.getter(name)
        for name in _WRITES:
            columns.setter(name)
    except (KeyError, ValueError):
        return False
    return True


def _row_of_label(store: Any, labels: list[Any]) -> list[int] | None:
    """Return the store row of each raw label, or ``None`` unless that is one-to-one."""

    fid = store.layout.fid_by_name["_label_raw"]
    rows = store._rows
    by_label = {cells[fid]: row for row, cells in enumerate(rows)}
    if len(by_label) != len(rows):
        return None
    try:
        keys = [by_label[label] for label in labels]
    except (KeyError, TypeError):
        return None
    return keys if len(set(keys)) == len(keys) else None


def _plain_column(rows: list[Any], fid: int, keys: list[int]) -> list[Any]:
    """Return one column's cells at ``keys``, an immutable default marker read as its value."""

    return [
        cell.source if cell.__class__ is LazyCell and not cell.make else cell
        for cell in [rows[key][fid] for key in keys]
    ]


def _empty_calls(value: Any) -> bool:
    """Return whether a module-call cell is provably an empty list."""

    if value.__class__ is LazyCell:
        return bool(value.make and value.source is list)
    return (value.__class__ is list or value.__class__ is tuple) and not value


class _Scan:
    """One step-11 scan over a building row store.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    store:
        Its building ``OpRowStore``.
    """

    def __init__(self, trace: Any, store: Any) -> None:
        """Bind the scan to one trace and its building store."""

        self.trace, self.store = trace, store
        self.rows = store._rows
        fid = store.layout.fid_by_name
        self.fids = tuple(fid[name] for name in _READS)
        self.write_fids = tuple(fid[name] for name in _WRITES)
        self.columns = OpColumns(store)
        self.relabel = trace._options_mode in _RELABEL_MODES
        self.num_calls = trace._module_capture_ws.module_build_data["module_num_calls"]
        self.new_keys: dict[Any, int] = {}
        self.names: dict[str, str] = {}
        self.fx_paths: dict[str, str] = {}

    def orphans_and_saved(self, keys: list[int]) -> tuple[list[bool], int]:
        """Return each op's orphan flag and how many kept ops hold an activation (-1: unknown)."""

        flags = _plain_column(self.rows, self.fids[7], keys)
        if [flag for flag in flags if flag.__class__ is not bool]:
            orphan = self.columns.getter("is_orphan", False)
            flags = [
                flag if flag.__class__ is bool else bool(orphan(key))
                for flag, key in zip(flags, keys)
            ]
        f_saved = self.fids[8]
        saved = [self.rows[key][f_saved] for key, flag in zip(keys, flags) if not flag]
        if [value for value in saved if value.__class__ is not bool]:
            return flags, -1
        return flags, len([value for value in saved if value])

    def pin_table(self, records: list[Any], keys: list[int]) -> Any:
        """Return the op view table ``records`` can be held in at ``keys``, else ``None``."""

        owner = op_rows(self.trace)
        if owner._store is not self.store:
            owner.bind()
        table = owner._tables[0]
        if owner._store is not self.store or table.__class__ is not PinnedViews:
            return None
        try:
            stores = list(map(_slot, records, repeat("_core")))
            rows = list(map(_slot, records, repeat("_row")))
        except AttributeError:
            return None
        if rows != keys or [store for store in stores if store is not self.store]:
            return None
        size = len(table)
        held = [table[key] if key < size else None for key in keys]
        if [1 for cell, record in zip(held, records) if cell is not None and cell is not record]:
            return None
        return table

    def keys_of(self, key: int, position: int, total: int) -> list[Any] | None:
        """Return the sorted lookup keys of one op, writing its step-11 cells; ``None`` if unusual."""
        # One flat function on purpose (complexity): a helper per field is a counted call per op.
        cells = self.rows[key]
        f_raw, f_label, f_layer, f_short_layer, f_short, f_index = self.fids[:6]
        f_buffer, f_input, f_output, f_path, f_call = self.fids[9:]
        raw = cells[f_raw]
        path, call = cells[f_path], cells[f_call]
        if (
            {
                raw.__class__,
                cells[f_label].__class__,
                cells[f_layer].__class__,
                cells[f_short_layer].__class__,
                cells[f_short].__class__,
            }
            != _STR
            or cells[f_index].__class__ is not int
            or (cells[f_buffer], cells[f_input], cells[f_output]) != _NO_ROLES
            or (path is not None and (path.__class__ is not str or call.__class__ is not int))
        ):
            return None
        module_keys = self.module_keys(cells)
        if module_keys is None:
            return None
        keys = [
            raw,
            raw[:-_SUFFIX_LEN] if raw[-_SUFFIX_LEN:] == _RAW_LABEL_SUFFIX else raw,
            cells[f_layer],
            cells[f_short_layer],
            cells[f_label],
            cells[f_short],
            position,
            position - total,
            *module_keys,
        ]
        if path is not None:
            try:
                fx_path = self.fx_paths[path]
            except KeyError:
                fx_path = self.fx_paths[path] = path.replace(".", "_")
            keys.append(f"{fx_path}_{call}")
        keys = sorted(keys, key=str)
        f_keys, f_ordinal = self.write_fids[:2]
        cells[f_keys] = keys
        cells[f_ordinal] = position
        return keys

    def module_keys(self, cells: list[Any]) -> list[str] | None:
        """Relabel one op's module calls as ``_relabel_module_calls`` does; return its module keys.

        ``None`` (with nothing written) when a cell is not in a shape the scan
        reproduces exactly; the caller then uses the facade function.
        """
        # One flat function on purpose (complexity): it stands in for three facade helpers per op.
        f_module, *f_lists = self.write_fids[2:]
        if not self.relabel:
            return [] if _empty_calls(cells[f_lists[0]]) else None
        relabeled = []
        for fid in f_lists:
            value = cells[fid]
            if value.__class__ is LazyCell and value.make and value.source is list:
                value = ()
            if value.__class__ is not list and value.__class__ is not tuple:
                return None
            if [1 for call in value if call.__class__ is not str and call.__class__ is not tuple]:
                return None
            if [1 for call in value if call.__class__ is tuple and len(call) != 2]:
                return None
            relabeled.append(
                [call if call.__class__ is str else f"{call[0]}:{call[1]}" for call in value]
            )
        exits = relabeled[0]
        if [1 for call in exits if ":" not in call]:
            return None
        module = cells[f_module]
        if module.__class__ is LazyCell and not module.make:
            module = module.source
        if module is not None and module.__class__ is not str:
            if module.__class__ is not tuple or len(module) != 2:
                return None
            module = f"{module[0]}:{module[1]}"
        if module is None and relabeled[2]:
            module = relabeled[2][-1]
        names, num_calls, keys = self.names, self.num_calls, []
        for call in exits:
            try:
                name = names[call]
            except KeyError:
                name = names[call] = call.rsplit(":", 1)[0]
            keys.append(call)
            if num_calls[name] == 1:
                keys.append(name)
        for fid, value in zip(f_lists, relabeled):
            cells[fid] = value
        if module is not None:
            cells[f_module] = module
        return keys

    def register(self, keys: list[Any], key: int) -> None:
        """Register one op's keys as ``_add_lookup_keys_for_layer_entry`` does."""
        # One flat loop on purpose (complexity): a helper per key would be a counted call per key.
        trace, new_keys, cells = self.trace, self.new_keys, self.rows[key]
        f_raw, f_label, f_layer, f_short_layer, f_short, f_index = self.fids[:6]
        index, raw, label = cells[f_index], cells[f_raw], cells[f_label]
        public_raw = raw[:-_SUFFIX_LEN] if raw[-_SUFFIX_LEN:] == _RAW_LABEL_SUFFIX else raw
        by_key = trace._lookup_keys_to_layer_num_dict
        for lookup_key in keys:
            if lookup_key not in by_key:
                by_key[lookup_key] = index
                new_keys[lookup_key] = key
            elif by_key[lookup_key] != index:
                if lookup_key in (raw, public_raw, label):
                    self.flush()
                    existing = trace._layer_dict_all_keys[lookup_key]
                    raise AmbiguousOpLookupError(
                        f"Exact Op lookup key {lookup_key!r} collides between "
                        f"{existing.label!r} and {label!r}."
                    )
                indices = trace._ambiguous_lookup_keys.setdefault(lookup_key, [by_key[lookup_key]])
                if index not in indices:
                    indices.append(index)
                if lookup_key in (cells[f_layer], cells[f_short_layer], label, cells[f_short]):
                    by_key[lookup_key] = index
                    new_keys[lookup_key] = key
        by_index = trace._layer_num_to_lookup_keys_dict
        if index in by_index:
            by_index[index] += keys
        else:
            by_index[index] = keys[:]

    def flush(self) -> None:
        """Write the keys registered so far into ``trace._layer_dict_all_keys``."""

        if self.new_keys:
            dict.update(self.trace._layer_dict_all_keys, self.new_keys)
            self.new_keys = {}

    def column(self, name: str, keys: list[int], kind: type) -> list[Any]:
        """Return field ``name`` at ``keys`` as the facade reads it."""

        fid = self.store.layout.fid_by_name[name]
        values = [self.rows[key][fid] for key in keys]
        if [value for value in values if value.__class__ is not kind]:
            return list(map(self.columns.getter(name), keys))
        return values


def scan_lookup_keys(trace: Any, add_keys: Any) -> bool:
    """Run step 11's label pass over raw rows; return ``False`` to keep the facade loop.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    add_keys:
        The facade loop's per-op function (``_add_lookup_keys_for_layer_entry``),
        used for an op the scan cannot describe from plain cells.

    Returns
    -------
    bool
        ``True`` when the scan did the whole step-11 label pass; ``False``
        (with nothing changed) when the facade loop must run instead.
    """

    store = _building_store(trace)
    if store is None or trace._tracing_finished or not _checked_columns(store):
        return False
    workspace = trace._raw_graph_ws
    labels = list(workspace.raw_layer_labels_list)
    raw_dict = workspace.raw_layer_dict
    keys = _row_of_label(store, labels) if len(labels) == len(raw_dict) else None
    if keys is None:
        return False
    scan = _Scan(trace, store)
    orphans, saved = scan.orphans_and_saved(keys)
    try:
        records = [raw_dict[label] for label in labels]
    except KeyError:
        return False
    kept_keys = [key for key, orphan in zip(keys, orphans) if not orphan]
    kept = [record for record, orphan in zip(records, orphans) if not orphan]
    table = scan.pin_table(kept, kept_keys) if saved >= 0 else None
    if table is None:
        return False
    _finish(trace, scan, table, kept, kept_keys, saved, add_keys)
    return True


def _finish(
    trace: Any,
    scan: _Scan,
    table: Any,
    kept: list[Any],
    keys: list[int],
    saved: int,
    add_keys: Any,
) -> None:
    """Hold every kept op, register its keys and fill the trace's ordered op containers.

    Trace attributes are assigned in the facade loop's order.
    """

    trace._num_saved_ops = saved
    restart_op_holders(trace)
    trace.layer_labels = []
    trace.op_labels = []
    trace._layer_num_calls = {}
    if keys:
        table.pad(max(keys) + 1)
        for record, key in zip(kept, keys):
            table[key] = record
    total = len(keys)
    try:
        for position, (record, key) in enumerate(zip(kept, keys)):
            op_keys = scan.keys_of(key, position, total)
            if op_keys is None:
                scan.flush()
                add_keys(trace, record, position, total)
            else:
                scan.register(op_keys, key)
    finally:
        scan.flush()
    labels = scan.column("label", keys, str)
    layer_labels = scan.column("layer_label", keys, str)
    list.extend(trace._layer_list, keys)
    dict.update(trace._layer_dict_main_keys, zip(labels, keys))
    trace.layer_labels.extend(dict.fromkeys(layer_labels))
    trace.op_labels.extend(labels)
    trace._layer_num_calls.update(zip(layer_labels, scan.column("num_passes", keys, int)))
    if keys:
        op_rows(trace)._last = keys[-1]
    trace._layers_logged = True
    trace._layers_saved = saved == total
