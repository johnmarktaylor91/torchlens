"""Read every public field of every record of a trace into canonical data."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

from .canonical import Canonicalizer, error_record
from .mask import NOT_READ, Mask

#: Record class name -> the constant in ``torchlens._constants`` that lists its fields.
FIELD_ORDER_CONSTANTS: dict[str, str] = {
    "Trace": "MODEL_LOG_FIELD_ORDER",
    "Op": "LAYER_PASS_LOG_FIELD_ORDER",
    "Layer": "LAYER_LOG_FIELD_ORDER",
    "Module": "MODULE_LOG_FIELD_ORDER",
    "ModuleCall": "MODULE_PASS_LOG_FIELD_ORDER",
    "Param": "PARAM_LOG_FIELD_ORDER",
    "Buffer": "BUFFER_LOG_FIELD_ORDER",
    "GradFn": "GRAD_FN_LOG_FIELD_ORDER",
    "GradFnCall": "GRAD_FN_PASS_LOG_FIELD_ORDER",
    "BackwardPass": "BACKWARD_PASS_FIELD_ORDER",
    "FuncCallLocation": "FUNC_CALL_LOCATION_FIELD_ORDER",
}

#: Record classes written as a reference when they appear inside a field.
REFERENCED_CLASSES = frozenset(set(FIELD_ORDER_CONSTANTS) - {"FuncCallLocation"})

_ROUTINES = (staticmethod, classmethod)


def field_order(class_name: str) -> tuple[str, ...]:
    """Return the declared field list of a record class, empty when it has none."""

    from torchlens import _constants

    constant = FIELD_ORDER_CONSTANTS.get(class_name)
    return tuple(getattr(_constants, constant, ())) if constant else ()


def storage_attrs(class_name: str) -> dict[str, str]:
    """Map a persisted field name to the attribute that reads it (grouped members)."""

    try:
        from torchlens._utils import _record_group_table as table
    except ImportError:
        return {}
    name = {"Op": "OP_STORAGE_ATTRS", "Layer": "LAYER_STORAGE_ATTRS"}.get(class_name)
    return dict(getattr(table, name, {})) if name else {}


def public_names(cls: type) -> tuple[str, ...]:
    """Return every public data member of a record class.

    The declared field list comes first, in its order; then every other
    public class member that is not a method (properties and descriptors),
    sorted.
    """

    declared = field_order(cls.__name__)
    seen = set(declared)
    extra = []
    for name in dir(cls):
        if name.startswith("_") or name in seen:
            continue
        member = inspect.getattr_static(cls, name)
        if isinstance(member, _ROUTINES) or inspect.isroutine(member) or isinstance(member, type):
            continue
        extra.append(name)
    return declared + tuple(sorted(extra))


def _is_group(value: Any) -> bool:
    """Return whether a value is a concern group (``op.cost``) of a record."""

    return any(base.__name__ in ("_RecordGroup", "_TraceGroup") for base in type(value).__mro__)


class Reader:
    """Read record objects into canonical data under the mask.

    Parameters
    ----------
    canon:
        Canonicaliser holding the identity tables of the trace being read.
    mask:
        The fixed mask.
    skip_names:
        Names left out of the dump on purpose (the planted missing-field fault).
    """

    def __init__(
        self, canon: Canonicalizer, mask: Mask, skip_names: frozenset[str] = frozenset()
    ) -> None:
        """Start with an empty warning list."""

        self.canon = canon
        self.mask = mask
        self.skip_names = skip_names
        self.warnings: list[dict[str, Any]] = []

    def read(self, obj: Any, attr: str, kind: str, key: str, site: str) -> tuple[Any, Any]:
        """Read ``obj.attr``; return its canonical form and the raw value.

        A raised error and every warning are recorded with the field they
        came from. The raw value is ``None`` when the read raised.
        """

        leaf = key.rsplit(".", 1)[-1]
        if leaf in NOT_READ:
            return {"__not_read__": NOT_READ[leaf]}, None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = getattr(obj, attr)
            except Exception as error:  # noqa: BLE001 - raising is part of the surface
                value, canonical = None, error_record(error, self.canon.string)
            else:
                canonical = self.canon.at(site).value(value)
        for warning in caught:
            self.warnings.append(
                {
                    "at": site,
                    "category": warning.category.__name__,
                    "message": self.canon.string(str(warning.message))[:300],
                }
            )
        return self._masked(kind, leaf, canonical), value

    def _masked(self, kind: str, leaf: str, canonical: Any) -> Any:
        """Apply the mask to one field's canonical value."""

        return self.mask.apply(kind, leaf, canonical)

    def record(self, obj: Any, site: str) -> dict[str, Any]:
        """Read every public member of one record, grouped members included."""

        kind = type(obj).__name__
        storage = storage_attrs(kind)
        surface: dict[str, Any] = {}
        for name in public_names(type(obj)):
            if name in self.skip_names:
                continue
            canonical, value = self.read(obj, storage.get(name, name), kind, name, f"{site}.{name}")
            surface[name] = canonical
            if value is not None and _is_group(value):
                self._group(surface, value, kind, name, site)
        surface["__repr__"] = self.text(obj, repr, f"{site}.__repr__")
        return surface

    def _group(self, surface: dict[str, Any], group: Any, kind: str, name: str, site: str) -> None:
        """Read every public member of a concern group under ``name.member``."""

        for member in sorted(n for n in dir(group) if not n.startswith("_")):
            if inspect.isroutine(inspect.getattr_static(type(group), member, None)):
                continue
            key = f"{name}.{member}"
            surface[key], _ = self.read(group, member, kind, key, f"{site}.{key}")

    def text(self, obj: Any, render: Any, site: str) -> Any:
        """Render ``repr`` or ``str`` of an object under the text rules."""

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                rendered = self.mask.text(self.canon.string(render(obj)))
            except Exception as error:  # noqa: BLE001 - a failing repr is recorded
                rendered = error_record(error, self.canon.string)
        for warning in caught:
            self.warnings.append(
                {"at": site, "category": warning.category.__name__, "message": str(warning.message)}
            )
        return rendered

    def table(self, frame: Any, site: str) -> dict[str, Any]:
        """Canonicalise a ``to_pandas()`` table: columns, dtypes and every cell."""

        columns: dict[str, Any] = {}
        for column in frame.columns:
            name = str(column)
            cells = [
                self._masked("to_pandas", name, self.canon.at(f"{site}.{name}").value(_plain(cell)))
                for cell in frame[column].tolist()
            ]
            columns[name] = {"dtype": str(frame[column].dtype), "cells": cells}
        return {"columns": [str(c) for c in frame.columns], "rows": len(frame), "data": columns}


def _plain(cell: Any) -> Any:
    """Turn a NumPy scalar from a table cell into the matching Python value."""

    item = getattr(cell, "item", None)
    if item is not None and type(cell).__module__ == "numpy":
        try:
            return item()
        except (ValueError, TypeError):
            return cell
    return cell


def trace_families(trace: Any) -> dict[str, Any]:
    """Return every record family of a trace as ``name -> {key: record}``.

    Ops are read from the trace's own op list, which includes input, output
    and buffer rows that the counted ``trace.ops`` view leaves out.
    """

    def _door(read: Any) -> Any:
        try:
            accessor = read()
        except Exception as error:  # noqa: BLE001 - a refusing door is recorded
            return error
        if accessor is None:
            return None
        return {str(key): record for key, record in accessor.items()}

    def _ops() -> Any:
        rows = getattr(trace, "_layer_list", None)
        if rows is None:
            return trace.ops
        return {op.label: op for op in rows}

    return {
        "ops": _door(_ops),
        "layers": _door(lambda: trace.layers),
        "modules": _door(lambda: trace.modules),
        "module_calls": _door(lambda: trace.module_calls),
        "params": _door(lambda: trace.params),
        "buffers": _door(lambda: trace.buffers),
        "grad_fns": _door(lambda: trace.autograd.grad_fns),
        "backward_passes": _door(lambda: trace.autograd.backward_passes),
    }


def snapshot_trace(
    trace: Any,
    roots: list[tuple[str, str]],
    mask: Mask | None = None,
    skip_names: frozenset[str] = frozenset(),
    with_tables: bool = True,
) -> tuple[dict[str, Any], Canonicalizer]:
    """Dump the trace, every record family, the reprs and the tables.

    Parameters
    ----------
    trace:
        A finished trace.
    roots:
        Path prefixes replaced inside strings (see ``Canonicalizer``).
    mask:
        The mask to apply; the fixed mask when omitted.
    skip_names:
        Field names left out on purpose (the planted missing-field fault).
    with_tables:
        Whether to include ``to_pandas()`` of the trace.

    Returns
    -------
    tuple[dict[str, Any], Canonicalizer]
        The canonical dump and the canonicaliser, whose ``tensor_sites``
        lists every tensor read and the field it was read from.
    """

    canon = Canonicalizer(roots, REFERENCED_CLASSES)
    reader = Reader(canon, mask or Mask(), skip_names)
    dump: dict[str, Any] = {"trace": reader.record(trace, "trace")}
    dump["trace"]["__str__"] = reader.text(trace, str, "trace.__str__")
    for family, records in trace_families(trace).items():
        if records is None or isinstance(records, BaseException):
            dump[family] = None if records is None else error_record(records, canon.string)
            continue
        dump[family] = {
            key: reader.record(record, f"{family}[{key}]") for key, record in records.items()
        }
    if with_tables:
        try:
            dump["to_pandas"] = reader.table(trace.to_pandas(), "to_pandas")
        except Exception as error:  # noqa: BLE001 - a refusing table is recorded
            dump["to_pandas"] = error_record(error, canon.string)
    dump["warnings"] = reader.warnings
    return dump, canon
