"""Record tables of the 3.0 saved format: reading, digest checks and the column contract.

One record file per kind. ``read_record_table`` returns the raw rows after the digest and
shape checks; the column checks refuse a column the bindings do not persist
(``unknown_persisted_field``) and a table that does not carry exactly its save level's
columns (``record_schema_invalid``). Design: ``docs/maintainers/saved_format_3_design.md``,
section 3.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from ..._data_classes._schema_bindings import STORAGE_BINDINGS
from ..._data_classes.field_policy import StorageKind
from ..field_registry import RECORD_CONTRACT_CLASSES
from ..format_contract import TLSPEC_VERSION
from ..format_errors import ArtifactError, UnknownPersistedFieldError
from ..prerelease import prerelease_fields_active, registered_prerelease_fields
from ..state_contract import _unknown_field_remedy
from . import contract
from .schema import level_columns, record_schema
from .validate import TLSPEC_KEY, integrity_error, read_json_file

_CODES: Final = contract.REFUSAL_CODES
_CORRUPT: Final = _CODES["record_file_corrupt"]
_SCHEMA_KEY_BY_RECORD: Final[dict[str, str]] = {"aten_op": "primitive_op"}
#: Columns the trace table carries beyond its bindings (transport state the reader consumes).
_TRACE_TRANSPORT_COLUMNS: Final = frozenset({"_io_module_accessor_state"})


def record_class(kind: str) -> type:
    """Return the record class a record kind names in the field registry."""

    module_name, class_name = RECORD_CONTRACT_CLASSES[kind]
    return getattr(import_module(module_name), class_name)


def read_record_table(root: Path, manifest: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    """Read ``records/<kind>.json`` after its digest check; return one raw dict per row."""

    entry = manifest["records"][kind]
    target = root / entry["path"]
    data, raw = read_json_file(target, code=_CORRUPT, what="record file")
    listed = manifest["files"][entry["path"]]
    if len(raw) != listed["bytes"] or hashlib.sha256(raw).hexdigest() != listed["sha256"]:
        raise integrity_error(target, "bytes differ from the manifest's digest", code=_CORRUPT)
    if not isinstance(data, dict) or data.get("record") != kind:
        raise integrity_error(target, f"not a {kind!r} record object", code=_CORRUPT)
    if data.get(TLSPEC_KEY) != manifest[TLSPEC_KEY]:
        raise integrity_error(target, "tlspec_version differs from the manifest's", code=_CORRUPT)
    if kind == "trace":
        return _trace_fields(data, entry, target)
    return _columnar_rows(data, entry, target)


def _trace_fields(
    data: dict[str, Any], entry: Mapping[str, Any], target: Path
) -> list[dict[str, Any]]:
    """The trace record is one row: ``{record, tlspec_version, fields}``."""

    fields = data.get("fields")
    if set(data) != {"record", TLSPEC_KEY, "fields"} or not isinstance(fields, dict):
        raise integrity_error(
            target, "trace record must be {record, tlspec_version, fields}", code=_CORRUPT
        )
    if entry["rows"] != 1:
        raise integrity_error(target, "the trace record counts one row", code=_CORRUPT)
    return [dict(fields)]


def _columnar_rows(
    data: dict[str, Any], entry: Mapping[str, Any], target: Path
) -> list[dict[str, Any]]:
    """A columnar table is ``{record, tlspec_version, rows, columns}`` with equal-length columns."""

    rows, columns = data.get("rows"), data.get("columns")
    if set(data) != {"record", TLSPEC_KEY, "rows", "columns"} or not isinstance(columns, dict):
        raise integrity_error(
            target, "record table must be {record, tlspec_version, rows, columns}", code=_CORRUPT
        )
    if rows != entry["rows"]:
        raise integrity_error(
            target, f"rows {rows!r} differs from the manifest's {entry['rows']}", code=_CORRUPT
        )
    for name, column in columns.items():
        if not isinstance(column, list) or len(column) != rows:
            raise integrity_error(
                target, f"column {name!r} is not an array of {rows} cells", code=_CORRUPT
            )
    return [{name: column[index] for name, column in columns.items()} for index in range(rows)]


def _allowed_extra_columns(kind: str) -> frozenset[str]:
    """Columns a table may carry beyond its bindings: trace transport state and, under the
    test-only switch, the registered pre-release fields (the marker the reader already
    validated proves the artifact is one of ours)."""

    allowed = _TRACE_TRANSPORT_COLUMNS if kind == "trace" else frozenset()
    if prerelease_fields_active():
        owner = record_class(kind).__name__
        allowed = allowed | frozenset(registered_prerelease_fields().get(owner, ()))
    return allowed


def _unknown_columns(kind: str, names: Mapping[str, Any]) -> list[str]:
    """Return the table columns the field bindings do not declare."""

    bindings = STORAGE_BINDINGS[_SCHEMA_KEY_BY_RECORD.get(kind, kind)]
    allowed_extra = _allowed_extra_columns(kind)
    unknown = []
    for name in names:
        if name in allowed_extra:
            continue
        binding = bindings.get(name)
        if binding is None or binding.kind is StorageKind.RUNTIME:
            unknown.append(name)
    return unknown


def refuse_unknown_columns(kind: str, names: Mapping[str, Any], *, declared_version: int) -> None:
    """Refuse a column the record's bindings do not persist (``unknown_persisted_field``)."""

    unknown = _unknown_columns(kind, names)
    if not unknown:
        return
    cls = record_class(kind)
    raise UnknownPersistedFieldError(
        f"{cls.__name__} record table carries {len(unknown)} column(s) unknown to this "
        f"reader's persisted-state contract: {', '.join(sorted(unknown)[:12])}. Unknown "
        "columns refuse in both writer directions.",
        code=_CODES["record_key_unknown"],
        record_type=cls.__name__,
        unknown_fields=sorted(unknown),
        declared_tlspec_version=declared_version,
        runtime_tlspec_version=TLSPEC_VERSION,
        remedy=_unknown_field_remedy(declared_version),
    )


def refuse_level_columns(
    kind: str,
    names: Mapping[str, Any],
    *,
    payload_policy: Mapping[str, bool],
    declared_version: int,
) -> None:
    """Refuse a table that does not carry its save level's columns (``record_schema_invalid``).

    A column without a payload flag, or whose flag is on, must be present; a column whose
    flag is off must be absent (the writer persists exactly ``schema.level_columns``). A file
    written at an older tlspec version may lack columns added since: the one forward-
    compatibility rule of the 3 major.
    """

    expected = set(level_columns(kind, payload_policy))
    present = set(names) - _allowed_extra_columns(kind)
    forbidden = sorted(present - expected)
    missing = sorted(expected - present) if declared_version >= TLSPEC_VERSION else []
    if not forbidden and not missing:
        return
    cls = record_class(kind)
    raise ArtifactError(
        f"{cls.__name__} record table does not carry the columns of its save level: "
        f"{len(missing)} required column(s) absent ({', '.join(missing[:12])}) and "
        f"{len(forbidden)} column(s) present whose payload flag is off in payload_policy "
        f"({', '.join(forbidden[:12])}).",
        code=_CODES["record_schema_invalid"],
        record_type=cls.__name__,
        missing_columns=missing,
        forbidden_columns=forbidden,
        remedy="re-save the trace; a record table or payload_policy edited by hand is refused",
    )


def refuse_table_columns(kind: str, names: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Both column checks on one table: unknown columns first, then the level's exact set."""

    declared = manifest[TLSPEC_KEY]
    refuse_unknown_columns(kind, names, declared_version=declared)
    refuse_level_columns(
        kind, names, payload_policy=manifest["payload_policy"], declared_version=declared
    )


# ---- typed scalar cells ------------------------------------------------------

#: The declared annotations the reader enforces on a scalar cell. A binding whose annotation
#: is anything else (a record class, a tuple grammar, ``Any``) is typed by its own decoder.
_LEAF_CHECKS: Final[dict[str, Callable[[Any], bool]]] = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "None": lambda v: v is None,
    "list[str]": lambda v: isinstance(v, list) and all(isinstance(i, str) for i in v),
    "list[int]": lambda v: (
        isinstance(v, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in v)
    ),
}
_CLASS_ANNOTATION = re.compile(r"<class '([a-z]+)'>")
_TYPED_STORAGE: Final = frozenset(
    {StorageKind.SCALAR.value, StorageKind.INTERNED.value, StorageKind.COMPUTED.value}
)


def _annotation_leaves(annotation: str | None) -> tuple[str, ...] | None:
    """Return the union members of a simple annotation, or ``None`` when it is not one."""

    if annotation is None:
        return None
    match = _CLASS_ANNOTATION.fullmatch(annotation)
    leaves = tuple(part.strip() for part in (match.group(1) if match else annotation).split("|"))
    return leaves if all(leaf in _LEAF_CHECKS for leaf in leaves) else None


def typed_columns(kind: str) -> dict[str, tuple[str, ...]]:
    """Return the scalar columns of ``kind`` whose declared annotation the reader enforces."""

    typed: dict[str, tuple[str, ...]] = {}
    for name, column in record_schema(kind).items():
        if column.storage not in _TYPED_STORAGE:
            continue
        leaves = _annotation_leaves(column.annotation)
        if leaves is not None:
            typed[name] = leaves
    return typed


def refuse_cell_types(kind: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a decoded scalar cell that is not of its column's declared type.

    A ``str`` column holding ``null``, a number, an array or a reference is a forged cell:
    the record it would build then derives other columns from it (a module call's module
    link from its address), so it refuses ``record_schema_invalid`` before any record builds.
    """

    for name, leaves in typed_columns(kind).items():
        for index, row in enumerate(rows):
            if name not in row:
                continue
            value = row[name]
            if any(_LEAF_CHECKS[leaf](value) for leaf in leaves):
                continue
            cls = record_class(kind)
            raise ArtifactError(
                f"{cls.__name__} record table cell {kind}[{index}].{name} is a "
                f"{type(value).__name__}; the column is declared {' | '.join(leaves)}.",
                code=_CODES["record_schema_invalid"],
                record_type=cls.__name__,
                column=name,
                row=index,
                remedy="re-save the trace; a record cell edited by hand is refused",
            )
