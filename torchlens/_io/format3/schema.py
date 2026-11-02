"""Record schemas of the 3.0 saved format, derived from the field bindings.

One schema per record kind: the ``columns`` of ``records/<record>.json`` are the
record's persisted field names, each cell typed by its ``StorageKind`` and
annotation; nested families (timing provenance, checkpoint witnesses,
intervention audit rows, kernel launches, capture advisories, ...) are
declarative sub-schemas in ``FAMILIES`` that the one decoder applies. The
schema digest is stamped into the manifest so a binding change is visible
without a contract bump. Design: ``docs/maintainers/saved_format_3_design.md``,
sections 3, 4 and 9.

Which columns a file carries depends on the payload policy: a ``required``
key is a column at every level; a key gated by a payload flag
(``FLAG_FIELDS``) is a column only when that flag is on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from ..._data_classes._schema_bindings import STORAGE_BINDINGS
from ..._data_classes.field_policy import StorageKind
from ..field_registry import persisted_field_names
from .contract import JSON_TYPE_BY_STORAGE_KIND, RECORD_INDEX

#: Nested-family sub-schemas, keyed by family name. Empty until the grammar
#: validators move here from ``_io/forgery_validation.py`` (design section 9);
#: the checked load still runs those validators until then.
FAMILIES: dict[str, dict[str, Any]] = {}

#: Payload include flags and the stored keys each one gates (the scrub's policy
#: resolver; mirrored by ``tools/docs/persisted_fields.py`` for the page).
FLAG_FIELDS: dict[str, tuple[str, ...]] = {
    "include_outs": ("out", "transformed_out"),
    "include_grads": ("grad", "transformed_grad"),
    "include_saved_args": (
        "forward_args",
        "forward_kwargs",
        "out_versions_by_child",
        "saved_args",
        "saved_kwargs",
    ),
    "include_rng_states": ("func_rng_states",),
    "include_buffer_values": ("_buffer_initial_values",),
    "include_custom_attributes": ("custom_attributes",),
}
#: The bindings key where it differs from the record contract key.
_BINDING_KEY_ALIASES = {"aten_op": "primitive_op"}
_FLAG_BY_FIELD = {field: flag for flag, fields in FLAG_FIELDS.items() for field in fields}


@dataclass(frozen=True)
class Column:
    """Schema of one record column.

    Attributes
    ----------
    name:
        Stored key, spelled as stored (leading underscore included).
    storage:
        ``StorageKind`` value of the binding.
    json_type:
        ``contract.JSON_TYPE_BY_STORAGE_KIND`` class of the cells.
    flag:
        Payload include flag that gates the column, or ``None`` when required.
    annotation:
        Declared annotation text, or ``None``.
    """

    name: str
    storage: str
    json_type: str
    flag: str | None
    annotation: str | None


@cache
def record_schema(record: str) -> dict[str, Column]:
    """Return the typed schema of one record kind (``op``, ``trace``, ...).

    Parameters
    ----------
    record:
        Record kind from ``contract.RECORD_INDEX``.

    Returns
    -------
    dict[str, Column]
        Persisted columns in bindings order (persisted keys the bindings do
        not list follow alphabetically, typed ``value``).
    """

    bindings = STORAGE_BINDINGS.get(_BINDING_KEY_ALIASES.get(record, record), {})
    persisted = persisted_field_names(record)
    ordered = [name for name in bindings if name in persisted]
    ordered.extend(sorted(persisted.difference(bindings)))
    columns: dict[str, Column] = {}
    for name in ordered:
        binding = bindings.get(name)
        kind = binding.kind if binding is not None else StorageKind.SCALAR
        if kind is StorageKind.RUNTIME:
            continue
        columns[name] = Column(
            name=name,
            storage=kind.value,
            json_type=JSON_TYPE_BY_STORAGE_KIND[kind],
            flag=_FLAG_BY_FIELD.get(name),
            annotation=None if binding is None else binding.annotation,
        )
    return columns


def level_columns(record: str, payload_policy: Mapping[str, bool]) -> list[str]:
    """Return the columns one record file carries under a payload policy.

    Parameters
    ----------
    record:
        Record kind.
    payload_policy:
        Resolved include flags; a flag absent from the mapping counts as off.

    Returns
    -------
    list[str]
        Column names in schema order.
    """

    return [
        name
        for name, column in record_schema(record).items()
        if column.flag is None or bool(payload_policy.get(column.flag, False))
    ]


@cache
def schema_digest() -> str:
    """Return the digest of every record schema, as stamped into the manifest.

    Returns
    -------
    str
        ``sha256`` hex digest of the canonical JSON of all twelve schemas.
    """

    payload = {
        record: [asdict(column) for column in record_schema(record).values()]
        for record, _, _ in RECORD_INDEX
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
