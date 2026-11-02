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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from importlib import import_module
from typing import Any

from ..._data_classes._accessor_base import Accessor
from ..._data_classes._schema_bindings import STORAGE_BINDINGS
from ..._data_classes.field_policy import StorageKind
from ..field_registry import RECORD_CONTRACT_CLASSES, persisted_field_names
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


#: Computed columns that aggregate over the trace link (module call memory and
#: durations, descendant counts, parameter tallies, the owning module). The
#: scrubbed copy has no link and computes 0, None or [] for them, so the writer
#: reads them from the live record the copy was made from.
_LINK_AGGREGATES = (
    "forward_args_template",
    "forward_kwargs_template",
    "param_memory",
    "has_trainable_params",
    "has_frozen_params",
    "backward_duration",
    "func_calls_duration",
    "num_descendant_calls",
    "max_descendant_depth",
)
LIVE_COMPUTED_COLUMNS: dict[str, frozenset[str]] = {
    "module": frozenset(
        {
            *_LINK_AGGREGATES,
            "num_param_tensors_trainable",
            "num_param_tensors_frozen",
            "recursive_params",
            "num_recursive_params",
            "num_recursive_param_tensors",
            "num_recursive_param_tensors_trainable",
            "num_recursive_param_tensors_frozen",
            "forward_duration",
            "total_forward_duration",
            "total_backward_duration",
            "total_func_calls_duration",
            "total_output_activation_memory",
            "total_internal_activation_memory",
            "total_output_gradient_memory",
            "total_internal_gradient_memory",
            "total_autograd_memory",
        }
    ),
    "module_call": frozenset(
        {
            *_LINK_AGGREGATES,
            "module",
            "num_param_tensors",
            "num_param_tensors_trainable",
            "num_param_tensors_frozen",
            "output_activation_memory",
            "internal_activation_memory",
            "output_gradient_memory",
            "internal_gradient_memory",
            "autograd_memory",
        }
    ),
}
#: Computed columns resolved from source text: read live only when the save
#: keeps source (``include_source``); otherwise the scrubbed copy's value stands.
SOURCE_COMPUTED_COLUMNS: dict[str, frozenset[str]] = {"op": frozenset({"arg_expressions"})}
#: Trace fields the scrub normalizes for the 2.x pickle (the capture outcome
#: becomes its string payload); the 3.0 writer writes the live value object.
LIVE_TRACE_FIELDS = frozenset({"_capture_outcome"})
_COMPUTED_ERRORS = (AttributeError, LookupError, RuntimeError, TypeError, ValueError)


@cache
def _record_kinds_by_class() -> dict[type, str]:
    """Return record class to record kind for the twelve record contracts."""

    return {
        getattr(import_module(module), name): kind
        for kind, (module, name) in RECORD_CONTRACT_CLASSES.items()
    }


def record_kind_of(value: Any) -> str | None:
    """Return the record kind of a record object, or ``None`` for any other value."""

    kinds = _record_kinds_by_class()
    kind = kinds.get(type(value))
    if kind is not None:
        return kind
    for cls in type(value).__mro__[1:]:
        if cls in kinds:
            return kinds[cls]
    return None


class LiveCells:
    """Resolve the cell values the writer encodes, reading live where the copy cannot.

    Parameters
    ----------
    pairs:
        ``(scrubbed copy, live original)`` for every record the scrub copied,
        plus ``(scrubbed state mapping, live trace)`` for the trace.
    include_source:
        Whether the save keeps source text (gates ``SOURCE_COMPUTED_COLUMNS``).

    Notes
    -----
    A live value that holds live records is mapped to their scrubbed copies, so
    the cell references rows. An accessor of records (``layer.ops``,
    ``module.params``) is an edge: its cell is the list of its members, which the
    encoder writes as row references.
    """

    def __init__(
        self, pairs: Sequence[tuple[Any, Any]] = (), *, include_source: bool = True
    ) -> None:
        # Objects the walk kept as they were pair with themselves; drop them.
        self._pairs = [
            (copy, live) for copy, live in pairs if copy is not None and copy is not live
        ]
        self._live = {id(copy): live for copy, live in self._pairs}
        self._copy = {id(live): copy for copy, live in self._pairs}
        self._include_source = include_source

    def cell(self, kind: str, record: Any, state: Mapping[str, Any], column: str) -> Any:
        """Return the value of one cell of ``record`` (``state`` is its stored state)."""

        live = self._live.get(id(record))
        if kind == "trace":
            from_live = column in LIVE_TRACE_FIELDS and live is not None
            value = getattr(live, column, None) if from_live else state.get(column)
        elif column in state:
            value = state[column]
        elif record_schema(kind)[column].storage != StorageKind.COMPUTED.value:
            value = None
        else:
            reads_live = column in LIVE_COMPUTED_COLUMNS.get(kind, ()) or (
                self._include_source and column in SOURCE_COMPUTED_COLUMNS.get(kind, ())
            )
            source = live if reads_live and live is not None else record
            try:
                value = getattr(source, column)
            except _COMPUTED_ERRORS:
                value = None
        return self._to_copies(_edge_members(value))

    def _to_copies(self, value: Any) -> Any:
        """Replace live records inside ``value`` by their scrubbed copies."""

        if id(value) in self._copy:
            return self._copy[id(value)]
        if isinstance(value, list):
            return [self._to_copies(item) for item in value]
        if isinstance(value, tuple) and not hasattr(value, "_fields"):
            return tuple(self._to_copies(item) for item in value)
        if isinstance(value, dict):
            return {key: self._to_copies(item) for key, item in value.items()}
        return value


def _edge_members(value: Any) -> Any:
    """Return an accessor of records (or an empty one) as its member list.

    The members are recoverable keys of their own (pass index, call index,
    address), so the list loses nothing the accessor's lookup map held.
    """

    if isinstance(value, Accessor):
        members = list(value._list)
        if all(record_kind_of(member) is not None for member in members):
            return members
    return value
