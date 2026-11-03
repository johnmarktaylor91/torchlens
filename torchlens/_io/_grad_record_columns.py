"""Columnar persisted form of an op's per-backward-pass gradient records.

An ``Op`` keeps one ``GradientRecord`` per backward pass that reached it. Pickled as one
object per record, ``metadata.pkl`` spends tens of opcodes on every record, so a trace with
many ops and many backward passes (a batched Jacobian fit over a long sequential scan) passes
the safe unpickler's opcode ceiling while its byte size stays modest. This module writes the
records of one op as ONE :class:`GradientRecordColumns` table instead: little-endian
fixed-width numeric columns held in ``bytes``, deduplicated shape and dtype tables, and the
saved-payload references packed the same way. The opcode cost per op is then a small constant,
whatever the number of passes; only the byte size grows with the record count.

Seams. Save: the canonical metadata pickler (:mod:`torchlens._io._canonical_pickle`) asks
:func:`reduce_with_grad_record_columns` for every object; for an ``Op`` whose pickle state
holds a non-empty list of records it returns the ordinary reduction with that one state entry
packed. The scrubbed object model keeps its record lists, so every walk after the scrub
(streamed blob-id reuse, skipped-blob replacement) still sees ``GradientRecord`` objects, and an
op without records pickles byte-identically to before. Load: ``Op.__setstate__`` calls
:func:`unpack_grad_records_state` before any other restore step, so every load path (eager,
lazy, nested materialization) sees a ``list[GradientRecord]`` again; ``owner`` is not stored
(``WEAKREF_STRIP``) and is re-pointed at the op by the accessor rebuild, as for the per-object
form. A record list that does not fit the columns exactly keeps the per-object form.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .._quantities import Bytes
from .format_contract import BlobRef
from .format_errors import ArtifactError

__all__ = (
    "GradientRecordColumns",
    "pack_gradient_records",
    "reduce_with_grad_record_columns",
    "unpack_grad_records_state",
    "unpack_gradient_records",
)

#: The persisted fields of one record, in ``GradientRecord._PORTABLE_STATE_SPEC`` order.
_RECORD_FIELDS = frozenset(
    {
        "owner",
        "ordinal",
        "backward_pass_number",
        "grad",
        "transformed_grad",
        "shape",
        "dtype",
        "memory",
        "timestamp",
    }
)
_INT32 = ("i", 4, -(2**31), 2**31 - 1)
_INT64 = ("q", 8, -(2**63), 2**63 - 1)
_FLOAT64_CODE, _FLOAT64_SIZE = "d", 8
#: Separator of the packed blob ids (a blob id is a short ASCII token, never multi-line).
_ID_SEPARATOR = "\n"

#: Pickle identities of ``Op`` (it has no subclasses) and ``GradientRecord``. Names, not
#: imports: this module stays a leaf of the import graph (``op.py`` imports it).
_OP_MODULE, _OP_NAME = "torchlens._data_classes.op", "Op"
_RECORD_TYPE = ("torchlens._data_classes.op", "GradientRecord")

#: A packed run of blob references: ``(kind, int32 record positions, separator-joined ids)``.
RefColumn = tuple[str, bytes, bytes]


@dataclass(frozen=True)
class GradientRecordColumns:
    """The gradient records of one op, persisted as one columnar table.

    Attributes
    ----------
    count:
        Number of records.
    ordinal, backward_pass_number, shape_index, dtype_index:
        Little-endian 4-byte signed integer columns, one value per record.
    memory:
        Little-endian 8-byte signed integer column of gradient bytes (restored as ``Bytes``).
    timestamp:
        Little-endian 8-byte float column of event timestamps.
    shapes, dtypes:
        Deduplicated value tables the index columns point into.
    grads, transformed_grads:
        Sparse payloads by record position, holding the scrubbed value of every record whose
        payload is set and is not packed in the matching reference column.
    grad_refs, transformed_grad_refs:
        Blob references of one kind packed as ``(kind, positions, ids)``, or ``None``.
    """

    count: int
    ordinal: bytes
    backward_pass_number: bytes
    memory: bytes
    timestamp: bytes
    shape_index: bytes
    dtype_index: bytes
    shapes: tuple[Any, ...]
    dtypes: tuple[Any, ...]
    grads: dict[int, Any]
    transformed_grads: dict[int, Any]
    grad_refs: RefColumn | None
    transformed_grad_refs: RefColumn | None


def _int_column(values: list[Any], spec: tuple[str, int, int, int]) -> bytes | None:
    """Pack exact ``int`` values into a little-endian column, or ``None`` when one does not fit."""

    code, _size, low, high = spec
    if any(type(value) is not int or not low <= value <= high for value in values):
        return None
    return struct.pack(f"<{len(values)}{code}", *values)


def _index_column(values: list[Any]) -> tuple[bytes, tuple[Any, ...]] | None:
    """Return the 4-byte index column and the deduplicated table for hashable ``values``."""

    table: dict[tuple[type, Any], int] = {}
    indices: list[int] = []
    try:
        for value in values:
            indices.append(table.setdefault((type(value), value), len(table)))
    except TypeError:  # an unhashable shape or dtype keeps the per-object form
        return None
    column = _int_column(indices, _INT32)
    if column is None:
        return None
    return column, tuple(value for _key_type, value in table)


def _ref_parts(value: Any, kind: str | None) -> tuple[str, str] | None:
    """Return ``(blob_id, kind)`` of a blob reference the reference column can hold."""

    if type(value) is not BlobRef:
        return None
    blob_id, ref_kind = value
    if type(blob_id) is not str or type(ref_kind) is not str or _ID_SEPARATOR in blob_id:
        return None
    return (blob_id, ref_kind) if kind in (None, ref_kind) else None


def _pack_payloads(values: list[Any]) -> tuple[dict[int, Any], RefColumn | None]:
    """Split one payload field into packed blob references and a sparse remainder."""

    sparse: dict[int, Any] = {}
    kind: str | None = None
    positions: list[int] = []
    blob_ids: list[str] = []
    for position, value in enumerate(values):
        if value is None:
            continue
        parts = _ref_parts(value, kind)
        if parts is None:
            sparse[position] = value
            continue
        blob_ids.append(parts[0])
        kind = parts[1]
        positions.append(position)
    if kind is None:
        return sparse, None
    packed_positions = _int_column(positions, _INT32)
    assert packed_positions is not None  # positions are list indices, far below int32
    return sparse, (kind, packed_positions, _ID_SEPARATOR.join(blob_ids).encode("utf-8"))


def _record_states(records: Any) -> list[Mapping[str, Any]] | None:
    """Return each record's state when ``records`` is a list the columns hold exactly."""

    if type(records) is not list or not records:
        return None
    states: list[Mapping[str, Any]] = []
    for record in records:
        if (type(record).__module__, type(record).__qualname__) != _RECORD_TYPE:
            return None
        state = vars(record)
        if state.keys() != _RECORD_FIELDS or state["owner"] is not None:
            return None
        if type(state["memory"]) is not Bytes or type(state["timestamp"]) is not float:
            return None
        states.append(state)
    return states


def pack_gradient_records(records: Any) -> GradientRecordColumns | None:
    """Pack one op's scrubbed gradient records into a columnar table.

    Parameters
    ----------
    records:
        The op's ``_grad_records`` value from its scrubbed pickle state.

    Returns
    -------
    GradientRecordColumns | None
        The table, or ``None`` when ``records`` is empty, is not a plain list of
        ``GradientRecord`` objects with exactly the persisted fields and an unset ``owner``, or
        holds a value the columns cannot represent exactly; the caller then keeps the list.
    """

    states = _record_states(records)
    if states is None:
        return None
    ordinal = _int_column([state["ordinal"] for state in states], _INT32)
    pass_index = _int_column([state["backward_pass_number"] for state in states], _INT32)
    memory = _int_column([int(state["memory"]) for state in states], _INT64)
    shapes = _index_column([state["shape"] for state in states])
    dtypes = _index_column([state["dtype"] for state in states])
    if ordinal is None or pass_index is None or memory is None or not shapes or not dtypes:
        return None
    grads, grad_refs = _pack_payloads([state["grad"] for state in states])
    transformed, transformed_refs = _pack_payloads([state["transformed_grad"] for state in states])
    timestamps = [state["timestamp"] for state in states]
    return GradientRecordColumns(
        count=len(states),
        ordinal=ordinal,
        backward_pass_number=pass_index,
        memory=memory,
        timestamp=struct.pack(f"<{len(timestamps)}{_FLOAT64_CODE}", *timestamps),
        shape_index=shapes[0],
        dtype_index=dtypes[0],
        shapes=shapes[1],
        dtypes=dtypes[1],
        grads=grads,
        transformed_grads=transformed,
        grad_refs=grad_refs,
        transformed_grad_refs=transformed_refs,
    )


def _corrupt(detail: str) -> ArtifactError:
    """Return the load refusal for a malformed columnar gradient-record table."""

    return ArtifactError(f"Corrupt gradient-record columns in bundle metadata: {detail}.")


def _read_column(data: Any, count: int, code: str, size: int, name: str) -> tuple[Any, ...]:
    """Decode one fixed-width column after checking its type and exact length."""

    if type(data) is not bytes or len(data) != count * size:
        raise _corrupt(f"column {name!r} does not hold {count} values")
    return struct.unpack(f"<{count}{code}", data)


def _read_indices(data: Any, count: int, table: Any, name: str) -> list[Any]:
    """Decode an index column and resolve it against its value table."""

    if type(table) is not tuple:
        raise _corrupt(f"table for {name!r} is not a tuple")
    indices = _read_column(data, count, _INT32[0], _INT32[1], name)
    if any(not 0 <= index < len(table) for index in indices):
        raise _corrupt(f"column {name!r} points outside its table")
    return [table[index] for index in indices]


def _read_payloads(sparse: Any, refs: Any, count: int, name: str) -> list[Any]:
    """Rebuild one payload field from its sparse entries and packed references."""

    values: list[Any] = [None] * count
    if type(sparse) is not dict:
        raise _corrupt(f"payloads {name!r} are not a dict")
    for position, value in sparse.items():
        if type(position) is not int or not 0 <= position < count:
            raise _corrupt(f"payloads {name!r} name a record outside the table")
        values[position] = value
    if refs is None:
        return values
    if type(refs) is not tuple or len(refs) != 3 or type(refs[0]) is not str:
        raise _corrupt(f"references {name!r} are malformed")
    kind, packed_positions, packed_ids = refs
    if type(packed_ids) is not bytes or type(packed_positions) is not bytes:
        raise _corrupt(f"references {name!r} are malformed")
    try:
        blob_ids = packed_ids.decode("utf-8").split(_ID_SEPARATOR)
    except UnicodeDecodeError as error:
        raise _corrupt(f"references {name!r} hold undecodable ids") from error
    positions = _read_column(packed_positions, len(blob_ids), _INT32[0], _INT32[1], name)
    for position, blob_id in zip(positions, blob_ids, strict=True):
        if not 0 <= position < count or values[position] is not None:
            raise _corrupt(f"references {name!r} name an invalid record")
        values[position] = BlobRef(blob_id=blob_id, kind=kind)
    return values


def unpack_gradient_records(
    columns: GradientRecordColumns, record_type: Callable[..., Any]
) -> list[Any]:
    """Rebuild the ``GradientRecord`` list a :class:`GradientRecordColumns` table holds.

    Parameters
    ----------
    columns:
        Table read from bundle metadata (untrusted: every column is checked).
    record_type:
        ``GradientRecord`` (passed in by ``Op`` so this module never imports it).

    Returns
    -------
    list[GradientRecord]
        Records in their saved order, ``owner`` unset (re-pointed by the accessor rebuild).

    Raises
    ------
    ArtifactError
        A column has the wrong type or length, or an index or position is out of range.
    """

    count = columns.count
    if type(count) is not int or count < 0:
        raise _corrupt("record count is not a non-negative int")
    ordinals = _read_column(columns.ordinal, count, _INT32[0], _INT32[1], "ordinal")
    passes = _read_column(columns.backward_pass_number, count, _INT32[0], _INT32[1], "pass")
    memories = _read_column(columns.memory, count, _INT64[0], _INT64[1], "memory")
    stamps = _read_column(columns.timestamp, count, _FLOAT64_CODE, _FLOAT64_SIZE, "timestamp")
    shapes = _read_indices(columns.shape_index, count, columns.shapes, "shape_index")
    dtypes = _read_indices(columns.dtype_index, count, columns.dtypes, "dtype_index")
    grads = _read_payloads(columns.grads, columns.grad_refs, count, "grads")
    transformed = _read_payloads(
        columns.transformed_grads, columns.transformed_grad_refs, count, "transformed_grads"
    )
    return [
        record_type(
            owner=None,
            ordinal=ordinals[position],
            backward_pass_number=passes[position],
            grad=grads[position],
            transformed_grad=transformed[position],
            shape=shapes[position],
            dtype=dtypes[position],
            memory=memories[position],
            timestamp=stamps[position],
        )
        for position in range(count)
    ]


def unpack_grad_records_state(state: dict[str, Any], record_type: Callable[..., Any]) -> None:
    """Restore a persisted ``_grad_records`` table in an ``Op`` pickle state, in place.

    Parameters
    ----------
    state:
        ``Op.__setstate__`` state; any other ``_grad_records`` value is left as it is.
    record_type:
        ``GradientRecord``.
    """

    columns = state.get("_grad_records")
    if type(columns) is GradientRecordColumns:
        state["_grad_records"] = unpack_gradient_records(columns, record_type)


def reduce_with_grad_record_columns(obj: Any, cls: type, proto: int) -> Any:
    """Return an ``Op`` reduction whose ``_grad_records`` state is columnar.

    Parameters
    ----------
    obj:
        Object the canonical metadata pickler is about to save.
    cls:
        ``type(obj)``.
    proto:
        Pickle protocol in use.

    Returns
    -------
    Any
        ``NotImplemented`` (the pickler's default reduction) unless ``obj`` is an ``Op``
        whose state holds records :func:`pack_gradient_records` packs; then the ordinary
        ``__reduce_ex__`` tuple with only that state entry replaced.
    """

    if cls.__name__ != _OP_NAME or cls.__module__ != _OP_MODULE:
        return NotImplemented
    reduced = obj.__reduce_ex__(proto)
    if type(reduced) is not tuple or len(reduced) < 3 or type(reduced[2]) is not dict:
        return NotImplemented
    columns = pack_gradient_records(reduced[2].get("_grad_records"))
    if columns is None:
        return NotImplemented
    state = dict(reduced[2])
    state["_grad_records"] = columns
    return (*reduced[:2], state, *reduced[3:])
