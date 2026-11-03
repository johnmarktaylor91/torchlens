"""Columnar persisted form of the grad_fn calls of one ``GradFn``, and packed pass references.

Every fire of an autograd node during a backward pass is one ``GradFnCall``. Pickled as one
object per call, ``metadata.pkl`` spends about 35 opcodes on each, so a trace with many
grad_fns and many backward passes passes the safe unpickler's opcode ceiling while its byte
size stays modest. The calls of one ``GradFn`` persist as ONE :class:`GradFnCallColumns` table
instead (little-endian fixed-width columns in ``bytes``, a deduplicated label table, packed blob
references, sparse remainders), and the call list of one ``BackwardPass`` as ONE
:class:`GradFnCallRefs` table of ``(owner, key)`` references into those tables, so the opcode
cost is a constant per owner whatever the number of calls.

This module holds the two tables and their load side: ``GradFn.__setstate__`` calls
:func:`unpack_grad_fn_calls_state` and ``BackwardPass.__setstate__`` calls
:func:`unpack_grad_fn_call_refs_state` before any other restore step, so every load path sees
``GradFnCall`` objects again, the same objects in both owners. Every column is checked: the
tables come from an untrusted file. The save side is :mod:`torchlens._io._grad_fn_call_packing`.
It imports nothing from ``_data_classes`` (record classes are passed in), so it stays a leaf of
the import graph.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, cast

from .format_contract import BlobRef
from .format_errors import ArtifactError

__all__ = (
    "GradFnCallColumns",
    "GradFnCallRefs",
    "unpack_grad_fn_call_refs_state",
    "unpack_grad_fn_calls_state",
)

#: Pickle identities, by name: this module stays a leaf of the import graph.
_GRAD_FN = ("torchlens._data_classes.grad_fn", "GradFn")
_PASS = ("torchlens._data_classes.backward_pass", "BackwardPass")
_CALL = ("torchlens._data_classes.grad_fn_call", "GradFnCall")
_ACCESSOR = ("torchlens._data_classes.grad_fn", "GradFnCallAccessor")
#: The pickle state keys of one call (``GradFnCall.__getstate__``).
_CALL_KEYS = frozenset(
    {
        "call_index",
        "ordinal",
        "backward_pass_index",
        "grad_fn_label",
        "grad_inputs",
        "grad_outputs",
        "fire_record",
        "timestamp",
        "_time_started",
        "_time_finished",
        "_source_trace_ref",
        "tlspec_version",
    }
)
_OPTIONAL_INTS = ("ordinal", "backward_pass_index")
_OPTIONAL_FLOATS = ("timestamp", "_time_started", "_time_finished")
_PAYLOADS = ("grad_inputs", "grad_outputs")
_SPARSE = (*_PAYLOADS, "fire_record")
_INT64, _FLOAT64 = "q", "d"
_ID_SEPARATOR = "\n"
#: Arity code of a call whose payload is not in the reference column (else its tuple size).
_ABSENT = -1

#: ``(values, null mask)``: one packed value and one 0/1 byte per call (1 marks ``None``).
OptColumn = tuple[bytes, bytes]
#: ``(kind, int64 arity per call, separator-joined ids)``; an empty id is a ``None`` slot.
RefColumn = tuple[str, bytes, bytes]


@dataclass(frozen=True)
class GradFnCallColumns:
    """The calls of one ``GradFn``, persisted as one columnar table.

    ``keys`` and ``call_index`` are int64 columns (the accessor key and the call's own index),
    the optional fields are :data:`OptColumn` pairs, ``label_index`` points into ``labels``,
    ``grad_inputs``/``grad_outputs`` pack blob references, and ``sparse`` holds every payload or
    ``fire_record`` value the columns do not, by field name and call position.
    """

    count: int
    tlspec_version: int
    keys: bytes
    call_index: bytes
    optional: dict[str, OptColumn]
    label_index: bytes
    labels: tuple[str, ...]
    refs: dict[str, RefColumn]
    sparse: dict[str, dict[int, Any]]


@dataclass(frozen=True)
class GradFnCallRefs:
    """The calls of one ``BackwardPass``, persisted as references into ``GradFn`` tables.

    ``owners`` is a tuple of ``GradFn`` objects shared by every pass of one artifact (one memo
    entry); ``owner_index`` and ``keys`` are int64 columns naming ``owners[i].calls[key]``.
    """

    count: int
    owners: tuple[Any, ...]
    owner_index: bytes
    keys: bytes


def _ints(values: list[Any]) -> bytes | None:
    """Pack exact int64 ``int`` values, or ``None`` when one does not fit."""

    if any(type(value) is not int or not -(2**63) <= value < 2**63 for value in values):
        return None
    return struct.pack(f"<{len(values)}{_INT64}", *values)


def _corrupt(detail: str) -> ArtifactError:
    """Return the load refusal for a malformed grad_fn-call table."""

    return ArtifactError(f"Corrupt grad_fn-call columns in bundle metadata: {detail}.")


def _read(data: Any, count: int, code: str, name: str) -> tuple[Any, ...]:
    """Decode one fixed-width column after checking its type and exact length."""

    if type(data) is not bytes or len(data) != count * struct.calcsize(f"<{code}"):
        raise _corrupt(f"column {name!r} does not hold {count} values")
    return struct.unpack(f"<{count}{code}", data)


def _read_optional(column: Any, count: int, code: str, name: str) -> list[Any]:
    """Decode an :data:`OptColumn` into values with ``None`` where the mask is set."""

    if type(column) is not tuple or len(column) != 2:
        raise _corrupt(f"column {name!r} is malformed")
    values = _read(column[0], count, code, name)
    mask = column[1]
    if type(mask) is not bytes or len(mask) != count or any(flag > 1 for flag in mask):
        raise _corrupt(f"null mask of {name!r} is malformed")
    return [None if flag else value for value, flag in zip(values, mask, strict=True)]


def _read_payload(column: Any, values: list[Any], name: str) -> None:
    """Fill ``values`` from a packed reference column, in place."""

    if type(column) is not tuple or len(column) != 3 or type(column[0]) is not str:
        raise _corrupt(f"references {name!r} are malformed")
    kind, packed, joined = column
    arities = _read(packed, len(values), _INT64, name)
    if type(joined) is not bytes:
        raise _corrupt(f"references {name!r} are malformed")
    try:
        ids = joined.decode("utf-8").split(_ID_SEPARATOR) if joined else []
    except UnicodeDecodeError as error:
        raise _corrupt(f"references {name!r} hold undecodable ids") from error
    cursor = 0
    for position, arity in enumerate(arities):
        if arity == _ABSENT:
            continue
        if arity < 0 or values[position] is not None or cursor + arity > len(ids):
            raise _corrupt(f"references {name!r} name an invalid call")
        chunk = ids[cursor : cursor + arity]
        values[position] = tuple(BlobRef(blob_id=i, kind=kind) if i else None for i in chunk)
        cursor += arity
    if cursor != len(ids):
        raise _corrupt(f"references {name!r} hold unused ids")


def _read_optional_fields(optional: Any, count: int) -> dict[str, list[Any]]:
    """Decode every optional int and float column of a table."""

    if type(optional) is not dict or set(optional) != {*_OPTIONAL_INTS, *_OPTIONAL_FLOATS}:
        raise _corrupt("optional columns are malformed")
    return {
        name: _read_optional(column, count, _INT64 if name in _OPTIONAL_INTS else _FLOAT64, name)
        for name, column in optional.items()
    }


def _read_sparse(sparse: Any, fields: dict[str, list[Any]], count: int) -> None:
    """Fill the sparse payload and ``fire_record`` values into ``fields``, in place."""

    if type(sparse) is not dict or not set(sparse) <= set(_SPARSE):
        raise _corrupt("sparse values are malformed")
    for name, entries in sparse.items():
        if type(entries) is not dict:
            raise _corrupt(f"sparse {name!r} is not a dict")
        for position, value in entries.items():
            if type(position) is not int or not 0 <= position < count or value is None:
                raise _corrupt(f"sparse {name!r} names an invalid call")
            fields[name][position] = value


def _read_states(columns: GradFnCallColumns) -> list[dict[str, Any]]:
    """Rebuild the pickle state of every call a table holds (untrusted input, all checked)."""

    count = columns.count
    if type(count) is not int or count < 0 or type(columns.tlspec_version) is not int:
        raise _corrupt("count or version is malformed")
    labels = columns.labels
    if type(labels) is not tuple or any(type(label) is not str for label in labels):
        raise _corrupt("label table is malformed")
    label_index = _read(columns.label_index, count, _INT64, "label_index")
    if any(not 0 <= index < len(labels) for index in label_index):
        raise _corrupt("label index points outside its table")
    fields: dict[str, list[Any]] = {
        "call_index": list(_read(columns.call_index, count, _INT64, "call_index")),
        "grad_fn_label": [labels[index] for index in label_index],
    }
    fields.update(_read_optional_fields(columns.optional, count))
    fields.update({name: [None] * count for name in _SPARSE})
    _read_sparse(columns.sparse, fields, count)
    if type(columns.refs) is not dict or not set(columns.refs) <= set(_PAYLOADS):
        raise _corrupt("reference columns are malformed")
    for name, ref_column in columns.refs.items():
        _read_payload(ref_column, fields[name], name)
    fixed = {"_source_trace_ref": None, "tlspec_version": columns.tlspec_version}
    return [{**fixed, **{name: v[i] for name, v in fields.items()}} for i in range(count)]


def unpack_grad_fn_calls_state(state: dict[str, Any], record_type: type) -> None:
    """Restore a persisted ``calls`` table in a ``GradFn`` pickle state, in place.

    Each call is rebuilt the way pickle rebuilds it (``record_type.__new__`` then
    ``__setstate__``) and ``state["calls"]`` becomes the ``{key: call}`` dict the ``GradFn``
    wraps in its accessor. Any other ``calls`` value is left as it is.

    Raises
    ------
    ArtifactError
        A column has the wrong type or length, or an index or position is out of range.
    """

    columns = state.get("calls")
    if type(columns) is not GradFnCallColumns:
        return
    call_states = _read_states(columns)
    keys = _read(columns.keys, columns.count, _INT64, "keys")
    calls: dict[int, Any] = {}
    for key, call_state in zip(keys, call_states, strict=True):
        call = cast(Any, record_type).__new__(record_type)  # as pickle: cls.__new__(cls)
        call.__setstate__(call_state)
        calls[key] = call
    if len(calls) != len(keys):
        raise _corrupt("call keys repeat")
    state["calls"] = calls


def unpack_grad_fn_call_refs_state(state: dict[str, Any], record_type: type) -> None:
    """Resolve a persisted ``grad_fn_calls`` reference table in a ``BackwardPass`` state.

    Raises
    ------
    ArtifactError
        A reference names no owner, no call, or an object that is not a ``record_type``.
    """

    refs = state.get("grad_fn_calls")
    if type(refs) is not GradFnCallRefs:
        return
    owners = refs.owners
    if type(refs.count) is not int or refs.count < 0 or type(owners) is not tuple:
        raise _corrupt("pass references are malformed")
    positions = _read(refs.owner_index, refs.count, _INT64, "owner_index")
    keys = _read(refs.keys, refs.count, _INT64, "keys")
    resolved: list[Any] = []
    for position, key in zip(positions, keys, strict=True):
        if not 0 <= position < len(owners):
            raise _corrupt("a pass reference names no grad_fn")
        mapping = getattr(getattr(owners[position], "calls", None), "_dict", None)
        call = mapping.get(key) if type(mapping) is dict else None
        if type(call) is not record_type:
            raise _corrupt("a pass reference names no call")
        resolved.append(call)
    state["grad_fn_calls"] = resolved
