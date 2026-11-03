"""Save side of the columnar grad_fn-call tables (:mod:`torchlens._io._grad_fn_call_columns`).

The canonical metadata pickler (:mod:`torchlens._io._canonical_pickle`) owns one
:class:`GradFnCallPacker` per dump and asks :func:`reduce_with_grad_fn_call_columns` for every
object. A ``GradFn`` is packed only when none of its calls was pickled before; a
``BackwardPass`` is packed only when every call it lists sits in a ``GradFn`` table whose save
already finished (the pickler reports each finished reduction through
:meth:`GradFnCallPacker.close`), so on load each referenced ``GradFn`` is complete before the
pass resolves it. Anything the tables cannot hold exactly keeps the per-object form, and an owner
without calls pickles byte-identically.
"""

from __future__ import annotations

import copyreg
import struct
from collections.abc import Mapping
from typing import Any

from ._grad_fn_call_columns import (
    _ABSENT,
    _ACCESSOR,
    _CALL,
    _CALL_KEYS,
    _FLOAT64,
    _GRAD_FN,
    _ID_SEPARATOR,
    _INT64,
    _OPTIONAL_FLOATS,
    _OPTIONAL_INTS,
    _PASS,
    _PAYLOADS,
    _SPARSE,
    GradFnCallColumns,
    GradFnCallRefs,
    OptColumn,
    RefColumn,
    _ints,
)
from .format_contract import BlobRef

__all__ = ("GradFnCallPacker", "reduce_with_grad_fn_call_columns")


def _identity(obj: Any) -> tuple[str, str]:
    """Return ``(module, qualname)`` of an object's exact type."""

    return type(obj).__module__, type(obj).__qualname__


def _optional(values: list[Any], value_type: type) -> OptColumn | None:
    """Pack values of exactly ``value_type`` or ``None`` as values plus a null mask."""

    if any(value is not None and type(value) is not value_type for value in values):
        return None
    filled = [value_type() if value is None else value for value in values]
    code = _INT64 if value_type is int else _FLOAT64
    if value_type is int and _ints(filled) is None:
        return None
    mask = bytes(value is None for value in values)
    return struct.pack(f"<{len(filled)}{code}", *filled), mask


def _ref_id(value: Any, kind: str | None) -> tuple[str, str] | None:
    """Return ``(blob_id, kind)`` of a reference the column holds, else ``None``."""

    if type(value) is not BlobRef:
        return None
    blob_id, ref_kind = value
    if type(blob_id) is not str or type(ref_kind) is not str or not blob_id:
        return None
    if _ID_SEPARATOR in blob_id or kind not in (None, ref_kind):
        return None
    return blob_id, ref_kind


def _payload_ids(value: Any, kind: str | None) -> tuple[list[str], str | None] | None:
    """Return ``(ids, kind)`` for a tuple of blob references and ``None`` slots, else ``None``."""

    if type(value) is not tuple:
        return None
    ids: list[str] = []
    for item in value:
        parts = None if item is None else _ref_id(item, kind)
        if item is not None and parts is None:
            return None
        if parts is not None:
            kind = parts[1]
        ids.append("" if parts is None else parts[0])
    return ids, kind


def _pack_payload(values: list[Any], sparse: dict[int, Any]) -> RefColumn | None:
    """Pack the reference-shaped payloads of one field; the rest go to ``sparse``."""

    kind: str | None = None
    arities: list[int] = []
    ids: list[str] = []
    for position, value in enumerate(values):
        parts = None if value is None else _payload_ids(value, kind)
        if parts is None:
            arities.append(_ABSENT)
            if value is not None:
                sparse[position] = value
            continue
        arities.append(len(parts[0]))
        ids.extend(parts[0])
        kind = parts[1] if parts[1] is not None else kind
    if all(arity == _ABSENT for arity in arities):
        return None
    packed = struct.pack(f"<{len(arities)}{_INT64}", *arities)  # tuple sizes: always int64
    return kind or "", packed, _ID_SEPARATOR.join(ids).encode("utf-8")


def _call_state(call: Any, proto: int) -> dict[str, Any] | None:
    """Return a call's pickle state when it is a plain ``GradFnCall`` the table holds."""

    if _identity(call) != _CALL:
        return None
    reduced = call.__reduce_ex__(proto)
    if type(reduced) is not tuple or len(reduced) not in (3, 5) or any(reduced[3:]):
        return None
    if reduced[0] is not copyreg.__newobj__ or reduced[1] != (type(call),):  # type: ignore[attr-defined]
        return None
    state = reduced[2]
    if type(state) is not dict or state.keys() != _CALL_KEYS:
        return None
    if state["_source_trace_ref"] is not None or type(state["grad_fn_label"]) is not str:
        return None
    return state


def _accessor_calls(accessor: Any, label: Any) -> dict[Any, Any] | None:
    """Return the call dict of a plain ``GradFnCallAccessor`` the table can rebuild exactly."""

    if _identity(accessor) != _ACCESSOR or not set(vars(accessor)) <= {
        "_dict",
        "_list",
        "_label",
        "_source_ref",
    }:
        return None
    calls, ordered = vars(accessor).get("_dict"), vars(accessor).get("_list")
    if type(calls) is not dict or not calls or type(ordered) is not list:
        return None
    if vars(accessor).get("_source_ref") is not None or vars(accessor).get("_label") != label:
        return None
    try:
        expected = [call for _key, call in sorted(calls.items())]
    except TypeError:
        return None
    if len(expected) != len(ordered) or any(
        a is not b for a, b in zip(expected, ordered, strict=True)
    ):
        return None
    return calls


def _pack_calls(calls: dict[Any, Any], proto: int) -> GradFnCallColumns | None:
    """Pack the calls of one ``GradFn`` (``None`` when any value does not fit exactly)."""

    states = [_call_state(call, proto) for call in calls.values()]
    versions = {(type(s["tlspec_version"]), s["tlspec_version"]) for s in states if s}
    if any(state is None for state in states) or len(versions) != 1:
        return None
    rows: list[dict[str, Any]] = states  # type: ignore[assignment]
    version = next(iter(versions))[1]
    keys, indices = _ints(list(calls)), _ints([row["call_index"] for row in rows])
    optional = {name: _optional([row[name] for row in rows], int) for name in _OPTIONAL_INTS}
    optional.update(
        {name: _optional([row[name] for row in rows], float) for name in _OPTIONAL_FLOATS}
    )
    if type(version) is not int or keys is None or indices is None or None in optional.values():
        return None
    labels: dict[str, int] = {}
    positions = [labels.setdefault(row["grad_fn_label"], len(labels)) for row in rows]
    label_index = struct.pack(f"<{len(positions)}{_INT64}", *positions)  # small, always int64
    sparse: dict[str, dict[int, Any]] = {name: {} for name in _SPARSE}
    refs = {}
    for name in _PAYLOADS:
        column = _pack_payload([row[name] for row in rows], sparse[name])
        if column is not None:
            refs[name] = column
    for position, row in enumerate(rows):
        if row["fire_record"] is not None:
            sparse["fire_record"][position] = row["fire_record"]
    return GradFnCallColumns(
        count=len(rows),
        tlspec_version=version,
        keys=keys,
        call_index=indices,
        optional=optional,  # type: ignore[arg-type]
        label_index=label_index,
        labels=tuple(labels),
        refs=refs,
        sparse={name: values for name, values in sparse.items() if values},
    )


class GradFnCallPacker:
    """Per-dump bookkeeping that links packed ``GradFn`` tables to ``BackwardPass`` references.

    Parameters
    ----------
    memo:
        The pickler's memo (keyed by ``id``): a call already pickled is never packed again.
    """

    def __init__(self, memo: Mapping[int, Any]) -> None:
        self._memo = memo
        self._calls: dict[int, tuple[Any, int, int]] = {}  # id(call) -> (call, owner pos, key)
        self._owners: list[Any] = []
        self._open: set[int] = set()
        self._owner_tuple: tuple[Any, ...] = ()

    def pack_grad_fn(self, grad_fn: Any, state: dict[str, Any], proto: int) -> Any:
        """Return ``state`` with its calls packed, or ``None`` to keep the per-object form."""

        calls = _accessor_calls(state.get("calls"), state.get("label"))
        if calls is None or any(id(call) in self._memo for call in calls.values()):
            return None
        if any(type(key) is not int for key in calls):
            return None
        columns = _pack_calls(calls, proto)
        if columns is None:
            return None
        position = len(self._owners)
        self._owners.append(grad_fn)
        self._open.add(id(grad_fn))
        for key, call in calls.items():
            self._calls[id(call)] = (call, position, key)
        return {**state, "calls": columns}

    def close(self, obj: Any) -> None:
        """Record that the pickler finished saving ``obj`` (a packed ``GradFn``'s BUILD is out)."""

        self._open.discard(id(obj))

    def pack_pass(self, state: dict[str, Any]) -> Any:
        """Return a pass state with its call list as references, or ``None`` to keep it."""

        calls = state.get("grad_fn_calls")
        if type(calls) is not list or not calls:
            return None
        entries = [self._calls.get(id(call)) for call in calls]
        if any(
            entry is None or entry[0] is not call
            for entry, call in zip(entries, calls, strict=True)
        ):
            return None
        found: list[tuple[Any, int, int]] = entries  # type: ignore[assignment]
        if any(id(self._owners[entry[1]]) in self._open for entry in found):
            return None
        if len(self._owner_tuple) != len(self._owners):
            self._owner_tuple = tuple(self._owners)
        refs = GradFnCallRefs(
            count=len(found),
            owners=self._owner_tuple,
            owner_index=_ints([entry[1] for entry in found]) or b"",
            keys=_ints([entry[2] for entry in found]) or b"",
        )
        return {**state, "grad_fn_calls": refs}


def reduce_with_grad_fn_call_columns(
    packer: GradFnCallPacker, obj: Any, cls: type, proto: int
) -> Any:
    """Return a ``GradFn`` or ``BackwardPass`` reduction with its calls packed.

    Parameters
    ----------
    packer:
        The dump's :class:`GradFnCallPacker`.
    obj, cls:
        Object the canonical metadata pickler is about to save, and ``type(obj)``.
    proto:
        Pickle protocol in use.

    Returns
    -------
    Any
        ``NotImplemented`` (the default reduction) unless ``obj`` is a ``GradFn`` or a
        ``BackwardPass`` whose calls the tables hold exactly; then the ordinary
        ``__reduce_ex__`` tuple with only that state entry replaced.
    """

    identity = (cls.__module__, cls.__qualname__)
    if identity not in (_GRAD_FN, _PASS):
        return NotImplemented
    reduced = obj.__reduce_ex__(proto)
    if type(reduced) is not tuple or len(reduced) < 3 or type(reduced[2]) is not dict:
        return NotImplemented
    if identity == _GRAD_FN:
        state = packer.pack_grad_fn(obj, reduced[2], proto)
    else:
        state = packer.pack_pass(reduced[2])
    if state is None:
        return NotImplemented
    return (*reduced[:2], state, *reduced[3:])
