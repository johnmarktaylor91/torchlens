"""Typed JSON encoding of scrubbed trace state for the 3.0 writer.

The scrubbed state is an object graph: record objects (ops, params, modules,
module calls, call locations, ...) are shared by many owners. The encoder
flattens it into one row table per record kind and replaces every record
object with a row reference, so the JSON stays shallow and every shared
object is written once. Everything else becomes ordinary JSON or a typed
object tagged with ``contract.TYPED_VALUE_KEY``:

- JSON scalars, lists and string-keyed dicts are written as themselves;
- tuples, sets, frozensets, non-string-keyed or marker-keyed dicts,
  bytes, complex numbers, non-finite floats, out-of-range ints, torch dtypes
  and devices, classes and enum members carry their tag;
- a tensor payload (a scrubbed ``BlobRef``) becomes a tensor reference
  object resolved against the written blob entries;
- a TorchLens value object (quantities, dtype and device refs, accessors,
  spec objects that are not records) becomes an ``object`` with its class
  path and encoded state; the reader rebuilds only classes on its table;
- a callable becomes a ``callable_placeholder`` (never imported on read);
- any other value is stringified into an ``unencodable`` disclosure and
  listed in the manifest, never pickled.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import json
import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping
from typing import Any

import torch

from ..._data_classes._state_adapter import state_items
from ..format_contract import BlobRef
from ..format_errors import ArtifactError
from .contract import (
    RECORD_INDEX,
    TYPED_VALUE_KEY,
    UNENCODABLE_REPR_LIMIT,
)

#: Largest integer magnitude JSON readers carry exactly (IEEE double mantissa).
_JSON_SAFE_INT = 2**53
#: Tag of a reference to a row of a record table (added by the writer lane).
ROW_TAG = "row"
#: Tag of a TorchLens value object rebuilt from its class path (writer lane).
OBJECT_TAG = "object"
#: Deepest inline nesting the encoder writes; the bounded JSON reader refuses
#: at 200, and record rows never nest each other, so real values stay far below.
_MAX_INLINE_DEPTH = 120

_RECORD_KIND_BY_CLASS_PATH: dict[str, str] = {path: kind for kind, path, _ in RECORD_INDEX}


def class_path(cls: type) -> str:
    """Return the ``module.qualname`` spelling the format uses for a class.

    Parameters
    ----------
    cls:
        Class to name.

    Returns
    -------
    str
        Dotted module path and qualified name.
    """

    return f"{cls.__module__}.{cls.__qualname__}"


def record_kind_of(value: Any) -> str | None:
    """Return the record kind an object is stored as, or ``None``.

    Parameters
    ----------
    value:
        Any value from the scrubbed state.

    Returns
    -------
    str | None
        One of the twelve record kinds of ``contract.RECORD_INDEX``.
    """

    return _RECORD_KIND_BY_CLASS_PATH.get(class_path(type(value)))


def _tagged(tag: str, **fields: Any) -> dict[str, Any]:
    """Return a typed JSON object with ``tag`` and its fields."""

    return {TYPED_VALUE_KEY: tag, **fields}


class StateEncoder:
    """Encode scrubbed state into record tables plus typed JSON values.

    Parameters
    ----------
    tensor_refs:
        Blob id to tensor reference fields (``contract.TENSOR_REFERENCE_FIELDS``)
        for every blob the save wrote.
    inline_tensor:
        Stores a tensor the scrub left inline (annotation tensors) in the
        inline blob file and returns its reference fields.
    """

    def __init__(
        self,
        tensor_refs: Mapping[str, dict[str, Any]],
        *,
        inline_tensor: Callable[[torch.Tensor], dict[str, Any]],
    ) -> None:
        self._tensor_refs = tensor_refs
        self._inline_tensor = inline_tensor
        self.tables: dict[str, list[dict[str, Any] | None]] = {
            kind: [] for kind, _, _ in RECORD_INDEX
        }
        self.unencodable: list[dict[str, Any]] = []
        self._rows: dict[int, tuple[str, int]] = {}
        self._pinned: list[Any] = []
        self._pending: list[tuple[str, int, Any]] = []
        self._active: set[int] = set()

    def encode_trace_state(self, state: Mapping[str, Any]) -> None:
        """Encode the scrubbed trace state as row 0 of the ``trace`` table.

        Parameters
        ----------
        state:
            Scrubbed top-level trace state from the save scrub.
        """

        self.tables["trace"].append(None)
        self.tables["trace"][0] = self._encode_fields(state, "trace[0]")
        self._drain()

    def row_reference(self, value: Any, kind: str) -> dict[str, Any]:
        """Return the row reference for a record object, registering it once.

        Parameters
        ----------
        value:
            Record object (an ``Op``, ``Param``, ...).
        kind:
            Its record kind.

        Returns
        -------
        dict[str, Any]
            ``{"$tl": "row", "record": kind, "row": index}``.
        """

        if kind == "trace":
            return _tagged(ROW_TAG, record="trace", row=0)
        known = self._rows.get(id(value))
        if known is None:
            index = len(self.tables[kind])
            self.tables[kind].append(None)
            self._pinned.append(value)
            known = (kind, index)
            self._rows[id(value)] = known
            self._pending.append((kind, index, value))
        return _tagged(ROW_TAG, record=known[0], row=known[1])

    def _drain(self) -> None:
        """Encode every registered record row (rows may register more rows)."""

        while self._pending:
            kind, index, value = self._pending.pop(0)
            self.tables[kind][index] = self._encode_fields(
                dict(state_items(value)), f"{kind}[{index}]"
            )

    def _encode_fields(self, state: Mapping[str, Any], at: str) -> dict[str, Any]:
        """Encode one record's field mapping (keys are field names)."""

        return {str(name): self.encode(value, f"{at}.{name}", 0) for name, value in state.items()}

    def encode(self, value: Any, at: str, depth: int) -> Any:
        """Encode one value to JSON-ready data.

        Parameters
        ----------
        value:
            Value to encode.
        at:
            Human-readable position, used by disclosures and refusals.
        depth:
            Current inline nesting depth.

        Returns
        -------
        Any
            JSON-ready data.
        """

        if value is None or type(value) in (bool, str):
            return value
        if depth > _MAX_INLINE_DEPTH:
            return self._unencodable(value, at, reason="nesting deeper than the format allows")
        handler = _EXACT_HANDLERS.get(type(value))
        if handler is not None:
            return handler(self, value, at, depth)
        return self._encode_other(value, at, depth)

    def _encode_other(self, value: Any, at: str, depth: int) -> Any:
        """Encode a value whose exact type has no direct handler."""

        if isinstance(value, BlobRef):
            return self._tensor_reference(value, at)
        if isinstance(value, torch.Tensor):
            return _tagged("tensor", **self._inline_tensor(value))
        kind = record_kind_of(value)
        if kind is not None:
            return self.row_reference(value, kind)
        if isinstance(value, enum.Enum):
            return _tagged(
                "enum",
                **{"class": class_path(type(value))},
                value=self.encode(value.value, f"{at}.value", depth + 1),
            )
        if isinstance(value, type):
            return _tagged("type", value=class_path(value))
        if isinstance(value, (dict, OrderedDict, defaultdict)) and _is_torchlens_or_builtin(value):
            return self._encode_mapping(value, at, depth)
        if _is_torchlens_value_object(value):
            return self._encode_object(value, at, depth)
        if callable(value):
            return self._callable_placeholder(value, at)
        return self._unencodable(value, at, reason="no typed encoding")

    def _enter(self, value: Any, at: str) -> None:
        """Refuse a container that contains itself (JSON has no cycles)."""

        if id(value) in self._active:
            raise ArtifactError(
                f"The 3.0 writer met a self-referencing value at {at}; the saved format "
                "stores record objects as rows but cannot store an inline cycle.",
                code="metadata_value_unportable",
                remedy="break the cycle in the value before saving",
            )
        self._active.add(id(value))

    def _encode_items(self, items: Any, at: str, depth: int) -> list[Any]:
        """Encode a sequence of items, guarding against cycles."""

        return [self.encode(item, f"{at}[{i}]", depth + 1) for i, item in enumerate(items)]

    def _encode_list(self, value: list[Any], at: str, depth: int) -> list[Any]:
        """Encode a list as a JSON array."""

        self._enter(value, at)
        try:
            return self._encode_items(value, at, depth)
        finally:
            self._active.discard(id(value))

    def _encode_tuple(self, value: tuple[Any, ...], at: str, depth: int) -> dict[str, Any]:
        """Encode a plain tuple as a ``tuple`` typed object."""

        return _tagged("tuple", items=self._encode_items(value, at, depth))

    def _encode_set(self, value: set[Any] | frozenset[Any], at: str, depth: int) -> dict[str, Any]:
        """Encode a set or frozenset with members in canonical order."""

        items = self._encode_items(value, at, depth)
        items.sort(key=_canonical_sort_key)
        tag = "frozenset" if isinstance(value, frozenset) else "set"
        return _tagged(tag, items=items)

    def _encode_mapping(self, value: Mapping[Any, Any], at: str, depth: int) -> Any:
        """Encode a mapping: a JSON object when it can be, else a ``dict`` object."""

        self._enter(value, at)
        try:
            plain = type(value) is dict and all(
                type(key) is str and key != TYPED_VALUE_KEY for key in value
            )
            if plain:
                return {
                    key: self.encode(item, f"{at}[{key!r}]", depth + 1)
                    for key, item in value.items()
                }
            fields: dict[str, Any] = {
                "items": [
                    [
                        self.encode(key, f"{at}<key>", depth + 1),
                        self.encode(item, f"{at}[{key!r}]", depth + 1),
                    ]
                    for key, item in value.items()
                ]
            }
            fields.update(_mapping_class_fields(value))
            return _tagged("dict", **fields)
        finally:
            self._active.discard(id(value))

    def _encode_object(self, value: Any, at: str, depth: int) -> dict[str, Any]:
        """Encode a TorchLens value object as its class path plus state."""

        self._enter(value, at)
        try:
            fields: dict[str, Any] = {"class": class_path(type(value))}
            scalar = _scalar_base_value(value)
            if scalar is not None:
                fields["value"] = self.encode(scalar, f"{at}.value", depth + 1)
            state = _object_state(value)
            if state or scalar is None:
                fields["state"] = {
                    name: self.encode(item, f"{at}.{name}", depth + 1)
                    for name, item in state.items()
                }
            return _tagged(OBJECT_TAG, **fields)
        finally:
            self._active.discard(id(value))

    def _tensor_reference(self, value: BlobRef, at: str) -> dict[str, Any]:
        """Resolve a scrubbed blob reference against the written blob entries."""

        fields = self._tensor_refs.get(value.blob_id)
        if fields is None:
            raise ArtifactError(
                f"The scrubbed state references blob {value.blob_id!r} at {at}, but the "
                "save wrote no such blob.",
                code="bundle_save_failed",
                remedy="re-save; the blob selection and the metadata disagree",
            )
        return _tagged("tensor", **fields)

    def _callable_placeholder(self, value: Any, at: str) -> dict[str, Any]:
        """Record a callable as a placeholder; the reader never imports it."""

        text = _bounded_repr(value)
        qualname = getattr(value, "__qualname__", None) or type(value).__qualname__
        module = getattr(value, "__module__", None) or ""
        return _tagged(
            "callable_placeholder",
            position=at,
            qualname=f"{module}:{qualname}" if module else str(qualname),
            repr_length=len(text),
        )

    def _unencodable(self, value: Any, at: str, *, reason: str) -> dict[str, Any]:
        """Stringify a value with a disclosure record (never pickled)."""

        text = _bounded_repr(value)
        disclosure = {
            "type": class_path(type(value)),
            "repr": text[:UNENCODABLE_REPR_LIMIT],
            "repr_length": len(text),
            "at": at,
        }
        self.unencodable.append({**disclosure, "reason": reason})
        return _tagged("unencodable", **disclosure)


def _encode_int(encoder: StateEncoder, value: int, at: str, depth: int) -> Any:
    """Encode a plain int; outside the exact JSON range it is a tagged string."""

    if -_JSON_SAFE_INT <= value <= _JSON_SAFE_INT:
        return value
    return _tagged("int", value=str(value))


def _encode_float(encoder: StateEncoder, value: float, at: str, depth: int) -> Any:
    """Encode a plain float; non-finite values are tagged strings."""

    if math.isfinite(value):
        return value
    return _tagged("float", value=repr(value))


def _encode_bytes(encoder: StateEncoder, value: bytes, at: str, depth: int) -> Any:
    """Encode bytes as base64 text."""

    return _tagged("bytes", value=base64.b64encode(bytes(value)).decode("ascii"))


def _encode_complex(encoder: StateEncoder, value: complex, at: str, depth: int) -> Any:
    """Encode a complex number by its parts."""

    return _tagged(
        "complex",
        real=_encode_float(encoder, value.real, at, depth),
        imag=_encode_float(encoder, value.imag, at, depth),
    )


def _encode_dtype(encoder: StateEncoder, value: torch.dtype, at: str, depth: int) -> Any:
    """Encode a torch dtype by its name (``float32``)."""

    return _tagged("dtype", value=str(value).replace("torch.", ""))


def _encode_device(encoder: StateEncoder, value: torch.device, at: str, depth: int) -> Any:
    """Encode a torch device by its string form."""

    return _tagged("device", value=str(value))


_EXACT_HANDLERS: dict[type, Callable[[StateEncoder, Any, str, int], Any]] = {
    int: _encode_int,
    float: _encode_float,
    bytes: _encode_bytes,
    bytearray: _encode_bytes,
    complex: _encode_complex,
    torch.dtype: _encode_dtype,
    torch.device: _encode_device,
    list: StateEncoder._encode_list,
    tuple: StateEncoder._encode_tuple,
    set: StateEncoder._encode_set,
    frozenset: StateEncoder._encode_set,
    dict: StateEncoder._encode_mapping,
    OrderedDict: StateEncoder._encode_mapping,
    defaultdict: StateEncoder._encode_mapping,
}


def _mapping_class_fields(value: Mapping[Any, Any]) -> dict[str, Any]:
    """Return the ``class`` (and ``factory``) fields of a non-plain mapping."""

    if isinstance(value, defaultdict):
        factory = value.default_factory
        return {
            "class": "collections.defaultdict",
            "factory": None if factory is None else class_path(factory),
        }
    if isinstance(value, OrderedDict):
        return {"class": "collections.OrderedDict"}
    return {}


def _is_torchlens_or_builtin(value: Any) -> bool:
    """Return whether a mapping's type is a builtin or collections mapping."""

    return type(value).__module__ in ("builtins", "collections")


def _is_torchlens_value_object(value: Any) -> bool:
    """Return whether a value is an instance of a TorchLens-defined class."""

    module = type(value).__module__
    return module == "torchlens" or module.startswith("torchlens.")


def _scalar_base_value(value: Any) -> int | float | str | None:
    """Return the builtin scalar a scalar subclass wraps (quantities)."""

    for base in (bool, int, float, str):
        if isinstance(value, base):
            return base(value)
    return None


def _object_state(value: Any) -> dict[str, Any]:
    """Return the field state of a TorchLens value object."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    fields = getattr(type(value), "_fields", None)
    if isinstance(value, tuple) and isinstance(fields, tuple):
        return dict(zip(fields, value, strict=True))
    if isinstance(value, (int, float, str)):
        return dict(getattr(value, "__dict__", {}) or {})
    return dict(state_items(value))


def _bounded_repr(value: Any) -> str:
    """Return ``repr(value)``, or a fallback when ``repr`` itself fails."""

    try:
        return repr(value)
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as exc:
        return f"<{class_path(type(value))} repr failed: {type(exc).__name__}>"


def _canonical_sort_key(item: Any) -> str:
    """Return a stable sort key for an encoded set member."""

    return json.dumps(item, sort_keys=True, default=str)
