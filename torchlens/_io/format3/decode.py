"""Typed JSON cells back to Python values: the decoder half of the 3.0 value codec.

The encoder and the tag table live in ``values.py`` (writer lane); this module
is the reader lane's strict inverse. Every cell is checked against the closed
vocabulary before anything is built: a tag outside ``TYPED_VALUE_TAGS``, a
field outside the tag's ``TAG_FIELDS``, a dotted name outside the code-built
tables, an over-long string or an unhashable key refuses typed with the cell
path, and the decoder imports nothing by a name it reads. Design:
``docs/maintainers/saved_format_3_design.md``, sections 4, 5 and 8.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import enum
import math
import re
from collections.abc import Callable, Mapping
from typing import Any, Final

import torch

from ..._data_classes._state_adapter import state_new, state_restore
from ..field_registry import component_spec_tables
from ..format_errors import ArtifactError
from .contract import (
    PROVISIONAL_REFUSALS,
    RECORD_INDEX,
    TENSOR_REFERENCE_FIELDS,
    TYPED_VALUE_KEY,
    TYPED_VALUE_TAGS,
    UNENCODABLE_REPR_LIMIT,
)
from .values import (
    BYTES_LIMIT,
    TAG_FIELDS,
    RowRef,
    class_path,
    dtype_table,
    enum_table,
    record_type_table,
    scalar_base,
)

# Decoding: the closed vocabulary back to Python values
# ---------------------------------------------------------------------------

#: The one record-cell refusal code (provisional spelling, contract PROVISIONAL_REFUSALS).
CELL_CODE: Final = PROVISIONAL_REFUSALS["record_schema_invalid"]
#: Longest decimal text an ``int`` cell may carry (CPython's own int-parse limit).
INT_TEXT_LIMIT: Final = 4300
#: Deepest nesting the decoder follows inside one cell.
MAX_CELL_DEPTH: Final = 64
#: Device strings the format accepts: a lowercase type with an optional index.
DEVICE_TEXT: Final = re.compile(r"[a-z_]+(:\d{1,3})?")
_RECORD_KINDS: Final = frozenset(name for name, _, _ in RECORD_INDEX)
_INT_TEXT = re.compile(r"-?[0-9]+")
_DOTTED_NAME = re.compile(r"[A-Za-z_][\w.]{0,255}")
_PLACEHOLDER_POSITION = re.compile(r"\d+(\.[\w\[\]\.]+)?")
#: Fields a tag may carry beyond its required ones (the tensor cell's payload fields).
_OPTIONAL_FIELDS: Final[dict[str, tuple[str, ...]]] = {"tensor": TENSOR_REFERENCE_FIELDS}


@dataclasses.dataclass(frozen=True)
class Unencodable:
    """A value the writer could not encode: its type, bounded repr and location (section 5)."""

    type: str
    repr: str
    repr_length: int
    at: str


@dataclasses.dataclass(frozen=True)
class CallablePlaceholder:
    """A callable the recipe could not persist: rule position and qualname (section 8)."""

    position: int | str
    qualname: str
    repr_length: int


@dataclasses.dataclass
class PendingRecord:
    """A ``record`` object whose decoded fields still hold ``ref`` cells for the reader to settle."""

    cls: type
    fields: dict[str, Any]
    at: str


def cell_error(at: str, detail: str, *, code: str = CELL_CODE) -> ArtifactError:
    """The one record-cell refusal: the path, the detail, the re-save remedy."""

    return ArtifactError(
        f"Record cell {at} does not satisfy the 3.0 saved-format contract: {detail}.",
        code=code,
        remedy="re-save the trace with trace.save(); the artifact was edited or corrupted",
        at=at,
        detail=detail,
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_fields(data: Mapping[str, Any], tag: str, at: str) -> None:
    """Refuse a typed object missing a required field or carrying one outside its tag."""

    allowed = TAG_FIELDS[tag] + _OPTIONAL_FIELDS.get(tag, ())
    missing = [name for name in TAG_FIELDS[tag] if name not in data]
    extra = [name for name in data if name != TYPED_VALUE_KEY and name not in allowed]
    if missing or extra:
        raise cell_error(
            at,
            f"{tag} value is missing {missing} or carries unknown fields {extra}",
            code=CELL_CODE,
        )


def record_fields_allowed(cls: type) -> frozenset[str] | None:
    """Return the field names a ``record`` of ``cls`` may carry, or ``None`` when unbounded.

    Dataclasses and namedtuples declare their fields; a component with a
    ``_PORTABLE_STATE_SPEC`` declares them in the writer contract; a scalar
    quantity carries ``value`` plus any instance attributes.
    """

    if dataclasses.is_dataclass(cls):
        return frozenset(field.name for field in dataclasses.fields(cls))
    fields = getattr(cls, "_fields", None)
    if isinstance(fields, tuple):
        return frozenset(fields)
    spec = component_spec_tables().get(class_path(cls))
    return frozenset(spec) if spec is not None and scalar_base(cls) is None else None


class Decoder:
    """Decode JSON cells into Python values through the closed tag vocabulary.

    Parameters
    ----------
    tensor_ref:
        Receives each ``tensor`` cell (fields, position) and returns the value
        the cell decodes to; ``None`` refuses any tensor reference (a decode
        outside a record table).
    defer_records:
        Return :class:`PendingRecord` for ``record``-tagged objects instead of
        building them, so the table reader can settle ``ref`` cells inside
        their fields first.
    """

    def __init__(
        self,
        tensor_ref: Callable[[Mapping[str, Any], str], Any] | None = None,
        *,
        defer_records: bool = False,
    ) -> None:
        self.tensor_ref = tensor_ref
        self.defer_records = defer_records

    def decode(self, data: Any, *, at: str, depth: int = 0) -> Any:
        if depth > MAX_CELL_DEPTH:
            raise cell_error(at, f"nesting deeper than {MAX_CELL_DEPTH}", code=CELL_CODE)
        if data is None or isinstance(data, (bool, int, str)):
            return data
        if isinstance(data, float):
            if not math.isfinite(data):
                raise cell_error(
                    at,
                    "a bare non-finite float; nan and inf travel as float values",
                    code=CELL_CODE,
                )
            return data
        if isinstance(data, list):
            return [self.decode(v, at=f"{at}[{i}]", depth=depth + 1) for i, v in enumerate(data)]
        if not isinstance(data, dict):
            raise cell_error(
                at,
                f"JSON value of type {type(data).__name__} is outside the vocabulary",
                code=CELL_CODE,
            )
        tag = data.get(TYPED_VALUE_KEY)
        if tag is None:
            return {k: self.decode(v, at=f"{at}.{k}", depth=depth + 1) for k, v in data.items()}
        if not isinstance(tag, str) or tag not in TYPED_VALUE_TAGS:
            raise cell_error(at, f"unknown typed-value tag {tag!r}", code=CELL_CODE)
        _check_fields(data, tag, at)
        handler: Callable[[Mapping[str, Any], str, int], Any] = getattr(self, f"_{tag}")
        return handler(data, at, depth + 1)

    # -- containers ---------------------------------------------------------

    def _items(self, data: Mapping[str, Any], at: str, depth: int, tag: str) -> list[Any]:
        items = data["items"]
        if not isinstance(items, list):
            raise cell_error(at, f"{tag} items must be an array", code=CELL_CODE)
        decoded = [self.decode(v, at=f"{at}[{i}]", depth=depth) for i, v in enumerate(items)]
        if tag != "tuple":
            try:
                hash(tuple(decoded))
            except TypeError:
                raise cell_error(at, f"{tag} members must be hashable", code=CELL_CODE) from None
        return decoded

    def _tuple(self, data: Mapping[str, Any], at: str, depth: int) -> tuple[Any, ...]:
        return tuple(self._items(data, at, depth, "tuple"))

    def _set(self, data: Mapping[str, Any], at: str, depth: int) -> set[Any]:
        return set(self._items(data, at, depth, "set"))

    def _frozenset(self, data: Mapping[str, Any], at: str, depth: int) -> frozenset[Any]:
        return frozenset(self._items(data, at, depth, "frozenset"))

    def _dict(self, data: Mapping[str, Any], at: str, depth: int) -> dict[Any, Any]:
        items = data["items"]
        if not isinstance(items, list):
            raise cell_error(
                at, "dict items must be an array of [key, value] pairs", code=CELL_CODE
            )
        result: dict[Any, Any] = {}
        for index, pair in enumerate(items):
            if not isinstance(pair, list) or len(pair) != 2:
                raise cell_error(
                    at, f"dict item {index} is not a [key, value] pair", code=CELL_CODE
                )
            key = self.decode(pair[0], at=f"{at}<key {index}>", depth=depth)
            try:
                hash(key)
            except TypeError:
                raise cell_error(at, f"dict key {index} is not hashable", code=CELL_CODE) from None
            if key in result:
                raise cell_error(at, f"dict key {key!r} repeats", code=CELL_CODE)
            result[key] = self.decode(pair[1], at=f"{at}[{index}]", depth=depth)
        return result

    # -- scalars ------------------------------------------------------------

    def _bytes(self, data: Mapping[str, Any], at: str, depth: int) -> bytes:
        text, length = data["base64"], data["bytes"]
        if not isinstance(text, str) or len(text) > (BYTES_LIMIT * 4) // 3 + 4:
            raise cell_error(
                at, "bytes base64 must be a string under the 1 MiB ceiling", code=CELL_CODE
            )
        try:
            raw = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            raise cell_error(at, "bytes base64 is not valid base64", code=CELL_CODE) from None
        if len(raw) > BYTES_LIMIT or not _is_int(length) or length != len(raw):
            raise cell_error(
                at, "bytes length disagrees with its base64 or exceeds 1 MiB", code=CELL_CODE
            )
        return raw

    def _float(self, data: Mapping[str, Any], at: str, depth: int) -> float:
        text = data["v"]
        if text not in ("nan", "inf", "-inf"):
            raise cell_error(at, f"float v {text!r} is not nan, inf or -inf", code=CELL_CODE)
        return float(text)

    def _int(self, data: Mapping[str, Any], at: str, depth: int) -> int:
        text = data["v"]
        if not isinstance(text, str) or len(text) > INT_TEXT_LIMIT or not _INT_TEXT.fullmatch(text):
            raise cell_error(at, "int v must be a bounded decimal string", code=CELL_CODE)
        return int(text)

    def _complex(self, data: Mapping[str, Any], at: str, depth: int) -> complex:
        parts: list[float] = []
        for name in ("re", "im"):
            part = self.decode(data[name], at=f"{at}.{name}", depth=depth)
            if isinstance(part, bool) or not isinstance(part, (int, float)):
                raise cell_error(at, f"complex {name} must be a number", code=CELL_CODE)
            parts.append(float(part))
        return complex(parts[0], parts[1])

    def _dtype(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        name = data["name"]
        dtype = dtype_table().get(name) if isinstance(name, str) else None
        if dtype is None:
            raise cell_error(at, f"dtype name {name!r} is outside the dtype table", code=CELL_CODE)
        return dtype

    def _device(self, data: Mapping[str, Any], at: str, depth: int) -> torch.device:
        name = data["name"]
        if not isinstance(name, str) or not DEVICE_TEXT.fullmatch(name):
            raise cell_error(at, f"device name {name!r} is not a device string", code=CELL_CODE)
        try:
            return torch.device(name)
        except (RuntimeError, ValueError):
            raise cell_error(
                at, f"device name {name!r} is not a torch device", code=CELL_CODE
            ) from None

    def _type(self, data: Mapping[str, Any], at: str, depth: int) -> str:
        name = data["name"]
        if not isinstance(name, str) or not _DOTTED_NAME.fullmatch(name):
            raise cell_error(at, "type name must be a dotted identifier", code=CELL_CODE)
        return name

    def _enum(self, data: Mapping[str, Any], at: str, depth: int) -> enum.Enum:
        type_name, member = data["type"], data["name"]
        cls = enum_table().get(type_name) if isinstance(type_name, str) else None
        if cls is None:
            raise cell_error(
                at, f"enum class {type_name!r} is outside the enum table", code=CELL_CODE
            )
        value = cls.__members__.get(member) if isinstance(member, str) else None
        if value is None:
            raise cell_error(at, f"{type_name} has no member named {member!r}", code=CELL_CODE)
        return value

    # -- objects ------------------------------------------------------------

    def _record(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        type_name, fields = data["type"], data["fields"]
        cls = record_type_table().get(type_name) if isinstance(type_name, str) else None
        if cls is None:
            raise cell_error(
                at,
                f"record type {type_name!r} is outside the record-type allowlist",
                code=CELL_CODE,
            )
        if not isinstance(fields, dict):
            raise cell_error(at, "record fields must be an object", code=CELL_CODE)
        allowed = record_fields_allowed(cls)
        unknown = sorted(set(fields) - allowed) if allowed is not None else []
        if unknown:
            raise cell_error(
                at, f"record {type_name} carries fields outside its spec: {unknown}", code=CELL_CODE
            )
        decoded = {k: self.decode(v, at=f"{at}.{k}", depth=depth) for k, v in fields.items()}
        if self.defer_records:
            return PendingRecord(cls, decoded, at)
        return build_component(cls, decoded, at=at)

    def _ref(self, data: Mapping[str, Any], at: str, depth: int) -> RowRef:
        record, row = data["record"], data["row"]
        if record not in _RECORD_KINDS or not _is_int(row) or row < 0:
            raise cell_error(
                at, f"ref must name a record kind and a row index, not {data!r}", code=CELL_CODE
            )
        return RowRef(record, row)

    def _tensor(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        if self.tensor_ref is None:
            raise cell_error(at, "a tensor reference outside a record table", code=CELL_CODE)
        return self.tensor_ref(data, at)

    @staticmethod
    def _unencodable(data: Mapping[str, Any], at: str, depth: int) -> Unencodable:
        typed = isinstance(data["type"], str) and isinstance(data["repr"], str)
        counted = _is_int(data["repr_length"]) and data["repr_length"] >= 0
        if not (typed and counted and isinstance(data["at"], str)):
            raise cell_error(at, "unencodable disclosure fields are mistyped", code=CELL_CODE)
        if len(data["repr"]) > UNENCODABLE_REPR_LIMIT:
            raise cell_error(at, "unencodable repr is over the repr limit", code=CELL_CODE)
        return Unencodable(data["type"], data["repr"], data["repr_length"], data["at"])

    @staticmethod
    def _callable_placeholder(data: Mapping[str, Any], at: str, depth: int) -> CallablePlaceholder:
        position = data["position"]
        indexed = _is_int(position) and position >= 0
        pathed = isinstance(position, str) and _PLACEHOLDER_POSITION.fullmatch(position)
        if not (indexed or pathed):
            raise cell_error(
                at,
                "callable_placeholder position must be a rule index or index.path",
                code=CELL_CODE,
            )
        if not isinstance(data["qualname"], str) or not _is_int(data["repr_length"]):
            raise cell_error(
                at, "callable_placeholder qualname or repr_length is mistyped", code=CELL_CODE
            )
        return CallablePlaceholder(position, data["qualname"], data["repr_length"])


def build_component(cls: type, fields: Mapping[str, Any], *, at: str = "value") -> Any:
    """Construct one value object from its decoded fields, mirroring :func:`object_state`.

    Dataclasses and namedtuples take their fields as keywords; a scalar
    quantity wraps ``value`` and keeps its instance attributes; any other class
    is allocated without ``__init__`` and restored through its ``__setstate__``
    (``Edit`` rebuilds its builtin factory there) or the state adapter.
    """

    state = dict(fields)
    try:
        if dataclasses.is_dataclass(cls) or isinstance(getattr(cls, "_fields", None), tuple):
            return cls(**state)
        if scalar_base(cls) is not None:
            instance = cls(state.pop("value"))
            instance.__dict__.update(state)
            return instance
    except (KeyError, TypeError, ValueError) as exc:
        raise cell_error(
            at, f"{class_path(cls)} rejects its fields: {exc}", code=CELL_CODE
        ) from None
    obj = state_new(cls)
    setstate = getattr(cls, "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, state)
        return obj
    return state_restore(obj, state)


def decode_value(data: Any, *, schema: Any = None, at: str = "value") -> Any:
    """Decode one tagged JSON value through the closed vocabulary (the encoder's inverse).

    ``tensor`` cells come back as their reference fields and ``ref`` cells as
    :class:`RowRef`; the table reader binds payloads and settles rows itself.
    ``schema`` is reserved for the schema-directed families of design section 9.
    """

    del schema
    return Decoder(lambda fields, _at: dict(fields)).decode(data, at=at)
