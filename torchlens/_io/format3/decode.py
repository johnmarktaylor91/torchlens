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
from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any, Final

import torch

from ..._data_classes._state_adapter import state_new, state_restore
from ..field_registry import component_spec_tables
from ._families import (
    CELL_CODE,
    PendingRecord,
    _apply_family,
    _FamilyContext,
    cell_error,
)
from .contract import (
    RECORD_INDEX,
    TENSOR_REFERENCE_FIELDS,
    TYPED_VALUE_KEY,
    TYPED_VALUE_TAGS,
    UNENCODABLE_REPR_LIMIT,
)
from .values import (
    _DEFAULT_FACTORIES,
    BYTES_LIMIT,
    TAG_FIELDS,
    RowRef,
    _scalar_base as scalar_base,
    class_path,
    dtype_table,
    enum_table,
    record_type_table,
    tuple_subtypes,
)

# Decoding: the closed vocabulary back to Python values
# ---------------------------------------------------------------------------

#: The one record-cell refusal code: a cell that fails its typed schema.
#: Longest decimal text an ``int`` cell may carry (CPython's own int-parse limit).
INT_TEXT_LIMIT: Final = 4300
#: Deepest nesting the decoder follows inside one cell.
MAX_CELL_DEPTH: Final = 64
#: Device strings the format accepts: a lowercase type with an optional index.
DEVICE_TEXT: Final = re.compile(r"[a-z_]+(:\d{1,3})?")
#: Dtypes a safetensors chunk can hold (a closed vocabulary; quantized, bit-packed and
#: sub-byte dtypes are in torch's table but have no storable element). Complex tensors are
#: written as their real view with the ``complex`` flag.
STORABLE_DTYPES: Final[frozenset[str]] = frozenset(
    {
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "complex32",
        "complex64",
        "complex128",
    }
)
#: The most elements one tensor reference may declare (2**48: a petabyte of float32).
#: A shape bomb declares more and refuses before any allocation.
MAX_TENSOR_NUMEL: Final = 1 << 48
_RECORD_KINDS: Final = frozenset(name for name, _, _ in RECORD_INDEX)
_INT_TEXT = re.compile(r"-?[0-9]+")
_DOTTED_NAME = re.compile(r"[A-Za-z_][\w.]{0,255}")
_PLACEHOLDER_POSITION = re.compile(r"\d+(\.[\w\[\]\.]+)?")
#: Fields a tag may carry beyond its required ones (the tensor cell's payload fields).
_OPTIONAL_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "tensor": TENSOR_REFERENCE_FIELDS,
    "tuple": ("subtype",),
    "dict": ("subtype", "default_factory"),
}
#: The one mapping subtype the ``dict`` tag may declare.
_DEFAULTDICT_SUBTYPE: Final = "collections.defaultdict"
#: Builtin factories a defaultdict cell may name; the artifact string only selects here.
_FACTORY_TABLE: Final[dict[str, type]] = {cls.__name__: cls for cls in _DEFAULT_FACTORIES}


@dataclasses.dataclass(frozen=True)
class UnencodableValue:
    """A value the writer could not encode: its type name, repr text and where it was.

    ``_repr_length`` (the unbounded repr's length) is carried for round trips, not public.
    """

    type_name: str
    repr_text: str
    at: str
    _repr_length: int = dataclasses.field(default=0, repr=False, compare=False)


@dataclasses.dataclass(frozen=True)
class CallablePlaceholder:
    """A callable the recipe could not persist: rule position and qualname (section 8)."""

    position: int | str
    qualname: str
    repr_length: int


def _is_int(value: Any) -> bool:
    """Return whether a value is an integer with boolean values excluded.

    Parameters
    ----------
    value:
        Candidate scalar.
    """

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
        self._family_context: Any = None

    def decode(self, data: Any, *, at: str, depth: int = 0) -> Any:
        """Decode one cell and apply its bound grammar before building records.

        Parameters
        ----------
        data:
            Encoded cell value.
        at:
            Record, row and column position used in refusals.
        depth:
            Current typed-value nesting depth.
        """

        value = self._decode(data, at=at, depth=depth)
        if depth == 0:
            if self._family_context is None:
                self._family_context = _FamilyContext()
            self._family_context.check(value, at)
        return value

    def _decode(self, data: Any, *, at: str, depth: int) -> Any:
        """Decode one value through the closed tag vocabulary.

        Parameters
        ----------
        data:
            Encoded JSON value.
        at:
            Diagnostic path.
        depth:
            Current nesting depth.
        """

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
        """Decode container items and check the set-member hashability contract.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        tag:
            Tuple, set or frozenset container tag.
        """

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
        """Decode a tuple from its ordered item cells.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        items = tuple(self._items(data, at, depth, "tuple"))
        subtype = data.get("subtype")
        if subtype is None:
            return items
        # The wire string only selects from the code-owned constructor table; nothing is
        # imported or looked up by name from the artifact.
        cls = tuple_subtypes().get(subtype) if isinstance(subtype, str) else None
        if cls is None:
            raise cell_error(
                at, f"tuple subtype {subtype!r} is not a known tuple type", code=CELL_CODE
            )
        try:
            return cls(items)
        except (TypeError, ValueError):
            raise cell_error(
                at, f"tuple subtype {subtype!r} does not accept these items", code=CELL_CODE
            ) from None

    def _set(self, data: Mapping[str, Any], at: str, depth: int) -> set[Any]:
        """Decode a set after checking member hashability.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        return set(self._items(data, at, depth, "set"))

    def _frozenset(self, data: Mapping[str, Any], at: str, depth: int) -> frozenset[Any]:
        """Decode an immutable set after checking member hashability.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        return frozenset(self._items(data, at, depth, "frozenset"))

    def _dict(self, data: Mapping[str, Any], at: str, depth: int) -> dict[Any, Any]:
        """Decode mapping pairs with unique, hashable keys.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
        return self._mapping_subtype(data, result, at, depth)

    def _mapping_subtype(
        self, data: Mapping[str, Any], result: dict[Any, Any], at: str, depth: int
    ) -> dict[Any, Any]:
        """Apply a declared ``dict`` subtype: a defaultdict whose factory is a builtin name
        rebuilds with that factory; a disclosed foreign factory reloads as a plain dict (the
        disclosure is counted by the manifest's load advisory). No factory ever executes here.
        """

        subtype = data.get("subtype")
        if subtype is None:
            if "default_factory" in data:
                raise cell_error(at, "default_factory needs a dict subtype", code=CELL_CODE)
            return result
        if subtype != _DEFAULTDICT_SUBTYPE:
            raise cell_error(at, f"dict subtype {subtype!r} is not a known mapping", code=CELL_CODE)
        if "default_factory" not in data:
            raise cell_error(at, "a defaultdict cell names its default_factory", code=CELL_CODE)
        factory = data["default_factory"]
        if isinstance(factory, str):
            cls = _FACTORY_TABLE.get(factory)
            if cls is None:
                raise cell_error(at, f"default_factory {factory!r} is not builtin", code=CELL_CODE)
            return defaultdict(cls, result)
        disclosed = self.decode(factory, at=f"{at}.default_factory", depth=depth)
        if not isinstance(disclosed, UnencodableValue):
            raise cell_error(
                at, "default_factory must be a builtin name or a disclosure", code=CELL_CODE
            )
        return result

    # -- scalars ------------------------------------------------------------

    def _bytes(self, data: Mapping[str, Any], at: str, depth: int) -> bytes:
        """Decode bounded base64 text and verify its declared byte count.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
        """Decode a tagged non-finite floating-point value.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        text = data["v"]
        if text not in ("nan", "inf", "-inf"):
            raise cell_error(at, f"float v {text!r} is not nan, inf or -inf", code=CELL_CODE)
        return float(text)

    def _int(self, data: Mapping[str, Any], at: str, depth: int) -> int:
        """Decode a bounded decimal integer representation.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        text = data["v"]
        if not isinstance(text, str) or len(text) > INT_TEXT_LIMIT or not _INT_TEXT.fullmatch(text):
            raise cell_error(at, "int v must be a bounded decimal string", code=CELL_CODE)
        return int(text)

    def _complex(self, data: Mapping[str, Any], at: str, depth: int) -> complex:
        """Decode the numeric real and imaginary parts of a complex value.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        parts: list[float] = []
        for name in ("re", "im"):
            part = self.decode(data[name], at=f"{at}.{name}", depth=depth)
            if isinstance(part, bool) or not isinstance(part, (int, float)):
                raise cell_error(at, f"complex {name} must be a number", code=CELL_CODE)
            parts.append(float(part))
        return complex(parts[0], parts[1])

    def _dtype(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        """Resolve a dtype through the code-owned dtype table.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        name = data["name"]
        dtype = dtype_table().get(name) if isinstance(name, str) else None
        if dtype is None:
            raise cell_error(at, f"dtype name {name!r} is outside the dtype table", code=CELL_CODE)
        return dtype

    def _device(self, data: Mapping[str, Any], at: str, depth: int) -> torch.device:
        """Build a device from the bounded device-string grammar.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
        """Return a dotted type identity without importing it.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        name = data["name"]
        if not isinstance(name, str) or not _DOTTED_NAME.fullmatch(name):
            raise cell_error(at, "type name must be a dotted identifier", code=CELL_CODE)
        return name

    def _enum(self, data: Mapping[str, Any], at: str, depth: int) -> enum.Enum:
        """Resolve a member of a code-owned enum class.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
        """Decode the fields of an allowlisted value-object class.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
        """Decode a nonnegative row reference to a known record table.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        record, row = data["record"], data["row"]
        if record not in _RECORD_KINDS or not _is_int(row) or row < 0:
            raise cell_error(
                at, f"ref must name a record kind and a row index, not {data!r}", code=CELL_CODE
            )
        return RowRef(record, row)

    def _tensor(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        """Delegate a tensor reference to the record-table payload sink.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        if self.tensor_ref is None:
            raise cell_error(at, "a tensor reference outside a record table", code=CELL_CODE)
        return self.tensor_ref(data, at)

    @staticmethod
    def _unencodable(data: Mapping[str, Any], at: str, depth: int) -> UnencodableValue:
        """Decode an unencodable-value disclosure with bounded repr text.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

        typed = isinstance(data["type"], str) and isinstance(data["repr"], str)
        counted = _is_int(data["repr_length"]) and data["repr_length"] >= 0
        if not (typed and counted and isinstance(data["at"], str)):
            raise cell_error(at, "unencodable disclosure fields are mistyped", code=CELL_CODE)
        if len(data["repr"]) > UNENCODABLE_REPR_LIMIT:
            raise cell_error(at, "unencodable repr is over the repr limit", code=CELL_CODE)
        return UnencodableValue(data["type"], data["repr"], data["at"], data["repr_length"])

    @staticmethod
    def _callable_placeholder(data: Mapping[str, Any], at: str, depth: int) -> CallablePlaceholder:
        """Decode a callable disclosure with a valid recipe position.

        Parameters
        ----------
        data:
            Encoded typed-value fields.
        at:
            Record-cell diagnostic path.
        depth:
            Current typed-value nesting depth.
        """

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
    obj: Any = state_new(cls)
    setstate = getattr(cls, "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, state)
        return obj
    return state_restore(obj, state)


class TensorCellFields(dict[str, Any]):
    """The reference fields of a ``tensor`` cell :func:`decode_value` meets, hashed by identity
    like the tensor it stands for, so a set or frozenset member decodes without a payload."""

    __slots__ = ()
    __hash__ = object.__hash__  # type: ignore[assignment]


def decode_value(data: Any, *, schema: Any = None, at: str = "value") -> Any:
    """Decode one tagged JSON value through the closed vocabulary (the encoder's inverse).

    ``tensor`` cells come back as their reference fields and ``ref`` cells as
    :class:`RowRef`; the table reader binds payloads and settles rows itself.
    ``schema`` optionally applies one family node after decoding.
    """

    value = Decoder(lambda fields, _at: TensorCellFields(fields)).decode(data, at=at)
    if schema is not None:
        _apply_family(schema, value, at)
    return value


@cache
def _element_size(dtype_name: str) -> int:
    """Return the storage bytes of one element of a storable dtype."""

    return torch.empty((), dtype=dtype_table()[dtype_name]).element_size()
