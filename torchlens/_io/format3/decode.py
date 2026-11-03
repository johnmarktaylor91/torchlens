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

import ast
import base64
import binascii
import dataclasses
import enum
import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import cache
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
from .schema import FAMILIES
from .values import (
    BYTES_LIMIT,
    TAG_FIELDS,
    RowRef,
    _scalar_base as scalar_base,
    class_path,
    dtype_table,
    enum_table,
    record_type_table,
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
_CELL_POSITION = re.compile(r"(\w+)\[\d+\]\.(\w+)")
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
        position = _CELL_POSITION.fullmatch(at) if depth == 0 else None
        if position is not None:
            for node in _bound_families(*position.groups()):
                selected, selected_at = value, at
                for key in node.get("path", ()):
                    if not isinstance(selected, Mapping) or key not in selected:
                        break
                    selected, selected_at = selected[key], f"{selected_at}.{key}"
                else:
                    _apply_family(node, selected, selected_at)
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
    obj: Any = state_new(cls)
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


@cache
def _element_size(dtype_name: str) -> int:
    """Return the storage bytes of one element of a storable dtype."""

    return torch.empty((), dtype=dtype_table()[dtype_name]).element_size()


@cache
def _bound_families(record: str, column: str) -> tuple[dict[str, Any], ...]:
    """Return the grammar nodes bound to one record column.

    Parameters
    ----------
    record, column:
        Record kind and stored column name.
    """

    return tuple(
        node
        for node in FAMILIES.values()
        if node.get("record") == record and node.get("column") == column
    )


def _apply_family(node: Mapping[str, Any], value: Any, at: str) -> None:
    """Check one decoded value against a declarative grammar node.

    Parameters
    ----------
    node:
        Schema data with an inherited refusal code and optional family reference.
    value:
        Decoded value; record-tagged values can still be pending construction.
    at:
        Complete record, row, column and nested item path.
    """

    if "family" in node:
        node = {**FAMILIES[node["family"]], **node}
    code = node.get("code", CELL_CODE)
    if value is None and node.get("nullable"):
        return
    if "vocab" in node and value not in _family_vocab(node["vocab"]):
        raise cell_error(at, f"value {value!r} is outside the family vocabulary", code=code)
    if "any_of" in node:
        for option in node["any_of"]:
            try:
                _apply_family({"code": code, **option}, value, at)
            except ArtifactError:
                continue
            return
        raise cell_error(at, "value matches none of the family schemas", code=code)
    kind = node.get("kind", "scalar")
    if kind == "object":
        _family_object(node, value, at, code)
    elif kind in ("list", "tuple"):
        _family_sequence(node, value, at, code)
    else:
        _family_scalar(node, value, at, code)


@cache
def _family_vocab(source: Any) -> tuple[Any, ...]:
    """Resolve a closed vocabulary from data or its code-owned source.

    Parameters
    ----------
    source:
        Literal vocabulary or the PIPELINE reference declared by the contract.
    """

    if source == "PIPELINE":
        from importlib import import_module

        from .contract import PHASE_TIMING_KEY_SOURCE

        module, _, attribute = PHASE_TIMING_KEY_SOURCE.rpartition(".")
        return tuple(step.name for step in getattr(import_module(module), attribute))
    return tuple(source)


def _family_scalar(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Apply scalar kinds, bounds and literal syntax from a node.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate scalar, diagnostic path and inherited refusal code.
    """

    types = {"str": str, "int": int, "bool": bool, "number": (int, float)}
    declared = node.get("type", ())
    declared = (declared,) if isinstance(declared, str) else declared
    typed = not declared or any(
        isinstance(value, types[name])
        and not (name in ("int", "number") and isinstance(value, bool))
        for name in declared
    )
    if not typed:
        raise cell_error(
            at, f"expected scalar kind {declared}, got {type(value).__name__}", code=code
        )
    for bound, compare in (
        ("minimum", lambda a, b: a >= b),
        ("min_length", lambda a, b: len(a) >= b),
        ("max_length", lambda a, b: len(a) <= b),
    ):
        if bound in node and not compare(value, node[bound]):
            raise cell_error(at, f"value violates {bound}={node[bound]}", code=code)
    if node.get("finite"):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise cell_error(at, "value must be finite", code=code)
    if "pattern" in node and re.match(node["pattern"], value) is None:
        raise cell_error(at, "string does not match the family's syntax", code=code)
    if node.get("literal_tuple"):
        try:
            parsed = ast.literal_eval(value)
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            raise cell_error(at, "string is not a literal tuple", code=code) from None
        if not isinstance(parsed, tuple):
            raise cell_error(at, "literal must be a tuple", code=code)


def _family_object(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Check mapping fields, discriminated variants and within-cell rules.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate object, diagnostic path and inherited refusal code.
    """

    keys = node.get("keys", {})
    if node.get("attributes") and not isinstance(value, Mapping):
        if isinstance(value, PendingRecord):
            value = value.fields
        else:
            value = {key: getattr(value, key) for key in keys if hasattr(value, key)}
    if not isinstance(value, Mapping):
        raise cell_error(at, "family value must be a mapping", code=code)
    required = set(keys) - set(node.get("optional", ()))
    if not required <= set(value) or (
        "keys" in node and not node.get("open") and set(value) - set(keys)
    ):
        raise cell_error(at, f"family fields differ from {sorted(keys)}", code=code)
    for key, item in value.items():
        if "key" in node:
            _apply_family({"code": code, **node["key"]}, key, f"{at}<key {key!r}>")
        child = keys.get(key, node.get("values"))
        if child is not None:
            _apply_family({"code": code, **child}, item, f"{at}.{key}")
    if "discriminator" in node:
        variant = value.get(node["discriminator"])
        selected = next((case for name, case in node["cases"].items() if variant == name), None)
        if selected is None:
            raise cell_error(at, f"unknown family variant {variant!r}", code=code)
        _apply_family({"code": code, **selected}, value, at)
    for rule in node.get("rules", ()):
        if all(_family_condition(value.get(key), test) for key, test in rule["when"].items()):
            _apply_family(
                {"kind": "object", "open": True, "code": code, "keys": rule["keys"]}, value, at
            )
    if "ordinal_keys" in node:
        table, count = node["ordinal_keys"]
        # Cardinality plus bounded integer keys avoids allocating an attacker-sized range.
        members, total = value[table], value[count]
        if len(members) != total or any(
            not isinstance(k, int) or not 1 <= k <= total for k in members
        ):
            raise cell_error(at, "mapping keys must be contiguous ordinals 1..count", code=code)
    if "sum" in node:
        total = value[node["sum"]["total"]]
        if total is not None:
            parts = [
                entry[index] or 0 for field, index in node["sum"]["terms"] for entry in value[field]
            ]
            if sum(parts) != total:
                raise cell_error(at, "item counts do not sum to the declared total", code=code)


def _family_condition(value: Any, condition: str) -> bool:
    """Evaluate a data-only presence or truth condition.

    Parameters
    ----------
    value:
        Field value in the current mapping.
    condition:
        Truthiness or presence predicate named by schema data.
    """

    if condition == "present":
        return value is not None
    return bool(value) if condition == "truthy" else not bool(value)


def _family_sequence(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Apply sequence items, positional fields and uniqueness constraints.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate sequence, diagnostic path and inherited refusal code.
    """

    allowed = isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    if node.get("kind") == "tuple":
        allowed = isinstance(value, (list, tuple))
    if not allowed:
        raise cell_error(at, "family value must be a sequence", code=code)
    if "length" in node and len(value) != node["length"]:
        raise cell_error(at, f"sequence must have length {node['length']}", code=code)
    for index, item in enumerate(value):
        child = node.get("positions", {}).get(index, node.get("items"))
        if child is not None:
            _apply_family({"code": code, **child}, item, f"{at}[{index}]")
    if node.get("sorted_unique") and list(value) != sorted(set(value)):
        raise cell_error(at, "sequence must be sorted and unique", code=code)
    if "unique_by" in node:
        keys = [tuple(item[key] for key in node["unique_by"]) for item in value]
        if len(set(keys)) != len(keys):
            raise cell_error(at, "sequence repeats an identity", code=code)
