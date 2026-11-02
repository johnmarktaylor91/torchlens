"""Typed values of the 3.0 saved format: the closed tag vocabulary, both directions.

Both directions live in one module so the closed tag vocabulary
(``contract.TYPED_VALUE_TAGS``) stays symmetric. Decoding never evaluates,
unpickles or imports by a name from the file: dtypes, enums and value-object
classes resolve through tables built at import time, and a name outside a
table refuses ``record_schema_invalid`` (provisional spelling
``manifest_schema_invalid``) naming the record, row and column path. A value
outside the vocabulary is written as an ``unencodable`` disclosure. Design:
``docs/maintainers/saved_format_3_design.md``, sections 4 and 5. The encode
half is the writer stream's; the decode half is the reader stream's.
"""

from __future__ import annotations

import base64
import binascii
import enum
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Final, NoReturn

import torch

from ..field_registry import COMPONENT_SPEC_MODULES, RECORD_CONTRACT_CLASSES, component_spec_tables
from ..format_contract import BlobRef
from ..format_errors import ArtifactError
from . import contract
from .contract import refuse_writer_not_built

__all__ = [
    "CELL_CODE",
    "CallablePlaceholder",
    "Decoder",
    "PendingRecord",
    "TensorRefSink",
    "Unencodable",
    "build_component",
    "cell_error",
    "decode_value",
    "encode_value",
]

_LANE_MODULE = "torchlens/_io/format3/values.py"

#: The record-cell refusal code; provisional spelling until ``record_schema_invalid`` is ruled.
CELL_CODE: Final = contract.PROVISIONAL_REFUSALS["record_schema_invalid"]
#: A device string is a backend name with an optional small ordinal; nothing else reaches torch.
DEVICE_TEXT: Final = re.compile(r"^[a-z_]+(:\d{1,3})?$")
#: Ceiling on a ``bytes`` typed value after base64 decoding.
BYTES_VALUE_LIMIT: Final = 1 << 20
#: Ceiling on the decimal text of an ``int`` typed value (python's own int-string limit).
INT_TEXT_LIMIT: Final = 4300
#: Ceiling on the nesting of one decoded cell.
MAX_CELL_DEPTH: Final = 64
_BLOB_PATH = re.compile(r"^blobs/[A-Za-z0-9_.-]+\.safetensors$")
_BLOB_KEY = re.compile(r"^t\d{1,12}$")
_RECORD_CLASS_NAMES: Final = frozenset(name for _, name in RECORD_CONTRACT_CLASSES.values())


def encode_value(value: Any, *, at: str) -> NoReturn:
    """Encode one cell value to JSON-ready data (the writer stream fills this).

    Parameters
    ----------
    value:
        The cell value.
    at:
        The ``record[row].column`` path, stamped into any disclosure.
    """

    refuse_writer_not_built("encode_value", _LANE_MODULE)


@dataclass(frozen=True)
class Unencodable:
    """A value the writer could not encode: its type, bounded repr and location (section 5)."""

    type: str
    repr: str
    repr_length: int
    at: str


@dataclass(frozen=True)
class CallablePlaceholder:
    """A callable the recipe could not persist: rule position and qualname (section 8)."""

    position: int | str
    qualname: str
    repr_length: int


@dataclass
class PendingRecord:
    """A ``record``-tagged value object whose fields still hold cross-record references.

    The table reader resolves the references (identity strings and row indices
    into the record tables) and then builds the object through
    :func:`build_component`; a decode outside a table builds immediately.
    """

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


# ---------------------------------------------------------------------------
# Import-time tables
# ---------------------------------------------------------------------------


def _torch_dtype_table() -> dict[str, torch.dtype]:
    table: dict[str, torch.dtype] = {}
    for name in dir(torch):
        value = getattr(torch, name, None)
        if isinstance(value, torch.dtype):
            table.setdefault(name, value)
            table.setdefault(str(value).removeprefix("torch."), value)
    return table


def _numpy_dtype_table() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with torch
        return {}
    names = (
        "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64", "complex64", "complex128",
    )  # fmt: skip
    table: dict[str, Any] = {}
    for name in names:
        dtype = np.dtype(getattr(np, name))
        for key in (name, dtype.str, dtype.name):
            table.setdefault(key, dtype)
    return table


def _enum_table() -> dict[str, type[enum.Enum]]:
    """TorchLens-owned enum classes, keyed by ``module.Class`` and by unambiguous bare name."""

    import pkgutil

    import torchlens._vocab as vocab

    modules = list(COMPONENT_SPEC_MODULES) + [
        f"{vocab.__name__}.{info.name}" for info in pkgutil.iter_modules(vocab.__path__)
    ]
    table: dict[str, type[enum.Enum]] = {}
    bare: dict[str, type[enum.Enum] | None] = {}
    for module_name in modules:
        module = import_module(module_name)
        for attr in dir(module):
            obj = getattr(module, attr)
            if not (isinstance(obj, type) and issubclass(obj, enum.Enum)):
                continue
            if not obj.__module__.startswith("torchlens."):
                continue
            table[f"{obj.__module__}.{obj.__qualname__}"] = obj
            previous = bare.get(obj.__qualname__, obj)
            bare[obj.__qualname__] = obj if previous is obj else None
    for name, obj in bare.items():
        if obj is not None:
            table.setdefault(name, obj)
    return table


def _record_type_table() -> dict[str, tuple[type, frozenset[str]]]:
    """The ``record`` allowlist: the writer contract's component classes and their fields.

    The twelve record classes are in the writer contract too, but a record
    never travels inline (section 3), so they are left out: an inline ``Op``
    refuses.
    """

    table: dict[str, tuple[type, frozenset[str]]] = {}
    bare: dict[str, str | None] = {}
    for qualname, fields in component_spec_tables().items():
        module_name, _, class_name = qualname.rpartition(".")
        if class_name in _RECORD_CLASS_NAMES:
            continue
        cls = getattr(import_module(module_name), class_name)
        allowed = frozenset(name for name, policy in fields.items() if policy != "drop")
        table[qualname] = (cls, allowed)
        bare[class_name] = None if class_name in bare else qualname
    for class_name, qualname in bare.items():
        if qualname is not None:
            table.setdefault(class_name, table[qualname])
    return table


class _Tables:
    """Import-time lookup tables; built once on first use, read-only afterwards."""

    _torch_dtypes: dict[str, torch.dtype] | None = None
    _numpy_dtypes: dict[str, Any] | None = None
    _enums: dict[str, type[enum.Enum]] | None = None
    _records: dict[str, tuple[type, frozenset[str]]] | None = None

    @classmethod
    def torch_dtypes(cls) -> dict[str, torch.dtype]:
        if cls._torch_dtypes is None:
            cls._torch_dtypes = _torch_dtype_table()
        return cls._torch_dtypes

    @classmethod
    def numpy_dtypes(cls) -> dict[str, Any]:
        if cls._numpy_dtypes is None:
            cls._numpy_dtypes = _numpy_dtype_table()
        return cls._numpy_dtypes

    @classmethod
    def enums(cls) -> dict[str, type[enum.Enum]]:
        if cls._enums is None:
            cls._enums = _enum_table()
        return cls._enums

    @classmethod
    def records(cls) -> dict[str, tuple[type, frozenset[str]]]:
        if cls._records is None:
            cls._records = _record_type_table()
        return cls._records


def _require_fields(data: Mapping[str, Any], names: tuple[str, ...], *, at: str, tag: str) -> None:
    missing = [name for name in names if name not in data]
    extra = [name for name in data if name != contract.TYPED_VALUE_KEY and name not in names]
    if missing or extra:
        raise cell_error(
            at,
            f"{tag} value is missing {missing} or carries unknown fields {extra}",
            code=CELL_CODE,
        )


def _is_plain_json(value: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_plain_json(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and key != contract.TYPED_VALUE_KEY
            and _is_plain_json(item, depth + 1)
            for key, item in value.items()
        )
    return False


# ---------------------------------------------------------------------------
# Tensor references
# ---------------------------------------------------------------------------


class TensorRefSink:
    """Collects the tensor references a decode meets, as ``BlobRef`` markers plus entries.

    The reader hands the entries to the existing lazy layer as a manifest
    index (blob id to tensor entry), so eager and lazy materialization run
    through ``LazyActivationRef`` unchanged. A structure-only reference (no
    ``blob`` and no ``key``) decodes to ``None`` and is counted.

    Parameters
    ----------
    codecs:
        The manifest's codec list; a reference naming another codec refuses.
    files:
        The manifest's ``files`` table; a reference into an unlisted chunk refuses.
    """

    def __init__(self, *, codecs: tuple[str, ...], files: Mapping[str, Mapping[str, Any]]) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.structure_only_refs = 0
        self._codecs = codecs
        self._files = files

    def add(self, ref: Mapping[str, Any], *, at: str, kind: str) -> Any:
        _require_fields(ref, contract.TENSOR_REFERENCE_FIELDS, at=at, tag="tensor")
        shape = ref["shape"]
        if not isinstance(shape, list) or not all(_is_int(dim) and dim >= 0 for dim in shape):
            raise cell_error(
                at, "tensor shape must be an array of non-negative integers", code=CELL_CODE
            )
        if len(shape) > 64:
            raise cell_error(at, "tensor shape has more than 64 dimensions", code=CELL_CODE)
        device = ref["device"]
        if not isinstance(device, str) or not DEVICE_TEXT.match(device):
            raise cell_error(at, f"tensor device {device!r} is not a device string", code=CELL_CODE)
        if not isinstance(ref["requires_grad"], bool) or not isinstance(ref["complex"], bool):
            raise cell_error(
                at, "tensor requires_grad and complex must be booleans", code=CELL_CODE
            )
        codec_id, codec_metadata = _split_codec(ref["codec"], at=at)
        if codec_id not in self._codecs:
            raise cell_error(
                at, f"tensor codec {codec_id!r} is not among the manifest's codecs", code=CELL_CODE
            )
        dtype_name = ref["dtype"]
        dtype_table = (
            _Tables.torch_dtypes() if codec_id == "torch_safetensors_v1" else _Tables.numpy_dtypes()
        )
        if not isinstance(dtype_name, str) or dtype_name not in dtype_table:
            raise cell_error(
                at,
                f"tensor dtype {dtype_name!r} is outside the {codec_id} dtype table",
                code=CELL_CODE,
            )
        blob, key = ref["blob"], ref["key"]
        if blob is None and key is None:
            self.structure_only_refs += 1
            return None
        if not isinstance(blob, str) or not _BLOB_PATH.match(blob):
            raise cell_error(
                at,
                f"tensor blob {blob!r} is not a blobs/<codec>-<nnnn>.safetensors path",
                code=CELL_CODE,
            )
        if not isinstance(key, str) or not _BLOB_KEY.match(key):
            raise cell_error(at, f"tensor key {key!r} is not an opaque t<k> key", code=CELL_CODE)
        listed = self._files.get(blob)
        if not isinstance(listed, Mapping):
            raise cell_error(
                at, f"tensor blob {blob} is not listed in the manifest's files", code=CELL_CODE
            )
        blob_id = f"{blob}#{key}"
        if blob_id in self.entries:
            raise cell_error(at, f"tensor {blob_id} is referenced twice", code=CELL_CODE)
        self.entries[blob_id] = {
            "blob_id": blob_id,
            "kind": kind,
            "label": at,
            "relative_path": blob,
            "entry_key": key,
            "backend": "torch" if codec_id == "torch_safetensors_v1" else "numpy",
            "shape": list(shape),
            "dtype": dtype_name,
            "device_at_save": device,
            "layout": "strided",
            "bytes": listed["bytes"],
            "sha256": listed["sha256"],
            "requires_grad": ref["requires_grad"],
            "logical_backend": "torch" if codec_id == "torch_safetensors_v1" else "numpy",
            "codec": codec_id,
            "codec_metadata": codec_metadata,
        }
        return BlobRef(blob_id, kind)


def _split_codec(codec: Any, *, at: str) -> tuple[str, dict[str, Any] | None]:
    if isinstance(codec, str):
        return codec, None
    if isinstance(codec, dict) and isinstance(codec.get("id"), str):
        metadata = {key: value for key, value in codec.items() if key != "id"}
        if not _is_plain_json(metadata):
            raise cell_error(at, "tensor codec metadata must be plain JSON", code=CELL_CODE)
        return codec["id"], metadata or None
    raise cell_error(at, "tensor codec must be a codec id or {id, ...hints}", code=CELL_CODE)


def _tensor_kind(at: str) -> str:
    """The column a tensor reference belongs to: ``grad`` for gradient columns, else ``out``."""

    column = at.split(".", 1)[1] if "." in at else at
    head = re.split(r"[.\[{]", column, maxsplit=1)[0]
    return "grad" if "grad" in head else "out"


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------


class Decoder:
    """Decode JSON cells into Python values through the closed tag vocabulary.

    Parameters
    ----------
    sink:
        Receives tensor references; ``None`` refuses any tensor reference
        (a decode outside a record table).
    defer_records:
        Return :class:`PendingRecord` for ``record``-tagged objects instead of
        building them, so the table reader can resolve cross-record references
        inside their fields first.
    """

    def __init__(self, sink: TensorRefSink | None = None, *, defer_records: bool = False) -> None:
        self.sink = sink
        self.defer_records = defer_records
        self._handlers: dict[str, Callable[[Mapping[str, Any], str, int], Any]] = {
            "tensor": self._tensor,
            "tuple": self._tuple,
            "set": self._set,
            "frozenset": self._frozenset,
            "dict": self._dict,
            "bytes": self._bytes,
            "float": self._float,
            "int": self._int,
            "complex": self._complex,
            "dtype": self._dtype,
            "device": self._device,
            "type": self._type,
            "enum": self._enum,
            "record": self._record,
            "unencodable": self._unencodable,
            "callable_placeholder": self._callable_placeholder,
        }
        drift = set(self._handlers) ^ set(contract.TYPED_VALUE_TAGS)
        if drift:  # pragma: no cover - a contract/decoder drift is a programming error
            raise RuntimeError(f"typed-value tags and decoder handlers differ: {sorted(drift)}")

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
            return [
                self.decode(item, at=f"{at}[{i}]", depth=depth + 1) for i, item in enumerate(data)
            ]
        if isinstance(data, dict):
            tag = data.get(contract.TYPED_VALUE_KEY)
            if tag is None:
                return {
                    key: self.decode(value, at=f"{at}.{key}", depth=depth + 1)
                    for key, value in data.items()
                }
            handler = self._handlers.get(tag) if isinstance(tag, str) else None
            if handler is None:
                raise cell_error(at, f"unknown typed-value tag {tag!r}", code=CELL_CODE)
            return handler(data, at, depth + 1)
        raise cell_error(
            at,
            f"JSON value of type {type(data).__name__} is outside the vocabulary",
            code=CELL_CODE,
        )

    # -- containers ---------------------------------------------------------

    def _items(self, data: Mapping[str, Any], at: str, depth: int, tag: str) -> list[Any]:
        _require_fields(data, ("items",), at=at, tag=tag)
        items = data["items"]
        if not isinstance(items, list):
            raise cell_error(at, f"{tag} items must be an array", code=CELL_CODE)
        return [self.decode(item, at=f"{at}[{i}]", depth=depth) for i, item in enumerate(items)]

    def _tuple(self, data: Mapping[str, Any], at: str, depth: int) -> tuple[Any, ...]:
        return tuple(self._items(data, at, depth, "tuple"))

    def _set(self, data: Mapping[str, Any], at: str, depth: int) -> set[Any]:
        return set(self._hashable(self._items(data, at, depth, "set"), at))

    def _frozenset(self, data: Mapping[str, Any], at: str, depth: int) -> frozenset[Any]:
        return frozenset(self._hashable(self._items(data, at, depth, "frozenset"), at))

    @staticmethod
    def _hashable(items: list[Any], at: str) -> list[Any]:
        try:
            hash(tuple(items))
        except TypeError:
            raise cell_error(at, "set members must be hashable", code=CELL_CODE) from None
        return items

    def _dict(self, data: Mapping[str, Any], at: str, depth: int) -> dict[Any, Any]:
        _require_fields(data, ("items",), at=at, tag="dict")
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
            key = self.decode(pair[0], at=f"{at}{{{index}}}.key", depth=depth)
            try:
                hash(key)
            except TypeError:
                raise cell_error(at, f"dict key {index} is not hashable", code=CELL_CODE) from None
            if key in result:
                raise cell_error(at, f"dict key {key!r} repeats", code=CELL_CODE)
            result[key] = self.decode(pair[1], at=f"{at}{{{index}}}", depth=depth)
        return result

    # -- scalars ------------------------------------------------------------

    def _bytes(self, data: Mapping[str, Any], at: str, depth: int) -> bytes:
        _require_fields(data, ("b64",), at=at, tag="bytes")
        text = data["b64"]
        if not isinstance(text, str) or len(text) > (BYTES_VALUE_LIMIT * 4) // 3 + 4:
            raise cell_error(
                at, "bytes b64 must be a string under the 1 MiB ceiling", code=CELL_CODE
            )
        try:
            raw = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            raise cell_error(at, "bytes b64 is not valid base64", code=CELL_CODE) from None
        if len(raw) > BYTES_VALUE_LIMIT:
            raise cell_error(at, "bytes value exceeds the 1 MiB ceiling", code=CELL_CODE)
        return raw

    def _float(self, data: Mapping[str, Any], at: str, depth: int) -> float:
        _require_fields(data, ("text",), at=at, tag="float")
        text = data["text"]
        if text not in ("nan", "inf", "-inf"):
            raise cell_error(at, f"float text {text!r} is not nan, inf or -inf", code=CELL_CODE)
        return float(text)

    def _int(self, data: Mapping[str, Any], at: str, depth: int) -> int:
        _require_fields(data, ("text",), at=at, tag="int")
        text = data["text"]
        if (
            not isinstance(text, str)
            or len(text) > INT_TEXT_LIMIT
            or not re.fullmatch(r"-?[0-9]+", text)
        ):
            raise cell_error(at, "int text must be a bounded decimal string", code=CELL_CODE)
        return int(text)

    def _complex(self, data: Mapping[str, Any], at: str, depth: int) -> complex:
        _require_fields(data, ("re", "im"), at=at, tag="complex")
        parts: list[float] = []
        for name in ("re", "im"):
            part = self.decode(data[name], at=f"{at}.{name}", depth=depth)
            if isinstance(part, bool) or not isinstance(part, (int, float)):
                raise cell_error(at, f"complex {name} must be a number", code=CELL_CODE)
            parts.append(float(part))
        return complex(parts[0], parts[1])

    def _dtype(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        _require_fields(data, ("name",), at=at, tag="dtype")
        name = data["name"]
        if isinstance(name, str):
            if name in _Tables.torch_dtypes():
                return _Tables.torch_dtypes()[name]
            if name in _Tables.numpy_dtypes():
                return _Tables.numpy_dtypes()[name]
        raise cell_error(at, f"dtype name {name!r} is outside the dtype table", code=CELL_CODE)

    def _device(self, data: Mapping[str, Any], at: str, depth: int) -> torch.device:
        _require_fields(data, ("text",), at=at, tag="device")
        text = data["text"]
        if not isinstance(text, str) or not DEVICE_TEXT.match(text):
            raise cell_error(at, f"device text {text!r} is not a device string", code=CELL_CODE)
        try:
            return torch.device(text)
        except (RuntimeError, ValueError):
            raise cell_error(
                at, f"device text {text!r} is not a torch device", code=CELL_CODE
            ) from None

    def _type(self, data: Mapping[str, Any], at: str, depth: int) -> str:
        _require_fields(data, ("name",), at=at, tag="type")
        name = data["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][\w.]{0,255}", name):
            raise cell_error(at, "type name must be a dotted identifier", code=CELL_CODE)
        return name

    def _enum(self, data: Mapping[str, Any], at: str, depth: int) -> enum.Enum:
        _require_fields(data, ("name", "value"), at=at, tag="enum")
        name = data["name"]
        cls = _Tables.enums().get(name) if isinstance(name, str) else None
        if cls is None:
            raise cell_error(at, f"enum class {name!r} is outside the enum table", code=CELL_CODE)
        value = self.decode(data["value"], at=f"{at}.value", depth=depth)
        try:
            return cls(value)
        except (ValueError, TypeError):
            raise cell_error(
                at, f"{name} has no member with value {value!r}", code=CELL_CODE
            ) from None

    # -- objects ------------------------------------------------------------

    def _record(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        _require_fields(data, ("type", "fields"), at=at, tag="record")
        type_name = data["type"]
        row = _Tables.records().get(type_name) if isinstance(type_name, str) else None
        if row is None:
            raise cell_error(
                at,
                f"record type {type_name!r} is outside the record-type allowlist",
                code=CELL_CODE,
            )
        cls, allowed = row
        fields = data["fields"]
        if not isinstance(fields, dict):
            raise cell_error(at, "record fields must be an object", code=CELL_CODE)
        unknown = sorted(set(fields) - allowed)
        if unknown:
            raise cell_error(
                at, f"record {type_name} carries fields outside its spec: {unknown}", code=CELL_CODE
            )
        decoded = {
            key: self.decode(value, at=f"{at}.{key}", depth=depth) for key, value in fields.items()
        }
        if self.defer_records:
            return PendingRecord(cls, decoded, at)
        return build_component(cls, decoded)

    def _tensor(self, data: Mapping[str, Any], at: str, depth: int) -> Any:
        if self.sink is None:
            raise cell_error(at, "a tensor reference outside a record table", code=CELL_CODE)
        return self.sink.add(data, at=at, kind=_tensor_kind(at))

    @staticmethod
    def _unencodable(data: Mapping[str, Any], at: str, depth: int) -> Unencodable:
        _require_fields(data, contract.UNENCODABLE_FIELDS, at=at, tag="unencodable")
        if not (
            isinstance(data["type"], str)
            and isinstance(data["repr"], str)
            and len(data["repr"]) <= contract.UNENCODABLE_REPR_LIMIT
            and _is_int(data["repr_length"])
            and data["repr_length"] >= 0
            and isinstance(data["at"], str)
        ):
            raise cell_error(
                at,
                "unencodable disclosure fields are mistyped or over the repr limit",
                code=CELL_CODE,
            )
        return Unencodable(data["type"], data["repr"], data["repr_length"], data["at"])

    @staticmethod
    def _callable_placeholder(data: Mapping[str, Any], at: str, depth: int) -> CallablePlaceholder:
        _require_fields(
            data, contract.CALLABLE_PLACEHOLDER_FIELDS, at=at, tag="callable_placeholder"
        )
        position = data["position"]
        if not (
            (_is_int(position) and position >= 0)
            or (isinstance(position, str) and re.fullmatch(r"\d+(\.[\w\[\]\.]+)?", position))
        ):
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


def build_component(cls: type, fields: Mapping[str, Any]) -> Any:
    """Construct one nested value object from its decoded fields, never calling ``__init__``.

    A class with its own ``__setstate__`` (``Edit`` rebuilds its builtin
    factory there) gets the fields through it; otherwise the state adapter
    restores them (``__tl_state_restore__`` hook, ``__dict__`` or slots).
    """

    from ..._data_classes._state_adapter import state_new, state_restore

    obj = state_new(cls)
    setstate = getattr(cls, "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, dict(fields))
        return obj
    return state_restore(obj, dict(fields))


def decode_value(data: Any, *, schema: Any = None, at: str = "value") -> Any:
    """Decode one JSON cell with no tensor references allowed.

    Parameters
    ----------
    data:
        The JSON-decoded cell.
    schema:
        The schema node (reserved for the schema-directed families of design
        section 9; cells are self-typed, so it is not consulted yet).
    at:
        The ``record[row].column`` path named by any refusal.
    """

    del schema
    return Decoder(None).decode(data, at=at)
