"""The typed-value codec of the 3.0 saved format: Python values to tagged JSON and back.

Plain JSON carries ``None``, bools, finite numbers (ints within 2^53), strings,
lists and string-keyed dicts. Everything else is an object whose marker key
``contract.TYPED_VALUE_KEY`` names a tag from ``contract.TYPED_VALUE_TAGS``;
both directions live here so the tags stay symmetric. A value outside the
vocabulary becomes an ``unencodable`` disclosure (type, bounded repr, length,
position); nothing is ever pickled, and the decoder imports nothing by a name
it reads: dtypes, enums and value-object classes resolve through tables built
from code. Design: ``docs/maintainers/saved_format_3_design.md``, sections 4
and 5.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import math
from collections.abc import Callable, Mapping
from functools import cache
from importlib import import_module
from typing import Any, NamedTuple

import torch

from ..._data_classes._state_adapter import state_items
from ..field_registry import RECORD_CONTRACT_CLASSES, component_spec_tables
from ..format_errors import ArtifactError
from .contract import (
    CALLABLE_PLACEHOLDER_FIELDS,
    TYPED_VALUE_KEY,
    UNENCODABLE_FIELDS,
    UNENCODABLE_REPR_LIMIT,
)

#: Largest integer magnitude JSON readers carry exactly (IEEE double mantissa).
_JSON_SAFE_INT = 2**53
#: Largest ``bytes`` value written inline (base64), per the contract.
BYTES_LIMIT = 1 << 20
#: Deepest inline nesting the encoder writes; the bounded JSON reader refuses
#: at 200 and a record cell sits a few levels down in its file.
_MAX_DEPTH = 150
#: Required payload fields per tag (beyond the marker key).
TAG_FIELDS: dict[str, tuple[str, ...]] = {
    "tensor": ("dtype", "shape", "device", "requires_grad", "codec"),
    "tuple": ("items",),
    "set": ("items",),
    "frozenset": ("items",),
    "dict": ("items",),
    "bytes": ("base64", "bytes"),
    "float": ("v",),
    "int": ("v",),
    "complex": ("re", "im"),
    "dtype": ("name",),
    "device": ("name",),
    "type": ("name",),
    "enum": ("type", "name"),
    "record": ("type", "fields"),
    "ref": ("record", "row"),
    "unencodable": UNENCODABLE_FIELDS,
    "callable_placeholder": CALLABLE_PLACEHOLDER_FIELDS,
}


def class_path(cls: type) -> str:
    """Return the dotted ``module.qualname`` spelling the format uses for a class."""

    return f"{cls.__module__}.{cls.__qualname__}"


def tagged(tag: str, **fields: Any) -> dict[str, Any]:
    """Return a typed JSON object with ``tag`` and its fields."""

    return {TYPED_VALUE_KEY: tag, **fields}


#: TorchLens value-object classes the format holds inline that declare no
#: ``_PORTABLE_STATE_SPEC`` of their own (so ``component_spec_tables`` does not
#: list them): unit quantities, neutral dtype and device refs, edge-use rows,
#: source text, container walk results, facet and intervention value objects.
#: Enums are not here (the ``enum`` tag carries them); records are tables.
_VALUE_CLASSES_BEYOND_COMPONENT_SPECS: tuple[tuple[str, str], ...] = (
    ("torchlens._quantities", "Bytes"),
    ("torchlens._quantities", "Duration"),
    ("torchlens._quantities", "Flops"),
    ("torchlens._quantities", "Macs"),
    ("torchlens._ir.refs", "DeferredRef"),
    ("torchlens._ir.refs", "DeviceRef"),
    ("torchlens._ir.refs", "DtypeRef"),
    ("torchlens._ir.refs", "ParamRef"),
    ("torchlens._ir.refs", "ReservedLabel"),
    ("torchlens._ir.refs", "TensorRef"),
    ("torchlens._ir.container_registry", "ContainerRegistry"),
    ("torchlens._ir.container_registry", "IdentityEntry"),
    ("torchlens._ir.container_registry", "WalkResult"),
    ("torchlens._visualization.code_panel", "SourceText"),
    ("torchlens._data_classes.module", "HookInfo"),
    ("torchlens._data_classes.backward_pass", "BackwardPassAccessor"),
    ("torchlens._data_classes.grad_fn", "GradFnAccessor"),
    ("torchlens._data_classes.op", "GradientRecordAccessor"),
    ("torchlens._data_classes.trace", "ConditionalArm"),
    ("torchlens._data_classes.trace", "ConditionalRoleRef"),
    ("torchlens._data_classes.trace", "ResolvedPostprocessing"),
    ("torchlens._data_classes.trace", "ResolvedPreprocessing"),
    ("torchlens._capture.stop", "StopDirective"),
    ("torchlens._semantic.facets", "FacetCapabilityFlags"),
    ("torchlens._semantic.facets", "FacetContribution"),
    ("torchlens._semantic.facets", "FacetMenuItem"),
    ("torchlens._semantic.facets", "FacetRegistrySnapshot"),
    ("torchlens._semantic.facets", "FacetSpec"),
    ("torchlens.intervention.types", "CapturedArgTemplate"),
    ("torchlens.intervention.types", "EdgeUseRecord"),
    ("torchlens.intervention.types", "FunctionRegistryKey"),
    ("torchlens.intervention.types", "LiteralValue"),
    ("torchlens.intervention.types", "ParentRef"),
    ("torchlens.intervention.types", "SamplingRecord"),
    ("torchlens.intervention.types", "TensorSliceSpec"),
    ("torchlens.intervention.types", "Unsupported"),
)


class RowRef(NamedTuple):
    """A decoded ``ref`` cell: row ``row`` of the ``record`` table."""

    record: str
    row: int


@cache
def record_type_table() -> dict[str, type]:
    """Return the ``record`` tag allowlist: dotted class name to class.

    The source is the writer contract's ``component_specs`` table
    (``field_registry.component_spec_tables``) plus
    ``_VALUE_CLASSES_BEYOND_COMPONENT_SPECS``, minus the twelve record classes
    (tables of their own, referenced by ``ref``) and minus enums (``enum``).
    The classes come from code tables, never from a saved file.

    Returns
    -------
    dict[str, type]
        Every TorchLens value-object class the format may hold inline.
    """

    record_paths = {f"{module}.{name}" for module, name in RECORD_CONTRACT_CLASSES.values()}
    pairs = [tuple(path.rsplit(".", 1)) for path in component_spec_tables()]
    pairs.extend(_VALUE_CLASSES_BEYOND_COMPONENT_SPECS)
    table: dict[str, type] = {}
    for module_name, qualname in pairs:
        path = f"{module_name}.{qualname}"
        cls = getattr(import_module(module_name), qualname, None)
        if path in record_paths or not isinstance(cls, type) or issubclass(cls, enum.Enum):
            continue
        table[path] = cls
    return table


@cache
def dtype_table() -> dict[str, torch.dtype]:
    """Return torch dtype names (without the ``torch.`` prefix) to dtypes."""

    return {
        str(value).replace("torch.", ""): value
        for value in vars(torch).values()
        if isinstance(value, torch.dtype)
    }


def bounded_repr(value: Any) -> str:
    """Return ``repr(value)``, or ``"<unrepresentable>"`` when ``repr`` raises."""

    try:
        return repr(value)
    except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return "<unrepresentable>"


def _set_sort_key(value: Any) -> tuple[str, str]:
    """Return the encoder's total order on set members: type name, then repr."""

    return (type(value).__name__, bounded_repr(value))


def object_state(value: Any) -> dict[str, Any]:
    """Return the field state of a TorchLens value object, in declared order."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    fields = getattr(type(value), "_fields", None)
    if isinstance(value, tuple) and isinstance(fields, tuple):
        return dict(zip(fields, value, strict=True))
    scalar = scalar_base(type(value))
    if scalar is not None:
        return {"value": scalar(value), **dict(getattr(value, "__dict__", None) or {})}
    return dict(state_items(value))


def scalar_base(cls: type) -> type | None:
    """Return the builtin scalar type a scalar subclass (a quantity) wraps."""

    for base in (bool, int, float, str):
        if issubclass(cls, base):
            return base
    return None


class ValueEncoder:
    """Encode Python values into the closed typed-JSON vocabulary.

    Parameters
    ----------
    record_ref:
        Returns the reference cell (label string or row index) of a record
        object, or ``None`` when the value is not a record.
    tensor_ref:
        Stores a tensor payload (a tensor or a scrubbed blob reference) found
        in a payload position and returns its reference fields; ``None`` makes
        every tensor a disclosure.
    callables:
        ``"unencodable"`` (record cells) or ``"placeholder"`` (the
        intervention recipe, where a callable becomes a placeholder).
    """

    def __init__(
        self,
        *,
        record_ref: Callable[[Any], Any] | None = None,
        tensor_ref: Callable[[Any, str], dict[str, Any] | None] | None = None,
        callables: str = "unencodable",
    ) -> None:
        self._record_ref = record_ref
        self._tensor_ref = tensor_ref
        self._callables = callables
        self._active: set[int] = set()
        self.disclosures: list[dict[str, Any]] = []
        self.payload_position = False

    def encode(self, value: Any, at: str, *, payload: bool = False) -> Any:
        """Encode one cell value.

        Parameters
        ----------
        value:
            The cell value.
        at:
            The ``record[row].column`` path, stamped into any disclosure.
        payload:
            Whether tensors met in this cell are payloads (written to chunks)
            rather than metadata (disclosed).

        Returns
        -------
        Any
            JSON-ready data.
        """

        self.payload_position = payload
        return self._encode(value, at, 0)

    def _encode(self, value: Any, at: str, depth: int) -> Any:
        """Encode one value at a nesting depth."""

        kind = type(value)
        if value is None or kind is bool or kind is str:
            return value
        if kind is int:
            return value if abs(value) <= _JSON_SAFE_INT else tagged("int", v=str(value))
        if kind is float:
            return value if math.isfinite(value) else tagged("float", v=repr(value))
        if depth > _MAX_DEPTH:
            return self.unencodable(value, at)
        if self._record_ref is not None:
            reference = self._record_ref(value)
            if reference is not None:
                return reference
        handler = _HANDLERS.get(kind)
        if handler is not None:
            return handler(self, value, at, depth)
        return self._encode_other(value, at, depth)

    def _encode_other(self, value: Any, at: str, depth: int) -> Any:
        """Encode a value whose exact type has no direct handler."""

        if _is_tensor_payload(value):
            return self._tensor(value, at)
        if isinstance(value, enum.Enum):
            return self._enum(value, at, depth)
        if isinstance(value, type):
            return tagged("type", name=class_path(value))
        record_cls = record_type_table().get(class_path(type(value)))
        if record_cls is not None:
            return self._record_value(value, at, depth)
        if isinstance(value, Mapping) and type(value).__module__ == "collections":
            return self._mapping(value, at, depth)
        for base in (bool, int, float, str):
            if isinstance(value, base) and not _is_torchlens_class(type(value)):
                return self._encode(base(value), at, depth)
        if callable(value) and self._callables == "placeholder":
            return callable_placeholder(value, position=at)
        return self.unencodable(value, at)

    def _enter(self, value: Any, at: str) -> None:
        """Refuse a container that contains itself (JSON has no cycles)."""

        if id(value) in self._active:
            raise ArtifactError(
                f"The 3.0 writer met a self-referencing value at {at}.",
                code="metadata_value_unportable",
                remedy="break the cycle in the value before saving",
            )
        self._active.add(id(value))

    def _items(self, items: Any, at: str, depth: int) -> list[Any]:
        """Encode a sequence of items."""

        return [self._encode(item, f"{at}[{i}]", depth + 1) for i, item in enumerate(items)]

    def _list(self, value: list[Any], at: str, depth: int) -> list[Any]:
        """Encode a list as a JSON array."""

        self._enter(value, at)
        try:
            return self._items(value, at, depth)
        finally:
            self._active.discard(id(value))

    def _tuple(self, value: tuple[Any, ...], at: str, depth: int) -> dict[str, Any]:
        """Encode a plain tuple."""

        return tagged("tuple", items=self._items(value, at, depth))

    def _set(self, value: set[Any] | frozenset[Any], at: str, depth: int) -> dict[str, Any]:
        """Encode a set or frozenset with members in the canonical order."""

        ordered = sorted(value, key=_set_sort_key)
        tag = "frozenset" if isinstance(value, frozenset) else "set"
        return tagged(tag, items=self._items(ordered, at, depth))

    def _mapping(self, value: Mapping[Any, Any], at: str, depth: int) -> Any:
        """Encode a mapping: a JSON object when every key is a plain string."""

        self._enter(value, at)
        try:
            if all(type(key) is str and key != TYPED_VALUE_KEY for key in value):
                return {
                    key: self._encode(item, f"{at}[{key!r}]", depth + 1)
                    for key, item in value.items()
                }
            pairs = [
                [
                    self._encode(key, f"{at}<key {index}>", depth + 1),
                    self._encode(item, f"{at}[{key!r}]", depth + 1),
                ]
                for index, (key, item) in enumerate(value.items())
            ]
            return tagged("dict", items=pairs)
        finally:
            self._active.discard(id(value))

    def _enum(self, value: enum.Enum, at: str, depth: int) -> Any:
        """Encode a TorchLens-owned enum member; any other enum is disclosed."""

        if not _is_torchlens_class(type(value)):
            return self.unencodable(value, at)
        return tagged("enum", type=class_path(type(value)), name=value.name)

    def _record_value(self, value: Any, at: str, depth: int) -> dict[str, Any]:
        """Encode an allowlisted TorchLens value object field by field."""

        self._enter(value, at)
        try:
            fields = {
                name: self._encode(item, f"{at}.{name}", depth + 1)
                for name, item in object_state(value).items()
            }
            return tagged("record", type=class_path(type(value)), fields=fields)
        finally:
            self._active.discard(id(value))

    def _tensor(self, value: Any, at: str) -> Any:
        """Store a payload tensor, or disclose a tensor in a metadata position.

        A scrubbed blob reference is a payload wherever it sits (the scrub
        puts them in non-payload columns too: buffer initial values, gradient
        records); a raw tensor counts only in a payload column.
        """

        is_blob_ref = not isinstance(value, torch.Tensor)
        if (is_blob_ref or self.payload_position) and self._tensor_ref is not None:
            fields = self._tensor_ref(value, at)
            if fields is not None:
                return tagged("tensor", **fields)
        return self.unencodable(value, at)

    def unencodable(self, value: Any, at: str) -> dict[str, Any]:
        """Stringify a value with a disclosure (never pickled).

        Parameters
        ----------
        value:
            The value with no typed encoding.
        at:
            Its position.

        Returns
        -------
        dict[str, Any]
            The ``unencodable`` typed object.
        """

        text = bounded_repr(value)
        fields = {
            "type": class_path(type(value)),
            "repr": text[:UNENCODABLE_REPR_LIMIT],
            "repr_length": len(text),
            "at": at,
        }
        self.disclosures.append(fields)
        return tagged("unencodable", **fields)


def callable_placeholder(value: Any, *, position: int | str) -> dict[str, Any]:
    """Return the placeholder object that stands in for a callable.

    Parameters
    ----------
    value:
        The callable (never imported back on read).
    position:
        Rule index, or ``"<rule index>.<arg path>"`` inside edit arguments.

    Returns
    -------
    dict[str, Any]
        The ``callable_placeholder`` typed object.
    """

    module = getattr(value, "__module__", None) or type(value).__module__
    qualname = getattr(value, "__qualname__", None) or type(value).__qualname__
    return tagged(
        "callable_placeholder",
        position=position,
        qualname=f"{module}.{qualname}",
        repr_length=len(bounded_repr(value)),
    )


def _encode_bytes(encoder: ValueEncoder, value: bytes, at: str, depth: int) -> Any:
    """Encode bytes as base64 (over the size limit they are disclosed)."""

    if len(value) > BYTES_LIMIT:
        return encoder.unencodable(value, at)
    return tagged("bytes", base64=base64.b64encode(bytes(value)).decode("ascii"), bytes=len(value))


def _encode_complex(encoder: ValueEncoder, value: complex, at: str, depth: int) -> Any:
    """Encode a complex number by its parts."""

    return tagged(
        "complex",
        re=encoder._encode(value.real, at, depth + 1),
        im=encoder._encode(value.imag, at, depth + 1),
    )


def _encode_dtype(encoder: ValueEncoder, value: torch.dtype, at: str, depth: int) -> Any:
    """Encode a torch dtype by its name."""

    return tagged("dtype", name=str(value).replace("torch.", ""))


def _encode_device(encoder: ValueEncoder, value: torch.device, at: str, depth: int) -> Any:
    """Encode a torch device by its string form."""

    return tagged("device", name=str(value))


_HANDLERS: dict[type, Callable[[ValueEncoder, Any, str, int], Any]] = {
    bytes: _encode_bytes,
    bytearray: _encode_bytes,
    complex: _encode_complex,
    torch.dtype: _encode_dtype,
    torch.device: _encode_device,
    list: ValueEncoder._list,
    tuple: ValueEncoder._tuple,
    set: ValueEncoder._set,
    frozenset: ValueEncoder._set,
    dict: ValueEncoder._mapping,
}


def _is_torchlens_class(cls: type) -> bool:
    """Return whether a class is defined inside the ``torchlens`` package."""

    module = cls.__module__
    return module == "torchlens" or module.startswith("torchlens.")


def _is_tensor_payload(value: Any) -> bool:
    """Return whether a value is a tensor or a scrubbed blob reference."""

    return isinstance(value, torch.Tensor) or type(value).__name__ == "BlobRef"


@cache
def enum_table() -> dict[str, type[enum.Enum]]:
    """Return every TorchLens enum class reachable from the value-object modules."""

    table: dict[str, type[enum.Enum]] = {}
    modules = {cls.__module__ for cls in record_type_table().values()}
    modules.update(module for module, _ in RECORD_CONTRACT_CLASSES.values())
    for module_name in sorted(modules):
        for value in vars(import_module(module_name)).values():
            if isinstance(value, type) and issubclass(value, enum.Enum):
                if _is_torchlens_class(value):
                    table[class_path(value)] = value
    return table
