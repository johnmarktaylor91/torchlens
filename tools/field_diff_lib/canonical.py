"""Canonical, comparable forms of the values a trace exposes.

Every value read from a record is turned into plain JSON data. Tensors keep
more than their content: dtype, device, layout, strides, storage offset,
``requires_grad``, whether an autograd node is attached, which fields hold the
same tensor object and which distinct tensors share storage. Object and
storage identities are numbered in order of first appearance within one
trace, so two runs can be compared without comparing addresses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from enum import Enum
from typing import Any

import torch

MAX_DEPTH = 8
_ADDRESS = re.compile(r"0x[0-9a-fA-F]{6,}")
_MESSAGE_LIMIT = 300


def error_record(error: BaseException, normalize: Any = None) -> dict[str, Any]:
    """Describe a raised error by type, stable code and message.

    Parameters
    ----------
    error:
        The exception a read or a step raised.
    normalize:
        Optional string normaliser applied to the message.

    Returns
    -------
    dict[str, Any]
        ``{"__raises__": type name, "code": code or None, "message": text}``.
    """

    fields = getattr(error, "fields", None)
    code = fields.get("code") if isinstance(fields, dict) else getattr(error, "code", None)
    message = _ADDRESS.sub("<addr>", str(error))
    if normalize is not None:
        message = normalize(message)
    message = message[:_MESSAGE_LIMIT]
    return {
        "__raises__": type(error).__name__,
        "code": code if isinstance(code, (str, int)) else None,
        "message": message,
    }


class Canonicalizer:
    """Turn values into canonical JSON data for one trace.

    Parameters
    ----------
    roots:
        Ordered ``(path prefix, placeholder)`` pairs replaced inside strings,
        so a dump does not depend on where a checkout or a scratch folder is.
    record_classes:
        Names of the record classes; a nested instance is written as a
        reference (class and label) and not expanded.
    """

    def __init__(self, roots: list[tuple[str, str]], record_classes: frozenset[str]) -> None:
        """Start with empty identity tables."""

        self._roots = [(prefix, mark) for prefix, mark in roots if prefix]
        self._record_classes = record_classes
        self._tensor_index: dict[int, int] = {}
        self._storage_index: dict[int, int] = {}
        self._keep: list[Any] = []
        self.tensor_sites: list[tuple[str, torch.Tensor]] = []
        self._site = ""

    def at(self, site: str) -> Canonicalizer:
        """Name the field being read, for the tensor site list."""

        self._site = site
        return self

    def string(self, value: str) -> str:
        """Replace known path prefixes and hex addresses inside a string."""

        for prefix, mark in self._roots:
            if prefix in value:
                value = value.replace(prefix, mark)
        if "/site-packages/" in value:
            value = re.sub(r"[^\s'\"]*/site-packages/", "<site>/", value)
        return _ADDRESS.sub("<addr>", value)

    def tensor(self, value: torch.Tensor) -> dict[str, Any]:
        """Describe one tensor: content hash, metadata and identities."""

        self._keep.append(value)
        self.tensor_sites.append((self._site, value))
        record: dict[str, Any] = {
            "__tensor__": type(value).__name__,
            "object": f"obj#{self._tensor_index.setdefault(id(value), len(self._tensor_index))}",
        }
        try:
            record.update(_tensor_metadata(value))
            record["storage"] = self._storage_of(value)
            record["sha256"] = _tensor_hash(value)
        except Exception as error:  # noqa: BLE001 - an unreadable tensor is itself a fact
            record["tensor_error"] = type(error).__name__
        return record

    def _storage_of(self, value: torch.Tensor) -> str | None:
        """Number the tensor's storage by first appearance; ``None`` when it has none."""

        if value.layout is not torch.strided:
            return None
        storage = value.untyped_storage()
        key = getattr(storage, "_cdata", None) or storage.data_ptr()
        self._keep.append(storage)
        return f"sto#{self._storage_index.setdefault(key, len(self._storage_index))}"

    def value(self, value: Any, depth: int = 0) -> Any:  # noqa: C901, PLR0911, PLR0912
        """Return the canonical form of any value.

        The branch per value kind is the whole job of this function, so its
        branching is essential and kept in one place.
        """

        if depth > MAX_DEPTH:
            return "<max-depth>"
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return {"__float__": repr(value)}
        if isinstance(value, str):
            return self.string(value)
        if isinstance(value, bytes):
            return {"__bytes__": hashlib.sha256(value).hexdigest(), "len": len(value)}
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if isinstance(value, (torch.dtype, torch.device, torch.Size, torch.layout)):
            return {"__torch__": str(value)}
        if isinstance(value, Enum):
            return {"__enum__": f"{type(value).__name__}.{value.name}"}
        if type(value).__name__ in self._record_classes:
            return self.record_ref(value)
        if isinstance(value, dict):
            return self._mapping(value, depth)
        if isinstance(value, (set, frozenset)):
            members = sorted(_sort_key(self.value(member, depth + 1)) for member in value)
            return {"__set__": type(value).__name__, "members": members}
        if isinstance(value, (list, tuple)):
            items = [self.value(member, depth + 1) for member in value]
            if type(value) in (list, tuple):
                return items if isinstance(value, list) else {"__tuple__": items}
            return {"__seq__": type(value).__name__, "items": items}
        if isinstance(value, slice):
            return {"__slice__": [repr(value.start), repr(value.stop), repr(value.step)]}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._dataclass(value, depth)
        return self._opaque(value, depth)

    def record_ref(self, value: Any) -> dict[str, Any]:
        """Write a nested record as its class and label."""

        label = None
        for name in ("label", "layer_label", "address", "name"):
            try:
                candidate = getattr(value, name, None)
            except Exception:  # noqa: BLE001 - a reference must never fail the dump
                candidate = None
            if isinstance(candidate, str):
                label = candidate
                break
        return {"__ref__": type(value).__name__, "label": label}

    def _mapping(self, value: dict[Any, Any], depth: int) -> dict[str, Any]:
        """Keep a mapping's type, key order, keys and values."""

        items = [
            [self.value(key, depth + 1), self.value(entry, depth + 1)]
            for key, entry in list(value.items())
        ]
        return {"__dict__": type(value).__name__, "items": items}

    def _dataclass(self, value: Any, depth: int) -> dict[str, Any]:
        """Expand a dataclass instance field by field."""

        fields: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            try:
                fields[field.name] = self.value(getattr(value, field.name), depth + 1)
            except Exception as error:  # noqa: BLE001 - raising is recorded, not hidden
                fields[field.name] = error_record(error, self.string)
        return {"__dataclass__": type(value).__name__, "fields": fields}

    def _opaque(self, value: Any, depth: int) -> dict[str, Any]:
        """Describe any other object by type, and by content where it is a view or a callable."""

        kind = f"{type(value).__module__}.{type(value).__qualname__}"
        if callable(value) and hasattr(value, "__qualname__"):
            return {"__callable__": str(value.__qualname__)}
        items = _view_items(value)
        if items is not None:
            return {"__view__": kind, "items": self.value(items, depth + 1)}
        try:
            text = self.string(repr(value))[:_MESSAGE_LIMIT]
        except Exception as error:  # noqa: BLE001 - a failing repr is recorded
            return {"__object__": kind, "repr_raises": type(error).__name__}
        return {"__object__": kind, "repr": text}


def _view_items(value: Any) -> Any:
    """Return the contents of a mapping-like or sequence-like view, else ``None``."""

    from collections.abc import Mapping, Sequence

    try:
        if isinstance(value, Mapping):
            return dict(value.items())
        if isinstance(value, Sequence):
            return list(value)
    except Exception:  # noqa: BLE001 - fall back to the repr form
        return None
    return None


def _sort_key(canonical: Any) -> str:
    """Give set members a stable order."""

    import json

    return json.dumps(canonical, sort_keys=True)


def _tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    """Return the metadata compared beside a tensor's content."""

    meta: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "layout": str(value.layout),
        "requires_grad": bool(value.requires_grad),
        "has_grad_fn": value.grad_fn is not None,
        "is_leaf": bool(value.is_leaf),
    }
    if value.layout is torch.strided:
        meta["strides"] = list(value.stride())
        meta["storage_offset"] = int(value.storage_offset())
    return meta


def _tensor_hash(value: torch.Tensor) -> str:
    """Hash a tensor's content; sparse layouts hash their indices and values."""

    detached = value.detach()
    digest = hashlib.sha256()
    if detached.layout is torch.sparse_coo:
        coalesced = detached.coalesce()
        parts = [coalesced.indices(), coalesced.values()]
    elif detached.layout is not torch.strided:
        parts = [detached.to_dense()]
    else:
        parts = [detached]
    for part in parts:
        dense = part.cpu().contiguous()
        try:
            digest.update(dense.numpy().tobytes())
        except (TypeError, RuntimeError):
            digest.update(repr(dense.flatten().tolist()).encode())
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    """Hash one tensor's content (public form of the content hash)."""

    return _tensor_hash(value)
