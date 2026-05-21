"""Reference dataclasses for backend-neutral captured objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import BlobRef


@dataclass(frozen=True, slots=True)
class DeferredRef:
    """Backend-private lazy payload reference."""

    backend: str
    handle_id: str
    blob_ref: BlobRef | None
    inferred_shape: tuple[int, ...] | None
    inferred_dtype: str | None
    materialize_fn: object | None


@dataclass(frozen=True, slots=True)
class TensorRef:
    """Backend-neutral metadata for a captured tensor-like value."""

    label_raw: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    requires_grad: bool | None
    memory: int | None
    payload: object | DeferredRef | None
    blob_ref: BlobRef | None
    backend_handle_id: str | None


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Backend-neutral metadata for a captured parameter."""

    barcode: str
    address: str
    shape: tuple[int, ...] | None
    dtype: str | None
    trainable: bool
    module_address: str | None


@dataclass(frozen=True, slots=True)
class ReservedLabel:
    """Atomic raw-label reservation for a loggable output site."""

    label: str
    label_raw: str
    raw_index: int
    type_index: int
    layer_type: str
    site: object
