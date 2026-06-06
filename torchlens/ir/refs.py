"""Reference dataclasses for backend-neutral captured objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import BlobRef


@dataclass(frozen=True, slots=True)
class DtypeRef:
    """Backend-neutral dtype reference."""

    backend: str
    name: str

    @classmethod
    def from_value(cls, value: Any) -> "DtypeRef | None":
        """Create a dtype reference from a backend dtype-like value.

        Parameters
        ----------
        value
            Backend dtype object or string.

        Returns
        -------
        DtypeRef | None
            Neutral dtype reference, or ``None`` when ``value`` is ``None``.
        """

        if value is None:
            return None
        text = str(value)
        backend = text.split(".", 1)[0] if "." in text else "unknown"
        return cls(backend=backend, name=text)

    def __str__(self) -> str:
        """Return the canonical dtype name."""

        return self.name


@dataclass(frozen=True, slots=True)
class DeviceRef:
    """Backend-neutral device reference."""

    backend: str
    name: str

    @classmethod
    def from_value(cls, value: Any) -> "DeviceRef | None":
        """Create a device reference from a backend device-like value.

        Parameters
        ----------
        value
            Backend device object or string.

        Returns
        -------
        DeviceRef | None
            Neutral device reference, or ``None`` when ``value`` is ``None``.
        """

        if value is None:
            return None
        text = str(value)
        backend = text.split(":", 1)[0] if ":" in text else text
        return cls(backend=backend, name=text)

    def __str__(self) -> str:
        """Return the canonical device name."""

        return self.name


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
