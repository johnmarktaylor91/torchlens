"""Persistent buffer entities and dict-like accessors."""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, cast

from .._io import FieldPolicy
from ..constants import BUFFER_LOG_FIELD_ORDER
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd

    from .op import Op
    from .trace import Trace


def _buffer_log_to_row(buffer_log: "Buffer") -> Dict[str, Any]:
    """Convert a Buffer into one DataFrame row.

    Parameters
    ----------
    buffer_log:
        Buffer metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value.
    """

    return {field: getattr(buffer_log, field) for field in BUFFER_LOG_FIELD_ORDER}


class Buffer:
    """Persistent metadata entity for one registered buffer address."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "address": FieldPolicy.KEEP,
        "module_address": FieldPolicy.KEEP,
        "versions": FieldPolicy.KEEP,
        "_initial_value": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        address: str,
        versions: Sequence["Op"],
        initial_value: Any | None = None,
        source_trace: "Trace | None" = None,
    ) -> None:
        """Initialize a buffer entity from graph version nodes.

        Parameters
        ----------
        address:
            Registered buffer address.
        versions:
            Ordered graph nodes representing observed buffer versions.
        source_trace:
            Trace that owns the graph nodes.
        """

        self.address = address
        self.module_address = address.rsplit(".", 1)[0] if "." in address else ""
        self.versions = list(versions)
        self._initial_value = initial_value
        self._source_ref = weakref.ref(source_trace) if source_trace is not None else None

    @property
    def source_trace(self) -> "Trace | None":
        """Return the owning trace if it is still alive."""

        if self._source_ref is None:
            return None
        return self._source_ref()

    @property
    def trace(self) -> "Trace | None":
        """Compatibility alias for ``source_trace``."""

        return self.source_trace

    @property
    def handle(self) -> Any | None:
        """Return the live registered buffer tensor when reachable.

        Returns
        -------
        Any | None
            Live torch buffer object from the source model, or ``None`` when
            this computed runtime handle is unavailable or non-portable.
        """

        trace = self.source_trace
        source_ref = getattr(trace, "_source_model_ref", None) if trace is not None else None
        if source_ref is None:
            return None
        model = source_ref()
        if model is None:
            return None
        try:
            return dict(model.named_buffers()).get(self.address)
        except Exception:
            return None

    @property
    def name(self) -> str:
        """Buffer name, the final component of ``address``."""

        return self.address.rsplit(".", 1)[-1]

    @property
    def all_addresses(self) -> list[str]:
        """Return all registered addresses represented by this entity."""

        return [self.address]

    @property
    def has_multiple_addresses(self) -> bool:
        """Return whether this entity represents shared registered addresses."""

        return len(self.all_addresses) > 1

    @property
    def initial_value(self) -> Any:
        """Initial observed buffer value."""

        if self._initial_value is not None:
            return self._initial_value
        return None if not self.versions else self.versions[0].out

    @property
    def final_value(self) -> Any:
        """Final observed buffer value."""

        return None if not self.versions else self.versions[-1].out

    @property
    def num_overwrites(self) -> int:
        """Number of write events recorded for this buffer address."""

        return sum(1 for version in self.versions if version.buffer_write_kind is not None)

    @property
    def is_overwritten(self) -> bool:
        """Return whether the buffer was written during the forward pass."""

        return self.num_overwrites > 0

    @property
    def write_versions(self) -> list["Op"]:
        """Return versions produced by writes, excluding static initial reads."""

        return [version for version in self.versions if version.buffer_write_kind is not None]

    @property
    def buffer_overwrite_index(self) -> int | None:
        """Return the most recent version's overwrite index."""

        return self.buffer_pass

    @property
    def buffer_pass(self) -> int | None:
        """Return the most recent graph version pass index."""

        return None if not self.versions else self.versions[-1].buffer_pass

    @property
    def call_index(self) -> int | None:
        """Return the most recent graph version call index."""

        return self.buffer_pass

    @property
    def layer_label(self) -> str | None:
        """Return the most recent graph version layer label."""

        return None if not self.versions else self.versions[-1].layer_label

    @property
    def shape(self) -> Any:
        """Return the final observed buffer shape."""

        return None if not self.versions else self.versions[-1].shape

    @property
    def dtype(self) -> Any:
        """Return the final observed buffer dtype."""

        return None if not self.versions else self.versions[-1].dtype

    @property
    def activation_memory(self) -> Any:
        """Return final observed buffer activation memory."""

        return None if not self.versions else self.versions[-1].activation_memory

    @property
    def has_saved_activation(self) -> bool:
        """Return whether the final version has a saved activation."""

        return bool(self.versions and self.versions[-1].has_saved_activation)

    @property
    def has_grad(self) -> bool:
        """Return whether the final version has a saved gradient."""

        return bool(self.versions and self.versions[-1].has_grad)

    @property
    def grad_shape(self) -> Any:
        """Return final observed gradient shape."""

        return None if not self.versions else self.versions[-1].grad_shape

    @property
    def grad_dtype(self) -> Any:
        """Return final observed gradient dtype."""

        return None if not self.versions else self.versions[-1].grad_dtype

    @property
    def gradient_memory(self) -> Any:
        """Return final observed gradient memory."""

        return None if not self.versions else self.versions[-1].gradient_memory

    @property
    def buffer_source(self) -> str | None:
        """Return the most recent write producer label, if any."""

        for version in reversed(self.versions):
            if version.buffer_source is not None:
                return cast(str, version.buffer_source)
        return None

    @property
    def module(self) -> Any:
        """Return the final graph version's module field."""

        return None if not self.versions else self.versions[-1].module

    @property
    def modules(self) -> Any:
        """Return the final graph version's module stack."""

        return [] if not self.versions else self.versions[-1].modules

    def value_at(self, version_index: int) -> Any:
        """Return the value at a 1-based buffer version index."""

        return self.versions[version_index - 1].out

    def value_after(self, overwrite_index: int) -> Any:
        """Return the value after a 1-based overwrite index."""

        writes = self.write_versions
        return writes[overwrite_index - 1].out

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Buffer as a one-row pandas DataFrame."""

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        return pd.DataFrame([_buffer_log_to_row(self)], columns=BUFFER_LOG_FIELD_ORDER)

    def __repr__(self) -> str:
        """Return a concise multi-line buffer summary."""

        lines = [f"Buffer: {self.address}"]
        if self.shape is not None:
            lines.append(f"  shape: {list(self.shape)}")
        if self.dtype is not None:
            lines.append(f"  dtype: {self.dtype}")
        lines.append(f"  versions: {len(self.versions)}")
        lines.append(f"  num_overwrites: {self.num_overwrites}")
        return "\n".join(lines)


class BufferAccessor(Accessor["Buffer"]):
    """Dict-like accessor for persistent Buffer entities."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        buffer_dict: Dict[str, "Buffer"],
        source_trace: "Trace | None" = None,
    ) -> None:
        """Initialize the accessor from address-keyed buffer entities."""

        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(buffer_dict, source_ref=source_ref)

    def _resolve_substring(self, key: str) -> "Buffer | None":
        """Resolve an unambiguous buffer short name."""

        matches = [bl for bl in self._list if bl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' -- use full address")
        return None

    def _resolve_pass_qualified(self, key: str) -> "Buffer | None":
        """Resolve pass-qualified notation to the parent Buffer."""

        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        if base in self._dict:
            return self._dict[base]
        return self._resolve_substring(base)

    def __contains__(self, key: object) -> bool:
        """Check membership by full address or short name."""

        try:
            self[key]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            return False
        return True

    def __repr__(self) -> str:
        """Format as a dict-like string of buffer addresses with shapes and dtypes."""

        if len(self) == 0:
            return "{}"
        items = []
        for bl in self._list:
            shape_str = str(list(bl.shape)) if bl.shape is not None else "?"
            dtype_str = str(bl.dtype) if bl.dtype is not None else "?"
            items.append(f"'{bl.address}': Buffer {shape_str} {dtype_str}")
        inner = ",\n ".join(items)
        return "{" + inner + "}"

    def to_pandas(self) -> "pd.DataFrame":
        """Export buffer metadata as a pandas DataFrame."""

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [_buffer_log_to_row(buffer_log) for buffer_log in self._list]
        return pd.DataFrame(rows, columns=BUFFER_LOG_FIELD_ORDER)
