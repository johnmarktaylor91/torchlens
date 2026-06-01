"""Buffer and BufferAccessor: per-buffer metadata and dict-like accessor for model buffers.

Buffer extends Op to represent a model buffer (e.g. BatchNorm's
``running_mean``).  Buffers participate in the computation graph just like
regular tensors but have additional identity: a ``address`` (e.g.
``"features.0.running_mean"``) and an owning module.

**Why name/address live on Buffer, not Layer**: These are
buffer-specific identifiers that don't apply to general layers.  A Layer
is too generic — it could be any operation.  Only Buffer entries have a
meaningful ``address``.  For single-pass buffers, the parent Layer
can access these fields via ``__getattr__`` delegation.
"""

import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from .._io import FieldPolicy
from ..constants import BUFFER_LOG_FIELD_ORDER
from ._accessor_base import Accessor
from .op_log import Op

if TYPE_CHECKING:
    import pandas as pd

    from .model_log import Trace


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


class Buffer(Op):
    """A Op entry representing a registered model buffer.

    Subclasses Op and participates in the computation graph
    identically to regular tensor operations.  Adds ``name`` derived
    from the ``address`` field inherited from Op.

    No additional constructor arguments — the buffer identity comes
    from the ``address`` field in the fields_dict passed to
    the parent ``Op.__init__``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = dict(Op.PORTABLE_STATE_SPEC)

    @property
    def all_addresses(self) -> list[str]:
        """Return all addresses associated with this buffer.

        Returns
        -------
        list[str]
            Buffer addresses.
        """

        return [] if self.address is None else [self.address]

    @property
    def has_multiple_addresses(self) -> bool:
        """Return whether this buffer is registered at multiple addresses.

        Returns
        -------
        bool
            Whether this buffer is shared.
        """

        return len(self.all_addresses) > 1

    @property
    def buffer_overwrite_index(self) -> int | None:
        """Return this buffer's 1-based overwrite index for its address.

        Returns
        -------
        int | None
            Overwrite index, or ``None`` before buffer postprocessing assigns it.
        """

        return self.buffer_pass

    @property
    def call_index(self) -> int | None:
        """Return this buffer's 1-based call index for its address.

        Returns
        -------
        int | None
            Call index, or ``None`` before buffer postprocessing assigns it.
        """

        return self.buffer_pass

    @property
    def buffer_source_label(self) -> str | None:
        """Stable Op label for the Op that overwrote this buffer value.

        Glossary stored-form name; ``buffer_source`` resolves the Op record.
        ``None`` for static buffers (no overwriting Op).
        """

        source = self.buffer_source
        return None if source is None else str(source)

    def _address_buffers(self) -> list["Buffer"]:
        """Return all Buffer records sharing this buffer's address, in order."""

        trace = self.source_trace
        if trace is None or self.address is None:
            return [self]
        try:
            buffers = list(trace.buffers)
        except Exception:  # noqa: BLE001 - defensive: detached/partial trace
            return [self]
        same = [b for b in buffers if getattr(b, "address", None) == self.address]
        return same or [self]

    @property
    def is_overwritten(self) -> bool:
        """True when this buffer is overwritten during forward (vs. static)."""

        return self.buffer_source_label is not None or self.num_overwrites > 0

    @property
    def num_overwrites(self) -> int:
        """Total overwrites of this buffer's address during the trace.

        Counts the Buffer boundary records sharing this address that carry an
        overwriting Op source. Static buffers (no overwriting Op) yield ``0``.
        """

        return sum(1 for b in self._address_buffers() if b.buffer_source is not None)

    @property
    def last_overwrite_source_label(self) -> str | None:
        """Stable Op label of the most recent Op that overwrote this address.

        Glossary stored-form name; ``last_overwrite_source`` resolves the record.
        ``None`` for static buffers.
        """

        last: str | None = None
        for b in self._address_buffers():
            source = b.buffer_source
            if source is not None:
                last = str(source)
        return last

    @property
    def last_overwrite_source(self) -> "Op | None":
        """Op record that most recently overwrote this buffer's address, or ``None``."""

        label = self.last_overwrite_source_label
        if label is None:
            return None
        trace = self.source_trace
        if trace is None:
            return None
        try:
            return cast("Op", trace.ops[label])
        except (KeyError, TypeError):
            return None

    @property
    def name(self) -> str:
        """Buffer name (last segment of address), e.g. 'running_mean'."""
        addr = self.address
        if addr is None:
            return ""
        return cast(str, addr.rsplit(".", 1)[-1])

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Buffer as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``BUFFER_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        return pd.DataFrame([_buffer_log_to_row(self)], columns=BUFFER_LOG_FIELD_ORDER)

    def __repr__(self) -> str:
        """Multi-line summary showing address, shape, dtype, size, module, and pass number."""
        lines = [f"Buffer: {self.address or self.layer_label}"]
        if self.shape is not None:
            lines.append(f"  shape: {list(self.shape)}")
        if self.dtype is not None:
            lines.append(f"  dtype: {self.dtype}")
        if self.activation_memory is not None:
            lines.append(f"  size: {self.activation_memory}")
        if self.module:
            lines.append(f"  module: {self.module[0]}")
        if self.buffer_pass is not None:
            lines.append(f"  pass: {self.buffer_pass}")
        lines.append(f"  has_saved_activation: {self.has_saved_activation}")
        if self.has_grad:
            lines.append("  has_grad: True")
        if self.layer_label is not None:
            lines.append(f"  layer_label: {self.layer_label}")
        return "\n".join(lines)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state for this buffer log."""
        return super().__getstate__()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state for this buffer log."""
        super().__setstate__(state)


class BufferAccessor(Accessor["Buffer"]):
    """Dict-like accessor for Buffer objects.

    Supports indexing by:
    * **full buffer address** (str) -- e.g. ``"features.0.running_mean"``.
    * **short name** (str) -- e.g. ``"running_mean"`` (must be unambiguous).
    * **ordinal position** (int) -- index into insertion-order list.

    Available as ``trace.buffers`` and ``module_log.buffers``.
    """

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
        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(buffer_dict, source_ref=source_ref)

    def _resolve_substring(self, key: str) -> "Buffer | None":
        """Resolve an unambiguous buffer short name."""
        # Fallback: match by short name (e.g. 'running_mean')
        matches = [bl for bl in self._list if bl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' -- use full address")
        return None

    def __contains__(self, key: object) -> bool:
        """Check membership by full address or short name."""
        try:
            return super().__contains__(key)
        except KeyError:
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
        """Export buffer metadata as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per buffer, ordered by ``BUFFER_LOG_FIELD_ORDER``.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = [_buffer_log_to_row(buffer_log) for buffer_log in self._list]
        return pd.DataFrame(rows, columns=BUFFER_LOG_FIELD_ORDER)
