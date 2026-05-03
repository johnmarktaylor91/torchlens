"""BufferLog and BufferAccessor: per-buffer metadata and dict-like accessor for model buffers.

BufferLog extends OpLog to represent a model buffer (e.g. BatchNorm's
``running_mean``).  Buffers participate in the computation graph just like
regular tensors but have additional identity: a ``buffer_address`` (e.g.
``"features.0.running_mean"``) and an owning module.

**Why name/address live on BufferLog, not LayerLog**: These are
buffer-specific identifiers that don't apply to general layers.  A LayerLog
is too generic — it could be any operation.  Only BufferLog entries have a
meaningful ``buffer_address``.  For single-pass buffers, the parent LayerLog
can access these fields via ``__getattr__`` delegation.
"""

import weakref
from collections.abc import Iterator
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from .._io import FieldPolicy
from ..constants import BUFFER_LOG_FIELD_ORDER
from ..utils.display import human_readable_size
from .op_log import OpLog

if TYPE_CHECKING:
    import pandas as pd

    from .model_log import Trace


def _buffer_log_to_row(buffer_log: "BufferLog") -> Dict[str, Any]:
    """Convert a BufferLog into one DataFrame row.

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


class BufferLog(OpLog):
    """A OpLog entry representing a registered model buffer.

    Subclasses OpLog and participates in the computation graph
    identically to regular tensor operations.  Adds ``name`` and
    ``address`` computed properties derived from the
    ``buffer_address`` field (inherited from OpLog).

    No additional constructor arguments — the buffer identity comes
    from the ``buffer_address`` field in the fields_dict passed to
    the parent ``OpLog.__init__``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = dict(OpLog.PORTABLE_STATE_SPEC)

    @property
    def all_buffer_addresses(self) -> list[str]:
        """Return all addresses associated with this buffer.

        Returns
        -------
        list[str]
            Buffer addresses.
        """

        return [] if self.buffer_address is None else [self.buffer_address]

    @property
    def is_shared(self) -> bool:
        """Return whether this buffer is registered at multiple addresses.

        Returns
        -------
        bool
            Whether this buffer is shared.
        """

        return len(self.all_buffer_addresses) > 1

    @property
    def name(self) -> str:
        """Buffer name (last segment of address), e.g. 'running_mean'."""
        addr = self.buffer_address
        if addr is None:
            return ""
        return cast(str, addr.rsplit(".", 1)[-1])

    @property
    def address(self) -> str:
        """Module address (everything before last dot), e.g. 'features.0'."""
        addr = self.buffer_address
        if addr is None:
            return ""
        return cast(str, addr.rsplit(".", 1)[0]) if "." in addr else ""

    def __repr__(self) -> str:
        """Multi-line summary showing address, shape, dtype, size, module, and pass number."""
        lines = [f"BufferLog: {self.buffer_address or self.layer_label}"]
        if self.shape is not None:
            lines.append(f"  shape: {list(self.shape)}")
        if self.dtype is not None:
            lines.append(f"  dtype: {self.dtype}")
        if self.memory is not None:
            lines.append(f"  size: {human_readable_size(self.memory)}")
        if self.address:
            lines.append(f"  module: {self.address}")
        if self.buffer_pass is not None:
            lines.append(f"  pass: {self.buffer_pass}")
        lines.append(f"  has_saved_outs: {self.has_saved_outs}")
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


class BufferAccessor:
    """Dict-like accessor for BufferLog objects.

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
        buffer_dict: Dict[str, "BufferLog"],
        source_trace: "Trace | None" = None,
    ) -> None:
        self._dict = buffer_dict  # address -> BufferLog
        self._list = list(buffer_dict.values())  # insertion-order list
        # Store as weakref to avoid preventing Trace GC.
        self._source_ref = weakref.ref(source_trace) if source_trace is not None else None

    def __getitem__(self, key: Union[int, str]) -> "BufferLog":
        """Retrieve a buffer by integer index, full address, or short name."""
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        # Fallback: match by short name (e.g. 'running_mean')
        matches = [bl for bl in self._list if bl.name == key]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(f"Ambiguous short name '{key}' -- use full address")
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check membership by full address or short name."""
        if key in self._dict:
            return True
        # Also check short names
        return any(bl.name == key for bl in self._list)

    def __len__(self) -> int:
        """Return the number of buffers."""
        return len(self._dict)

    def __iter__(self) -> Iterator["BufferLog"]:
        """Iterate over BufferLog objects in insertion order."""
        return iter(self._list)

    def __repr__(self) -> str:
        """Format as a dict-like string of buffer addresses with shapes and dtypes."""
        if len(self) == 0:
            return "{}"
        items = []
        for bl in self._list:
            shape_str = str(list(bl.shape)) if bl.shape is not None else "?"
            dtype_str = str(bl.dtype) if bl.dtype is not None else "?"
            items.append(f"'{bl.buffer_address}': BufferLog {shape_str} {dtype_str}")
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

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the buffer table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the buffer table to Parquet.

        Parameters
        ----------
        filepath:
            Output Parquet path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is unavailable.
        """
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc
        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_parquet(filepath, **kwargs)

    def to_json(
        self,
        filepath: str | PathLike[str],
        *,
        orient: Literal["split", "records", "index", "columns", "values", "table"] = "records",
        **kwargs: Any,
    ) -> None:
        """Write the buffer table to JSON.

        Parameters
        ----------
        filepath:
            Output JSON path.
        orient:
            JSON orientation passed to ``DataFrame.to_json``.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_json``.
        """
        self.to_pandas().to_json(filepath, orient=orient, **kwargs)
