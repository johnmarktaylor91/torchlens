"""BufferLog and BufferAccessor: per-buffer metadata and dict-like accessor for model buffers.

BufferLog extends LayerPassLog to represent a model buffer (e.g. BatchNorm's
``running_mean``).  Buffers participate in the computation graph just like
regular tensors but have additional identity: a ``buffer_address`` (e.g.
``"features.0.running_mean"``) and an owning module.

**Why name/module_address live on BufferLog, not LayerLog**: These are
buffer-specific identifiers that don't apply to general layers.  A LayerLog
is too generic — it could be any operation.  Only BufferLog entries have a
meaningful ``buffer_address``.  For single-pass buffers, the parent LayerLog
can access these fields via ``__getattr__`` delegation.
"""

import weakref
from typing import Dict, List, Optional, Union

from .layer_pass_log import LayerPassLog
from ..utils.display import human_readable_size


class BufferLog(LayerPassLog):
    """A LayerPassLog entry representing a registered model buffer.

    Subclasses LayerPassLog and participates in the computation graph
    identically to regular tensor operations.  Adds ``name`` and
    ``module_address`` computed properties derived from the
    ``buffer_address`` field (inherited from LayerPassLog).

    No additional constructor arguments — the buffer identity comes
    from the ``buffer_address`` field in the fields_dict passed to
    the parent ``LayerPassLog.__init__``.
    """

    @property
    def name(self) -> str:
        """Buffer name (last segment of address), e.g. 'running_mean'."""
        addr = self.buffer_address
        if addr is None:
            return ""
        return addr.rsplit(".", 1)[-1]

    @property
    def module_address(self) -> str:
        """Module address (everything before last dot), e.g. 'features.0'."""
        addr = self.buffer_address
        if addr is None:
            return ""
        return addr.rsplit(".", 1)[0] if "." in addr else ""

    def __repr__(self) -> str:
        """Multi-line summary showing address, shape, dtype, size, module, and pass number."""
        lines = [f"BufferLog: {self.buffer_address or self.layer_label}"]
        if self.tensor_shape is not None:
            lines.append(f"  shape: {list(self.tensor_shape)}")
        if self.tensor_dtype is not None:
            lines.append(f"  dtype: {self.tensor_dtype}")
        if self.tensor_memory is not None:
            lines.append(f"  size: {human_readable_size(self.tensor_memory)}")
        if self.module_address:
            lines.append(f"  module: {self.module_address}")
        if self.buffer_pass is not None:
            lines.append(f"  pass: {self.buffer_pass}")
        lines.append(f"  has_saved_activations: {self.has_saved_activations}")
        if self.has_gradient:
            lines.append("  has_gradient: True")
        if self.layer_label is not None:
            lines.append(f"  layer_label: {self.layer_label}")
        return "\n".join(lines)


class BufferAccessor:
    """Dict-like accessor for BufferLog objects.

    Supports indexing by:
    * **full buffer address** (str) -- e.g. ``"features.0.running_mean"``.
    * **short name** (str) -- e.g. ``"running_mean"`` (must be unambiguous).
    * **ordinal position** (int) -- index into insertion-order list.

    Available as ``model_log.buffers`` and ``module_log.buffers``.
    """

    def __init__(
        self,
        buffer_dict: Dict[str, "BufferLog"],
        source_model_log=None,
    ) -> None:
        self._dict = buffer_dict  # address -> BufferLog
        self._list = list(buffer_dict.values())  # insertion-order list
        # Store as weakref to avoid preventing ModelLog GC.
        self._source_ref = weakref.ref(source_model_log) if source_model_log is not None else None

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

    def __iter__(self):
        """Iterate over BufferLog objects in insertion order."""
        return iter(self._list)

    def __repr__(self) -> str:
        """Format as a dict-like string of buffer addresses with shapes and dtypes."""
        if len(self) == 0:
            return "{}"
        items = []
        for bl in self._list:
            shape_str = str(list(bl.tensor_shape)) if bl.tensor_shape is not None else "?"
            dtype_str = str(bl.tensor_dtype) if bl.tensor_dtype is not None else "?"
            items.append(f"'{bl.buffer_address}': BufferLog {shape_str} {dtype_str}")
        inner = ",\n ".join(items)
        return "{" + inner + "}"
