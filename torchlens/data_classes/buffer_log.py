from typing import Dict, List, Optional, Union

from .tensor_log import TensorLog
from ..helper_funcs import human_readable_size


class BufferLog(TensorLog):
    """A TensorLog entry representing a buffer tensor.

    Subclasses TensorLog -- participates in the computation graph identically.
    Adds a focused __repr__ and convenience properties.
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
        lines = [f"BufferLog: {self.buffer_address or self.layer_label}"]
        if self.tensor_shape is not None:
            lines.append(f"  shape: {list(self.tensor_shape)}")
        if self.tensor_dtype is not None:
            lines.append(f"  dtype: {self.tensor_dtype}")
        if self.tensor_fsize is not None:
            lines.append(f"  size: {human_readable_size(self.tensor_fsize)}")
        if self.module_address:
            lines.append(f"  module: {self.module_address}")
        if self.buffer_pass is not None:
            lines.append(f"  pass: {self.buffer_pass}")
        lines.append(f"  has_saved_activations: {self.has_saved_activations}")
        if self.has_saved_grad:
            lines.append("  has_saved_grad: True")
        if self.layer_label is not None:
            lines.append(f"  layer_label: {self.layer_label}")
        return "\n".join(lines)


class BufferAccessor:
    """Dict-like accessor for BufferLog objects.

    Supports indexing by full buffer address, short name, or ordinal position.
    """

    def __init__(
        self,
        buffer_dict: Dict[str, "BufferLog"],
        source_model_log=None,
    ):
        self._dict = buffer_dict
        self._list = list(buffer_dict.values())
        self._source = source_model_log

    def __getitem__(self, key: Union[int, str]) -> "BufferLog":
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
        if key in self._dict:
            return True
        # Also check short names
        return any(bl.name == key for bl in self._list)

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._list)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "{}"
        items = []
        for bl in self._list:
            shape_str = str(list(bl.tensor_shape)) if bl.tensor_shape is not None else "?"
            dtype_str = str(bl.tensor_dtype) if bl.tensor_dtype is not None else "?"
            items.append(f"'{bl.buffer_address}': BufferLog {shape_str} {dtype_str}")
        inner = ",\n ".join(items)
        return "{" + inner + "}"
