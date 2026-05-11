"""External metadata store for MLX arrays.

MLX arrays do not expose Torch-style arbitrary Python attributes, so the MLX
backend keeps TorchLens labels in a side table keyed by object identity.
"""

from __future__ import annotations

import weakref
from typing import Any


class MLXTensorLabelStore:
    """Side table mapping MLX array identities to raw TorchLens labels."""

    def __init__(self) -> None:
        """Initialize an empty label store."""

        self._labels: dict[int, str] = {}
        self._live_arrays: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        self._finalizers: dict[int, weakref.finalize[Any, Any]] = {}

    def set_label(self, array: object, label: str) -> None:
        """Associate a raw TorchLens label with an MLX array.

        Parameters
        ----------
        array:
            MLX array object.
        label:
            Raw TorchLens label.
        """

        array_id = id(array)
        self._labels[array_id] = label
        try:
            self._live_arrays[array_id] = array
            self._finalizers[array_id] = weakref.finalize(array, self.discard_id, array_id)
        except TypeError:
            # Keep the label for this capture if a future MLX array build is not weakrefable.
            self._finalizers.pop(array_id, None)

    def get_label(self, array: object) -> str | None:
        """Return the raw label associated with an MLX array, if any.

        Parameters
        ----------
        array:
            MLX array object.

        Returns
        -------
        str | None
            Stored raw label or ``None``.
        """

        return self._labels.get(id(array))

    def discard_id(self, array_id: int) -> None:
        """Remove metadata for a dead array identity.

        Parameters
        ----------
        array_id:
            Python object identity to remove.
        """

        self._labels.pop(array_id, None)
        self._live_arrays.pop(array_id, None)
        self._finalizers.pop(array_id, None)

    def clear(self) -> None:
        """Clear all labels and weak references."""

        for finalizer in self._finalizers.values():
            finalizer.detach()
        self._labels.clear()
        self._live_arrays.clear()
        self._finalizers.clear()


__all__ = ["MLXTensorLabelStore"]
