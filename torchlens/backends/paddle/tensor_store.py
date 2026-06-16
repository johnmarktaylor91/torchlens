"""External metadata store for Paddle tensors."""

from __future__ import annotations

import weakref
from typing import Any


class PaddleTensorLabelStore:
    """Side table mapping Paddle tensor identities to raw TorchLens labels."""

    def __init__(self) -> None:
        """Initialize an empty label store."""

        self._labels: dict[int, str] = {}
        self._live_tensors: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
        self._finalizers: dict[int, weakref.finalize[Any, Any]] = {}

    def set_label(self, tensor: object, label: str) -> None:
        """Associate a raw TorchLens label with a Paddle tensor.

        Parameters
        ----------
        tensor
            Paddle tensor object.
        label
            Raw TorchLens label.
        """

        tensor_id = id(tensor)
        self._labels[tensor_id] = label
        try:
            self._live_tensors[tensor_id] = tensor
            self._finalizers[tensor_id] = weakref.finalize(tensor, self.discard_id, tensor_id)
        except TypeError:
            self._finalizers.pop(tensor_id, None)

    def set_label_if_unlabeled(self, tensor: object, label: str) -> bool:
        """Associate ``label`` only when ``tensor`` is not already labeled.

        Parameters
        ----------
        tensor
            Paddle tensor object.
        label
            Raw TorchLens label.

        Returns
        -------
        bool
            True when a new label was stored.
        """

        if self.is_labeled(tensor):
            return False
        self.set_label(tensor, label)
        return True

    def get_label(self, tensor: object) -> str | None:
        """Return the raw label associated with a Paddle tensor, if any.

        Parameters
        ----------
        tensor
            Paddle tensor object.

        Returns
        -------
        str | None
            Stored raw label or ``None``.
        """

        return self._labels.get(id(tensor))

    def is_labeled(self, tensor: object) -> bool:
        """Return whether ``tensor`` already has a side-table label.

        Parameters
        ----------
        tensor
            Paddle tensor object.

        Returns
        -------
        bool
            True when a label exists.
        """

        return id(tensor) in self._labels

    def discard_id(self, tensor_id: int) -> None:
        """Remove metadata for a dead tensor identity.

        Parameters
        ----------
        tensor_id
            Python object identity to remove.
        """

        self._labels.pop(tensor_id, None)
        self._live_tensors.pop(tensor_id, None)
        self._finalizers.pop(tensor_id, None)

    def clear(self) -> None:
        """Clear all labels and weak references."""

        for finalizer in self._finalizers.values():
            finalizer.detach()
        self._labels.clear()
        self._live_tensors.clear()
        self._finalizers.clear()


__all__ = ["PaddleTensorLabelStore"]
