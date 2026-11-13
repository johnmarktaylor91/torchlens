"""Capture-time retention for the TensorFlow eager path.

The eager session needs every op output's identity while the forward runs (to
link a consumer's inputs to their producers), but not the live tensor itself.
Keeping the live tensor, or a ``tensor.ref()`` key that holds it, pins every
activation for the whole capture, on top of the NumPy payload snapshot.

This module decides what a capture keeps:

* :class:`WeakProducerIndex` maps a tensor to its producer label without
  holding the tensor. An entry dies with its tensor, so a later tensor that
  reuses the address never inherits the old label.
* :func:`retention_policy` says, per op record, whether its live tensors must
  survive the forward (later backward passes and the derived-gradient tap need
  them; nothing else does).
* :func:`release_live_tensors` drops a record's live tensors once its edges are
  recorded, keeping literal inputs, which replay reads as data.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import replace
from typing import Any


class WeakProducerIndex(MutableMapping[object, str]):
    """Producer labels keyed by TensorFlow ``tensor.ref()`` keys, holding tensors weakly.

    Callers keep using ``tensor.ref()`` keys (``index.get(tensor.ref())``); the
    index stores the referenced tensor's address and a weak reference, never
    the key. Keys that do not reference a weak-referenceable tensor are kept
    as given.
    """

    def __init__(self) -> None:
        self._by_id: dict[int, tuple[weakref.ref[Any], str]] = {}
        self._strong: dict[object, str] = {}

    @staticmethod
    def _referent(key: object) -> Any:
        """The tensor ``key`` references, or ``None`` for a plain key."""

        deref = getattr(key, "deref", None)
        return deref() if callable(deref) else None

    def _forget(self, ident: int) -> Callable[[weakref.ref[Any]], None]:
        """Weak-reference callback removing the entry for a tensor that died."""

        def forget(dead: weakref.ref[Any]) -> None:
            entry = self._by_id.get(ident)
            if entry is not None and entry[0] is dead:
                del self._by_id[ident]

        return forget

    def __setitem__(self, key: object, label: str) -> None:
        tensor = self._referent(key)
        try:
            ref = weakref.ref(tensor, self._forget(id(tensor))) if tensor is not None else None
        except TypeError:
            ref = None
        if ref is None:
            self._strong[key] = label
            return
        self._by_id[id(tensor)] = (ref, label)

    def __getitem__(self, key: object) -> str:
        tensor = self._referent(key)
        if tensor is not None:
            entry = self._by_id.get(id(tensor))
            if entry is not None and entry[0]() is tensor:
                return entry[1]
        return self._strong[key]

    def __delitem__(self, key: object) -> None:
        tensor = self._referent(key)
        if tensor is not None and id(tensor) in self._by_id:
            del self._by_id[id(tensor)]
            return
        del self._strong[key]

    def __iter__(self) -> Iterator[object]:
        for ref, _label in list(self._by_id.values()):
            tensor = ref()
            if tensor is not None:
                yield tensor.ref()
        yield from list(self._strong)

    def __len__(self) -> int:
        return sum(1 for ref, _ in self._by_id.values() if ref() is not None) + len(self._strong)


def retention_policy(session: Any) -> Callable[[Any], bool]:
    """Whether an op record's live tensors must outlive the forward.

    Parameters
    ----------
    session
        The eager capture session. ``retain_live_tensors`` (default ``True``)
        is turned off by the capture entry when no backward pass will run; an
        ``output_tap`` (the derived-gradient replay) always needs them.

    Returns
    -------
    Callable[[Any], bool]
        Predicate over a ``TFOpCapture``.
    """

    keep = bool(getattr(session, "retain_live_tensors", True)) or (
        getattr(session, "output_tap", None) is not None
    )
    return lambda _op_capture: keep


def release_live_tensors(op_capture: Any) -> Any:
    """Return ``op_capture`` without its live output and producer-linked input tensors.

    Literal inputs (no producer and no source label) keep their tensor, since
    replay validation reads them as data; every other input is rebuilt from
    its parent's saved payload.
    """

    inputs = tuple(
        item
        if item.producer_label_raw is None and item.source_label_raw is None
        else replace(item, tensor=None, ref_key=None)
        for item in op_capture.inputs
    )
    return replace(op_capture, output_tensor=None, inputs=inputs)


__all__ = ("WeakProducerIndex", "release_live_tensors", "retention_policy")
