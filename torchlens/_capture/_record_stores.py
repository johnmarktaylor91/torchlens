"""Record-store building blocks for fastlog projections.

``_GradFnContextMap`` keys record contexts by autograd node with weak references
when the node supports them, and ``_StorageBackend`` is the protocol every fastlog
storage backend implements. Both are leaf definitions: they import torch only.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from ..recording.options import _RecordingConfig
    from ..recording.types import (
        ActivationRecord,
        CaptureSpec,
        GradRecordContext,
        RecordContext,
        StorageIntent,
    )


class _GradFnContextMap:
    """Map autograd nodes to record contexts with weak keys when supported."""

    def __init__(self) -> None:
        """Initialize weak-key and object-key fallback storage."""

        self._weak: weakref.WeakKeyDictionary[Any, RecordContext] = weakref.WeakKeyDictionary()
        self._strong: dict[Any, RecordContext] = {}

    def __setitem__(self, key: Any, value: RecordContext) -> None:
        """Store a context by grad_fn_handle object."""

        try:
            self._weak[key] = value
        except TypeError:
            self._strong[key] = value

    def get(self, key: Any, default: RecordContext | None = None) -> RecordContext | None:
        """Return the context for a grad_fn_handle object, if present."""

        try:
            value = self._weak.get(key)
        except TypeError:
            value = None
        if value is not None:
            return value
        return self._strong.get(key, default)


class _StorageBackend(Protocol):
    """Protocol implemented by fastlog storage backends."""

    def append(self, record: ActivationRecord) -> None:
        """Append one retained record."""

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: _RecordingConfig,
        ctx: RecordContext | GradRecordContext | None,
        kind: str = "activation",
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads for one selected tensor."""

    def finalize(self) -> None:
        """Finalize storage."""

    def abort(self, reason: str) -> None:
        """Abort storage."""


class _RetainedRecordKeys:
    """``(raw_label, pass_number)`` keys of the retained records, kept across reads.

    The module-boundary retention step needs this set on every module exit; a
    fresh comprehension over the record list made it quadratic in model size.
    ``sync`` keys only the records appended since the last call, so the set
    equals the fresh comprehension at every read. A shorter list than last
    seen means the list was rebuilt, and the keys are rebuilt with it.
    """

    __slots__ = ("_count", "keys")

    def __init__(self) -> None:
        self.keys: set[tuple[str, int]] = set()
        self._count = 0

    def sync(self, records: Sequence[Any]) -> set[tuple[str, int]]:
        """Extend the keys over ``records`` and return the live set."""

        count = len(records)
        start = self._count
        keys = self.keys
        if count < start:
            keys.clear()
            start = 0
        for index in range(start, count):
            ctx = records[index].ctx
            if ctx.raw_label is not None:
                keys.add((ctx.raw_label, ctx.pass_number))
        self._count = count
        return keys
