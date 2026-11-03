"""Record-store building blocks for fastlog projections.

``_GradFnContextMap`` keys record contexts by autograd node with weak references
when the node supports them, and ``_StorageBackend`` is the protocol every fastlog
storage backend implements. Both are leaf definitions: they import torch only.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from ..recording._options import _RecordingConfig
    from ..recording._types import (
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
