"""The profiler span registry: one stack, three altitudes (torchnative W1.1).

Spans are registered AT ENTRY with identity-carrying names -- a crash
mid-span leaves an honest open record instead of nothing -- and popped in
``finally``. Every span carries ``(pid, tid, device, rank, clock_domain)``
and an owner classification including the MANDATORY named
``torchlens_internal`` bucket (TN-D11: half the markers on a real capture
are our own bookkeeping; without the bucket they would be published under
torch's name).

Timestamps are ``time.monotonic_ns()`` and stay in that clock domain; they
are never projected onto Kineto's clock (W1.4 boundary).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from ._errors import SpanError

__tl_layer__ = "L5"

#: The three span altitudes (torchnative 4.1). ``op`` covers forward
#: torch-function brackets and user regions; ``grad_fn_fire`` one autograd
#: node execution; ``aten`` the dispatch-lane primitive refinement.
ALTITUDES = ("op", "grad_fn_fire", "aten")

#: Closed owner classification. ``torchlens_internal`` is mandatory
#: vocabulary (TN-D11); ``user_region`` is the tl.region bucket; ``model``
#: is captured model work; ``unknown`` never masquerades as any of them.
OWNER_KINDS = ("model", "user_region", "torchlens_internal", "unknown")

#: Label bound: longer names are truncated with a disclosed marker so a
#: runaway f-string can never bloat every span record.
MAX_LABEL_LENGTH = 512

_SCALARS = (bool, int, float, str)


def escape_label(name: Any) -> str:
    """Return the escaped, bounded span label.

    Control characters are escaped (``repr``-style) and labels longer than
    ``MAX_LABEL_LENGTH`` are truncated with an explicit ``...[truncated]``
    marker -- disclosure, never silent.
    """

    text = str(name)
    escaped = "".join(
        ch if ch.isprintable() or ch == " " else ch.encode("unicode_escape").decode("ascii")
        for ch in text
    )
    if len(escaped) > MAX_LABEL_LENGTH:
        escaped = escaped[: MAX_LABEL_LENGTH - 14] + "...[truncated]"
    return escaped


def validate_scalar_metadata(owner: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate scalar-only span/region metadata; refuse typed otherwise."""

    validated: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(value, _SCALARS):
            raise SpanError(
                f"{owner} metadata {key!r} has non-scalar type "
                f"{type(value).__name__}; span metadata rides every span "
                "record and export row, so values must be portable scalars.",
                code="region_metadata_invalid",
                key=key,
                value_type=type(value).__name__,
                remedy="Pass bool, int, float, or str metadata values only.",
            )
        validated[str(key)] = value
    return validated


@dataclass(frozen=True)
class SpanRecord:
    """One span: identity, altitude, owner, coordinates, interval."""

    span_id: int
    name: str
    altitude: str
    owner: str
    pid: int
    tid: int
    device: str | None
    rank: int | None
    clock_domain: str
    start_ns: int
    end_ns: int | None
    parent_span_id: int | None
    leaked: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class _ThreadStacks(threading.local):
    """Per-thread open-span stack."""

    def __init__(self) -> None:
        self.stack: list[int] = []


class SpanRegistry:
    """One span registry: entry registration, finally-pop, leak closure."""

    def __init__(self, *, device: str | None = None, rank: int | None = None) -> None:
        self.device = device
        self.rank = rank
        self._lock = threading.Lock()
        self._threads = _ThreadStacks()
        self._next_id = 1
        self._open: dict[int, SpanRecord] = {}
        self._finished: list[SpanRecord] = []

    def open(
        self,
        name: str,
        *,
        altitude: str,
        owner: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Register one span at entry; returns its id for the finally-pop."""

        if altitude not in ALTITUDES:
            raise SpanError(
                f"altitude={altitude!r} is not one of {ALTITUDES}.",
                code="span_altitude_invalid",
                altitude=altitude,
                remedy=f"Use one of {ALTITUDES}.",
            )
        if owner not in OWNER_KINDS:
            raise SpanError(
                f"owner={owner!r} is not one of {OWNER_KINDS}; the owner "
                "classification (incl. the mandatory torchlens_internal "
                "bucket) is what keeps our own bookkeeping from being "
                "published under torch's name (TN-D11).",
                code="span_owner_invalid",
                owner=owner,
                remedy=f"Use one of {OWNER_KINDS}.",
            )
        validated = validate_scalar_metadata("span", metadata or {})
        label = escape_label(name)
        stack = self._threads.stack
        parent = stack[-1] if stack else None
        with self._lock:
            span_id = self._next_id
            self._next_id += 1
            record = SpanRecord(
                span_id=span_id,
                name=label,
                altitude=altitude,
                owner=owner,
                pid=os.getpid(),
                tid=threading.get_ident(),
                device=self.device,
                rank=self.rank,
                clock_domain="monotonic",
                start_ns=time.monotonic_ns(),
                end_ns=None,
                parent_span_id=parent,
                metadata=MappingProxyType(validated),
            )
            self._open[span_id] = record
        stack.append(span_id)
        return span_id

    def close(self, span_id: int, *, leaked: bool = False) -> SpanRecord:
        """Finalize one span (the finally-pop); unknown ids refuse typed."""

        with self._lock:
            record = self._open.pop(span_id, None)
            if record is None:
                raise SpanError(
                    f"span id {span_id} is not open in this registry; spans "
                    "close exactly once, in the finally of their opener.",
                    code="span_not_open",
                    span_id=span_id,
                    remedy="Close each span exactly once, from the code path that opened it.",
                )
            finished = SpanRecord(
                span_id=record.span_id,
                name=record.name,
                altitude=record.altitude,
                owner=record.owner,
                pid=record.pid,
                tid=record.tid,
                device=record.device,
                rank=record.rank,
                clock_domain=record.clock_domain,
                start_ns=record.start_ns,
                end_ns=time.monotonic_ns(),
                parent_span_id=record.parent_span_id,
                leaked=leaked,
                metadata=record.metadata,
            )
            self._finished.append(finished)
        stack = self._threads.stack
        if span_id in stack:
            stack.remove(span_id)
        return finished

    def close_leaked(self) -> int:
        """Close every still-open span with the leak disclosure.

        Called at the session boundary: a raising body can leak an open
        marker (torchnative measured exactly this on grad_fn hooks), so the
        boundary cleanup closes anything still open and DISCLOSES it rather
        than dropping or silently completing it.
        """

        with self._lock:
            leaked_ids = list(self._open)
        for span_id in leaked_ids:
            self.close(span_id, leaked=True)
        return len(leaked_ids)

    def snapshot(self) -> tuple[SpanRecord, ...]:
        """Return finished-then-open records (open spans have end_ns=None)."""

        with self._lock:
            return (*self._finished, *self._open.values())

    @property
    def open_count(self) -> int:
        """Number of currently open spans."""

        with self._lock:
            return len(self._open)


__all__ = [
    "ALTITUDES",
    "MAX_LABEL_LENGTH",
    "OWNER_KINDS",
    "SpanRecord",
    "SpanRegistry",
    "escape_label",
    "validate_scalar_metadata",
]
