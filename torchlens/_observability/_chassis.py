"""The observer/check slot: one bounded immutable event stream (checks 4.7).

This is the chassis SEAM the checks (F23) and trackers (F26) families plug
into. Checks own detection, loop-phase and scale correctness, cadence,
severity/action; trackers own buffering, aggregation, namespaces, sinks.
Neither repeats the tensor scan; both consume the one stream published here.

Event schema contract (checks memo 4.7, load-bearing):

- ``kind`` is one of ``scalar | histogram_counts | verdict | unavailable``;
  ``unavailable`` is MANDATORY vocabulary -- the structural answer to silent
  empty panels. An unavailable event must carry its reason.
- Every gradient scalar carries ``stage`` + ``scale_provenance``; without
  them a dashboard cannot label its own y-axis.
- Values are already-reduced only (floats; histograms as bucket edges +
  counts). Subscribers never trigger a device sync from here.
- A throwing subscriber CANNOT corrupt collection: failures are caught,
  counted, and reported loudly once per subscriber.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..errors._base import TorchLensWarning
from ._errors import ObserverEventError
from ._schema import GRAD_SCALE_PROVENANCE

__tl_layer__ = "L5"

#: Closed event kinds (checks 4.7).
EVENT_KINDS = ("scalar", "histogram_counts", "verdict", "unavailable")

#: Closed step-axis provenance for events (D17 + hook-only mode).
AXIS_PROVENANCE = ("explicit", "implicit", "hook_only")

#: Default stream bound: old events fall off the front; the stream is a
#: seam, not an archive -- durable history lives in the history artifact.
DEFAULT_STREAM_CAPACITY = 4096

#: Consecutive-failure count after which a subscriber is disabled with a
#: final loud warning (it can never corrupt collection either way).
SUBSCRIBER_FAILURE_LIMIT = 8


@dataclass(frozen=True)
class ObserverEvent:
    """One immutable, already-reduced observer event (checks 4.7).

    ``global_step`` is nullable (hook-only mode); local backward /
    attempted-step / accepted-step ids ride where known; ``axis_provenance``
    says which step axis authority stamped them.
    """

    key: str
    kind: str
    value: float | None = None
    bucket_limits: tuple[float, ...] | None = None
    bucket_counts: tuple[int, ...] | None = None
    verdict: str | None = None
    reason: str | None = None
    global_step: int | None = None
    backward_id: int | None = None
    attempted_step_id: int | None = None
    accepted_step_id: int | None = None
    axis_provenance: str = "hook_only"
    scope: str | None = None
    coverage: str | None = None
    gradient: bool = False
    stage: str | None = None
    scale_provenance: str | None = None

    def __post_init__(self) -> None:
        """Validate the kind-specific required fields."""

        for field_name, value, vocabulary in (
            ("kind", self.kind, EVENT_KINDS),
            ("axis_provenance", self.axis_provenance, AXIS_PROVENANCE),
        ):
            if value not in vocabulary:
                raise ObserverEventError(
                    f"ObserverEvent {field_name}={value!r} is not in the closed "
                    f"vocabulary {vocabulary}.",
                    code="observer_event_invalid",
                    field=field_name,
                    value=value,
                    remedy=f"Use one of {vocabulary}.",
                )
        if not self.key:
            raise ObserverEventError(
                "ObserverEvent needs a non-empty key; subscribers route on it.",
                code="observer_event_invalid",
                field="key",
                remedy="Pass the check/stat key.",
            )
        if self.kind == "scalar" and self.value is None:
            raise ObserverEventError(
                f"scalar event {self.key!r} has no value; missing is never "
                "zero -- emit kind='unavailable' with a reason instead.",
                code="observer_event_invalid",
                field="value",
                key=self.key,
                remedy="Set value=, or emit kind='unavailable' with reason=.",
            )
        if self.kind == "histogram_counts" and (
            self.bucket_limits is None
            or self.bucket_counts is None
            or len(self.bucket_limits) != len(self.bucket_counts) + 1
        ):
            raise ObserverEventError(
                f"histogram_counts event {self.key!r} needs bucket_limits "
                "(N+1 edges) and bucket_counts (N counts).",
                code="observer_event_invalid",
                field="bucket_counts",
                key=self.key,
                remedy="Pass device-computed edges and counts with len(edges) == len(counts)+1.",
            )
        if self.kind == "verdict" and not self.verdict:
            raise ObserverEventError(
                f"verdict event {self.key!r} carries no verdict token.",
                code="observer_event_invalid",
                field="verdict",
                key=self.key,
                remedy="Pass verdict=.",
            )
        if self.kind == "unavailable" and not self.reason:
            raise ObserverEventError(
                f"unavailable event {self.key!r} must say WHY (checks 4.2: a "
                "check that cannot run emits unavailable with a reason, never "
                "silence).",
                code="observer_event_invalid",
                field="reason",
                key=self.key,
                remedy="Pass reason= naming what was unavailable.",
            )
        if (
            self.gradient
            and self.kind == "scalar"
            and (not self.stage or self.scale_provenance not in GRAD_SCALE_PROVENANCE)
        ):
            raise ObserverEventError(
                f"gradient scalar {self.key!r} must carry stage and "
                "scale_provenance ('scaled' | 'unscaled' | 'unknown'); "
                "without them a dashboard cannot label its own y-axis "
                "(checks 4.7).",
                code="observer_event_invalid",
                field="scale_provenance",
                key=self.key,
                remedy="Stamp stage= and scale_provenance= ('unknown' is a legal value).",
            )


class EventStream:
    """Bounded append-only event stream with a subscription slot.

    Snapshot reads are immutable tuples; subscribers are notified on
    publish and can never corrupt collection: a raising subscriber is
    caught, counted, warned about once, and disabled after
    ``SUBSCRIBER_FAILURE_LIMIT`` consecutive failures.
    """

    def __init__(self, capacity: int = DEFAULT_STREAM_CAPACITY) -> None:
        self._events: deque[ObserverEvent] = deque(maxlen=capacity)
        self._subscribers: dict[int, Callable[[ObserverEvent], None]] = {}
        self._failures: dict[int, int] = {}
        self._warned: set[int] = set()
        self._next_token = 1
        self.published_total = 0

    def subscribe(self, callback: Callable[[ObserverEvent], None]) -> int:
        """Register one subscriber; returns the unsubscribe token."""

        token = self._next_token
        self._next_token += 1
        self._subscribers[token] = callback
        self._failures[token] = 0
        return token

    def unsubscribe(self, token: int) -> None:
        """Remove one subscriber; unknown tokens are a no-op."""

        self._subscribers.pop(token, None)
        self._failures.pop(token, None)
        self._warned.discard(token)

    def publish(self, event: ObserverEvent) -> None:
        """Append one event and notify subscribers, collection-safe."""

        if not isinstance(event, ObserverEvent):
            raise ObserverEventError(
                f"EventStream.publish expects an ObserverEvent, got {type(event).__name__}.",
                code="observer_event_invalid",
                remedy="Publish ObserverEvent instances; they validate at construction.",
            )
        self._events.append(event)
        self.published_total += 1
        for token, callback in list(self._subscribers.items()):
            try:
                callback(event)
                self._failures[token] = 0
            except Exception as exc:  # noqa: BLE001 -- subscriber isolation is the contract
                failures = self._failures.get(token, 0) + 1
                self._failures[token] = failures
                if token not in self._warned:
                    self._warned.add(token)
                    warnings.warn(
                        TorchLensWarning(
                            f"Observer subscriber {callback!r} raised "
                            f"{type(exc).__name__}: {exc}. Collection continues; "
                            "the subscriber will be disabled after "
                            f"{SUBSCRIBER_FAILURE_LIMIT} consecutive failures. "
                            "Remedy: fix or unsubscribe the failing subscriber; "
                            "collection is never interrupted by subscriber bugs",
                            code="observer_subscriber_failed",
                        ),
                        stacklevel=2,
                    )
                if failures >= SUBSCRIBER_FAILURE_LIMIT:
                    self._subscribers.pop(token, None)
                    warnings.warn(
                        TorchLensWarning(
                            f"Observer subscriber {callback!r} disabled after "
                            f"{failures} consecutive failures; collection was "
                            "never interrupted. "
                            "Remedy: fix the subscriber and re-subscribe it; "
                            "failure_counts() carries the per-subscriber tallies",
                            code="observer_subscriber_disabled",
                        ),
                        stacklevel=2,
                    )

    def snapshot(self) -> tuple[ObserverEvent, ...]:
        """Return an immutable view of the retained events."""

        return tuple(self._events)

    def failure_counts(self) -> dict[int, int]:
        """Return per-subscriber consecutive-failure counts (diagnostics)."""

        return dict(self._failures)


__all__ = [
    "AXIS_PROVENANCE",
    "DEFAULT_STREAM_CAPACITY",
    "EVENT_KINDS",
    "SUBSCRIBER_FAILURE_LIMIT",
    "EventStream",
    "ObserverEvent",
]
