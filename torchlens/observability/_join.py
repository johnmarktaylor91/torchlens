"""The correlation-ID device-time join (torchnative W2.2 / W2.0 consumers).

Runtime correlation IDs are the ONLY device join; names are display
metadata. A launch is counted once -- several owners form a launch group,
never split evenly, never duplicated into additive rows. Kernels, copies,
and memsets stay separate; device-busy elapsed time is the interval union
per device; missing is ``None``, never zero.

The joiner is a pure function over normalized events (``_kineto``) plus an
optional trace for owner resolution, so every correctness property is
testable on synthetic fixtures with hand-set thread ids (the ONLY sanctioned
cross-thread negative control -- torch's profiler records nothing on a plain
Python thread, so a live-thread control would pass on an empty trace,
TN-D19).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any

from ._errors import ProfilerSessionError
from ._kineto import NormalizedEvent

__tl_layer__ = "L5"

#: Marker-name prefixes minted by TorchLens (exact keys, opaque to users;
#: user ``record_function`` labels cannot forge them because owner
#: resolution demands the user-annotation flag AND an exact parse).
OP_MARKER_PREFIX = "torchlens::op::"
INTERNAL_MARKER_PREFIX = "torchlens::internal::"
REGION_MARKER_PREFIX = "torchlens::region::"
GRADFN_MARKER_PREFIX = "torchlens::gradfn::"

#: Closed per-row attribution states (torchnative 4.1 rule 4).
ROW_STATUS = ("attributed", "ambiguous", "unattributed", "unavailable")

#: Device event kinds joined into launch rows (closed; frozenset for the
#: per-event membership test on the partition hot path).
_DEVICE_KINDS = frozenset(("kernel", "memcpy", "memset"))

#: Closed owner classes (forward wave; the backward taxonomy extends this
#: set with its seven evidence-bearing classes as FLIP-2 links land).
OWNER_CLASSES = (
    "forward_op",
    "grad_fn",
    "user_region",
    "torchlens_internal",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class MarkerSpan:
    """One TorchLens marker interval parsed from the event stream."""

    name: str
    owner_class: str
    owner_key: str
    start_ns: int
    end_ns: int
    tid: int | None


@dataclass(frozen=True, slots=True)
class LaunchRow:
    """One device event (kernel / memcpy / memset), counted exactly once."""

    launch_id: int
    kind: str
    name: str
    device_type: str
    device_index: int | None
    stream: int | str | None
    start_ns: int
    end_ns: int
    correlation_id: int | None
    status: str
    owners: tuple[str, ...]
    owner_labels: tuple[str, ...]


@dataclass(frozen=True)
class JoinCoverage:
    """The coverage ledger: availability, shares, and the NAMED remainder.

    ``device_busy_ns`` is the interval union of device events per device;
    inclusive columns are visibly non-additive. The two-part accounting
    (section 4.2) publishes exact model attribution against (a) device
    activity minus typed TorchLens-internal work and (b) total device
    activity with internal work owned -- overhead is excluded only when
    PROVEN by typed internal spans, never inferred from launch names.
    """

    availability: str
    reason: str | None
    device_busy_ns: dict[str, int]
    attributed_ns: dict[str, int]
    internal_ns: dict[str, int]
    unattributed_ns: dict[str, int]
    owner_share_ns: dict[str, int]
    residual: tuple[tuple[str, int], ...]
    extraction_path: str = "unavailable"

    def accounting(self, device: str) -> tuple[float | None, float | None]:
        """Return the two-part accounting for one device, or Nones.

        Returns
        -------
        tuple[float | None, float | None]
            ``(exact_over_model_activity, owned_over_total)`` -- part (a)
            excludes typed internal work from the denominator; part (b)
            counts it as owned. ``None`` when the device recorded nothing.
        """

        total = self.device_busy_ns.get(device)
        if not total:
            return (None, None)
        attributed = self.attributed_ns.get(device, 0)
        internal = self.internal_ns.get(device, 0)
        model_denominator = total - internal
        part_a = (attributed / model_denominator) if model_denominator > 0 else None
        part_b = (attributed + internal) / total
        return (part_a, part_b)


@dataclass(frozen=True)
class KinetoJoinResult:
    """The joined result: markers, launches, relations, coverage."""

    availability: str
    markers: tuple[MarkerSpan, ...]
    launches: tuple[LaunchRow, ...]
    coverage: JoinCoverage
    op_device_ns: dict[str, int] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _collection_paused() -> Iterator[None]:
    """Pause cyclic GC for the join's allocation burst (fork-build precedent).

    The join allocates a handful of tracked containers per event and frees
    none of them until the result is returned, so a mid-join collection is a
    young-gen (and periodically full-heap) scan whose cost scales with the
    WHOLE process heap rather than the event stream. Pausing keeps the join's
    cost a function of its input alone; state restores exactly, and a caller
    who already disabled gc keeps it disabled.
    """

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    """Return the total length of the union of half-open intervals."""

    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def parse_marker(event: NormalizedEvent) -> MarkerSpan | None:
    """Parse one TorchLens marker event; foreign annotations return None.

    Foreign ``record_function`` labels cannot forge TorchLens markers: the
    event must carry the profiler's user-annotation flag AND parse exactly
    against one of the minted prefixes.
    """

    if not event.is_user_annotation:
        return None
    name = event.name
    if name.startswith(OP_MARKER_PREFIX):
        owner_class, key = "forward_op", name[len(OP_MARKER_PREFIX) :]
    elif name.startswith(GRADFN_MARKER_PREFIX):
        owner_class, key = "grad_fn", name[len(GRADFN_MARKER_PREFIX) :]
    elif name.startswith(REGION_MARKER_PREFIX):
        owner_class, key = "user_region", name[len(REGION_MARKER_PREFIX) :]
    elif name.startswith(INTERNAL_MARKER_PREFIX):
        owner_class, key = "torchlens_internal", name[len(INTERNAL_MARKER_PREFIX) :]
    else:
        return None
    return MarkerSpan(
        name=name,
        owner_class=owner_class,
        owner_key=key,
        start_ns=event.start_ns,
        end_ns=event.end_ns,
        tid=event.tid,
    )


def _innermost_owner_by_runtime(
    markers: tuple[MarkerSpan, ...], runtime_events: list[NormalizedEvent]
) -> dict[int, set[MarkerSpan]]:
    """Map correlation id -> innermost same-thread enclosing marker(s).

    One O(N log N) sweep per thread: marker open/close boundaries and
    runtime-event start points are merged in time order; a stack of open
    markers makes the innermost owner a stack top. Marker spans and runtime
    events on DIFFERENT threads never match (the cross-thread
    misattribution defect class, TN-D2). A correlation id reached through
    SEVERAL runtime events can accumulate several distinct owners -- the
    caller publishes those rows as ``ambiguous``, never split or
    duplicated.
    """

    by_tid: dict[int | None, list[tuple[int, Any]]] = {}
    # Boundary tuples: (time_ns * 4 + kind, payload). Kind order at equal
    # time: open(0) < close(1) < runtime(2), so a runtime start exactly at
    # a marker's open joins it and one exactly at its close does not. The
    # historical seq tiebreaker is replaced by sort STABILITY: buckets are
    # appended in seq order and the key-only stable sort preserves it, so
    # payloads are never compared. Runtime payloads are the correlation id
    # itself (pre-filtered non-None); marker payloads are the span.
    for marker in markers:
        bucket = by_tid.get(marker.tid)
        if bucket is None:
            bucket = by_tid[marker.tid] = []
        bucket.append((marker.start_ns * 4, marker))
        bucket.append((marker.end_ns * 4 + 1, marker))
    for event in runtime_events:
        if event.correlation_id is None:
            continue
        bucket = by_tid.get(event.tid)
        if bucket is None:
            bucket = by_tid[event.tid] = []
        bucket.append((event.start_ns * 4 + 2, event.correlation_id))
    owners: dict[int, set[MarkerSpan]] = {}
    key_of = itemgetter(0)
    for boundaries in by_tid.values():
        boundaries.sort(key=key_of)
        stack: list[MarkerSpan] = []
        stack_append = stack.append
        owners_get = owners.get
        for key, payload in boundaries:
            kind = key & 3
            if kind == 0:
                stack_append(payload)
            elif kind == 1:
                # Well-nested spans close at the stack top; the linear scan
                # stays as the fallback for overlapping (non-nested) spans.
                if stack and stack[-1] is payload:
                    stack.pop()
                elif payload in stack:
                    stack.remove(payload)
            elif stack:
                owner_set = owners_get(payload)
                if owner_set is None:
                    owner_set = owners[payload] = set()
                owner_set.add(stack[-1])
    return owners


def _owner_disposition(
    owner_set: set[MarkerSpan], label_by_call_id: dict[str, str]
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Classify one launch's owner set (counted once, never split).

    Returns
    -------
    tuple[str, tuple[str, ...], tuple[str, ...]]
        ``(status, owner_classes, owner_labels)``; an empty owner set is
        ``unattributed`` and several distinct owners are one ``ambiguous``
        launch group.
    """

    def _label(marker: MarkerSpan) -> str:
        """Resolve one marker to its display label (op key -> op label)."""

        if marker.owner_class == "forward_op":
            return label_by_call_id.get(marker.owner_key, marker.owner_key)
        return marker.owner_key

    if not owner_set:
        return "unattributed", (), ()
    if len(owner_set) == 1:
        owner = next(iter(owner_set))
        return "attributed", (owner.owner_class,), (_label(owner),)
    if len({marker.owner_key for marker in owner_set}) > 1:
        group = sorted(owner_set, key=lambda marker: marker.owner_key)
        return (
            "ambiguous",
            tuple(marker.owner_class for marker in group),
            tuple(_label(marker) for marker in group),
        )
    owner = next(iter(owner_set))
    return "attributed", (owner.owner_class,), (_label(owner),)


def _availability_verdict(
    events: tuple[NormalizedEvent, ...],
    device_events: list[NormalizedEvent],
    markers: tuple[MarkerSpan, ...],
) -> tuple[str, str | None]:
    """Settle the five-state availability plus its reason (rule 4)."""

    if not events:
        return "empty", "the profiler session recorded no events"
    if not device_events:
        return (
            "empty",
            "no device activity recorded (CPU-only session or no kernels "
            "launched); device cells render '-'",
        )
    if not markers:
        return (
            "partial",
            "device activity recorded but no TorchLens markers found in the "
            "stream (if the model ran on a different thread than the "
            "session: torch's profiler is thread-scoped -- open the session "
            "on the thread that runs the model)",
        )
    return "joined", None


def join_events(
    events: tuple[NormalizedEvent, ...],
    *,
    extraction_path: str,
    trace: Any | None = None,
) -> KinetoJoinResult:
    """Join device events to TorchLens markers by runtime correlation only.

    Parameters
    ----------
    events:
        Normalized events from :func:`torchlens.observability._kineto.
        extract_events`.
    extraction_path:
        Which extraction path produced the events (disclosed on coverage).
    trace:
        Optional finished Trace; when given, ``forward_op`` marker keys
        (persisted ``func_call_id``) resolve to pass-qualified op labels and
        the per-op device table is populated.

    Returns
    -------
    KinetoJoinResult
        Marker spans, single-count launch rows, the coverage ledger, and
        (trace-resolved) per-op device nanoseconds.
    """

    if extraction_path == "unavailable":
        coverage = JoinCoverage(
            availability="unavailable",
            reason="profiler event extraction unavailable on this build/session",
            device_busy_ns={},
            attributed_ns={},
            internal_ns={},
            unattributed_ns={},
            owner_share_ns={},
            residual=(),
            extraction_path=extraction_path,
        )
        return KinetoJoinResult(
            availability="unavailable", markers=(), launches=(), coverage=coverage
        )

    with _collection_paused():
        # The is_user_annotation prefilter is parse_marker's own first check,
        # hoisted so non-annotation events skip the call entirely.
        markers = tuple(
            m for e in events if e.is_user_annotation and (m := parse_marker(e)) is not None
        )
        runtime_events = [e for e in events if e.kind == "runtime"]
        device_events = [e for e in events if e.kind in _DEVICE_KINDS]
        owners_by_correlation = _innermost_owner_by_runtime(markers, runtime_events)

        label_by_call_id: dict[str, str] = {}
        if trace is not None:
            for op in getattr(trace, "ops", ()):  # pragma: no branch
                call_id = getattr(op, "func_call_id", None)
                label = getattr(op, "label", None)
                if call_id is not None and label is not None:
                    label_by_call_id[str(call_id)] = str(label)

        launches: list[LaunchRow] = []
        busy: dict[str, list[tuple[int, int]]] = {}
        attributed: dict[str, list[tuple[int, int]]] = {}
        internal: dict[str, list[tuple[int, int]]] = {}
        unattributed: dict[str, list[tuple[int, int]]] = {}
        owner_share: Counter[str] = Counter()
        residual: Counter[str] = Counter()
        op_device_ns: Counter[str] = Counter()

        # A correlation id's owner set is fixed for the whole join, so its
        # disposition is computed once and reused across every device event
        # that shares the id (O(1) amortized on the per-launch hot path).
        _EMPTY_DISPOSITION: tuple[str, tuple[str, ...], tuple[str, ...]] = (
            "unattributed",
            (),
            (),
        )
        disposition_by_correlation: dict[int, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
        device_key_cache: dict[tuple[str, int | None], str] = {}

        launches_append = launches.append
        for index, event in enumerate(device_events):
            device_key = (event.device_type, event.device_index)
            device = device_key_cache.get(device_key)
            if device is None:
                device = device_key_cache[device_key] = f"{event.device_type}:{event.device_index}"
            start_ns = event.start_ns
            end_ns = event.end_ns
            interval = (start_ns, end_ns)
            busy.setdefault(device, []).append(interval)
            correlation = event.correlation_id
            if correlation is None:
                status, owner_classes, owner_labels = _EMPTY_DISPOSITION
            else:
                disposition = disposition_by_correlation.get(correlation)
                if disposition is None:
                    disposition = _owner_disposition(
                        owners_by_correlation.get(correlation, set()), label_by_call_id
                    )
                    disposition_by_correlation[correlation] = disposition
                status, owner_classes, owner_labels = disposition
            if status == "unattributed":
                unattributed.setdefault(device, []).append(interval)
                residual[event.name] += 1
            elif owner_classes == ("torchlens_internal",):
                internal.setdefault(device, []).append(interval)
                owner_share["torchlens_internal"] += end_ns - start_ns
            else:
                attributed.setdefault(device, []).append(interval)
                if status == "ambiguous":
                    owner_share["launch_group"] += end_ns - start_ns
                else:
                    owner_share[owner_classes[0]] += end_ns - start_ns
                    if owner_classes[0] == "forward_op":
                        op_device_ns[owner_labels[0]] += end_ns - start_ns
            launches_append(
                LaunchRow(
                    launch_id=index,
                    kind=event.kind,
                    name=event.name,
                    device_type=event.device_type,
                    device_index=event.device_index,
                    stream=event.stream,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    correlation_id=correlation,
                    status=status,
                    owners=owner_classes,
                    owner_labels=owner_labels,
                )
            )

    availability, reason = _availability_verdict(events, device_events, markers)

    coverage = JoinCoverage(
        availability=availability,
        reason=reason,
        device_busy_ns={d: _interval_union_ns(v) for d, v in busy.items()},
        attributed_ns={d: _interval_union_ns(v) for d, v in attributed.items()},
        internal_ns={d: _interval_union_ns(v) for d, v in internal.items()},
        unattributed_ns={d: _interval_union_ns(v) for d, v in unattributed.items()},
        owner_share_ns=dict(owner_share),
        residual=tuple(sorted(residual.items(), key=lambda kv: (-kv[1], kv[0]))),
        extraction_path=extraction_path,
    )
    return KinetoJoinResult(
        availability=availability,
        markers=markers,
        launches=tuple(launches),
        coverage=coverage,
        op_device_ns=dict(op_device_ns),
        facts={"n_events": len(events), "n_markers": len(markers)},
    )


def require_availability(result: KinetoJoinResult, *, needs: str = "device_time") -> None:
    """Refuse typed when a wholly inapplicable device column is requested.

    The honest-fallback ladder rung 3: a cell may be empty, but the reason
    never is, and the reason names the next action.
    """

    if result.availability == "joined":
        return
    reason = result.coverage.reason or "the join did not run"
    raise ProfilerSessionError(
        f"{needs} is unavailable on this capture: {reason}.",
        code="device_time_unavailable",
        availability=result.availability,
        remedy=(
            "Run the capture under torchlens.observability.native_profile "
            "(or an owned session) on a CUDA host; on CPU-only hosts device "
            "time does not exist and host-clock columns are the honest "
            "alternative."
        ),
    )


__all__ = [
    "GRADFN_MARKER_PREFIX",
    "INTERNAL_MARKER_PREFIX",
    "OP_MARKER_PREFIX",
    "OWNER_CLASSES",
    "REGION_MARKER_PREFIX",
    "ROW_STATUS",
    "JoinCoverage",
    "KinetoJoinResult",
    "LaunchRow",
    "MarkerSpan",
    "join_events",
    "parse_marker",
    "require_availability",
]
