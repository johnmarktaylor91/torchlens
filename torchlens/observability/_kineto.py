"""In-memory Kineto extraction adapter (torchnative W2.1).

The join consumes normalized profiler events extracted from in-memory
``_KinetoEvent`` objects -- integer nanoseconds, runtime correlation IDs,
thread IDs, the user-annotation flag, and the typed activity classification
-- behind ONE feature-detected adapter (`torchlens.utils._torch_compat.
kineto_events_from_profiler`, the sanctioned private-probe boundary). The
historical chrome-trace temp-file join path is DELETED: when the in-memory
field contract is unavailable the adapter demotes to a bounded chrome-event
extractor through the guarded JSON reader, and the extraction result records
WHICH path ran -- the adapter never guesses.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .._io import _json
from ..utils import _torch_compat

__tl_layer__ = "L5"

#: Closed normalized event kinds. Classification is by the profiler's TYPED
#: activity field (in-memory) or exact category tokens (chrome fallback) --
#: never launch-name matching.
EVENT_KINDS = ("marker", "op", "runtime", "kernel", "memcpy", "memset", "other")

#: Extraction paths the adapter can take, recorded on every result.
EXTRACTION_PATHS = ("in_memory", "chrome_stream", "unavailable")

# Typed activity -> normalized kind (in-memory path). Anything unlisted is
# "other" and stays out of totals rather than being guessed.
_ACTIVITY_KIND = {
    "user_annotation": "marker",
    "gpu_user_annotation": "marker",
    "cpu_op": "op",
    "cuda_runtime": "runtime",
    "cuda_driver": "runtime",
    "privateuse1_runtime": "runtime",
    "privateuse1_driver": "runtime",
    "kernel": "kernel",
    "concurrent_kernel": "kernel",
    "gpu_memcpy": "memcpy",
    "gpu_memset": "memset",
}

# Chrome-trace category tokens -> normalized kind (fallback path). Exact
# tokens only, mirroring the deleted kernel_telemetry classifier.
_CHROME_CATEGORY_KIND = {
    "user_annotation": "marker",
    "gpu_user_annotation": "marker",
    "cpu_op": "op",
    "cuda_runtime": "runtime",
    "cuda_driver": "runtime",
    "kernel": "kernel",
    "cuda_kernel": "kernel",
    "gpu_memcpy": "memcpy",
    "memcpy": "memcpy",
    "gpu_memset": "memset",
    "memset": "memset",
}


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    """One normalized profiler event in Kineto's own clock domain.

    ``start_ns``/``end_ns`` stay on the profiler's clock; they are never
    projected onto TorchLens's host clocks (W1.4 boundary). ``scope`` is the
    profiler's typed per-event record scope when the build exposes it
    (``None`` otherwise -- absence degrades the backward body witness, never
    the join).
    """

    name: str
    kind: str
    activity: str
    start_ns: int
    end_ns: int
    tid: int | None
    device_type: str
    device_index: int | None
    correlation_id: int | None
    is_user_annotation: bool
    scope: int | None
    stream: int | str | None = None

    @property
    def duration_ns(self) -> int:
        """Event duration in integer nanoseconds."""

        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class ExtractionResult:
    """Normalized events plus the disclosure of which path produced them.

    ``raw_chrome_bytes`` carries the exact bytes torch's profiler wrote when
    the ``chrome_stream`` fallback path ran its one allowed
    ``export_chrome_trace`` call (Kineto's own result object refuses a
    second ``save`` with ``RuntimeError: Trace is already saved.``); a
    native-chrome consumer reuses these bytes instead of re-exporting.
    ``None`` on every other path (nothing to reuse).
    """

    path: str
    events: tuple[NormalizedEvent, ...]
    notes: tuple[str, ...] = ()
    raw_chrome_bytes: bytes | None = None


def _normalize_inmemory_event(event: Any, *, has_scope: bool) -> NormalizedEvent | None:
    """Normalize one in-memory ``_KinetoEvent``; unusable rows become None."""

    try:
        start_ns = int(event.start_ns())
        duration_ns = int(event.duration_ns())
        activity = str(event.activity_type())
        name = str(event.name())
        correlation = event.correlation_id()
        tid = event.start_thread_id()
        device_type = str(event.device_type())
        device_index = event.device_index()
        is_user_annotation = bool(event.is_user_annotation())
        scope = int(event.scope()) if has_scope else None
    except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError):
        # A hostile or drifted event object demotes ITS row, never the
        # extraction; contract validation happened at the compat boundary.
        return None
    kind = _ACTIVITY_KIND.get(activity, "other")
    return NormalizedEvent(
        name=name,
        kind=kind,
        activity=activity,
        start_ns=start_ns,
        end_ns=start_ns + max(duration_ns, 0),
        tid=int(tid) if isinstance(tid, int) else None,
        device_type=device_type.rsplit(".", 1)[-1].lower(),
        device_index=int(device_index) if isinstance(device_index, int) else None,
        correlation_id=(
            int(correlation) if isinstance(correlation, int) and correlation >= 0 else None
        ),
        is_user_annotation=is_user_annotation,
        scope=scope,
    )


def _chrome_category_tokens(event: Mapping[str, Any]) -> frozenset[str]:
    """Return exact lower-case chrome category tokens for one event."""

    category = event.get("cat", "")
    if not isinstance(category, str):
        return frozenset()
    return frozenset(part.strip().lower() for part in category.split(",") if part.strip())


def _chrome_args(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return one chrome event's argument mapping, or an empty mapping."""

    args = event.get("args", {})
    return args if isinstance(args, Mapping) else {}


def _chrome_correlation(event: Mapping[str, Any]) -> int | None:
    """Read a chrome event's Kineto correlation id without name matching."""

    args = _chrome_args(event)
    for key in ("correlation", "Correlation id", "correlation_id"):
        value = args.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _normalize_chrome_event(event: Mapping[str, Any]) -> NormalizedEvent | None:
    """Normalize one chrome-trace event mapping; unusable rows become None."""

    name = event.get("name")
    ts = event.get("ts")
    if not isinstance(name, str) or isinstance(ts, bool) or not isinstance(ts, (int, float)):
        return None
    duration = event.get("dur", 0)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        duration = 0
    tokens = _chrome_category_tokens(event)
    kind = "other"
    activity = ",".join(sorted(tokens))
    for token in tokens:
        mapped = _CHROME_CATEGORY_KIND.get(token)
        if mapped is not None:
            kind = mapped
            break
    tid = event.get("tid")
    args = _chrome_args(event)
    device = args.get("device")
    stream = args.get("stream")
    start_ns = int(ts * 1000)
    return NormalizedEvent(
        name=name,
        kind=kind,
        activity=activity,
        start_ns=start_ns,
        end_ns=start_ns + int(max(float(duration), 0.0) * 1000),
        tid=int(tid) if isinstance(tid, int) and not isinstance(tid, bool) else None,
        device_type="cuda" if kind in ("kernel", "memcpy", "memset") else "cpu",
        device_index=int(device)
        if isinstance(device, int) and not isinstance(device, bool)
        else None,
        correlation_id=_chrome_correlation(event),
        is_user_annotation="user_annotation" in tokens or "gpu_user_annotation" in tokens,
        scope=None,
        stream=stream if isinstance(stream, (int, str)) and not isinstance(stream, bool) else None,
    )


def _extract_chrome_stream(profiler: Any) -> ExtractionResult:
    """Bounded chrome-event fallback through the guarded JSON reader."""

    try:
        with TemporaryDirectory(prefix="torchlens-kineto-") as temp_dir:
            trace_path = Path(temp_dir) / "kineto.json"
            profiler.export_chrome_trace(str(trace_path))
            raw_bytes = trace_path.read_bytes()
            raw = _json.read_bounded(trace_path)
    # The fallback path itself failing (export refusal, oversized trace,
    # parse refusal) is DISCLOSED as unavailable -- never a guess and never
    # a crash of the capture that owns the session.
    except Exception as exc:  # noqa: BLE001 - disclosed-unavailable degradation
        return ExtractionResult(
            path="unavailable",
            events=(),
            notes=(f"chrome fallback failed: {type(exc).__name__}",),
        )
    raw_events: Sequence[Any] = raw.get("traceEvents", ()) if isinstance(raw, Mapping) else ()
    events = tuple(
        normalized
        for event in raw_events
        if isinstance(event, Mapping) and (normalized := _normalize_chrome_event(event)) is not None
    )
    return ExtractionResult(path="chrome_stream", events=events, raw_chrome_bytes=raw_bytes)


def extract_events(profiler: Any) -> ExtractionResult:
    """Extract normalized events from one CLOSED profiler (W2.1).

    Tries the in-memory ``_KinetoEvent`` seam first (feature-detected at the
    ``_torch_compat`` boundary); demotes to the bounded chrome-stream
    extractor when the field contract is unavailable. The result records
    which path ran; a wholly failed extraction is ``path="unavailable"``
    with empty events, never a guess.

    Parameters
    ----------
    profiler:
        A closed ``torch.profiler.profile`` instance.

    Returns
    -------
    ExtractionResult
        Normalized events plus the extraction-path disclosure.
    """

    raw = _torch_compat.kineto_events_from_profiler(profiler)
    if raw is not None and _torch_compat.HAS_KINETO_INMEMORY_EVENTS:
        has_scope = _torch_compat.HAS_KINETO_EVENT_SCOPE
        events = tuple(
            normalized
            for event in raw
            if (normalized := _normalize_inmemory_event(event, has_scope=has_scope)) is not None
        )
        return ExtractionResult(path="in_memory", events=events)
    return _extract_chrome_stream(profiler)


__all__ = [
    "EVENT_KINDS",
    "EXTRACTION_PATHS",
    "ExtractionResult",
    "NormalizedEvent",
    "extract_events",
]
