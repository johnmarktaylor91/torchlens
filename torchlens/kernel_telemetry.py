"""Detachable CUDA kernel telemetry for the gated ATen execution profile.

This module is intentionally not imported by the TorchLens package or capture
core. Importing it installs the documented-unstable ``gpu_kernels``
descriptors, while the private profiling adapter instruments the existing ATen
observer only for its own scope. Removing this module therefore leaves the ATen
profile and every non-telemetry capture path untouched.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from ._io import FieldPolicy
from .data_classes.aten_op import AtenOp
from .data_classes.op import Op

_ANNOTATION_KEY = "_kernel_telemetry"
_MARKER_PREFIX = "torchlens::aten::"
_INSTRUMENTATION_LOCK = threading.RLock()
_ATEN_KERNELS: weakref.WeakKeyDictionary[AtenOp, tuple[KernelLaunch, ...]] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelLaunch:
    """One CUDA kernel or memory-copy event reported by Kineto.

    All fields are measured observations from one profiler session. ``None``
    values are used only by the single ``unavailable`` disclosure row; they
    never stand in for an observed launch.
    """

    launch_name: str | None
    device: str | None
    stream: int | str | None
    duration: float | None
    runtime_correlation: int | str | None
    attribution_status: str

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "launch_name": FieldPolicy.KEEP,
        "device": FieldPolicy.KEEP,
        "stream": FieldPolicy.KEEP,
        "duration": FieldPolicy.KEEP,
        "runtime_correlation": FieldPolicy.KEEP,
        "attribution_status": FieldPolicy.KEEP,
    }


@dataclass(frozen=True, slots=True)
class _TelemetryPayload:
    """Private DROP-gated annotation payload and primitive-row relation."""

    _available: bool
    _launches: tuple[KernelLaunch, ...]
    _relations: tuple[tuple[int, int], ...]

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "_available": FieldPolicy.KEEP,
        "_launches": FieldPolicy.KEEP,
        "_relations": FieldPolicy.KEEP,
    }


_UNAVAILABLE = KernelLaunch(
    launch_name=None,
    device=None,
    stream=None,
    duration=None,
    runtime_correlation=None,
    attribution_status="unavailable",
)


def _payload_from_events(
    events: Sequence[Any],
    marker_sequences: Mapping[str, int],
    *,
    telemetry_available: bool,
) -> _TelemetryPayload:
    """Join unique ATen markers to device events by runtime correlation (W2.1).

    Consumes NORMALIZED events from the one extraction adapter
    (``torchlens.observability._kineto.extract_events``); the historical
    chrome-trace temp-file path is deleted. Same-thread INNERMOST containment
    identifies the runtime API calls one redispatched ATen operation made
    (the cross-thread timestamp comparison defect class TN-D2 is gone);
    Kineto's runtime correlation identifier then follows asynchronous work
    onto device events, including events whose timestamps fall outside the
    CPU marker. No operator or launch-name substring is used.

    Parameters
    ----------
    events
        Normalized profiler events.
    marker_sequences
        Unique marker name to primitive-row sequence mapping.
    telemetry_available
        Whether the CUDA/CUPTI profiler session initialized successfully.

    Returns
    -------
    _TelemetryPayload
        Launch rows plus a many-to-many primitive-sequence relation.
    """

    if not telemetry_available:
        return _TelemetryPayload(
            _available=False,
            _launches=(_UNAVAILABLE,),
            _relations=tuple((sequence, 0) for sequence in sorted(set(marker_sequences.values()))),
        )

    from .observability._join import MarkerSpan, _innermost_owner_by_runtime

    markers = tuple(
        MarkerSpan(
            name=event.name,
            owner_class="torchlens_internal",
            owner_key=event.name,
            start_ns=event.start_ns,
            end_ns=event.end_ns,
            tid=event.tid,
        )
        for event in events
        if event.is_user_annotation and event.name in marker_sequences
    )
    runtime_events = [event for event in events if event.kind == "runtime"]
    owners = _innermost_owner_by_runtime(markers, runtime_events)
    sequences_by_correlation: dict[int, set[int]] = {
        correlation: {marker_sequences[marker.name] for marker in marker_set}
        for correlation, marker_set in owners.items()
    }

    launches: list[KernelLaunch] = []
    relations: list[tuple[int, int]] = []
    for event in events:
        if event.kind not in ("kernel", "memcpy", "memset"):
            continue
        correlation = event.correlation_id
        sequences = (
            set() if correlation is None else sequences_by_correlation.get(correlation, set())
        )
        status = "attributed" if sequences else "unattributed"
        launch = KernelLaunch(
            launch_name=event.name,
            device=(None if event.device_index is None else str(event.device_index)),
            stream=event.stream,
            duration=event.duration_ns / 1_000.0,
            runtime_correlation=correlation,
            attribution_status=status,
        )
        launch_index = len(launches)
        launches.append(launch)
        relations.extend((sequence, launch_index) for sequence in sorted(sequences))
    return _TelemetryPayload(
        _available=True,
        _launches=tuple(launches),
        _relations=tuple(relations),
    )


def _coerce_launch(value: Any) -> KernelLaunch:
    """Coerce a loaded mapping back to a ``KernelLaunch`` facade.

    Parameters
    ----------
    value
        Live or rehydrated launch value.

    Returns
    -------
    KernelLaunch
        Typed launch row.

    Raises
    ------
    TypeError
        If the loaded value has no recognized launch representation.
    """

    if isinstance(value, KernelLaunch):
        return value
    if isinstance(value, Mapping):
        return KernelLaunch(**{name: value[name] for name in KernelLaunch.PORTABLE_STATE_SPEC})
    raise TypeError("kernel telemetry launch row has an invalid representation")


def _coerce_payload(value: Any) -> _TelemetryPayload | None:
    """Coerce a live or loaded private annotation payload.

    Parameters
    ----------
    value
        Trace annotation value.

    Returns
    -------
    _TelemetryPayload | None
        Typed payload, or ``None`` when telemetry is absent.
    """

    if isinstance(value, _TelemetryPayload):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        raw_relations = value["_relations"]
        if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, str | bytes):
            return None
        relations: list[tuple[int, int]] = []
        for item in raw_relations:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
                return None
            relations.append((int(item[0]), int(item[1])))
        return _TelemetryPayload(
            _available=bool(value["_available"]),
            _launches=tuple(_coerce_launch(item) for item in value["_launches"]),
            _relations=tuple(relations),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _bind_trace_telemetry(trace: Any) -> None:
    """Bind one trace's relation rows to its live ``AtenOp`` facades.

    Parameters
    ----------
    trace
        Trace carrying the primitive profile and optional telemetry annotation.
    """

    profile = getattr(trace, "_primitive_op_profile", None)
    if profile is None:
        return
    annotations = getattr(trace, "annotations", {})
    payload = (
        _coerce_payload(annotations.get(_ANNOTATION_KEY))
        if isinstance(annotations, Mapping)
        else None
    )
    rows = tuple(getattr(profile, "primitive_ops", ()))
    if payload is None:
        for row in rows:
            _ATEN_KERNELS.pop(row, None)
        return
    indices_by_sequence: dict[int, list[int]] = {}
    for sequence, launch_index in payload._relations:
        indices_by_sequence.setdefault(sequence, []).append(launch_index)
    for row in rows:
        indices = indices_by_sequence.get(row.sequence, [])
        _ATEN_KERNELS[row] = tuple(
            payload._launches[index] for index in indices if 0 <= index < len(payload._launches)
        )


def _aten_gpu_kernels(row: AtenOp) -> tuple[KernelLaunch, ...]:
    """Return device events correlated to one primitive row.

    Parameters
    ----------
    row
        Primitive ATen row.

    Returns
    -------
    tuple[KernelLaunch, ...]
        Correlated rows, or one typed ``unavailable`` disclosure when no
        telemetry session has been bound.
    """

    return _ATEN_KERNELS.get(row, (_UNAVAILABLE,))


def _op_gpu_kernels(op: Op) -> tuple[KernelLaunch, ...]:
    """Return the deduplicated union of device events for one user Op.

    Parameters
    ----------
    op
        User-level Op facade.

    Returns
    -------
    tuple[KernelLaunch, ...]
        Correlated device rows. A single ``unavailable`` disclosure is
        returned when no telemetry session exists.
    """

    trace = op._source_trace_or_none()
    if trace is None:
        return (_UNAVAILABLE,)
    _bind_trace_telemetry(trace)
    profile = getattr(trace, "_primitive_op_profile", None)
    annotations = getattr(trace, "annotations", {})
    payload = (
        _coerce_payload(annotations.get(_ANNOTATION_KEY))
        if isinstance(annotations, Mapping)
        else None
    )
    if profile is None or payload is None or not payload._available:
        return (_UNAVAILABLE,)
    row_index = object.__getattribute__(op, "_row")
    launches: list[KernelLaunch] = []
    seen: set[int] = set()
    for aten_row in profile.primitive_ops:
        if not any(ref.op_row_index == row_index for ref in aten_row.parent_op_refs):
            continue
        for launch in _ATEN_KERNELS.get(aten_row, ()):
            identity = id(launch)
            if identity not in seen:
                seen.add(identity)
                launches.append(launch)
    return tuple(launches)


def _install_gpu_kernel_properties() -> None:
    """Install the two documented-unstable descriptors exactly once."""

    for owner, getter in ((AtenOp, _aten_gpu_kernels), (Op, _op_gpu_kernels)):
        existing = vars(owner).get("gpu_kernels")
        if existing is None:
            owner.gpu_kernels = property(getter)  # type: ignore[union-attr]
        elif not isinstance(existing, property) or existing.fget is not getter:
            raise RuntimeError(f"{owner.__name__}.gpu_kernels is already owned by another lane")


@contextmanager
def _instrument_aten_markers() -> Iterator[dict[str, int]]:
    """Temporarily bracket redispatched ATen calls with unique profiler markers.

    Yields
    ------
    dict[str, int]
        Mutable marker-name to finalized primitive-sequence relation.
    """

    from .backends.torch import _aten_capture

    with _INSTRUMENTATION_LOCK:
        original_prepare = _aten_capture._prepare_aten_call
        original_finish = _aten_capture._finish_aten_call
        pending_markers: dict[int, tuple[str, Any]] = {}
        marker_sequences: dict[str, int] = {}
        counter = 0

        def prepare(*args: Any, **kwargs: Any) -> Any:
            """Enter one marker after the core has prepared an ATen call."""

            nonlocal counter
            pending = original_prepare(*args, **kwargs)
            counter += 1
            name = f"{_MARKER_PREFIX}{counter}"
            marker = torch.profiler.record_function(name)
            marker.__enter__()
            pending_markers[id(pending)] = (name, marker)
            return pending

        def finish(state: Any, pending: Any, **kwargs: Any) -> None:
            """Exit the marker before materializing the primitive event."""

            marker_entry = pending_markers.pop(id(pending), None)
            if marker_entry is not None:
                marker_entry[1].__exit__(None, None, None)
            original_finish(state, pending, **kwargs)
            if marker_entry is not None and state.aten_events.aten_events:
                marker_sequences[marker_entry[0]] = state.aten_events.aten_events[-1].seq

        _aten_capture._prepare_aten_call = prepare
        _aten_capture._finish_aten_call = finish
        try:
            yield marker_sequences
        finally:
            _aten_capture._prepare_aten_call = original_prepare
            _aten_capture._finish_aten_call = original_finish
            for _, marker in pending_markers.values():
                marker.__exit__(None, None, None)


def _attach_payload(trace: Any, payload: _TelemetryPayload) -> None:
    """Attach a DROP-gated telemetry annotation and bind its live views.

    Parameters
    ----------
    trace
        Trace receiving the telemetry result.
    payload
        Parsed telemetry rows and relations.
    """

    trace.annotations[_ANNOTATION_KEY] = {
        "_available": payload._available,
        "_launches": tuple(
            {
                field_name: getattr(launch, field_name)
                for field_name in KernelLaunch.PORTABLE_STATE_SPEC
            }
            for launch in payload._launches
        ),
        "_relations": payload._relations,
    }
    _bind_trace_telemetry(trace)


def _profile_trace_with_cuda_kernels(factory: Callable[[], Any]) -> Any:
    """Run a private ATen-recording trace factory under CUDA kernel profiling.

    This is the non-public activation seam while ``record_aten=`` remains
    naming-gated. It routes through the ONE profiler session engine
    (``torchlens.observability.session``) and the in-memory extraction
    adapter -- the chrome-trace temp-file path is deleted (W2.1). On hosts
    without CUDA, the factory still runs once and every primitive row
    receives a typed ``unavailable`` disclosure; no launch is fabricated.

    Parameters
    ----------
    factory
        Zero-argument callable returning one Trace.

    Returns
    -------
    Any
        The factory's Trace with a gated telemetry annotation.
    """

    from .backends.torch._aten_capture import _activate_aten_recording_for_tests
    from .observability._kineto import extract_events
    from .observability._session import session

    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        with _activate_aten_recording_for_tests(), _instrument_aten_markers() as marker_sequences:
            trace = factory()
        payload = _payload_from_events((), marker_sequences, telemetry_available=False)
        _attach_payload(trace, payload)
        return trace

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with (
        _activate_aten_recording_for_tests(),
        _instrument_aten_markers() as marker_sequences,
        session(mode="owned", activities=activities) as active,
    ):
        trace = factory()
        torch.cuda.synchronize()
    extraction = extract_events(active.closed_profiler)
    payload = _payload_from_events(
        extraction.events,
        marker_sequences,
        telemetry_available=extraction.path != "unavailable",
    )
    _attach_payload(trace, payload)
    return trace


# The tlspec v8 coordinated bump retired the telemetry S3 pre-release rows:
# KernelLaunch/_TelemetryPayload declare FieldPolicy.KEEP directly and the
# private annotations section persists plainly, validated at load by
# torchlens/_io/forgery_validation.py.
_install_gpu_kernel_properties()

__all__ = ["KernelLaunch"]
