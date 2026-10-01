"""The one-call device-profiling door (torchnative 4.3; FLIP-1 consumer).

``native_profile(model, x)`` runs ONE execution at the save-nothing tier
(the measurement tier: cheaper and more attributable than a saving capture)
inside the ONE owned profiler session, then joins device events to the
captured ops by runtime correlation IDs. It ships BEFORE the >=95% GPU
acceptance gates pass -- the refusal ladder answering honestly ("device
cells render '-' on a CPU-only session") is itself the advertisement of the
joined table.

No hidden replay, ever: one door call describes exactly one execution.

The join result is registered session-time in a weak trace registry (the
``kernel_telemetry`` precedent) -- never persisted, never a Trace schema
field -- and the consumers (``draw(color_by="device_time")``,
``hot_path(by="device_time")``, the profile device column) read it through
:func:`join_result_for`.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ._join import KinetoJoinResult, join_events, require_availability
from ._kineto import extract_events
from ._session import ProfilerSession, session

__tl_layer__ = "L5"

#: Session-time join registry keyed by live Trace identity. Weak: a released
#: trace drops its join rows with it. NEVER persisted (device joins describe
#: one live session; a loaded artifact has no session).
_JOIN_TRACES: weakref.WeakKeyDictionary[Any, KinetoJoinResult] = weakref.WeakKeyDictionary()


def join_result_for(trace: Any) -> KinetoJoinResult | None:
    """Return the session-time join result for one trace, if any."""

    try:
        return _JOIN_TRACES.get(trace)
    except TypeError:  # non-weakref-able stand-ins in tests
        return None


def register_join_result(trace: Any, result: KinetoJoinResult) -> None:
    """Bind one trace's session-time join result (weak, never persisted)."""

    _JOIN_TRACES[trace] = result


@dataclass(frozen=True)
class NativeProfileResult:
    """One door call: one execution, one trace, one joined result."""

    trace: Any
    join: KinetoJoinResult
    session_facts: dict[str, Any]

    @property
    def availability(self) -> str:
        """Five-state session availability (missing is None, never zero)."""

        return self.join.availability

    def device_time_ns(self, op_label: str) -> int:
        """Exact-attribution device nanoseconds for one op label.

        Refuses typed (``device_time_unavailable``) when the join did not
        reach ``joined`` -- an explicitly requested inapplicable column
        raises with cause and remedy, never returns a silent zero.
        """

        require_availability(self.join)
        return self.join.op_device_ns.get(op_label, 0)

    def device_time_table(self) -> tuple[tuple[str, int], ...]:
        """Per-op exact device time, descending; refuses when unjoined."""

        require_availability(self.join)
        return tuple(sorted(self.join.op_device_ns.items(), key=lambda kv: (-kv[1], kv[0])))


def native_profile(
    model: Any,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    activities: Any | None = None,
    native_chrome_path: str | Path | None = None,
    **trace_kwargs: Any,
) -> NativeProfileResult:
    """Run one save-nothing capture under the owned session and join it.

    Parameters
    ----------
    model:
        The model to run (any ``tl.trace``-able callable).
    input_args, input_kwargs:
        Forwarded to ``tl.trace``.
    activities:
        Optional explicit ``torch.profiler`` activity list; the default asks
        for CPU plus CUDA when CUDA is available.
    native_chrome_path:
        When given, the NATIVE chrome trace (Kineto's clock, foreign events
        and all) is preserved at this path with a TorchLens mapping sidecar
        beside it (``<path>.torchlens-map.json``) carrying exact marker-name
        -> op-label rows. Host timestamps are never projected onto the
        profiler's clock.
    **trace_kwargs:
        Extra ``tl.trace`` keyword arguments. ``save=`` defaults to the
        save-nothing tier (measurement tier); passing an explicit ``save=``
        rides that capture instead, with the internal share named in the
        coverage ledger.

    Returns
    -------
    NativeProfileResult
        The trace, the joined result, and the session facts.
    """

    from .. import user_funcs

    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():  # pragma: no cover - CUDA leg
            activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_kwargs.setdefault("save", None)
    with session(mode="owned", activities=activities) as active:
        trace = user_funcs.trace(model, input_args, input_kwargs, **trace_kwargs)
        if torch.cuda.is_available():  # pragma: no cover - CUDA leg
            torch.cuda.synchronize()
    result = join_session(active, trace, native_chrome_path=native_chrome_path)
    return NativeProfileResult(
        trace=trace,
        join=result,
        session_facts=dict(active.result.facts if active.result is not None else {}),
    )


def join_session(
    active: ProfilerSession,
    trace: Any | None = None,
    *,
    native_chrome_path: str | Path | None = None,
) -> KinetoJoinResult:
    """Extract and join one CLOSED session's events (owned sessions only).

    Borrowed sessions are joined by their owner after the caller's profiler
    closes: pass the borrowed profiler's session through this function once
    the caller has exited it.
    """

    profiler = active.closed_profiler
    raw_chrome_bytes: bytes | None = None
    if profiler is None:
        extraction_path = "unavailable"
        events: tuple[Any, ...] = ()
    else:
        extraction = extract_events(profiler)
        extraction_path = extraction.path
        events = extraction.events
        raw_chrome_bytes = extraction.raw_chrome_bytes
    result = join_events(events, extraction_path=extraction_path, trace=trace)
    if trace is not None:
        register_join_result(trace, result)
    if active.result is not None:
        active.result.facts["kineto_join"] = result.availability
    if native_chrome_path is not None and profiler is not None:
        _write_native_chrome(
            profiler, result, Path(native_chrome_path), raw_chrome_bytes=raw_chrome_bytes
        )
    return result


def _write_native_chrome(
    profiler: Any,
    result: KinetoJoinResult,
    path: Path,
    *,
    raw_chrome_bytes: bytes | None = None,
) -> None:
    """Preserve the NATIVE chrome artifact plus the exact-ID mapping sidecar.

    The native file is torch's own export on Kineto's clock -- TorchLens
    neither rewrites nor re-times it. The sidecar carries the exact marker
    name -> owner rows so a viewer-side join needs no name matching.

    ``raw_chrome_bytes``, when given, is the exact export the event
    extraction already pulled from this same profiler (the chrome-stream
    fallback path, W2.1): torch's Kineto result object permits exactly ONE
    ``export_chrome_trace`` save per profiler and raises ``RuntimeError:
    Trace is already saved.`` on a second call, so this reuses those bytes
    verbatim instead of re-exporting.
    """

    import json as _stdlib_json

    from ..utils.display import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_chrome_bytes is not None:
        path.write_bytes(raw_chrome_bytes)
    else:
        profiler.export_chrome_trace(str(path))
    sidecar = {
        "schema": "torchlens.native_chrome_map.v1",
        "clock_note": (
            "the native trace is on Kineto's clock; TorchLens host-clock "
            "exports share no basis with it and are never projected onto it"
        ),
        "markers": [
            {
                "name": marker.name,
                "owner_class": marker.owner_class,
                "owner_key": marker.owner_key,
            }
            for marker in result.markers
        ],
    }
    atomic_write_text(
        path.with_name(path.name + ".torchlens-map.json"),
        _stdlib_json.dumps(sidecar, indent=2),
    )


__all__ = [
    "NativeProfileResult",
    "join_result_for",
    "join_session",
    "native_profile",
    "register_join_result",
]
