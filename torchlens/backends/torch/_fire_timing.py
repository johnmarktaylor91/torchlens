"""Per-fire backward timing helpers (L9 memo 1.3; split under the size ratchet).

The keyed per-node start-stamp LIFOs (weak-keyed by the owning trace,
cleared at every pass boundary), the lightweight timing prehook, and its
degrade-to-untimed registration. ``backends/torch/backward.py`` remains the
ONE node-hook registration path; these helpers build the hooks it registers.
"""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from typing import Any

from ._gradfn_markers import _push_gradfn_marker

_FIRE_TIMING_STAMPS: weakref.WeakKeyDictionary[Any, dict[int, list[tuple[int, float]]]]
_FIRE_TIMING_STAMPS = weakref.WeakKeyDictionary()
"""Per-trace ``grad_fn_object_id -> keyed start-stamp LIFO`` for per-fire timing.

Each list is the per-node keyed LIFO (L9 memo 1.3) shared by that node's
timing prehook and its posthook: the prehook appends ``(call_index,
time.perf_counter())`` and the posthook pops entries until it finds a
``call_index`` match, discarding stale entries above the match. Per-node
sequential firing on one engine worker thread is the same assumption the
shipped aten-marker LIFO already makes. The registry exists so pass
boundaries can clear retry debris; hooks close over their own list.
"""


def _fire_timing_stamp_list(trace: Any, grad_fn_object_id: int) -> list[tuple[int, float]]:
    """Return (and lazily build) one node's keyed start-stamp LIFO."""

    by_node = _FIRE_TIMING_STAMPS.get(trace)
    if by_node is None:
        by_node = {}
        _FIRE_TIMING_STAMPS[trace] = by_node
    return by_node.setdefault(grad_fn_object_id, [])


def _clear_fire_timing_stamps(trace: Any) -> None:
    """Clear every per-node start-stamp LIFO at a pass boundary.

    Stale entries (a prehook fired but its node raised; a caught-and-retried
    backward) must not survive into a later pass, where the restarting
    ``call_index`` sequence could otherwise collide with retry debris.
    """

    by_node = _FIRE_TIMING_STAMPS.get(trace)
    if not by_node:
        return
    for stamps in by_node.values():
        stamps.clear()


def _pop_matching_fire_start(stamps: list[tuple[int, float]], call_index: int) -> float | None:
    """Pop the keyed LIFO until a ``call_index`` match; return its stamp.

    Stale entries above the match are DISCARDED, never paired; no match or an
    empty LIFO returns ``None`` (an untimed fire), so a posthook that runs
    without its prehook can never inherit a stale stamp as a wrong positive
    span.
    """

    while stamps:
        key, stamp = stamps.pop()
        if key == call_index:
            return stamp
    return None


def _make_timing_grad_fn_prehook(
    trace: Any,
    grad_fn_object_id: int,
    fire_start_stamps: list[tuple[int, float]],
    gradfn_marker_tokens: list[Any] | None = None,
) -> Callable[..., None]:
    """Build the lightweight per-fire timing prehook (L9 memo 1.3).

    The prehook dispatches no tensor ops -- one ``perf_counter`` call and one
    list append -- so its registration order can never sweep a foreign aten
    dispatch into the marker bracket; it is registered BEFORE the aten marker
    prehook only so the measured span covers the whole fire. FLIP-2 (F27):
    when a profiler session is active, a ``torchlens::gradfn::`` marker
    opens AT this start-stamp line, so the marker span and the recorded host
    span are the same interval by construction.
    """

    trace_ref = weakref.ref(trace)

    def timing_prehook(*hook_args: Any) -> None:
        """Push a keyed ``(call_index, perf_counter)`` start stamp."""

        del hook_args
        live_trace = trace_ref()
        if live_trace is None:
            return None
        grad_fn_record = getattr(live_trace, "grad_fn_logs", {}).get(grad_fn_object_id)
        if grad_fn_record is None:
            return None
        call_index = len(grad_fn_record.calls) + 1
        if gradfn_marker_tokens is not None:
            _push_gradfn_marker(live_trace, gradfn_marker_tokens, grad_fn_object_id, call_index)
        fire_start_stamps.append((call_index, time.perf_counter()))
        return None

    return timing_prehook


def _register_fire_timing_prehook(
    trace: Any,
    grad_fn_handle: Any,
    grad_fn_object_id: int,
    fire_start_stamps: list[tuple[int, float]],
    gradfn_marker_tokens: list[Any] | None = None,
) -> Any | None:
    """Register the timing prehook on one node; failure degrades to untimed.

    The timing registration gets ITS OWN try/except, separate from the
    shipped registration block whose ``except RuntimeError`` converts a node
    into a fail-closed ``BackwardCoverageGap``: an optional measurement must
    never turn a complete-coverage node into a coverage gap (L9 memo 1.3,
    opus m2-r2). Failure returns ``None`` and the node's fires stay untimed
    (the posthook's keyed LIFO simply never matches).
    """

    try:
        return grad_fn_handle.register_prehook(
            _make_timing_grad_fn_prehook(
                trace, grad_fn_object_id, fire_start_stamps, gradfn_marker_tokens
            )
        )
    except RuntimeError:
        return None


__all__ = [
    "_clear_fire_timing_stamps",
    "_fire_timing_stamp_list",
    "_pop_matching_fire_start",
    "_register_fire_timing_prehook",
]
