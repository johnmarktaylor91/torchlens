"""FLIP-2 grad_fn-fire marker sink helpers (torchnative W2.0, lane F27).

The backward join's carrier state: the per-trace open-marker LIFOs
(weak-keyed by the owning trace, drained at every pass boundary), the
observability-session probe, and the push / pop / typed-internal-bracket
primitives ``backends/torch/backward.py`` rides. Split out of the backward
god-file under its size ratchet; backward.py remains the ONE node-hook
registration path (the no-parallel-hook-stack dependency test).
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any

import torch

_GRADFN_MARKER_TOKENS: weakref.WeakKeyDictionary[Any, dict[int, list[Any]]]
_GRADFN_MARKER_TOKENS = weakref.WeakKeyDictionary()
"""Per-trace ``grad_fn_object_id -> open profiler-marker LIFO`` (F27 FLIP-2).

The grad_fn-fire ``record_function`` sink rides the existing node-hook
bracket: the timing prehook pushes a marker at its start-stamp line, the
posthook pops at its finish-stamp line, so the marker span and the recorded
host span are the same interval by construction. Lifecycle is fail-closed
and MEASURED: a raising node never runs its posthook and leaks an open
marker, so the token rides this per-node keyed LIFO and the pass-boundary
cleanup closes anything still open (disclosed, never dropped).
"""


def _gradfn_marker_list(trace: Any, grad_fn_object_id: int) -> list[Any]:
    """Return (and lazily build) one node's open-marker LIFO (FLIP-2)."""

    by_node = _GRADFN_MARKER_TOKENS.get(trace)
    if by_node is None:
        by_node = {}
        _GRADFN_MARKER_TOKENS[trace] = by_node
    return by_node.setdefault(grad_fn_object_id, [])


def _obs_session_profiler() -> Any | None:
    """Return the active observability session's live profiler, if any.

    The grad_fn marker sink is on ONLY when its backend is: no session (or a
    session without a profiler) means no marker work at all.
    """

    try:
        from ...observability._session import active_session
    except ImportError:  # pragma: no cover - torn install without the substrate
        return None
    session = active_session()
    return session.profiler if session is not None else None


def _push_gradfn_marker(
    trace: Any, gradfn_marker_tokens: list[Any], grad_fn_object_id: int, call_index: int
) -> None:
    """Open one grad_fn-fire marker at the start-stamp line (FLIP-2).

    A registration failure emits a typed ``marker_registration_gap`` fact
    (a session-time counter the join facts disclose) and the fire stays
    unmarkered -- an optional measurement never becomes a coverage gap.
    """

    profiler = _obs_session_profiler()
    if profiler is None:
        return
    pass_index = int(getattr(trace, "_active_backward_pass_index", 0) or 0)
    try:
        marker = torch.profiler.record_function(
            f"torchlens::gradfn::{grad_fn_object_id}:{pass_index}:{call_index}"
        )
        marker.__enter__()
    # A registration failure of ANY kind degrades this fire to unmarkered
    # and emits the typed marker_registration_gap fact -- an optional
    # measurement never interrupts the user's backward (memo W2.0).
    except Exception:  # noqa: BLE001 - typed-gap degradation, never a swallowed verdict
        gaps = trace.__dict__.get("_tl_gradfn_marker_gaps", 0)
        trace.__dict__["_tl_gradfn_marker_gaps"] = gaps + 1
        return
    gradfn_marker_tokens.append(marker)


def _exit_gradfn_marker(marker: Any) -> None:
    """Close one marker token, swallowing sink-side errors."""

    if marker is None:
        return
    with contextlib.suppress(Exception):
        marker.__exit__(None, None, None)


def _enter_internal_gradfn_bracket(grad_fn_object_id: int) -> Any | None:
    """Bracket TorchLens's OWN posthook logging with a typed internal span.

    TN-D18: on a gradient-saving capture our own posthook logging outnumbers
    true engine work 1513:12 -- without the typed bracket the complement
    bucket would publish our clone storm under torch's name.
    """

    profiler = _obs_session_profiler()
    if profiler is None:
        return None
    try:
        marker = torch.profiler.record_function(
            f"torchlens::internal::gradfn_hook::{grad_fn_object_id}"
        )
        marker.__enter__()
    except Exception:  # noqa: BLE001 - sink-side failure degrades the bracket, never the hook
        return None
    return marker


def _drain_gradfn_markers(trace: Any) -> int:
    """Close every still-open grad_fn marker at a pass boundary (FLIP-2).

    A raising node never runs its posthook (reproduced: 5 nodes hooked, 3
    markers, 1 leak), so the boundary cleanup closes anything still open
    and DISCLOSES the count on the session-time leak counter rather than
    dropping it.
    """

    by_node = _GRADFN_MARKER_TOKENS.get(trace)
    if not by_node:
        return 0
    leaked = 0
    for tokens in by_node.values():
        while tokens:
            _exit_gradfn_marker(tokens.pop())
            leaked += 1
    if leaked:
        total = trace.__dict__.get("_tl_gradfn_marker_leaks", 0)
        trace.__dict__["_tl_gradfn_marker_leaks"] = total + leaked
    return leaked


__all__ = [
    "_drain_gradfn_markers",
    "_enter_internal_gradfn_bracket",
    "_exit_gradfn_marker",
    "_gradfn_marker_list",
    "_push_gradfn_marker",
]
