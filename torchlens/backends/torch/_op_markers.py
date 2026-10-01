"""Per-call op markers: NVTX ranges + FLIP-1 join markers (lane F27).

The two per-call marker sinks the wrapper timing bracket rides, split out
of the wrappers god-file under its size ratchet: identity-carrying NVTX
ranges (W0.2, with TorchLens-internal separation via the allowlist-by-
construction internal-read marker) and the ``record_function`` join markers
whose names are exact ``func_call_id`` keys (W2.2). Each sink is on only
when its backend is active; a sink-side failure degrades the range/join,
never the user op.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch


def _nvtx_range_push(name: str) -> bool:
    """Push an NVTX range if CUDA NVTX support is available.

    Parameters
    ----------
    name:
        Range label.

    Returns
    -------
    bool
        Whether a corresponding pop should be attempted.
    """

    try:
        torch.cuda.nvtx.range_push(name)
    except (RuntimeError, AttributeError, OSError):
        # Sink-side NVTX failure (no driver, torn cuda install) degrades the
        # range, never the user op.
        return False
    return True


def _nvtx_range_pop(enabled: bool) -> None:
    """Pop a previously pushed NVTX range.

    Parameters
    ----------
    enabled:
        Whether a push succeeded.
    """

    if not enabled:
        return
    with contextlib.suppress(RuntimeError, AttributeError, OSError):
        torch.cuda.nvtx.range_pop()


def _op_marker_labels(func_name: str, func_call_id: int) -> tuple[str, str]:
    """Return the (visual, join) marker names for one wrapped call.

    W0.2 (TN-D11/D13/D14): names carry identity -- twenty ``conv2d`` calls
    are twenty distinguishable ranges -- and TorchLens's OWN bookkeeping
    calls (an explicit :func:`internal_scalar_read` marker is live, e.g. the
    per-output ``register_hook`` install) are separated under
    ``torchlens::internal::`` so half the ranges on a real capture stop
    masquerading as model work. The classification is allowlist-BY-
    CONSTRUCTION (the same marker discipline as the completeness witness),
    never a stack-frame filename heuristic.

    Parameters
    ----------
    func_name:
        Recorded torch function name.
    func_call_id:
        Process-monotonic wrapped-call id (persisted on the op record as
        ``func_call_id``, which is what makes the join name an exact key).

    Returns
    -------
    tuple[str, str]
        ``(visual_name, join_name)``: the human-readable NVTX/Nsight range
        name and the opaque exact-key ``record_function`` marker name.
    """

    from ._completeness_cross_thread import _internal_read_active

    if _internal_read_active():
        visual = f"torchlens::internal::{func_name}#{func_call_id}"
        join = f"torchlens::internal::{func_call_id}"
    else:
        visual = f"torchlens::{func_name}#{func_call_id}"
        join = f"torchlens::op::{func_call_id}"
    return visual, join


def _no_active_session() -> Any:
    """Fallback session probe when the observability substrate is absent."""

    return None


#: Cached ``torchlens.observability._session.active_session`` (resolved on
#: first wrapped call so the hot path pays one global read, not an import).
_active_session_fn: Callable[[], Any] | None = None


def _push_op_markers(trace: Any, func_name: str, func_call_id: int) -> tuple[bool, Any]:
    """Open the per-call visual/join markers at the op clock's open (W0.3).

    Two independent sinks, each on only when its backend is active:

    - NVTX (``emit_nvtx=True``): Nsight-visual range with an
      identity-carrying name.
    - ``torch.profiler.record_function`` (an active
      ``torchlens.observability.session``): the FLIP-1 op-altitude join
      marker whose name is the exact ``func_call_id`` key.

    Parameters
    ----------
    trace:
        Active trace (carries ``emit_nvtx``).
    func_name:
        Recorded torch function name.
    func_call_id:
        Process-monotonic wrapped-call id.

    Returns
    -------
    tuple[bool, Any]
        ``(nvtx_pushed, record_function_cm)`` pop token consumed by
        :func:`_pop_op_markers`.
    """

    global _active_session_fn
    session_fn: Callable[[], Any] | None = _active_session_fn
    if session_fn is None:
        try:
            from ...observability._session import active_session
        except ImportError:  # pragma: no cover - torn install without the substrate
            session_fn = _no_active_session
        else:
            session_fn = active_session
        _active_session_fn = session_fn
    session = session_fn()
    wants_nvtx = bool(getattr(trace, "emit_nvtx", False))
    if session is None and not wants_nvtx:
        return (False, None)
    visual, join = _op_marker_labels(func_name, func_call_id)
    nvtx_pushed = _nvtx_range_push(visual) if wants_nvtx else False
    record_cm: Any = None
    if session is not None and session.profiler is not None:
        # Sink-side failure must never break the user op (suppress, not a
        # crash): an unmarkered call degrades the join, never the capture.
        with contextlib.suppress(Exception):
            cm = torch.profiler.record_function(join)
            cm.__enter__()
            record_cm = cm
    return (nvtx_pushed, record_cm)


def _pop_op_markers(tokens: tuple[bool, Any]) -> None:
    """Close the per-call markers pushed by :func:`_push_op_markers`.

    Parameters
    ----------
    tokens:
        ``(nvtx_pushed, record_function_cm)`` pop token.
    """

    nvtx_pushed, record_cm = tokens
    if record_cm is not None:
        with contextlib.suppress(Exception):
            record_cm.__exit__(None, None, None)
    _nvtx_range_pop(nvtx_pushed)


__all__ = ["_pop_op_markers", "_push_op_markers"]
