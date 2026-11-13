"""Session facts of the torch wrapper, read once per capture session.

The wrapper body (``wrappers.torch_func_decorator``) used to read a handful of
facts on every logged call that cannot change while one capture session runs:
whether the session records call-input container structure, whether it emits
NVTX ranges, whether it samples device memory, which observability-session
reader the op markers use, and, once Dynamo is loaded, that the clone-on-write
window stays closed. :func:`session_facts` reads them at the session's first
logged call and keeps them until the session ends.

A session is identified by the ``_state.CallFingerprint`` its capture context
mints on entry (a fresh object per entry) together with the trace's id, so a
fact changed between two captures is read again by the next one. Calls without
a fingerprint read the facts afresh every time. Facts that can change inside a
forward stay per-call reads: the observability session itself (only its
reader is kept), the functional-transform, compile and tracing-pass regions,
and the autocast state.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from ... import _state
from ..._utils import _compiled_write_guard
from ..._utils._torch_symbols import tensor_read
from ..._utils.tensor_utils import _DEFER_ENABLED as _COW_ENABLED

#: The pop token of a call that opened no op marker (``_push_op_markers``'s own value).
NO_MARKERS: tuple[bool, Any] = (False, None)


class SessionFacts:
    """The wrapper's per-session facts for one capture session."""

    __slots__ = (
        "container_structure",
        "cow_closed",
        "fingerprint",
        "markers_nvtx",
        "session_reader",
        "trace_id",
        "track_device_memory",
    )

    def __init__(self, trace: Any, fingerprint: Any, session_reader: Any = None) -> None:
        """Read every session fact of ``trace`` once."""

        #: The session key: the capture context's fingerprint and the trace's id.
        self.fingerprint = fingerprint
        self.trace_id = id(trace) if trace is not None else None
        #: ``register_call_input_container_snapshots`` returns at once without it.
        self.container_structure = bool(getattr(trace, "_capture_container_structure", False))
        #: ``_push_op_markers`` opens an NVTX range for every call with it.
        self.markers_nvtx = bool(getattr(trace, "_options_emit_nvtx", False))
        #: The observability-session reader; the session it returns is read per call.
        self.session_reader = session_reader
        self.track_device_memory = getattr(trace, "track_device_memory", False)
        #: Set once the clone-on-write window state is ``None`` for good this session.
        self.cow_closed = False


#: The facts of the session that logged the latest call (owner thread only).
FACTS = SessionFacts(None, None)


def session_facts(
    trace: Any, fingerprint: Any, resolve_session_reader: Callable[[], Any]
) -> SessionFacts:
    """Return the facts of the running session, reading them on its first call.

    Parameters
    ----------
    trace:
        The active capture trace.
    fingerprint:
        ``_state._call_fingerprint`` as the wrapper read it for this call.
    resolve_session_reader:
        ``_op_markers.resolve_active_session_fn``, passed in so this module stays
        out of the wrapper package's import cycle.

    Returns
    -------
    SessionFacts
        The kept facts when this call belongs to the session that read them;
        otherwise freshly read facts, kept only when the session has a fingerprint.
    """

    global FACTS
    facts = FACTS
    if fingerprint is not None and facts.fingerprint is fingerprint and facts.trace_id == id(trace):
        return facts
    facts = SessionFacts(trace, fingerprint, resolve_session_reader())
    if fingerprint is not None:
        FACTS = facts
    return facts


def cow_window_state(facts: SessionFacts, trace: Any, scalar_read: Callable[[], Any]) -> Any:
    """Return :func:`_cow_payload_state_ptrs` for this call, skipping it once it is final.

    Once Dynamo is loaded (``compiled_writes_seen``, which never turns back) and
    the trace's state is cached, every later read returns ``None``; the first
    read still runs in full, so its one-time materialization happens where it did.
    ``scalar_read`` is ``_completeness_origins.internal_scalar_read``, passed in so
    this module stays out of the wrapper package's import cycle.
    """

    if facts.cow_closed:
        return None
    state = _cow_payload_state_ptrs(trace, scalar_read)
    if state is None and _compiled_write_guard.compiled_writes_seen():
        facts.cow_closed = True
    return state


# Per-trace cache for the clone-on-write eligibility decision. Keyed weakly:
# a Trace attribute would trip the portable-state completeness scrub
# (_PORTABLE_STATE_SPEC), and the decision is wrapper-internal state anyway.
_COW_STATE_PTRS_CACHE: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
_COW_UNSET = object()
# The last trace asked and its cached state, one tuple swapped whole so a reader never pairs
# one trace with another's state; weak, so no trace is kept alive. The per-call read of the
# cache above without the ``suppress`` frames and the weak-dictionary lookup.
_COW_LAST_TRACE: list[Any] = [(None, _COW_UNSET)]


def _cow_payload_state_ptrs(trace: Any, scalar_read: Callable[[], Any]) -> Any:
    """Return the clone-on-write payload window state for one capture trace.

    Computed once per trace and cached weakly: ``None`` when deferred payload
    clones must stay disabled for this capture, else the frozenset of model
    param/buffer storage pointers that must never be deferred (their bytes can
    move through C++ side effects no wrapped-call signature announces, e.g.
    train-mode ``batch_norm`` running-stat updates).

    Eligibility is deliberately narrow — the plain default capture only. Every
    mode where saved payloads interact with other machinery (runnable
    witnesses, backward capture, gradient saving, user activation transforms
    that may mutate the payload in place, predicate recording, streaming
    writers, non-copy save modes) keeps the historical eager clone.
    ``inference_only`` (implied by ``no_grad``) only adds ``no_grad``: eligible.
    """
    last = _COW_LAST_TRACE[0]
    cached: Any = last[1] if last[0] is not None and last[0]() is trace else _COW_UNSET
    if cached is _COW_UNSET:
        with suppress(TypeError):
            cached = _COW_STATE_PTRS_CACHE.get(trace, _COW_UNSET)
            _COW_LAST_TRACE[0] = (weakref.ref(trace), cached)
    if cached is not _COW_UNSET:
        return None if _compiled_write_guard.compiled_writes_seen() else cached
    ptrs: Any = None
    if (
        _COW_ENABLED
        and not getattr(trace, "is_replayable", False)
        and not trace._options_backward_ready
        and trace._options_save_grads in (None, False)
        and trace._options_activation_transform is None
        and trace._options_mode != "predicate"
        and trace._options_save_mode == "copy"
        and getattr(trace, "_out_writer", None) is None
        and not _compiled_write_guard.compiled_writes_possible()
    ):
        model_ref = getattr(trace, "_source_model_ref", None)
        model = model_ref() if callable(model_ref) else None
        if model is not None and hasattr(model, "parameters"):
            try:
                with _state.pause_logging(), scalar_read():
                    collected: set[int] = set()
                    for p in model.parameters():
                        collected.add(tensor_read(p, "untyped_storage")(p).data_ptr())
                    for b in model.buffers():
                        collected.add(tensor_read(b, "untyped_storage")(b).data_ptr())
                ptrs = frozenset(collected)
            except Exception:
                ptrs = None
    with suppress(TypeError):
        _COW_STATE_PTRS_CACHE[trace] = ptrs
        _COW_LAST_TRACE[0] = (weakref.ref(trace), ptrs)
    return ptrs


__all__ = [
    "FACTS",
    "NO_MARKERS",
    "SessionFacts",
    "cow_window_state",
    "session_facts",
]
