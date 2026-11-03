"""Torch stopped and failed forward-session cleanup for the FINALIZE stage.

Moved verbatim from ``torchlens.backends.torch.backend`` by the capture seam: the bodies of
``TorchBackend.cleanup_stopped_forward_session`` and ``TorchBackend.cleanup_failed_forward_session``,
which bind these functions as methods.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from torchlens._errors import TraceCaptureError, attach_partial_trace, original_exception
from torchlens._utils._exception_notes import annotate_guarded
from torchlens.backends.torch import _tl

from ..._utils._record_group_table import record_member

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace
    from torchlens.backends.torch.backend import TorchBackend


def cleanup_stopped_forward_session(
    self: TorchBackend, session: object, prepared_model: object
) -> None:
    """Clean up torch metadata after a halted forward capture."""
    self.cleanup_model_session(session, prepared_model)
    # F5: the raw-graph workspace may already be gone when the halt fired
    # around a postprocess-tail boundary; a missing workspace must degrade
    # gracefully instead of masking the in-flight signal with
    # AttributeError.
    raw_graph_ws = getattr(session, "__dict__", {}).get("_raw_graph_ws")
    if raw_graph_ws is None:
        return
    raw_layer_dict = raw_graph_ws.raw_layer_dict
    for label in list(raw_layer_dict.keys()):
        entry = raw_layer_dict.get(label)
        if entry is not None and hasattr(entry, "out") and entry.out is not None:
            _tl.clear_meta(entry.out)


def _stamp_actual_buffer_writes(session: object, exc: Exception) -> None:
    """Stamp the failed forward's actual buffer-write record on ``exc``."""

    # Stamp the failed forward's ACTUAL buffer-write record (value-changing
    # journal events) on the exception while the journal is still live —
    # ``cleanup_model_session`` below clears ``capture_events``, and the
    # rescue driver needs this record to refuse a double-forward re-run
    # after an output-attribution failure (R16-2 for the failed-primary
    # trigger). Only an exhaustive session arms the tracker, so only there
    # is an empty journal proof of "no writes".
    if cast("Trace", session)._options_mode == "exhaustive":
        events = getattr(getattr(session, "capture_events", None), "buffer_write_events", None)
        if events is not None:
            with contextlib.suppress(Exception):
                exc._torchlens_actual_buffer_writes = tuple(  # type: ignore[attr-defined]
                    str(record_member(event, "address", None) or "?")
                    for event in events
                    if getattr(event, "value_changed", None) is not False
                )


def _snapshot_failed_predicate_events(session: object) -> None:
    """Snapshot the failing predicate-mode pass's capture events onto ``session``."""

    if cast("Trace", session)._options_mode == "predicate":
        from torchlens._ir.capture_events import CaptureEvents

        events = getattr(session, "capture_events", None)
        if events is not None:
            # Snapshot the failing pass through the sanctioned merge path:
            # concat clones and re-stamps every mergeable lane under the
            # declared merge law, keeping the snapshot's counter coherent
            # with its seq values (a direct lane splice preserved source
            # seqs that collided with the re-stamped op lane).
            failed_fastlog_events = CaptureEvents()
            failed_fastlog_events.concat(events)
            setattr(session, "_failed_fastlog_capture_events", failed_fastlog_events)


def _attach_partial_trace(
    self: TorchBackend,
    session: object,
    exc: Exception,
    partial_trace_cls: Any,
) -> None:
    """Build the partial trace and attach it to the escaping ``exc``, warning without masking.

    ``exc`` is the exception the capture door raises: a ``TraceCaptureError``
    wrapping the forward's exception, or a TorchLens error propagating as
    itself. The partial is built from the forward's own exception and lands on
    ``exc.partial_trace``; the forward's exception is never mutated.
    """
    try:
        partial_trace = partial_trace_cls.from_trace(
            cast("Trace", session), original_exception(exc)
        )
    except Exception as construction_error:
        self._warn_without_masking(
            exc,
            "TorchLens could not construct partial-trace recovery after the "
            f"forward failed: {type(construction_error).__name__}: "
            f"{construction_error}",
            RuntimeWarning,
            stacklevel=3,
        )
        with contextlib.suppress(Exception):
            annotate_guarded(
                exc,
                "TorchLens partial-trace construction also failed: "
                f"{type(construction_error).__name__}: {construction_error}",
            )
    else:
        try:
            attach_partial_trace(exc, partial_trace)
            # B8-45: the SUCCESS path tells the user the recovery exists. A
            # TraceCaptureError says so in its message; a TorchLens error
            # propagating as itself gets a note.
            if not isinstance(exc, TraceCaptureError):
                with contextlib.suppress(Exception):
                    annotate_guarded(
                        exc,
                        "TorchLens attached the partial capture to this error: "
                        "inspect exc.partial_trace.",
                    )
        except Exception as attachment_error:
            self._warn_without_masking(
                exc,
                "TorchLens could not attach the partial trace to the "
                f"{type(exc).__name__} it raises: "
                f"{type(attachment_error).__name__}: {attachment_error}",
                RuntimeWarning,
                stacklevel=3,
            )


def _clear_raw_graph_meta(session: object) -> None:
    """Strip TorchLens metadata from any leftover raw-graph output tensors."""

    # F5: a postprocess-tail failure (step 18-20 or the relation freeze)
    # arrives here AFTER the transient-state seam popped the raw-graph
    # workspace. The unguarded read used to double-fault with
    # AttributeError, MASKING the original exception (it survived only as
    # __context__) and losing partial diagnostics; degrade gracefully so
    # the original failure propagates.
    raw_graph_ws = getattr(session, "__dict__", {}).get("_raw_graph_ws")
    if raw_graph_ws is not None:
        raw_layer_dict = raw_graph_ws.raw_layer_dict
        for label in list(raw_layer_dict.keys()):
            entry = raw_layer_dict.get(label)
            if entry is not None and hasattr(entry, "out") and entry.out is not None:
                _tl.clear_meta(entry.out)


def _capture_attempt_failed_message(exc: Exception) -> str:
    """Return the routed capture-failure advisory text for ``exc``."""

    # B8-44 follow-up (quickstart memo 4.6): the restoration sentence is
    # scoped to what teardown actually restores -- TorchLens
    # instrumentation and the torch environment. Model state mutated by
    # the partially-executed forward (a materialized lazy module, updated
    # BatchNorm running stats) is NOT rolled back, so the historical
    # "the model ... restored" claim was false exactly when a forward
    # materialized a module and then failed.
    return (
        "TorchLens capture attempt failed "
        f"({type(original_exception(exc)).__name__}); TorchLens instrumentation is removed "
        "from the model and the torch environment restored. A default "
        "capture also rolls back the buffers, random state, training "
        "modes and plain attributes the partial forward changed; under "
        "CaptureOptions(carry_state=True) they are not rolled back, and "
        "materialized lazy modules never are. The partial capture rides the "
        "raised error's .partial_trace."
    )


def cleanup_failed_forward_session(
    self: TorchBackend,
    session: object,
    prepared_model: object,
    exc: Exception,
) -> None:
    """Attach the partial trace to the escaping error and clean up capture state.

    ``exc`` is the exception the capture door is about to raise: a
    :class:`~torchlens.errors.TraceCaptureError` wrapping a forward
    exception, or a TorchLens error propagating as itself. The partial
    capture lands on its ``partial_trace``; the forward's own exception
    is never mutated.
    """
    # active_logging's __exit__ already turned off the toggle.
    # Clean up model session state and strip TorchLens metadata from any
    # partially-constructed tensor entries to avoid stale references (#110).
    from ..._partial import PartialTrace

    _stamp_actual_buffer_writes(session, exc)

    _snapshot_failed_predicate_events(session)
    # Partial diagnostics are BEST EFFORT; the model-session teardown is not.
    # The arms below catch ``Exception``, so an interruption (KeyboardInterrupt,
    # SystemExit) raised while building or attaching the partial trace used to
    # escape straight past ``cleanup_model_session`` -- leaving the user's model
    # with TorchLens-forced ``requires_grad``, ``tl_*`` metadata and an
    # installed buffer tracker. The ``finally`` guarantees teardown without
    # swallowing the interruption.
    try:
        _attach_partial_trace(self, session, exc, PartialTrace)
    finally:
        self.cleanup_model_session(session, prepared_model)
    _clear_raw_graph_meta(session)
    # B8-35/B8-44: the historical unconditional stdout banner ("Feature
    # extraction failed; returning model and environment to normal") was
    # factually false on rescue-recovered captures and supported
    # return_partial flows, corrupted machine-readable stdout, and was
    # unfilterable. One accurate ROUTED warning replaces it, naming what
    # actually happened and where the diagnostics live; stacklevel targets
    # the user's tl.trace call through the driver frames. The dedicated
    # category (a RuntimeWarning subclass, so user filters keep matching)
    # lets the rescue driver defer it while a rescue re-run can still
    # swallow this failure — a successful rescue drops the advisory
    # instead of pointing users at an exception they never receive.
    from .rescue import CaptureAttemptFailedWarning

    self._warn_without_masking(
        exc,
        _capture_attempt_failed_message(exc),
        CaptureAttemptFailedWarning,
        stacklevel=4,
    )
