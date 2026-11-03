"""Exit arms of the FINALIZE stage.

Each entry runs one exit arm of the forward pass exactly as it ran inline in
``torchlens._capture.run.run_and_log_inputs_through_model`` (the bodies moved
here verbatim; the order of every arm is the one pinned by
``tests/test_capture_settlement.py``). RUN calls one entry per arm with the
``ForwardResult`` it built and nothing else; the ``finally`` teardown ladder
stays in RUN.

Arms and what each returns:

- ``finalize_completed``: the forward's outputs (the model's return value).
- ``finalize_halted``: the halted output when the halt frontier returns a
  partial trace; otherwise re-raises the stop signal.
- ``finalize_failed``: re-raises the forward's exception after settlement.
- ``finalize_interrupted``: re-raises the interrupt after settlement.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, NoReturn, cast

from ..._output_slot import returned
from ..._quantities import Duration
from ..._utils._exception_notes import annotate_guarded
from ..._utils.display import _vprint, user_stacklevel
from ..._utils.tensor_utils import synchronize_pending_cpu_async_copies
from .._trace_capture_error import failed_forward_error
from ..outcome import (
    CapturePhase,
    StopSignalSwallowedError,
    count_committed_ops,
    safe_exception_str,
    set_capture_phase,
    settle_completed,
    settle_failed,
    settle_halted,
)
from ..stop import cleared_inference_only_directive
from .settle import (
    _drop_semantic_output_transients,
    _finalize_halted_trace,
    _publish_streamed_bundle_at_settlement,
    _scrub_failed_capture_transients,
    _settle_interrupted_halted_arm,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ...recording._halt import StopSignal
    from ..session import CaptureSession
    from ..stages import ForwardResult


def _model_session_args(result: ForwardResult) -> tuple[Any, Any, tuple[Any, Any]]:
    """Return the ``(model, input_tensors, (args, kwargs))`` tuple the backend's cleanup takes."""
    return (result.model, result.input_tensors, (result.input_args, result.input_kwargs))


def _postprocess_requested(result: ForwardResult) -> bool:
    """Return whether this run materializes the final ``Trace`` immediately."""
    request = result.request
    return True if request is None else bool(request.postprocess)


def _raise_if_stop_swallowed(trace: Trace) -> None:
    """Refuse a normal return whose stop-request latch is still set (F6 boundary checkpoint).

    A "normal" forward return with the stop-request latch set means user code
    swallowed the control signal (halt or nonfinite abort) in a broad except.
    The capture must never be blessed COMPLETE; the typed error settles FAILED
    through the normal failure arm. Covers ``tl.trace`` and every Recorder pass.
    """
    swallowed_stop = trace.__dict__.get("_stop_requested")
    if swallowed_stop is None:
        return
    raise StopSignalSwallowedError(
        "TorchLens raised a "
        f"{'halt' if swallowed_stop.kind == 'halt' else 'non-finite abort'} "
        "stop signal during this forward, but the forward returned "
        "normally: user code swallowed the control signal (typically a "
        "broad `except:` or `except BaseException:` around the model "
        "body). The capture cannot be trusted as complete. Stop "
        f"boundary: {swallowed_stop.boundary_label or swallowed_stop.reason!r}.",
        kind=swallowed_stop.kind,
        boundary_label=swallowed_stop.boundary_label,
    )


def _fence_device_work(trace: Trace) -> None:
    """Fence async copies and drain deferred nonfinite flags before any host consumer.

    Every cpu_async D2H copy recorded this forward is fenced BEFORE any
    host-side consumer (finalize, postprocess digests, ``op.out``,
    ``Trace.save`` serialization) can observe partial bytes. Deferred
    track_nonfinite device flags settle here, after the forward is complete,
    so the record costs one batch of scalar reads instead of a per-op device
    synchronization (the flags' kernels are long done).
    """
    # R36-1 (the fence) and the deferred-flag drain, in that order.
    from ..._data_classes._nonfinite import drain_pending_nonfinite

    synchronize_pending_cpu_async_copies()
    drain_pending_nonfinite(trace)


def finalize_completed(result: ForwardResult) -> Any:
    """Run the completed arm and return the forward's outputs.

    Order (pinned): swallowed-stop check; ``cleared_inference_only_directive``;
    phase FINALIZE; ``synchronize_pending_cpu_async_copies``;
    ``drain_pending_nonfinite``; ``finalize_forward_session``; raw output
    through the output transform; ``decode_outputs_for_trace``;
    ``_drop_semantic_output_transients``; then the recorder path or the trace
    path.

    Parameters
    ----------
    result:
        Forward result on the completed arm; ``raw_output`` is the model's
        return value.

    Returns
    -------
    Any
        The model's outputs.
    """
    trace, backend, outputs = result.trace, result.backend, result.raw_output
    _raise_if_stop_swallowed(trace)
    # an implied inference_only holds only if the forward discarded its graph.
    cleared = cleared_inference_only_directive(trace._stop_directive, outputs)
    if cleared is not None:  # before settlement stamps the flag on the outcome
        trace._options_inference_only, trace._stop_directive = False, cleared
    set_capture_phase(trace, CapturePhase.FINALIZE)
    _fence_device_work(trace)
    backend.finalize_forward_session(trace, trace._raw_graph_ws)

    trace._output_slot = returned(outputs)
    output_transform = trace._output_transform
    trace.transformed_output = output_transform(outputs) if output_transform is not None else None
    from ..._autoroute._builtin_output import decode_outputs_for_trace

    decode_outputs_for_trace(trace, outputs)
    # Tight window on the normal arm: drop immediately after the only
    # consumer. RUN's outer ``finally`` repeats this for the arms that never
    # reach here (halt, failure, interrupt) -- see B1-02.
    _drop_semantic_output_transients(trace)

    trace._cost_forward_duration = Duration(
        time.time() - trace._provenance_start_time - trace._provenance_setup_duration
    )
    _vprint(
        trace,
        f"Forward pass complete ({trace._cost_forward_duration:.2f s}, "
        f"{len(trace.capture_events.op_events)} raw operations)",
    )
    if not _postprocess_requested(result):
        return _complete_recorder_path(result)
    return _complete_trace_path(result)


def _complete_recorder_path(result: ForwardResult) -> Any:
    """Finish a completed forward that skips postprocess (the Recorder path).

    Extract/mark output tensors BEFORE cleanup, mirroring the trace path.
    ``cleanup_model_session()`` strips TorchLens tensor metadata from every
    model-owned tensor (buffers included, via ``_undecorate_model_tensors``);
    extracting afterward would let output-attribution race against that wipe.
    Callers that skip postprocess (the Recorder) read these scratch results
    back off the trace and pop them immediately.
    """
    trace, backend, outputs = result.trace, result.backend, result.raw_output
    capture_session = cast("CaptureSession", result.session)
    output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(trace, outputs)
    trace._fastlog_output_tensors = list(output_tensors_any)
    trace._fastlog_output_tensor_addresses = output_tensor_addresses
    capture_session.snapshot_recording_projection(
        trace,
        output_tensors=list(output_tensors_any),
        output_tensor_addresses=output_tensor_addresses,
    )
    trace._fastlog_captured_run_core = capture_session.seal()
    trace.__dict__.pop("_output_attribution_input_tensors", None)
    backend.cleanup_model_session(trace, _model_session_args(result))
    trace._provenance_end_time = time.time()
    trace.__dict__.pop("_capture_producer_policy", None)
    settle_completed(trace, capture_session)
    return outputs


def _complete_trace_path(result: ForwardResult) -> Any:
    """Finish a completed forward that materializes the ``Trace`` now (the door path)."""
    trace, backend, outputs = result.trace, result.backend, result.raw_output
    capture_session = cast("CaptureSession", result.session)
    output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(trace, outputs)
    output_tensors = list(output_tensors_any)
    trace.__dict__.pop("_output_attribution_input_tensors", None)

    backend.cleanup_model_session(trace, _model_session_args(result))
    _vprint(trace, f"Postprocessing {len(trace.capture_events.op_events)} operations...")
    set_capture_phase(trace, CapturePhase.POSTPROCESS)
    trace._postprocess(output_tensors, output_tensor_addresses)
    trace.__dict__.pop("_capture_producer_policy", None)
    settle_completed(trace, capture_session)
    _publish_streamed_bundle_at_settlement(trace)
    return outputs


def _halt_returns_partial_trace(trace: Trace) -> bool:
    """Return whether the halt frontier hands back a partial trace instead of raising."""
    options = trace.__dict__.get("_predicate_save_options")
    return (
        options is not None
        and getattr(options, "halt", None) is not None
        and bool(trace._halt_returns_partial_trace)
    )


def finalize_halted(result: ForwardResult) -> Any:
    """Run the halted arm: return the halted output or re-raise the stop signal.

    Order (pinned): ``_finalize_halted_trace`` and ``settle_halted`` when the
    frontier returns a partial trace; otherwise the recorder seal, stopped
    session cleanup and ``settle_halted`` before the stop signal re-raises.

    Parameters
    ----------
    result:
        Forward result on the halted arm; ``exc`` is the ``StopSignal``.

    Returns
    -------
    Any
        The halted output, when the halt frontier returns a partial trace.
    """
    trace, halt_exc = result.trace, cast("StopSignal", result.exc)
    capture_session = result.session
    if _halt_returns_partial_trace(trace):
        halted_output = _finalize_halted_partial(result, halt_exc)
        trace.__dict__.pop("_capture_producer_policy", None)
        settle_halted(
            trace,
            capture_session,
            halt_exc,
            finalize_partial=True,
            postprocess_ran=_postprocess_requested(result),
        )
        _publish_streamed_bundle_at_settlement(trace)
        return halted_output
    _cleanup_halted_session(result, halt_exc)
    trace.__dict__.pop("_capture_producer_policy", None)
    settle_halted(
        trace,
        capture_session,
        halt_exc,
        finalize_partial=False,
        postprocess_ran=False,
    )
    raise halt_exc


def _finalize_halted_partial(result: ForwardResult, halt_exc: StopSignal) -> Any:
    """Finalize the partial trace of a halt; settle FAILED or interrupted when that fails."""
    trace, capture_session = result.trace, result.session
    try:
        return _finalize_halted_trace(
            trace,
            result.backend,
            halt_exc,
            result.model,
            list(result.input_tensors),
            _postprocess_requested(result),
        )
    except Exception as secondary_exc:
        # Halted-finalization secondary failure: the capture settles
        # FAILED (FINALIZE or POSTPROCESS per the finalizer's phase
        # markers) with a mandatory disclosure note; the secondary
        # exception propagates with the HaltSignal chained, exactly
        # as before settlement existed.
        trace.__dict__.pop("_capture_producer_policy", None)
        settle_failed(
            trace,
            capture_session,
            secondary_exc,
            note=(f"halted finalization failed after halt at {getattr(halt_exc, 'reason', '')!r}"),
        )
        _scrub_failed_capture_transients(trace)
        raise
    except BaseException as halt_interrupt_exc:
        # Halted-arm interrupt: ``except Exception`` above cannot see a
        # KeyboardInterrupt/SystemExit, which used to escape with NO
        # settlement stamp -- the product read UNKNOWN only through the
        # fail-closed no-sidecar default instead of a settled record.
        # Stamp FAILED/interrupted like the interrupted arm.
        trace.__dict__.pop("_capture_producer_policy", None)
        _settle_interrupted_halted_arm(trace, capture_session, halt_interrupt_exc, halt_exc)
        raise


def _cleanup_halted_session(result: ForwardResult, halt_exc: StopSignal) -> None:
    """Seal the recorder projection and clean the stopped session; settle a secondary failure."""
    trace, backend, capture_session = result.trace, result.backend, result.session
    try:
        # Same double-fault fence as the failed arm: a raising seal must not
        # skip the model-session teardown.
        try:
            if capture_session is not None and not _postprocess_requested(result):
                capture_session.snapshot_recording_projection(trace)
                trace._fastlog_captured_run_core = capture_session.seal()
        finally:
            backend.cleanup_stopped_forward_session(trace, _model_session_args(result))
    except Exception as secondary_exc:
        trace.__dict__.pop("_capture_producer_policy", None)
        settle_failed(
            trace,
            capture_session,
            secondary_exc,
            note=(f"halted cleanup failed after halt at {getattr(halt_exc, 'reason', '')!r}"),
        )
        _scrub_failed_capture_transients(trace)
        raise
    except BaseException as halt_interrupt_exc:
        # Same halted-arm interrupt stamp as the finalize path above: the
        # seal/cleanup seam's ``except Exception`` cannot see an interrupt,
        # which used to escape unsettled.
        trace.__dict__.pop("_capture_producer_policy", None)
        _settle_interrupted_halted_arm(trace, capture_session, halt_interrupt_exc, halt_exc)
        raise


def finalize_failed(result: ForwardResult) -> NoReturn:
    """Run the failed arm and re-raise the forward's exception.

    Order (pinned): ``count_committed_ops``; nonfinite-prefix finalization when
    postprocess; recorder seal then ``cleanup_failed_forward_session`` (a
    raising seal never skips cleanup); ``settle_failed`` and the transient
    scrub in ``finally``; a ``TraceCaptureError`` chained from a non-TorchLens
    forward exception (or the TorchLens error itself) re-raises.

    Parameters
    ----------
    result:
        Forward result on the failed arm; ``exc`` is the forward's exception.
    """
    trace, e = result.trace, result.exc
    assert e is not None
    # Boundary facts snapshot eagerly at cause time: cleanup below may pop
    # the event stream (or double-fault on already-popped workspaces), and
    # the stamp in ``finally`` must still carry the committed-op count.
    committed_ops = count_committed_ops(trace)
    # Aborted-nonfinite label unification (observe item 1): postprocess
    # the committed prefix with the offending tensor seeded as the output
    # frontier, so every public surface reads FINAL labels through the
    # step-8 identity map instead of leaking raw spellings. Failure-safe
    # by contract: any finalization failure attaches as secondary
    # evidence on ``e`` and this arm proceeds exactly as before.
    if _postprocess_requested(result):
        from .._nonfinite_prefix import maybe_finalize_nonfinite_prefix

        maybe_finalize_nonfinite_prefix(
            trace, result.backend, e, result.model, list(result.input_tensors)
        )
    # The door raises one typed TraceCaptureError for a non-TorchLens
    # forward exception (chained ``from e``; ``e`` itself is never
    # mutated) and TorchLens's own errors as themselves. Built BEFORE the
    # cleanup so the message can read the capture frontier, and handed to
    # the backend so the partial capture and every note ride it.
    escaping = failed_forward_error(trace, cast(Exception, e), committed_ops)
    try:
        _cleanup_failed_session(result, escaping)
    finally:
        # Guaranteed settlement: a cleanup double-fault still stamps the
        # terminal outcome before the (original or secondary) exception
        # escapes; exception identity/chaining is byte-identical to the
        # pre-settlement arms. The transient scrub runs after settlement
        # (R11/R32: the escaping partial must not pin live input tensors
        # or the payload-bearing event stream).
        settle_failed(trace, result.session, e, num_committed_ops=committed_ops)
        _scrub_failed_capture_transients(trace)
    if escaping is e:
        raise e
    raise escaping from e


def _cleanup_failed_session(result: ForwardResult, e: BaseException) -> None:
    """Seal the recorder projection and clean the failed session; note a double fault on ``e``.

    ``e`` is the exception the door is about to raise: a ``TraceCaptureError``
    wrapping the forward's exception, or a TorchLens error as itself.
    """
    trace, backend, capture_session = result.trace, result.backend, result.session
    # The seal runs FIRST (it reads live capture state that cleanup
    # strips) but must not be able to SKIP cleanup: a raising seal used
    # to bypass requires_grad restore, tl_* metadata stripping, buffer
    # tracker uninstall and end_label_session, leaving the user's model
    # permanently altered by a failed capture.
    try:
        try:
            if capture_session is not None and not _postprocess_requested(result):
                capture_session.snapshot_recording_projection(trace)
                trace._fastlog_captured_run_core = capture_session.seal()
        finally:
            backend.cleanup_failed_forward_session(trace, _model_session_args(result), e)
        trace.__dict__.pop("_capture_producer_policy", None)
    except Exception as cleanup_exc:
        # grind-r6 b1 R06 (sol MED, probe): the PRIMARY user error must
        # propagate. An ordinary seal/cleanup double-fault used to
        # escape INSTEAD of ``raise e`` -- the settled CaptureOutcome
        # named the primary while the escaping exception was the
        # secondary and carried no partial trace. Mirror the interrupt
        # arm: attach the secondary as a note and re-raise the primary.
        # A BaseException secondary (Ctrl-C during cleanup) keeps
        # escaping -- interrupts always win (B8-23 doctrine).
        annotate_guarded(
            e,
            "TorchLens failed-forward cleanup also failed while handling "
            f"this error: {type(cleanup_exc).__name__}: {safe_exception_str(cleanup_exc)}",
            stacklevel=user_stacklevel(1),
        )


def finalize_interrupted(result: ForwardResult) -> NoReturn:
    """Run the interrupted arm and re-raise the interrupt.

    ``except Exception`` handles ordinary failed-forward diagnostics, but user
    code may raise e.g. ``KeyboardInterrupt`` or a custom ``BaseException``.
    The torch session forces gradient-capable parameters to require grads, so
    its teardown must run before re-raising any such escape. Order (pinned):
    ``count_committed_ops``; ``cleanup_model_session`` (a failure attaches as
    a note); ``settle_failed(interrupted=True)`` and the scrub in ``finally``
    (a failure attaches as a note); the interrupt re-raises.

    Parameters
    ----------
    result:
        Forward result on the interrupted arm; ``exc`` is the interrupt.
    """
    trace, interrupt_exc = result.trace, result.exc
    assert interrupt_exc is not None
    committed_ops = count_committed_ops(trace)
    try:
        try:
            result.backend.cleanup_model_session(trace, _model_session_args(result))
        except Exception as cleanup_exc:
            # B8-23: the PRIMARY control-flow exception (KeyboardInterrupt /
            # SystemExit) must propagate. Letting an ordinary cleanup
            # Exception escape here demoted the KI to ``__context__``, and
            # a caller's ``except Exception`` retry loop swallowed Ctrl-C
            # outright. Attach the cleanup failure instead of raising it.
            annotate_guarded(
                interrupt_exc,
                "TorchLens model-session cleanup also failed while handling "
                f"this interrupt: {type(cleanup_exc).__name__}: "
                f"{safe_exception_str(cleanup_exc)}",
                stacklevel=user_stacklevel(1),
            )
        trace.__dict__.pop("_capture_producer_policy", None)
    finally:
        try:
            settle_failed(
                trace,
                result.session,
                interrupt_exc,
                interrupted=True,
                num_committed_ops=committed_ops,
            )
            _scrub_failed_capture_transients(trace)
        except Exception as settle_exc:
            # B8-23 applies here too: an ordinary settlement/scrub failure
            # inside this ``finally`` would replace the unwinding
            # KeyboardInterrupt/SystemExit (demoting it to __context__),
            # letting a caller's ``except Exception`` swallow Ctrl-C.
            # Attach the failure to the interrupt instead; an unsettled
            # outcome reads UNKNOWN (most restrictive), never blessed.
            annotate_guarded(
                interrupt_exc,
                "TorchLens settlement/scrub also failed while handling "
                f"this interrupt: {type(settle_exc).__name__}: {safe_exception_str(settle_exc)}",
                stacklevel=user_stacklevel(1),
            )
    raise interrupt_exc


__all__ = [
    "finalize_completed",
    "finalize_failed",
    "finalize_halted",
    "finalize_interrupted",
]
