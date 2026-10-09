"""Public Recorder context manager for fastlog sessions."""

from __future__ import annotations

import time
import traceback as traceback_module
import warnings
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import torch
from torch import nn

from .. import _state
from .._deprecations import MISSING, MissingType
from .._errors import CaptureContextError, KeywordConflictError
from .._training_validation import TrainingModeConfigError, reject_compiled_model
from ..capture._module_aliases import _capture_module_aliases
from ..capture.config import InternalCaptureConfig
from ..capture.outcome import safe_exception_repr, safe_exception_str
from ..capture.predicates import validate_followed_by_capability
from ..capture.projections import (
    RecordingState,
    _empty_recording,
    active_recording_state,
)
from ..capture.stop import StopDirective, stop_directive_for_trace
from ..capture.trace import _extract_and_mark_outputs
from ..data_classes.trace import Trace
from ..intervention.predicates import InterventionPredicate
from ..ir import CaptureEvents
from ..options import StreamingOptions
from ..types import ActivationPostfunc, GradientPostfunc
from ..utils._torch_compat import get_fsdp_wrapper_type
from ._halt import HaltSignal
from ._validation import validate_recording_options
from .exceptions import RecorderStateError
from .options import (
    ForwardErrorMode,
    GradPredicateFn,
    HaltPredicateFn,
    LookbackPayloadPolicy,
    PredicateErrorMode,
    PredicateFn,
    merge_recording_options,
)
from .types import CaptureSpec, Recording, _mark_recording_halted


def _rank_prefixed_streaming_options(
    streaming: StreamingOptions | None | MissingType,
) -> StreamingOptions | None | MissingType:
    """Return streaming options with a rank-local directory prefix.

    Parameters
    ----------
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    StreamingOptions | None | MissingType
        Options with ``bundle_path`` rewritten to include ``rank_NN`` when a
        bundle path is configured.
    """

    if isinstance(streaming, MissingType) or streaming is None or streaming.bundle_path is None:
        return streaming
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    bundle_path = Path(streaming.bundle_path)
    # Rewrite ONLY bundle_path: rebuilding from defaults silently dropped the
    # R62 include_* opt-outs (and now the async-write routing) for rank-split
    # recordings.
    return StreamingOptions(
        bundle_path=bundle_path.parent / f"rank_{rank:02d}" / bundle_path.name,
        retain_in_memory=streaming.retain_in_memory,
        out_callback=streaming.out_callback,
        include_custom_attributes=streaming.include_custom_attributes,
        include_buffer_values=streaming.include_buffer_values,
        async_writes=streaming.async_writes,
        max_pending_bytes=streaming.max_pending_bytes,
    )


def _warn_zero_match_capture_selectors(state: RecordingState) -> None:
    """Warn when sparse capture selectors matched no sites.

    Parameters
    ----------
    state:
        Completed recording state carrying all retained records and fire counts.

    Returns
    -------
    None
        Emits at most one warning for each configured selector slot.
    """

    from ..intervention.selectors import BaseSelector

    save_selector = state.options.keep_op
    if isinstance(save_selector, BaseSelector):
        save_matched = any(bool(save_selector(record.ctx)) for record in state.recording.records)
        if not save_matched:
            warnings.warn(
                f"Capture-time save selector {save_selector!r} matched zero sites; "
                "no activations were selected by it.",
                UserWarning,
                stacklevel=3,
            )
    intervene_selector = getattr(state.options.intervene, "selector", None)
    intervene_decision = getattr(state.options.intervene, "decision", None)
    if (
        isinstance(intervene_selector, BaseSelector)
        and getattr(intervene_decision, "direction", None) == "forward"
        and state.intervene_selector_fire_count == 0
    ):
        warnings.warn(
            f"Capture-time intervention selector {intervene_selector!r} matched zero sites; "
            "no intervention fired.",
            UserWarning,
            stacklevel=3,
        )


def _unwrap_ddp_for_fastlog(
    model: nn.Module,
    streaming: StreamingOptions | None | MissingType,
) -> tuple[nn.Module, StreamingOptions | None | MissingType]:
    """Unwrap DDP/DataParallel wrappers for rank-local fastlog capture.

    Parameters
    ----------
    model:
        Candidate model supplied to fastlog.
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    tuple[nn.Module, StreamingOptions | None | MissingType]
        The model to execute and possibly rewritten streaming options.
    """

    # The lazy probe never imports torch.distributed.fsdp on plain captures.
    fsdp_wrapper_type = get_fsdp_wrapper_type()
    if fsdp_wrapper_type is not None and isinstance(model, fsdp_wrapper_type):
        raise CaptureContextError(
            "torchlens.fastlog does not support FullyShardedDataParallel (FSDP): "
            "parameters are sharded across ranks and there is no unsharded module to log",
            code="fsdp_capture_unsupported",
            remedy="record the unsharded module before FSDP wrapping",
        )

    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        distributed_data_parallel: type[nn.Module] | None = None
    else:
        distributed_data_parallel = DistributedDataParallel

    if distributed_data_parallel is not None and isinstance(model, distributed_data_parallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    return model, streaming


def _resolve_train_mode_default(
    *,
    field_name: str,
    value: bool | CaptureSpec | MissingType,
    backward_ready: bool,
) -> bool | CaptureSpec | MissingType:
    """Resolve one default capture option for train-mode sugar."""

    if backward_ready and value is MISSING:
        return CaptureSpec(keep_grad=True, save_out=True, save_metadata=True)
    if not backward_ready or value is MISSING or value is False:
        return value
    if value is True:
        raise TrainingModeConfigError(
            f"backward_ready=True conflicts with {field_name}=True because True uses "
            "keep_grad=False; use CaptureSpec(keep_grad=True) or omit the default. "
            "Remedy: pass CaptureSpec(keep_grad=True) or omit the default.",
            code="backward_ready_conflict",
        )
    if isinstance(value, CaptureSpec) and not value.keep_grad:
        raise TrainingModeConfigError(
            f"backward_ready=True conflicts with {field_name}=CaptureSpec(keep_grad=False). "
            "Remedy: pass CaptureSpec(keep_grad=True) or omit the default.",
            code="backward_ready_conflict",
        )
    return value


class Recorder:
    """Context manager for explicitly captured fastlog forwards."""

    def __init__(
        self,
        model: nn.Module,
        *,
        save: PredicateFn | None | MissingType = MISSING,
        default_op: bool | CaptureSpec | MissingType = MISSING,
        default_module: bool | CaptureSpec | MissingType = MISSING,
        history_size: int | MissingType = MISSING,
        lookback: int | MissingType = MISSING,
        lookback_payload_policy: LookbackPayloadPolicy | MissingType = MISSING,
        include_source_events: bool | MissingType = MISSING,
        intervene: InterventionPredicate | None | MissingType = MISSING,
        halt: HaltPredicateFn | None | MissingType = MISSING,
        max_predicate_failures: int | MissingType = MISSING,
        on_predicate_error: PredicateErrorMode | MissingType = MISSING,
        on_forward_error: ForwardErrorMode | MissingType = MISSING,
        storage: StreamingOptions | None | MissingType = MISSING,
        streaming: StreamingOptions | None | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
        activation_transform: ActivationPostfunc | None | MissingType = MISSING,
        save_raw_activations: bool | MissingType = MISSING,
        save_grads: GradPredicateFn | bool | CaptureSpec | None | MissingType = MISSING,
        default_grad: bool | CaptureSpec | MissingType = MISSING,
        grad_transform: GradientPostfunc | None | MissingType = MISSING,
        save_raw_gradients: bool | MissingType = MISSING,
        backward_ready: bool = False,
        echo: Any | None | MissingType = MISSING,
    ) -> None:
        """Initialize a recorder and perform construction-time validation.

        Parameters
        ----------
        model:
            PyTorch module to record.
        save, default_op, default_module, history_size,
        lookback, lookback_payload_policy, include_source_events, max_predicate_failures,
        on_predicate_error, storage, streaming, random_seed:
            Fastlog recording options.
        on_forward_error:
            Controls failed-forward handling. ``"raise"`` preserves the
            historical behavior. ``"attach_partial"`` attaches a failed partial
            ``Recording`` to ``exc.partial_recording`` and re-raises.
            ``"return_partial"`` returns ``None`` from the failed ``log()`` and
            exposes the failed partial on ``recorder.recording``.
        halt:
            Optional predicate evaluated after each event's save decision. Returning
            ``True`` stops the active forward pass and marks the recording halted.
        activation_transform:
            Optional callable applied to each retained out copy after
            dtype/device transforms. The callable runs under ``pause_logging``
            and must return a ``torch.Tensor``. Errors are wrapped in
            :class:`torchlens.TorchLensPostfuncError`.
        save_raw_activations:
            When ``False`` and ``activation_transform`` is set, only the
            transformed payload is retained on the record. Defaults to
            ``True`` to mirror the slow path.
        backward_ready:
            If True, omitted defaults are promoted to ``CaptureSpec(keep_grad=True)``.
        """

        reject_compiled_model(model, api_name="torchlens.fastlog.Recorder")
        storage_supplied = storage is not MISSING and storage is not None
        streaming_supplied = streaming is not MISSING and streaming is not None
        if storage_supplied and streaming_supplied:
            raise KeywordConflictError(
                "Do not pass both `storage` and `streaming`",
                code="storage_argument_conflict",
                remedy="prefer storage=, or remove one of the two arguments",
            )
        resolved_streaming = storage if storage_supplied else streaming
        unwrapped_model, streaming = _unwrap_ddp_for_fastlog(model, resolved_streaming)
        default_op = _resolve_train_mode_default(
            field_name="default_op",
            value=default_op,
            backward_ready=backward_ready,
        )
        default_module = _resolve_train_mode_default(
            field_name="default_module",
            value=default_module,
            backward_ready=backward_ready,
        )
        self.model = unwrapped_model
        # echo= normalization runs at construction so the typed refusals
        # (bad spellings, finalized-label selectors) fire BEFORE any forward.
        if echo is MISSING or echo is None or echo is False:
            echo_normalized = None
        else:
            from ..snoop import normalize_echo

            echo_normalized = normalize_echo(echo)
        self.options = merge_recording_options(
            recording=None,
            keep_op=save,
            default_op=default_op,
            default_module=default_module,
            history_size=history_size,
            lookback=lookback,
            lookback_payload_policy=lookback_payload_policy,
            include_source_events=include_source_events,
            intervene=intervene,
            halt=halt,
            max_predicate_failures=max_predicate_failures,
            on_predicate_error=on_predicate_error,
            on_forward_error=on_forward_error,
            streaming=streaming,
            random_seed=random_seed,
            activation_transform=activation_transform,
            save_raw_activations=save_raw_activations,
            save_grads=save_grads,
            default_grad=default_grad,
            grad_transform=grad_transform,
            save_raw_gradients=save_raw_gradients,
            echo=echo_normalized if echo_normalized is not None else MISSING,
        )
        validate_recording_options(self.options)
        validate_followed_by_capability(
            self.options.keep_op,
            api_name="record(save=...)",
            supports_retroactive=False,
        )
        self._state: RecordingState | None = None
        self._recording: Recording | None = None
        self._capture_events: CaptureEvents | None = None
        self._output_tensors: list[torch.Tensor] = []
        self._output_tensor_addresses: list[str] = []
        self._captured_run_cores: list[Any] = []
        # Alias address -> canonical address of the recorded model's shared
        # modules, kept from each pass so the post-recording zero-match check
        # resolves a selector's alias spelling as the forward did.
        self._module_aliases: dict[str, str] = {}
        self._entered = False
        self._exited = False
        self._failed = False
        self._next_pass_index = 1
        self._echo_session: Any | None = None
        if self.options.echo is not None:
            from ..snoop import EchoSession

            # One narrator per Recorder: per-forward state resets each pass,
            # the configured scope/sink/stats persist (snoop D1).
            self._echo_session = EchoSession(self.options.echo, tier="record")

    def __enter__(self) -> Recorder:
        """Enter the recorder resource scope."""

        if self._entered or self._exited:
            raise RecorderStateError("Recorder cannot be re-entered")
        recording = _empty_recording(self.options)
        self._state = RecordingState(options=self.options, recording=recording)
        object.__setattr__(recording, "_recording_state", self._state)
        self._capture_events = CaptureEvents()
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize or abort the recorder resource scope."""

        _ = exc_type, traceback
        if not self._entered or self._exited or self._state is None:
            raise RecorderStateError("Recorder is not active")
        if self._failed:
            self._entered = False
            self._exited = True
            return
        if exc_value is None:
            self._state.finalize_storage()
            session = type("_FastlogCaptureSession", (), {})()
            session.capture_events = self._capture_events
            session.output_tensors = self._output_tensors
            session.output_tensor_addresses = self._output_tensor_addresses
            session._fastlog_recording = self._state.recording
            session.recording_state = self._state
            session.captured_run_cores = self._captured_run_cores
            self._recording = Recording.from_capture_events(session)
            # Settle the finalized product (path 14). A construction failure
            # above propagates productless -- the raise is the signal.
            from ..capture.outcome import (
                CaptureOutcome,
                CaptureStatus,
                stamp_recording_outcome,
            )

            if getattr(self._recording, "_outcome", None) is None:
                recording = self._recording
                if recording.halted:
                    stamped = CaptureOutcome(
                        status=CaptureStatus.HALTED,
                        reason=recording.halt_reason,
                        boundary_label=recording.halt_reason,
                    )
                else:
                    stamped = CaptureOutcome(status=CaptureStatus.COMPLETE)
                stamp_recording_outcome(recording, stamped)
        else:
            self._state.abort_storage(safe_exception_str(exc_value))
        self._entered = False
        self._exited = True
        if exc_value is None:
            if self._echo_session is not None:
                recording = self._state.recording
                self._echo_session.finish("halted" if recording.halted else "complete")
            from ..ir.selector_eval import module_alias_scope

            with module_alias_scope(self._module_aliases):
                _warn_zero_match_capture_selectors(self._state)
            self._state.raise_accumulated_predicate_error()
        elif self._echo_session is not None:
            # Interrupts and with-body failures get a best-effort flush; the
            # in-log() crash tail (if any) already fired.
            self._echo_session.on_interrupt()

    def log(
        self,
        input_args: Any,
        input_kwargs: dict[str, Any] | None = None,
        *,
        sample_id: str | int | None = None,
    ) -> Any:
        """Capture one forward pass and return the model output.

        Parameters
        ----------
        input_args:
            Tensor, tuple, or list of positional model inputs.
        input_kwargs:
            Optional keyword arguments for the model call.
        sample_id:
            Optional caller-provided sample identifier stored on event contexts.

        Returns
        -------
        Any
            Model forward output.
        """

        if not self._entered or self._exited or self._state is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        if self._failed:
            raise RecorderStateError(
                "Recorder is in a failed state after a forward exception; "
                "create a new Recorder for further captures."
            )
        output = self._run_unified_capture(input_args, input_kwargs, sample_id=sample_id)
        self._next_pass_index += 1
        return output

    def _run_unified_capture(
        self,
        input_args: Any,
        input_kwargs: dict[str, Any] | None,
        *,
        sample_id: str | int | None,
    ) -> Any:
        """Run one unified predicate capture pass and retain CaptureEvents."""

        if self._state is None or self._capture_events is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        trace = Trace(
            model_class_name=str(type(self.model).__name__),
            activation_transform=self.options.activation_transform,
            save_raw_activations=self.options.save_raw_activations,
            # The backward hook leg reads grad policy from the TRACE
            # (tensor_tracking._build_fastlog_grad_payloads), not from
            # RecordingOptions; omitting these silently ignored
            # grad_transform and save_raw_gradients on every Recorder /
            # record() backward (explorer rider: Recorder backward coverage).
            grad_transform=self.options.grad_transform,
            save_raw_gradients=self.options.save_raw_gradients,
            detach_saved_activations=False,
            backward_ready=True,
        )
        trace.capture_mode = "predicate"
        trace._fastlog_recording = self._state.recording
        trace._predicate_save_options = self.options
        trace._stop_directive = StopDirective(
            halt_options=self.options,
            raise_on_nan=False,
            forward_error_mode=self.options.on_forward_error,
            inference_only=False,
        )
        trace._capture_config = InternalCaptureConfig(
            capture_mode="predicate",
            layers_to_save=[],
            grad_layers_to_save=[],
            random_seed=self.options.random_seed,
            postprocess=False,
            stop=trace._stop_directive,
        )
        self._reset_state_for_pass(sample_id=sample_id)
        if self._echo_session is not None:
            self._echo_session.reset_pass(self._next_pass_index)
            self._echo_session.bind_trace(trace)
        self._state.recording.start_times.append(time.time())
        try:
            # The reservation must wrap the recording-state install: a refused
            # concurrent record() used to overwrite the admitted recorder's
            # RecordingState for the window until its inner refusal unwound,
            # projecting the winner's events into the loser's state. The inner
            # orchestration re-enters the reservation same-thread (passthrough).
            with (
                _state.capture_reservation() as reservation_token,
                active_recording_state(self._state),
            ):
                output = trace._run_and_log_inputs_through_model(
                    self.model,
                    input_args,
                    input_kwargs,
                    layers_to_save=[],
                    grad_layers_to_save=[],
                    random_seed=self.options.random_seed,
                    postprocess=False,
                    reservation_resume=reservation_token,
                )
        except HaltSignal as halt_exc:
            captured_run_core = trace.__dict__.pop("_fastlog_captured_run_core", None)
            if captured_run_core is not None:
                self._captured_run_cores.append(captured_run_core)
            self._absorb_pass_events(trace)
            object.__setattr__(
                self._state.recording,
                "n_ops",
                max(self._state.recording.n_ops, self._next_pass_index),
            )
            self._mark_halted_pass(self._next_pass_index, halt_exc)
            if self._echo_session is not None:
                # Halted is not failed: no crash tail, one typed disclosure.
                self._echo_session.note_line("-- echo: capture halted (not failed) --")
            output = None
            return output
        except Exception as exc:
            captured_run_core = trace.__dict__.pop("_fastlog_captured_run_core", None)
            if captured_run_core is not None:
                self._captured_run_cores.append(captured_run_core)
            forward_disposition = stop_directive_for_trace(trace).forward_disposition(exc)
            if self._echo_session is not None:
                # Crash tail (snoop D5 tail 2): flush synchronously, extend the
                # exception's note when it propagates. Original exception wins.
                echo_note = self._echo_session.on_forward_failure(exc)
                if echo_note is not None and forward_disposition != "return_partial":
                    add_note = getattr(exc, "add_note", None)
                    if add_note is not None:
                        add_note(echo_note)
            if forward_disposition == "raise":
                self._state.abort_storage(safe_exception_str(exc))
                raise
            partial_build_failed = False
            try:
                partial = self._mark_recording_failed(trace, exc)
                self._failed = True
                self._recording = partial
                if forward_disposition == "attach_partial":
                    exc.partial_recording = partial  # type: ignore[attr-defined]
            except Exception:
                partial_build_failed = True
            if partial_build_failed:
                raise
            if forward_disposition == "attach_partial":
                raise
            return None
        finally:
            self._state.recording.end_times.append(time.time())
            self._module_aliases.update(_capture_module_aliases(trace) or {})
        # Output tensors are extracted+marked inside _run_and_log_inputs_through_model
        # (postprocess=False branch) BEFORE it cleans up model session metadata, so
        # buffer-output attribution isn't racing the label wipe. Read the stashed
        # scratch results back rather than re-extracting from the now-cleaned-up model.
        output_tensors = trace.__dict__.pop("_fastlog_output_tensors", None)
        output_tensor_addresses = trace.__dict__.pop("_fastlog_output_tensor_addresses", None)
        captured_run_core = trace.__dict__.pop("_fastlog_captured_run_core", None)
        if captured_run_core is not None:
            self._captured_run_cores.append(captured_run_core)
        if output_tensors is None or output_tensor_addresses is None:
            # Defensive fallback only; the postprocess=False branch above always
            # populates these on a normal return.
            output_tensors, output_tensor_addresses = _extract_and_mark_outputs(trace, output)
        trace.__dict__.pop("_output_attribution_input_tensors", None)
        self._absorb_pass_events(trace)
        trace.capture_events = self._capture_events
        trace._capture_events = self._capture_events
        self._state.runtime_trace = trace
        self._state.intervene_selector_fire_count += int(
            getattr(trace, "_tl_intervene_selector_fire_count", 0)
        )
        self._output_tensors = output_tensors
        self._output_tensor_addresses = output_tensor_addresses
        object.__setattr__(
            self._state.recording,
            "n_ops",
            max(self._state.recording.n_ops, self._next_pass_index),
        )
        return output

    def _absorb_pass_events(self, trace: Trace) -> None:
        """Fold one pass's capture stream into the recorder's journal.

        The per-pass ``trace`` created in :meth:`_run_unified_capture` owns its
        own ``CaptureEvents`` while the forward runs: model preparation emits
        one ``ModulePrepEvent`` per module onto it (with each module's real
        ``address_children`` / source metadata), the wrapper hot path emits the
        op events, and user pre-hook provenance lands in its own lane.
        ``Recording.to_trace()`` rebuilds a fresh ``Trace`` from exactly the
        recorder's accumulated journal, so those structure facts must fold
        across or ``_build_root_module_log`` degrades to an address-children
        fallback that breaks the module-hierarchy invariant.

        The fold is one :meth:`CaptureEvents.concat` call under the declared
        merge law: module structure lanes are first-run-only (multi-pass
        recordings re-emit identical prep events every pass), op and pre-hook
        lanes append with re-stamped seq, and run-local lanes (output
        versions, buffer writes, backward) never merge.
        """

        if self._capture_events is None:
            return
        source = getattr(trace, "capture_events", None)
        if source is None or source is self._capture_events:
            return
        self._capture_events.concat(source)

    def _mark_halted_pass(self, pass_index: int, halt_exc: HaltSignal) -> None:
        """Persist halt state for the given pass."""

        if self._state is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        _mark_recording_halted(self._state.recording, pass_index, halt_exc.reason)
        from ..capture.outcome import CaptureOutcome, CaptureStatus, stamp_recording_outcome

        stamp_recording_outcome(
            self._state.recording,
            CaptureOutcome(
                status=CaptureStatus.HALTED,
                reason=halt_exc.reason,
                boundary_kind=getattr(halt_exc, "boundary_kind", None),
                boundary_label=getattr(halt_exc, "boundary_label", None) or halt_exc.reason,
            ),
        )

    def _mark_recording_failed(self, trace: Trace, exc: BaseException) -> Recording:
        """Build and stamp a failed partial recording for a forward exception.

        Parameters
        ----------
        trace:
            Live trace whose predicate event stream contains the events captured
            before the exception.
        exc:
            Original forward exception. Only string metadata is copied from it.

        Returns
        -------
        Recording
            Failed partial recording. user-op failures exclude the failing call;
            TL-side capture failures may include a skipped/partial current-call
            event.
        """

        if self._state is None or self._capture_events is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        self._state.abort_storage(safe_exception_str(exc))
        failed_events = getattr(trace, "_failed_fastlog_capture_events", None)
        if failed_events is None:
            raise RecorderStateError(
                "failed-capture event snapshot missing; cannot build a faithful partial"
            )
        combined_events = CaptureEvents()
        combined_events.concat(self._capture_events)
        if failed_events is not self._capture_events:
            # The failing pass contributes only its op and pre-hook facts;
            # module structure from a partially-executed forward is not
            # trusted (matching the historical recovery behavior).
            combined_events.concat(failed_events, lanes=("op_events", "pre_hook_events"))
        self._capture_events = combined_events
        trace.capture_events = combined_events
        trace._capture_events = combined_events
        self._state.runtime_trace = trace
        session = type("_FastlogCaptureSession", (), {})()
        session.capture_events = self._capture_events
        session.output_tensors = []
        session.output_tensor_addresses = []
        session._fastlog_recording = self._state.recording
        session.recording_state = self._state
        session.captured_run_cores = self._captured_run_cores
        recording = Recording.from_capture_events(session)
        self._stamp_failed_recording(recording, exc)
        return recording

    def _stamp_failed_recording(self, recording: Recording, exc: BaseException) -> None:
        """Stamp string-only failure metadata onto a frozen recording."""

        op_events = tuple(
            self._capture_events.amended_op_records() if self._capture_events is not None else ()
        )
        # B8-46: the exception unwind records module-exit events AFTER the
        # failure point (a mid-forward submodule failure leaves a trailing
        # ``boom:exit:1`` / ``root:exit:1`` run), so the raw last event names
        # the unwind, not the failure frontier. Skip trailing module-exit
        # events so the best-effort ``last_event_*`` metadata points at the
        # deepest event that actually ran before the failure; if every event
        # is a module exit, keep the raw tail rather than reporting nothing.
        last_event = None
        for event in reversed(op_events):
            event_kind = getattr(getattr(event, "record_context", None), "kind", None)
            if event_kind != "module_exit":
                last_event = event
                break
        if last_event is None and op_events:
            last_event = op_events[-1]
        last_ctx = getattr(last_event, "record_context", None)
        successful_op_labels = [
            str(getattr(event, "label_raw", getattr(event, "label", "")))
            for event in op_events
            if getattr(getattr(event, "record_context", None), "kind", None) == "op"
        ]
        object.__setattr__(recording, "status", "partial_error")
        object.__setattr__(recording, "failed", True)
        object.__setattr__(recording, "error_repr", safe_exception_repr(exc))
        object.__setattr__(
            recording,
            "error_traceback",
            "".join(traceback_module.format_exception(type(exc), exc, exc.__traceback__)),
        )
        object.__setattr__(recording, "n_ops_completed", len(successful_op_labels))
        object.__setattr__(
            recording,
            "last_successful_op_label",
            successful_op_labels[-1] if successful_op_labels else None,
        )
        object.__setattr__(
            recording,
            "last_event_label",
            None if last_ctx is None else str(getattr(last_ctx, "label", "")),
        )
        object.__setattr__(
            recording,
            "last_event_func",
            None if last_event is None else getattr(last_event, "func_name", None),
        )
        source_line = None if last_event is None else getattr(last_event, "source_line", None)
        object.__setattr__(
            recording, "last_event_source_line", str(source_line) if source_line else None
        )
        input_meta = None if last_ctx is None else getattr(last_ctx, "input_meta", None)
        object.__setattr__(
            recording,
            "last_event_input_meta",
            repr(input_meta) if input_meta is not None else None,
        )
        recoverable_path = self._recoverable_temp_bundle_path()
        if recoverable_path is not None:
            object.__setattr__(recording, "bundle_path", recoverable_path)
        # Settle the failed product: mirror the scratch trace's settled record
        # (the orchestrator's finally already classified phase/origin) with
        # the recorder-level committed-op count.
        from dataclasses import replace as dataclass_replace

        from ..capture.outcome import (
            CaptureOutcome,
            CaptureStatus,
            classify_failure_origin,
            outcome_for,
            stamp_recording_outcome,
        )

        runtime_trace = self._state.runtime_trace if self._state is not None else None
        settled = outcome_for(runtime_trace) if runtime_trace is not None else None
        if settled is not None and settled.status in (
            CaptureStatus.FAILED,
            CaptureStatus.ABORTED_NONFINITE,
        ):
            stamped = dataclass_replace(settled, n_ops_committed=recording.n_ops_completed)
        else:
            stamped = CaptureOutcome(
                status=CaptureStatus.FAILED,
                origin=classify_failure_origin(exc),
                reason=safe_exception_str(exc),
                error_type=type(exc).__name__,
                n_ops_committed=recording.n_ops_completed,
            )
        stamp_recording_outcome(recording, stamped)

    def _recoverable_temp_bundle_path(self) -> Path | None:
        """Return the fastlog temp bundle path when it has a recoverable index."""

        if self._state is None:
            return None
        writer = getattr(self._state.storage_backend, "writer", None)
        tmp_path = getattr(writer, "tmp_path", None)
        if not isinstance(tmp_path, Path):
            return None
        if not (tmp_path / "fastlog_index.jsonl").exists():
            return None
        return tmp_path

    def _reset_state_for_pass(self, *, sample_id: str | int | None) -> None:
        """Reset per-pass predicate state while preserving accumulated events."""

        if self._state is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        self._state.history.clear()
        self._state.op_counts.clear()
        self._state.module_stack.clear()
        self._state.sample_id = sample_id
        self._state.pass_index = self._next_pass_index
        self._state.event_index = 0
        self._state.step_index = 0

    def log_backward(
        self,
        loss: torch.Tensor,
        *,
        save_grads: (GradPredicateFn | bool | CaptureSpec | None) = None,
        default_grad: bool | CaptureSpec | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> Recording:
        """Run backward for the active recorder and retain selected gradients."""

        if not self._entered or self._exited or self._state is None:
            raise RecorderStateError("Recorder.log_backward() requires an active with-block")
        self._state.recording.log_backward(
            loss,
            save_grads=save_grads,
            default_grad=default_grad,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )
        return self._state.recording

    @property
    def recording(self) -> Recording:
        """Return the finalized recording after context-manager exit."""

        if self._recording is None:
            raise RecorderStateError("Recorder.recording is only available after __exit__")
        return self._recording
