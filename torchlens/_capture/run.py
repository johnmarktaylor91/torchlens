"""Forward-pass control for the RUN stage: ``run_and_log_inputs_through_model``.

The skeleton's order (it matters for correctness):

1. Seed pick, backend resolution, save-selector resolution.
2. Capture-slot reservation, then the user-RNG snapshot and reseed, then
   input placement (the reseed happens BEFORE ``active_logging``: entering
   the logging context may run decorated operations that consume RNG state).
3. Input tensor labeling, event stream, stop directive, capture sessions.
4. Model instrumentation (the WRAP stage's ``instrumented()``), runnable
   state snapshots.
5. The forward pass under ``active_logging``.
6. One of four exit arms: completed (outputs, cleanup,
   postprocess, settlement), halted (``StopSignal``), failed (``Exception``)
   or interrupted (``BaseException``), each ending in settlement and handing
   the trace to the FINALIZE stage through ``ForwardResult``.
7. The ``finally``: the ``_drop_semantic_output_transients`` backstop every
   arm passes through (B1-02; the completed arm drops earlier, right after the
   decode consumer, in ``finalize.arms.finalize_completed``), the teardown
   ladder, the RNG restore and the slot release.
"""

import contextlib
import random
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NoReturn, cast

from torch import get_default_dtype, nn

from torchlens import _state
from torchlens._ir.capture_events import CaptureEvents
from torchlens.backends.registry import BackendUnsupportedError

from .._capture_state_helpers import CompiledCapturePrep
from .._quantities import Duration
from ..backends.registry import (
    TORCH_BACKEND_NAME,
    CaptureBackend,
    get_backend_spec,
)
from ..recording._halt import StopSignal
from .outcome import (
    CapturePhase,
    demote_outcome,
    safe_exception_str,
    set_capture_phase,
)
from .session import (
    attach_capture_events_session,
    attach_legacy_capture_session,
    detach_capture_session,
)
from .stages import CaptureRequest, ForwardArm, ForwardResult, InternalCaptureConfig
from .stop import StopDirective

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
from .._utils._exception_notes import annotate_guarded
from .._utils.display import _timed_phase, _vprint, user_stacklevel
from .._utils.rng import RngStreamsRestore, host_rng_advanced, snapshot_host_rng
from ._forward_helpers import (
    _backend_name_for_trace,
    _capture_backend_from_registry,
    _cleanup_forward_memory_once,
    _clear_saved_activation_dedup_caches,
    _forward_peak_memory_bracket,
    _record_fused_attention_hold,
    _run_predicate_forward_with_root_frame,
    _stamp_host_rng_verdict,
    _structure_only_forward_boundary,
    _warn_unlogged_dynamo_region,
)
from ._input_records import (
    _get_op_nums_from_user_labels,
    _record_runnable_input_literal_leaves,
    _record_runnable_input_structure,
    _record_runnable_input_tensor_sites,
    _record_runnable_module_training_modes,
    _register_model_input_container_snapshots,
)
from ._run_state import _RunState
from .finalize.arms import (
    finalize_completed,
    finalize_failed,
    finalize_halted,
    finalize_interrupted,
)
from .finalize.settle import _drop_semantic_output_transients

_ACTIVE_CAPTURE_BACKEND: CaptureBackend | None = None

_AUTO_SEED_ENTROPY = random.Random()
"""Private entropy stream for ``seed=None`` capture seed picks (R57).

Seeded once from OS entropy at import. Drawing the auto seed from the user's
global ``random`` engine either advanced that stream past the capture's
restore bracket (breaking byte-exact RNG neutrality for a seeded host
process) or, if the draw were bracketed too, made consecutive auto-seeded
captures reuse one identical seed (silently correlating dropout patterns
across runs). A private stream preserves both guarantees; the chosen seed is
always disclosed on ``trace.options.seed``.
"""


def _fetch_label_move_input_tensors(
    session: object,
    input_args: list[Any],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
    model_device: object,
) -> tuple[list[Any], list[str]]:
    """Delegate input tensor movement and source labeling to the active backend.

    Parameters
    ----------
    session:
        Active capture session receiving input-boundary diagnostics.
    input_args:
        Copied positional inputs that may be mutated for internal device moves.
    input_arg_names:
        Forward signature names for positional inputs.
    input_kwargs:
        Copied keyword inputs that may be mutated for internal device moves.
    model_device:
        Device selected by backend input setup.

    Returns
    -------
    tuple[list[Any], list[str]]
        Backend input tensor leaves and their source-address labels.
    """

    backend = _ACTIVE_CAPTURE_BACKEND
    if backend is None:
        spec = get_backend_spec("torch")
        if spec.capture_backend is None:
            raise BackendUnsupportedError(
                f"backend={spec.name!r} does not expose a shared capture Protocol adapter."
            )
        backend = spec.capture_backend()
    return backend.fetch_label_move_input_tensors(
        session,
        input_args,
        input_arg_names,
        input_kwargs,
        model_device,
    )


def _begin_capture(
    self: "Trace",
    model: object,
    input_args: Any,
    input_kwargs: dict[Any, Any] | None,
    seed: int | None,
) -> tuple[CaptureBackend, int]:
    """Pick the seed, reset per-run caches and resolve the capture backend."""

    if seed is None:
        # R57 (neutrality half): the auto-seed pick draws from a PRIVATE
        # entropy stream, never the user's global ``random`` engine: drawing
        # from the global stream either leaked one ``randint`` advance past
        # the restore bracket (capture not byte-neutral to a seeded host
        # process) or, if bracketed, made every ``seed=None`` capture reuse
        # the identical seed (correlated dropout across runs).
        seed = _AUTO_SEED_ENTROPY.randint(1, 4294967294)
    self._options_seed = seed
    # The per-capture code-context cache (and the call-site anchor stored in
    # it) is valid for the stack of ONE capture run: a re-capture on a
    # carried-over Trace state (a live-refresh fork) must never consult a
    # prior run's anchor or locations.
    self._code_context_cache = {}
    backend = _capture_backend_from_registry(
        _backend_name_for_trace(self),
        model,
        input_args,
        input_kwargs,
    )
    backend.set_capture_producer_policy(self, self._options_mode)
    if self._source_model_ref is None:
        # Needed so unlabeled output tensors that are direct registered-buffer
        # reads (``forward`` returning ``self.running_mean`` untouched) can be
        # identified during output extraction. The exhaustive door sets this
        # before calling in; predicate/recorder callers do not, so set it here
        # once, idempotently, for every capture path.
        from ._run_setup import make_weak_model_ref

        self._source_model_ref = make_weak_model_ref(model)  # type: ignore[arg-type]
    return backend, seed


def _resolve_save_selectors(
    self: "Trace",
    layers_to_save: str | list[str | int] | None,
    grad_layers_to_save: str | list[str | int] | None,
) -> None:
    """Resolve the activation and gradient save selectors to raw op numbers."""

    if self._options_mode == "predicate":
        self._layer_nums_to_save = []
        self._grad_op_nums_to_save = []
    else:
        if hasattr(self, "_deferred_retention_selector"):
            self._layer_nums_to_save = []
        elif hasattr(self, "_refresh_resolved_layer_nums_to_save"):
            self._layer_nums_to_save = self.__dict__.pop("_refresh_resolved_layer_nums_to_save")
        else:
            self._layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)  # type: ignore[assignment]
        if hasattr(self, "_deferred_gradient_selector"):
            self._grad_op_nums_to_save = []
        elif hasattr(self, "_refresh_resolved_grad_layer_nums_to_save"):
            self._grad_op_nums_to_save = self.__dict__.pop(
                "_refresh_resolved_grad_layer_nums_to_save"
            )
        else:
            self._grad_op_nums_to_save = _get_op_nums_from_user_labels(self, grad_layers_to_save)

    # Selective captures retain output-layer parents so output payloads remain
    # available when the synthetic output node itself is requested (#46).
    layer_nums_to_save = cast(Any, self._layer_nums_to_save)
    if layer_nums_to_save != "all" and self._tracing_finished:
        output_parent_nums = set()
        for output_label in self.output_layers:
            output_entry = self._layer_dict_all_keys[output_label]
            for parent_label in output_entry.parents:
                parent_entry = self._layer_dict_all_keys[parent_label]
                output_parent_nums.add(parent_entry.raw_index)
        if output_parent_nums:
            combined = set(layer_nums_to_save) | output_parent_nums
            self._layer_nums_to_save = sorted(combined)


def _snapshot_user_rng() -> Callable[[], None]:
    """Snapshot the user's global RNG engines; return a restore-once callable.

    R57 (restore half): capture seeding reseeds the USER'S three global RNG
    engines (random / numpy / torch, plus CUDA). The primary capture used to
    leave the process reseeded permanently; the snapshot is restored on EVERY
    settlement path (the outermost ``finally`` plus the pre-``try`` failure
    windows). The reseed POLICY itself (the fenced R21 fork) is unchanged.
    """

    # The one restore door: construction snapshots, ``restore`` writes back once.
    return RngStreamsRestore().restore


def _claim_slot_and_seed(self: "Trace", run: _RunState, reservation_resume: object | None) -> None:
    """Reserve the capture slot, then seed the RNG and place the inputs.

    The slot is claimed BEFORE any capture-global side effect (label-session
    swap in model prep, compiled-submodule swaps, the recorder's fastlog
    state): a concurrent capture destined for ``ReentrantTraceError`` used to
    run those mutations first and orphan the admitted winner's session. R54
    moved the claim ahead of the RNG snapshot, reseed and input setup too, so
    a refused loser never reseeds the process-global engines mid-window. A
    same-thread re-entry passes only on the recorder's continuation token
    (R55); the reservation is released in the outermost ``finally``.
    """

    run.capture_slot = _state.capture_reservation(resume=reservation_resume)
    run.capture_slot.__enter__()
    try:
        run.restore_rng = _snapshot_user_rng()
        try:
            run.backend.seed_rng(self, run.seed)
            run.input_args, run.input_kwargs, run.input_arg_names, run.model_device = (
                run.backend.setup_inputs_and_device(
                    self, run.model, run.input_args, run.input_kwargs
                )
            )
            # B3R4-R12-1: a non-total namedtuple ``_fields`` schema on any input
            # site refuses typed HERE, for every capture: the tensor-extraction
            # BFS cannot see positional slots of tuple subclasses, so such an
            # input used to lose its tensor leaves silently while settling COMPLETE.
            from torchlens._input_walk import refuse_nontotal_namedtuple_inputs

            refuse_nontotal_namedtuple_inputs(run.input_args, run.input_kwargs)
        except BaseException:
            # A pre-outer-``finally`` setup failure: the seeded engines must
            # not leak to the user.
            run.restore_rng()
            raise
        self._provenance_start_time = time.time()
        # Settlement state for this pass: the phase marker attributes failures
        # to FORWARD/FINALIZE/POSTPROCESS, and any stale stop-request latch
        # from a prior pass on a carried-over Trace must never classify this one.
        set_capture_phase(self, CapturePhase.FORWARD)
        self.__dict__.pop("_stop_requested", None)
    except BaseException:
        # A raise between the reservation claim and the outer ``try`` would
        # otherwise leak the reservation and wedge every later admission.
        run.capture_slot.__exit__(None, None, None)
        run.restore_rng()
        raise


def _fetch_inputs(self: "Trace", run: _RunState) -> None:
    """Move and label the input tensors through the active backend."""

    global _ACTIVE_CAPTURE_BACKEND
    previous_capture_backend = _ACTIVE_CAPTURE_BACKEND
    _ACTIVE_CAPTURE_BACKEND = run.backend
    try:
        input_tensors_any, input_tensor_addresses = _fetch_label_move_input_tensors(
            self,
            run.input_args,
            run.input_arg_names,
            cast(dict[Any, Any], run.input_kwargs),
            run.model_device,
        )
    finally:
        _ACTIVE_CAPTURE_BACKEND = previous_capture_backend
    run.input_tensors = list(input_tensors_any)
    self._raw_graph_ws.input_tensor_addresses = list(input_tensor_addresses)
    self._output_attribution_input_tensors = run.input_tensors
    # RNG state snapshot for deterministic explicit refreshes and legacy
    # two-pass consistency (#58).
    if self._options_mode == "exhaustive":
        self._pre_forward_rng_states = run.backend.snapshot_rng(self)


def _attach_sessions(
    self: "Trace",
    run: _RunState,
    layers_to_save: str | list[str | int] | None,
    grad_layers_to_save: str | list[str | int] | None,
) -> None:
    """Open the event stream, fix the stop directive and attach the capture sessions."""

    self.capture_events = CaptureEvents()
    run.capture_events = self.capture_events
    if not isinstance(getattr(self, "_stop_directive", None), StopDirective):
        self._stop_directive = StopDirective(
            halt_options=getattr(self, "_predicate_save_options", None),
            raise_on_nan=bool(self._options_on_nonfinite),
            forward_error_mode=getattr(
                getattr(self, "_predicate_save_options", None),
                "on_forward_error",
                "raise",
            ),
            inference_only=bool(self._options_inference_only),
        )
    self._capture_config = InternalCaptureConfig(
        capture_mode=str(self._options_mode),
        layers_to_save=layers_to_save,
        grad_layers_to_save=grad_layers_to_save,
        random_seed=run.seed,
        postprocess=run.postprocess,
        stop=self._stop_directive,
    )
    run.capture_session = attach_legacy_capture_session(
        self,
        backend_token=run.backend,
        backend_name=str(_backend_name_for_trace(self)),
        layers_to_save=layers_to_save,
        grad_layers_to_save=grad_layers_to_save,
        seed=run.seed,
        postprocess=run.postprocess,
    )
    attach_capture_events_session(run.capture_events, run.capture_session)


def _enter_instrumentation(
    self: "Trace", run: _RunState, stack: contextlib.ExitStack
) -> CompiledCapturePrep | None:
    """Prepare the model on ``stack``; return its compiled-capture preparation.

    An ``nn.Module`` goes through the WRAP stage's ``instrumented()`` context
    (compiled-submodule unwrapping plus model preparation). A plain callable
    has no compiled slots to unwrap: it is prepared directly and the same
    once-only session cleanup is registered on the stack.
    """

    payload = run.cleanup_payload()
    if isinstance(run.model, nn.Module):
        from ..backends.torch.instrument import instrumented

        instrumented_model = stack.enter_context(
            instrumented(cast(Any, run.backend), self, run.model, payload)
        )
        compiled_prep = instrumented_model.compiled_prep
        return compiled_prep if isinstance(compiled_prep, CompiledCapturePrep) else None
    run.backend.prepare_model_once(run.model)
    run.backend.prepare_model_session(self, run.model)
    stack.callback(run.backend.cleanup_model_session, self, payload)
    return None


def _instrument_model(self: "Trace", run: _RunState) -> None:
    """Enter the WRAP stage: compiled-capture preparation plus model preparation.

    The context is held open on ``run.instrumentation`` and closed by the
    teardown ladder while the capture session is still attached, so its exit
    cleanup is the session's once-only ``model_session`` action (a no-op after
    the arm that already ran it) and the compiled-submodule swaps unwind
    before the forward memory is reclaimed.
    """

    run.instrumentation = contextlib.ExitStack()
    with _timed_phase(self, "ctx_build:model_prepare"):
        run.compiled_prep = _enter_instrumentation(self, run, run.instrumentation)
    self._provenance_setup_duration = Duration(time.time() - self._provenance_start_time)
    _vprint(self, f"Model prepared ({self._provenance_setup_duration:.2f s})")
    if self._options_verbose:
        devices = set()
        for t in run.input_tensors or ():
            if hasattr(t, "device"):
                devices.add(str(t.device))
        device_str = ", ".join(sorted(devices)) if devices else "unknown"
        _vprint(self, f"Inputs: {len(run.input_tensors or ())} tensor(s) on {device_str}")


def _snapshot_runnable_state(self: "Trace", run: _RunState) -> None:
    """Record the bound state a runnable descriptor needs, before the forward."""

    if bool(getattr(self, "is_replayable", False)):
        from .._runnable.state import (
            snapshot_capture_state,
            snapshot_capture_state_signatures,
            snapshot_persistent_buffer_universe,
            snapshot_state_alias_topology,
        )

        model = run.model
        # r37 corr2-4: the live bound-state alias topology (object identity,
        # storage overlap) must be captured BEFORE ``snapshot_capture_state``'s
        # clones erase it; the runnable producer refuses unsupported topologies
        # at save and reproduces identity groups from this record.
        self._runnable.state_alias_topology = snapshot_state_alias_topology(model)
        # r63 C1: per-slot metadata signatures are stamped from the LIVE tensors
        # PRE-clone: the clone compacts ``storage_offset`` and materializes
        # conj/neg, so a post-clone signature is blind to two of the four
        # transport-lossy physical dims.
        self._runnable.capture_state_signatures = snapshot_capture_state_signatures(model)
        self._runnable.capture_state = snapshot_capture_state(model)
        # r77 F2: the persistent-buffer NAME universe survives non-tensor state
        # (``get_extra_state()`` / packed entries), so a dead-model
        # include_weights=False save declares the SAME slot universe as the
        # live lane instead of silently dropping never-forward-used buffers.
        self._runnable.persistent_buffer_universe = snapshot_persistent_buffer_universe(model)

    if str(_backend_name_for_trace(self)) == TORCH_BACKEND_NAME:
        # The provenance manifest needs the capture-time default, never the
        # potentially different save-time default. Runnable-ready captures
        # replace this minimal snapshot with the complete ambient record.
        self._runnable.capture_ambient = {"default_dtype": str(get_default_dtype())}


def _log_inputs(self: "Trace", run: _RunState) -> None:
    """Log the source tensors and the runnable input records inside ``active_logging``."""

    input_tensors = run.input_tensors or []
    input_tensor_addresses = self._raw_graph_ws.input_tensor_addresses
    for i, t in enumerate(input_tensors):
        run.backend.log_source_tensor(self, t, "input", input_tensor_addresses[i])
    input_args, input_kwargs = run.input_args, cast(dict[Any, Any], run.input_kwargs)
    _register_model_input_container_snapshots(self, input_args, input_kwargs)
    _record_runnable_input_literal_leaves(self, input_args, input_kwargs)
    _record_runnable_input_tensor_sites(self, input_args, input_kwargs)
    _record_runnable_input_structure(self, input_args, input_kwargs)
    _record_runnable_module_training_modes(self, run.model)
    if bool(getattr(self, "is_replayable", False)):
        # r35 decision E: capture the ambient backend execution context the
        # forward is about to run under (defaults, matmul precision,
        # determinism, TF32/cuDNN flags, SDP toggles) so the sparse runnable
        # descriptor can restore it explicitly at replay.
        from .._utils._torch_compat import (
            AMBIENT_FP32_UNREPRESENTABLE_KEY,
            read_legacy_fp32_controls,
            snapshot_ambient_execution_context,
        )

        ambient = snapshot_ambient_execution_context()
        # An fp32_precision policy (torch >= 2.9) the legacy record fields
        # cannot express: capture proceeds and the runnable producer refuses
        # typed. The disclosure rides the session-only snapshot mapping.
        _, unrepresentable = read_legacy_fp32_controls()
        if unrepresentable:
            ambient[AMBIENT_FP32_UNREPRESENTABLE_KEY] = unrepresentable
        self._runnable.capture_ambient = ambient
    # The root forward runs undecorated: keep its arguments for the root call.
    self._module_capture_ws.root_forward_args = (input_args, input_kwargs)


def _run_exhaustive_forward(self: "Trace", run: _RunState) -> Any:
    """Run the user forward under the memory, inference and host-RNG brackets.

    The host-RNG bracket lets a runnable descriptor record honestly whether
    Python ``random`` / NumPy control flow ran (TorchLens itself draws no host
    RNG here). r37 hon1_2: the channel monitor additionally observes
    non-global channels (RNG instances, SystemRandom, os entropy, clocks, the
    default_rng factory); any touch is permanently unreplayable, and monitor
    uncertainty downgrades completeness, never reads as no-consumption. The
    monitor is armed ONLY for runnable-capable captures (``replayable``
    is the exact predicate); a disarmed capture stamps ``monitor_uncertain``
    fail-closed. The FAIL-CLOSED verdict is stamped BEFORE the forward: the
    real verdict lands several statements later, so anything raising in
    between used to leave the field ``None``, read as a proven-clean window.
    """

    model = cast(Callable[..., Any], run.model)
    input_args, input_kwargs = run.input_args, cast(dict[Any, Any], run.input_kwargs)
    with _timed_phase(self, "dispatch:forward_model"):
        with _forward_peak_memory_bracket(self, run.model_device):
            with run.backend.inference_context(self):
                self._runnable.rng_monitor_uncertain = True
                self._runnable.rng_monitor_uncertain_detail = ("monitor_verdict_not_stamped",)
                _host_rng_before = snapshot_host_rng()
                if bool(getattr(self, "is_replayable", False)):
                    from .._utils.rng import host_nondeterminism_monitor

                    with host_nondeterminism_monitor(cast(nn.Module, run.model)) as _rng_channels:
                        outputs = model(*input_args, **input_kwargs)
                else:
                    _rng_channels = None
                    with _structure_only_forward_boundary(self):
                        outputs = model(*input_args, **input_kwargs)
                _global_advanced = host_rng_advanced(_host_rng_before, snapshot_host_rng())
                _stamp_host_rng_verdict(self, _global_advanced, _rng_channels)
    return outputs


def _run_forward(self: "Trace", run: _RunState) -> Any:
    """Log the inputs and run the forward pass under ``active_logging``."""

    _vprint(self, f"Running {self._options_mode} forward pass...")
    with run.backend.active_logging(self):
        _warn_unlogged_dynamo_region(self, run.compiled_prep)
        _record_fused_attention_hold(self, run.compiled_prep)
        _log_inputs(self, run)
        if self._options_mode == "predicate":
            with _structure_only_forward_boundary(self):
                return _run_predicate_forward_with_root_frame(
                    self,
                    run.backend,
                    run.model,
                    run.input_args,
                    cast(dict[Any, Any], run.input_kwargs),
                    run.model_device,
                )
        return _run_exhaustive_forward(self, run)


def _forward_result(
    self: "Trace",
    run: _RunState,
    arm: ForwardArm,
    *,
    raw_output: Any = None,
    exc: BaseException | None = None,
) -> ForwardResult:
    """Bundle the forward's live state into the FINALIZE hand-off record.

    Every arm after the forward is one FINALIZE entry; RUN holds the forward
    state (``_RunState``) and FINALIZE the arm order, so the record is built
    here and read there (``finalize_completed`` raises the swallowed-stop
    error into the failed arm, exactly as the inline check did).
    """

    return ForwardResult(
        trace=self,
        request=CaptureRequest(
            model=cast(nn.Module, run.model),
            input_args=run.input_args,
            input_kwargs=run.input_kwargs,
            postprocess=run.postprocess,
            reservation_resume=run.reservation_resume,
        ),
        arm=arm,
        raw_output=raw_output,
        session=run.capture_session,
        exc=exc,
        backend=run.backend,
        model=run.model,
        input_tensors=tuple(run.input_tensors or ()),
        input_args=run.input_args,
        input_kwargs=run.input_kwargs,
    )


def _completed_arm(self: "Trace", run: _RunState, outputs: Any) -> Any:
    """Finish a forward pass that returned: outputs, cleanup, postprocess, settlement."""

    return finalize_completed(_forward_result(self, run, "completed", raw_output=outputs))


def _halted_arm(self: "Trace", run: _RunState, halt_exc: StopSignal) -> Any:
    """Dispatch a halt: return the partial trace's output, or clean up and re-raise."""

    return finalize_halted(_forward_result(self, run, "halted", exc=halt_exc))


def _failed_arm(self: "Trace", run: _RunState, e: Exception) -> NoReturn:
    """Settle a forward that raised an ordinary exception and re-raise it."""

    finalize_failed(_forward_result(self, run, "failed", exc=e))


def _interrupted_arm(self: "Trace", run: _RunState, interrupt_exc: BaseException) -> NoReturn:
    """Tear down and settle after a control-flow interrupt and re-raise it."""

    finalize_interrupted(_forward_result(self, run, "interrupted", exc=interrupt_exc))


def _release_instrumentation(run: _RunState) -> None:
    """Exit the WRAP context (once-only session cleanup, compiled swaps) while attached."""

    instrumentation, run.instrumentation = run.instrumentation, None
    if instrumentation is not None:
        instrumentation.__exit__(*run.exc_info)


def _teardown_ladder(self: "Trace", run: _RunState) -> None:
    """Run the post-settlement teardown; demote the outcome if it faults.

    Path 7: a teardown failure demotes the already-settled COMPLETE/HALTED
    outcome to FAILED/TEARDOWN in both homes and the teardown exception
    propagates. R63 (B8-23 one frame out): an ordinary teardown Exception
    must not replace an unwinding KeyboardInterrupt/SystemExit; the demotion
    stays, the failure is attached, and the ORIGINAL interrupt re-raises.
    """

    inflight_exc = sys.exc_info()[1]
    try:
        try:
            _clear_saved_activation_dedup_caches(self)
            # Release input tensor references so GC can reclaim backend memory.
            run.input_tensors = None
            _release_instrumentation(run)
            try:
                _cleanup_forward_memory_once(self, run.backend, run.capture_session)
            finally:
                if run.capture_session is not None and run.capture_events is not None:
                    detach_capture_session(self, run.capture_events, run.capture_session)
        finally:
            _release_instrumentation(run)
    except BaseException as teardown_exc:
        demote_outcome(
            self,
            run.capture_session,
            note=f"teardown failed: {type(teardown_exc).__name__}: {safe_exception_str(teardown_exc)}",
            exc=teardown_exc,
        )
        if (
            inflight_exc is not None
            and not isinstance(inflight_exc, Exception)
            and isinstance(teardown_exc, Exception)
        ):
            annotate_guarded(
                inflight_exc,
                "TorchLens post-settlement teardown also failed while handling "
                f"this interrupt: {type(teardown_exc).__name__}: {safe_exception_str(teardown_exc)}",
                stacklevel=user_stacklevel(1),
            )
            raise inflight_exc
        raise


def _finish_run(self: "Trace", run: _RunState) -> None:
    """The outermost ``finally``: transient drop, teardown, RNG restore, slot release."""

    # B1-02: the backstop every settlement path passes through (the completed
    # arm already dropped in ``finalize_completed``; the drop is idempotent),
    # placed before the teardown ladder so a teardown double-fault cannot skip it.
    _drop_semantic_output_transients(self)
    try:
        _teardown_ladder(self, run)
    finally:
        # Outermost: a teardown double-fault must not leak the capture
        # reservation, or every later admission refuses forever. The RNG
        # restore (R57) runs first but can never displace the release.
        try:
            run.restore_rng()
        finally:
            run.capture_slot.__exit__(None, None, None)


def run_and_log_inputs_through_model(
    self: "Trace",
    model: object,
    input_args: Any | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[str | int] | None = "all",
    grad_layers_to_save: str | list[str | int] | None = "all",
    seed: int | None = None,
    postprocess: bool = True,
    reservation_resume: object | None = None,
) -> Any:
    """Run one forward pass under capture and log everything into ``self``.

    The RUN stage's control skeleton; the step order is documented in the
    module docstring. Returns the forward's output (or the halted output).
    """

    backend, seed = _begin_capture(self, model, input_args, input_kwargs, seed)
    _resolve_save_selectors(self, layers_to_save, grad_layers_to_save)
    run = _RunState(
        backend=backend,
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        seed=seed,
        postprocess=postprocess,
        reservation_resume=reservation_resume,
    )
    _claim_slot_and_seed(self, run, reservation_resume)
    try:
        _fetch_inputs(self, run)
        _attach_sessions(self, run, layers_to_save, grad_layers_to_save)
        _instrument_model(self, run)
        _snapshot_runnable_state(self, run)
        outputs = _run_forward(self, run)
        return _completed_arm(self, run, outputs)
    except StopSignal as halt_exc:
        run.exc_info = sys.exc_info()
        return _halted_arm(self, run, halt_exc)
    except Exception as e:
        run.exc_info = sys.exc_info()
        _failed_arm(self, run, e)
        # Unreachable (the arm is ``NoReturn``: ``finalize_failed`` re-raises
        # ``e`` itself). Kept because the blind-except ratchet (BLE001) reads
        # a broad handler as legitimate only when the handler re-raises; the
        # explicit raise is what keeps these two handlers out of its count.
        raise e
    except BaseException as interrupt_exc:
        run.exc_info = sys.exc_info()
        _interrupted_arm(self, run, interrupt_exc)
        raise  # unreachable, kept for the blind-except ratchet (see above)
    finally:
        _finish_run(self, run)
