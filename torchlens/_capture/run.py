"""Forward-pass control for the RUN stage: ``run_and_log_inputs_through_model``.

Moved verbatim from ``torchlens._capture.trace`` by the capture seam.
"""

import contextlib
import random
import sys
import time
import warnings
from collections.abc import Callable
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from torch import get_default_dtype, nn

from torchlens import _state
from torchlens.backends.registry import BackendUnsupportedError

from .._capture_state_helpers import CompiledCapturePrep, prepare_compiled_capture
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
    CaptureSession,
    attach_capture_events_session,
    attach_legacy_capture_session,
    detach_capture_session,
)
from .stages import CaptureRequest, ForwardArm, ForwardResult, InternalCaptureConfig
from .stop import StopDirective

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
from .._utils.display import _timed_phase, _vprint
from .._utils.rng import (
    host_rng_advanced,
    log_current_rng_states,
    set_rng_from_saved_states,
    snapshot_host_rng,
)
from ._forward_helpers import (
    _backend_name_for_trace,
    _capture_backend_from_registry,
    _cleanup_forward_memory_once,
    _clear_saved_activation_dedup_caches,
    _forward_peak_memory_bracket,
    _run_predicate_forward_with_root_frame,
    _structure_only_forward_boundary,
)
from ._input_records import (
    _get_op_nums_from_user_labels,
    _record_runnable_input_literal_leaves,
    _record_runnable_input_structure,
    _record_runnable_input_tensor_sites,
    _record_runnable_module_training_modes,
    _register_model_input_container_snapshots,
)
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
always disclosed on ``trace.random_seed``.
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
    """Core orchestration: run a forward pass and log everything into Trace.

    Execution order (ordering matters for correctness):
      1. Set RNG seed (MUST happen before active_logging — see below).
      2. Resolve ``layers_to_save`` to internal tensor numbers.
      3. Normalize/copy inputs, detect device.
      4. Move inputs to model device.
      5. Capture RNG state for explicit-refresh reproducibility.
      6. Prepare model (one-time decoration + per-session hooks).
      7. Enter ``active_logging()`` context — toggles ``_state._logging_enabled``.
      8. Log source tensors (inputs), then run ``model(*args, **kwargs)``.
      9. Exit logging context, extract/mark outputs, clean up, postprocess.

    RNG ordering constraint: backend seeding and snapshots happen BEFORE
    ``active_logging()`` because entering the logging context may trigger
    decorated operations (e.g., module hooks) that consume RNG state. The fast
    pass restores the same pre-forward RNG state so stochastic layers produce
    identical graph structure.
    """
    if seed is None:
        # R57 (neutrality half): the auto-seed pick draws from a PRIVATE
        # entropy stream, never the user's global ``random`` engine. Drawing
        # from the global stream either leaked one ``randint`` advance past
        # the restore bracket below (capture not byte-neutral to a seeded
        # host process) or, if bracketed, made every ``seed=None``
        # capture reuse the identical seed (correlated dropout across runs).
        # The private stream keeps both properties: byte-exact global-engine
        # neutrality AND fresh seeds per capture.
        seed = _AUTO_SEED_ENTROPY.randint(1, 4294967294)
    self.random_seed = seed  # type: ignore[assignment]
    # The per-capture code-context cache (and the call-site anchor stored
    # inside it) is only valid for the stack of ONE capture run: the anchor
    # frame is proven alive by identity for the duration of a single capture,
    # but a re-capture on a carried-over Trace state (e.g. a live-refresh
    # fork) must never consult a prior run's anchor or locations.
    self._code_context_cache = {}
    backend = _capture_backend_from_registry(
        _backend_name_for_trace(self),
        model,
        input_args,
        input_kwargs,
    )
    backend.set_capture_producer_policy(self, self.capture_mode)

    if getattr(self, "_source_model_ref", None) is None:
        # Needed so unlabeled output tensors that are direct registered-buffer
        # reads (e.g. ``forward`` returning ``self.running_mean`` untouched)
        # can be identified during output extraction. The exhaustive
        # ``tl.trace()`` entry point (user_funcs.py) sets this before calling
        # into this function; predicate/fastlog callers (tl.record()) do not,
        # so set it here once, idempotently, for every capture path.
        from .._visualization.code_panel import make_weak_model_ref

        self._source_model_ref = make_weak_model_ref(model)  # type: ignore[arg-type]

    if self.capture_mode == "predicate":
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
            output_entry = self.layer_dict_all_keys[output_label]
            for parent_label in output_entry.parents:
                parent_entry = self.layer_dict_all_keys[parent_label]
                output_parent_nums.add(parent_entry.raw_index)
        if output_parent_nums:
            combined = set(layer_nums_to_save) | output_parent_nums
            self._layer_nums_to_save = sorted(combined)

    # Reserve the capture slot BEFORE any capture-global side effect (label
    # session swap in model prep, compiled-submodule swaps, the fastlog
    # recording state installed by the recorder around this call): a
    # concurrent capture destined for the typed ``ReentrantTraceError`` used
    # to run those mutations first and orphan the admitted winner's session
    # (runtime-probed ``capture_verified=False``). r8 R54 moved the claim
    # ahead of the RNG snapshot + reseed and input/device setup too -- a
    # refused loser used to reseed the process-global RNG engines mid-window
    # (corrupting the admitted winner's replay determinism) and the refusal
    # path skipped the R57 restore entirely; refusing FIRST covers that path
    # for free (nothing is seeded yet). Same-thread re-entry from the
    # recorder's outer reservation passes through only by presenting the
    # recorder's continuation token (R55: a bare same-thread re-entry is a
    # nested public capture from user code inside the window and refuses);
    # the reservation is released in the outermost ``finally`` below.
    capture_slot = _state.capture_reservation(resume=reservation_resume)
    capture_slot.__enter__()
    _rng_restorers: list[Any] = []
    try:
        # R57 (restore half): capture seeding reseeds the USER'S three global
        # RNG engines (random / numpy / torch, plus CUDA); the sibling refresh
        # and fast-run paths snapshot and restore around their reseeds, but
        # the primary capture used to leave the process reseeded permanently
        # -- code after ``tl.trace()`` silently continued from the capture's
        # stream, not the user's. Snapshot the pre-seed states here and
        # restore them on EVERY settlement path (the outermost ``finally``
        # below, plus the pre-``try`` failure windows). The reseed POLICY
        # itself (whether capture seeds at all) is the fenced R21 fork and is
        # deliberately unchanged.
        pre_capture_rng_states = log_current_rng_states()
        rng_restore_pending = [True]

        def _restore_user_global_rng() -> None:
            """Restore the user's pre-capture global RNG streams exactly once."""

            if rng_restore_pending[0]:
                rng_restore_pending[0] = False
                set_rng_from_saved_states(pre_capture_rng_states)

        _rng_restorers.append(_restore_user_global_rng)
        backend.seed_rng(self, seed)
        try:
            input_args, input_kwargs, input_arg_names, model_device = (
                backend.setup_inputs_and_device(
                    self,
                    model,
                    input_args,
                    input_kwargs,
                )
            )
            # B3R4-R12-1: a non-total namedtuple `_fields` schema on any input
            # site refuses typed HERE, for every capture. The tensor-extraction
            # BFS cannot see positional slots of tuple subclasses, so such an
            # input used to lose its tensor leaves silently (no input node,
            # parents dropped, the gap misattributed to a stale-reference
            # escape) while settling COMPLETE.
            from torchlens._input_walk import refuse_nontotal_namedtuple_inputs

            refuse_nontotal_namedtuple_inputs(input_args, input_kwargs)
        except BaseException:
            # A pre-outer-``finally`` setup failure: the seeded engines must
            # not leak to the user.
            _restore_user_global_rng()
            raise

        self.capture_start_time = time.time()
        # Settlement state for this pass: the phase marker attributes failures
        # to FORWARD/FINALIZE/POSTPROCESS, and any stale stop-request latch
        # from a prior pass on a carried-over Trace must never classify this
        # one.
        set_capture_phase(self, CapturePhase.FORWARD)
        self.__dict__.pop("_stop_requested", None)
    except BaseException:
        # A raise between the reservation claim and the outer ``try`` would
        # otherwise leak the reservation and wedge every later admission.
        for restorer in _rng_restorers:
            restorer()
        capture_slot.__exit__(None, None, None)
        raise
    input_tensors: list[Any] = []
    capture_session: CaptureSession | None = None
    capture_events: object | None = None
    compiled_unwrap_exception: tuple[
        type[BaseException] | None, BaseException | None, TracebackType | None
    ] = (None, None, None)
    try:
        compiled_capture_context = (
            prepare_compiled_capture(model)
            if isinstance(model, nn.Module)
            else contextlib.nullcontext()
        )
        compiled_capture_prep = compiled_capture_context.__enter__()
    except BaseException:
        # A raise between the reservation claim and the outer ``try`` would
        # otherwise leak the reservation and wedge every later admission.
        capture_slot.__exit__(None, None, None)
        _restore_user_global_rng()
        raise

    # Every arm after the forward is one FINALIZE entry; the record it
    # reads is built here so RUN holds the forward state and FINALIZE the
    # arm order (``finalize_completed`` raises the swallowed-stop error
    # into the failed arm, exactly as the inline check did).
    def _forward_result(
        arm: ForwardArm, *, raw_output: Any = None, exc: BaseException | None = None
    ) -> ForwardResult:
        """Bundle the forward's live state into the FINALIZE hand-off record."""
        return ForwardResult(
            trace=self,
            request=CaptureRequest(
                model=cast(nn.Module, model),
                input_args=input_args,
                input_kwargs=input_kwargs,
                postprocess=postprocess,
                reservation_resume=reservation_resume,
            ),
            arm=arm,
            raw_output=raw_output,
            session=capture_session,
            exc=exc,
            backend=backend,
            model=model,
            input_tensors=tuple(input_tensors),
            input_args=input_args,
            input_kwargs=input_kwargs,
        )

    try:
        # B8-25b: everything after ``__enter__`` runs INSIDE the try whose
        # ``finally`` exits the context -- a KeyboardInterrupt between enter
        # and try used to strand the compiled-submodule swaps on the user
        # model with no unwind.
        if not isinstance(compiled_capture_prep, CompiledCapturePrep):
            compiled_capture_prep = CompiledCapturePrep(sites=(), force_eager_stance=False)
        compiled_callable_sites = compiled_capture_prep.sites
        global _ACTIVE_CAPTURE_BACKEND
        previous_capture_backend = _ACTIVE_CAPTURE_BACKEND
        _ACTIVE_CAPTURE_BACKEND = backend
        try:
            (
                input_tensors_any,
                input_tensor_addresses,
            ) = _fetch_label_move_input_tensors(
                self,
                input_args,
                input_arg_names,
                input_kwargs,
                model_device,
            )
        finally:
            _ACTIVE_CAPTURE_BACKEND = previous_capture_backend
        input_tensors = list(input_tensors_any)
        self._raw_graph_ws.input_tensor_addresses = list(input_tensor_addresses)
        self._output_attribution_input_tensors = input_tensors

        # RNG state snapshot for deterministic explicit refreshes and legacy
        # two-pass consistency (#58).
        if self.capture_mode == "exhaustive":
            self._pre_forward_rng_states = backend.snapshot_rng(self)

        from torchlens._ir.capture_events import CaptureEvents

        self.capture_events = CaptureEvents()
        capture_events = self.capture_events
        if not isinstance(getattr(self, "_stop_directive", None), StopDirective):
            self._stop_directive = StopDirective(
                halt_options=getattr(self, "_predicate_save_options", None),
                raise_on_nan=bool(getattr(self, "raise_on_nan", False)),
                forward_error_mode=getattr(
                    getattr(self, "_predicate_save_options", None),
                    "on_forward_error",
                    "raise",
                ),
                inference_only=bool(getattr(self, "inference_only", False)),
            )
        self._capture_config = InternalCaptureConfig(
            capture_mode=str(self.capture_mode),
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=seed,
            postprocess=postprocess,
            stop=self._stop_directive,
        )
        capture_session = attach_legacy_capture_session(
            self,
            backend_token=backend,
            backend_name=str(_backend_name_for_trace(self)),
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            seed=seed,
            postprocess=postprocess,
        )
        attach_capture_events_session(capture_events, capture_session)

        with _timed_phase(self, "ctx_build:model_prepare"):
            # One-time model preparation + incremental sys.modules crawl
            backend.prepare_model_once(model)

            # Per-session model preparation
            backend.prepare_model_session(self, model)
        self.setup_duration = Duration(time.time() - self.capture_start_time)
        _vprint(self, f"Model prepared ({self.setup_duration:.2f s})")

        # Print input summary
        if getattr(self, "verbose", False):
            devices = set()
            for t in input_tensors:
                if hasattr(t, "device"):
                    devices.add(str(t.device))
            device_str = ", ".join(sorted(devices)) if devices else "unknown"
            _vprint(self, f"Inputs: {len(input_tensors)} tensor(s) on {device_str}")

        if bool(getattr(self, "intervention_ready", False)):
            from .._runnable_state import (
                snapshot_capture_state,
                snapshot_capture_state_signatures,
                snapshot_persistent_buffer_universe,
                snapshot_state_alias_topology,
            )

            # r37 corr2-4: the live bound-state alias topology (object identity,
            # storage overlap) must be captured BEFORE ``snapshot_capture_state``'s
            # clones erase it; the runnable producer refuses unsupported topologies
            # at save and reproduces identity groups from this record.
            self._runnable.state_alias_topology = snapshot_state_alias_topology(model)
            # r63 C1: per-slot metadata signatures are stamped from the LIVE tensors
            # PRE-clone -- the clone itself compacts ``storage_offset`` and
            # materializes conj/neg, so a post-clone signature is blind to two of
            # the four transport-lossy physical dims. Consumed by the escape-gated
            # ``producer_state_metadata`` preflight.
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
            # replace this minimal snapshot below with the complete ambient record.
            self._runnable.capture_ambient = {"default_dtype": str(get_default_dtype())}

        # Turn on the logging toggle and run the forward pass.
        # Inside this context, every decorated torch function will log its
        # inputs/outputs.  Source tensors (model inputs) are logged explicitly
        # before invoking the model; all subsequent operations are captured
        # automatically by the decorated wrappers.
        _vprint(self, f"Running {self.capture_mode} forward pass...")
        with backend.active_logging(self):
            # Under an active ``force_eager`` stance (torch >= 2.6) the
            # inventoried compiled callables run their original eager Python and
            # their interiors ARE logged, so the capture keeps full verified
            # semantics; the ceiling below is the honest pre-2.6 fallback.
            if compiled_callable_sites and not compiled_capture_prep.force_eager_stance:
                self._raw_dynamo_region_detected = True
                self._raw_transform_escape_detected = True
                _state._dynamo_warning_emitted = True
                warnings.warn(
                    "TorchLens detected a torch.compile (Dynamo) region on the captured model "
                    f"at {', '.join(compiled_callable_sites)}. Operations that run inside the "
                    "compiled region are not logged: on a cold compile the tensors there are "
                    "data-free FakeTensors, while a warm-cache execution can bypass Python "
                    "wrappers entirely. The returned Trace contains only operations that ran "
                    "OUTSIDE the compiled region. Use the eager callable during capture if you "
                    "need its interior logged (on torch >= 2.6, TorchLens instead runs compiled "
                    "callables eagerly via torch.compiler.set_stance and this gap does not "
                    "arise).",
                    UserWarning,
                    stacklevel=2,
                )
            for i, t in enumerate(input_tensors):
                backend.log_source_tensor(self, t, "input", input_tensor_addresses[i])
            _register_model_input_container_snapshots(self, input_args, input_kwargs)
            _record_runnable_input_literal_leaves(self, input_args, input_kwargs)
            _record_runnable_input_tensor_sites(self, input_args, input_kwargs)
            _record_runnable_input_structure(self, input_args, input_kwargs)
            _record_runnable_module_training_modes(self, model)
            if bool(getattr(self, "intervention_ready", False)):
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
            if self.capture_mode == "predicate":
                with _structure_only_forward_boundary(self):
                    outputs = _run_predicate_forward_with_root_frame(
                        self,
                        backend,
                        model,
                        input_args,
                        input_kwargs,
                        model_device,
                    )
            else:
                with _timed_phase(self, "dispatch:forward_model"):
                    with _forward_peak_memory_bracket(self, model_device):
                        with backend.inference_context(self):
                            # Bracket the user forward with host-RNG snapshots so a
                            # runnable descriptor can honestly record whether Python
                            # ``random`` / NumPy control flow (an unwitnessed branch)
                            # ran. TorchLens itself never draws host RNG here (its only
                            # host draw seeds before this point), so any advance is the
                            # user's. Reads are side-effect free -> capture unchanged.
                            # r37 hon1_2: the four-layer channel monitor additionally
                            # observes NON-global channels (RNG instances, SystemRandom,
                            # os entropy, clocks, the default_rng factory) over the
                            # frozen vocabulary. Any touch is permanently unreplayable
                            # (no identifiable seed); monitor uncertainty downgrades
                            # completeness, never reads as no-consumption.
                            #
                            # The channel monitor is armed ONLY for runnable-capable
                            # captures: ``intervention_ready`` is the exact predicate
                            # for "this capture can produce a passing sparse runnable
                            # descriptor" (the same predicate that gates the ambient
                            # execution-context snapshot above), and the descriptor
                            # builder is the witness verdict's only consumer. A
                            # disarmed capture stamps ``monitor_uncertain`` fail-closed
                            # so any unforeseen descriptor build ceilings through the
                            # existing RNG_MONITOR_UNCERTAIN witness gap
                            # (unverifiable, never a silent false VERIFIED) -- channel
                            # coverage on the disarmed lane is unknowable, not absent.
                            # Stamp the FAIL-CLOSED verdict before the forward runs.
                            # The real verdict is stamped after the monitor tears down,
                            # several statements away from its only consumer
                            # (``_io/runnable.py`` treats a falsy/``None`` flag as
                            # CERTAIN), so anything raising in between -- the swallowed-
                            # stop checkpoint, the global-engine diff, a Ctrl-C during
                            # teardown -- used to leave the field ``None`` and read as a
                            # proven-clean window. Pre-stamping means an unreached stamp
                            # ceilings through the existing RNG_MONITOR_UNCERTAIN
                            # witness gap instead.
                            self._runnable.rng_monitor_uncertain = True
                            self._runnable.rng_monitor_uncertain_detail = (
                                "monitor_verdict_not_stamped",
                            )
                            _host_rng_before = snapshot_host_rng()
                            if bool(getattr(self, "intervention_ready", False)):
                                from .._utils.rng import host_nondeterminism_monitor

                                with host_nondeterminism_monitor(model) as _rng_channels:
                                    outputs = cast(Callable[..., Any], model)(
                                        *input_args, **input_kwargs
                                    )
                            else:
                                _rng_channels = None
                                with _structure_only_forward_boundary(self):
                                    outputs = cast(Callable[..., Any], model)(
                                        *input_args, **input_kwargs
                                    )
                            _global_advanced = host_rng_advanced(
                                _host_rng_before, snapshot_host_rng()
                            )
                            if _rng_channels is not None:
                                # r65 CLUSTER Z stamping split: a torch RNG
                                # ``replayable_read`` (the ``initial_seed`` family --
                                # a host scalar fully determined by the capture seed)
                                # sets CONSUMED without poisoning the capture seed, so
                                # a run at the capture seed stays verified while any
                                # other/absent seed ceilings; ceiling ``channels``
                                # alone decide UNREPLAYABLE.
                                self._runnable.host_rng_consumed = (
                                    _global_advanced
                                    or bool(_rng_channels.channels)
                                    or bool(_rng_channels.replayable_reads)
                                )
                                self._runnable.host_rng_unreplayable = bool(_rng_channels.channels)
                                self._runnable.host_rng_channels = tuple(
                                    sorted(_rng_channels.channels)
                                )
                                self._runnable.host_rng_replayable_reads = tuple(
                                    sorted(_rng_channels.replayable_reads)
                                )
                                self._runnable.rng_monitor_uncertain = bool(_rng_channels.uncertain)
                                # r39 CLASS A: name the offending threads / coverage
                                # failure so the INCOMPLETE ceiling's readiness
                                # diagnostic is actionable.
                                self._runnable.rng_monitor_uncertain_detail = tuple(
                                    _rng_channels.uncertain_detail
                                )
                            else:
                                # Global engines are still bracketed (cheap); the
                                # channel verdict was never observed, so it must
                                # read as UNKNOWABLE, never as no-consumption.
                                self._runnable.host_rng_consumed = _global_advanced
                                self._runnable.rng_monitor_uncertain = True
                                self._runnable.rng_monitor_uncertain_detail = ("monitor_not_armed",)

        return finalize_completed(_forward_result("completed", raw_output=outputs))

    except StopSignal as halt_exc:
        compiled_unwrap_exception = sys.exc_info()
        return finalize_halted(_forward_result("halted", exc=halt_exc))

    except Exception as e:
        compiled_unwrap_exception = sys.exc_info()
        finalize_failed(_forward_result("failed", exc=e))

    except BaseException as interrupt_exc:
        compiled_unwrap_exception = sys.exc_info()
        finalize_interrupted(_forward_result("interrupted", exc=interrupt_exc))

    finally:
        # B1-02: the ONE site every settlement path passes through. The pop
        # block above lives only on the normal forward-return arm, so a halt
        # (or a failure, or a KeyboardInterrupt) used to exit with the live
        # tokenizer and metadata key still pinned to the escaping product.
        # Placed before the teardown ladder so a teardown double-fault cannot
        # skip it.
        _drop_semantic_output_transients(self)
        # Snapshot the exception this ``finally`` is unwinding through (None on
        # the normal return path): the teardown double-fault guard below needs
        # to know whether an ordinary teardown Exception would be replacing a
        # control-flow BaseException.
        inflight_exc = sys.exc_info()[1]
        try:
            try:
                _clear_saved_activation_dedup_caches(self)
                # Release input tensor references so GC can reclaim backend memory.
                input_tensors = None  # type: ignore[assignment]
                try:
                    _cleanup_forward_memory_once(self, backend, capture_session)
                finally:
                    if capture_session is not None and capture_events is not None:
                        detach_capture_session(self, capture_events, capture_session)
            finally:
                compiled_capture_context.__exit__(*compiled_unwrap_exception)
        except BaseException as teardown_exc:
            # Post-settlement teardown failure (path 7): the already-settled
            # COMPLETE/HALTED outcome demotes to FAILED/TEARDOWN in both homes
            # and the teardown exception propagates -- the raise preempts the
            # return, so no product escapes carrying an undemoted claim.
            demote_outcome(
                self,
                capture_session,
                note=(
                    f"teardown failed: {type(teardown_exc).__name__}: "
                    f"{safe_exception_str(teardown_exc)}"
                ),
                exc=teardown_exc,
            )
            if (
                inflight_exc is not None
                and not isinstance(inflight_exc, Exception)
                and isinstance(teardown_exc, Exception)
            ):
                # R63 (B8-23 one frame out): an ordinary teardown Exception
                # must not replace an unwinding KeyboardInterrupt/SystemExit
                # (demoting it to ``__context__``), or a caller's
                # ``except Exception`` retry loop swallows Ctrl-C. Keep the
                # demotion, attach the teardown failure, and re-raise the
                # ORIGINAL interrupt (the teardown exception stays visible as
                # its ``__context__``).
                note = (
                    "TorchLens post-settlement teardown also failed while handling "
                    f"this interrupt: {type(teardown_exc).__name__}: "
                    f"{safe_exception_str(teardown_exc)}"
                )
                add_note = getattr(inflight_exc, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:  # Python 3.10: no PEP 678 notes -- surface via warning.
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
                raise inflight_exc
            raise
        finally:
            # Outermost: a teardown double-fault must not leak the capture
            # reservation, or every later admission refuses forever. The RNG
            # restore (R57) runs first but can never displace the release.
            try:
                _restore_user_global_rng()
            finally:
                capture_slot.__exit__(None, None, None)
