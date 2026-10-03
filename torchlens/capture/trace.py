"""Forward-pass orchestration: runs the model, manages logging state, and saves outs.

This module implements the forward-pass architecture that TorchLens uses to extract
model outs:

1. **Exhaustive pass** (``capture_mode="exhaustive"``): Runs the model once,
   capturing every tensor operation's full metadata (shapes, dtypes, FLOPs,
   parent-child relationships, module context, etc.) into Op entries.
   This builds the complete computational graph.

2. **Predicate pass** (``capture_mode="predicate"``): Captures selectively
   while preserving the shared event journal and fixed-order kernel.

Key ordering constraint:
    RNG state must be captured/restored BEFORE ``active_logging()`` is entered,
    because the logging context manager itself may trigger decorated operations
    that consume RNG state.  See ``_pre_forward_rng_states`` handling.

Key functions:
    - ``normalize_input_args``: resolves tuple-vs-multi-arg ambiguity
    - ``safe_copy_args``: clones tensors to protect user inputs from in-place mutation
    - ``run_and_log_inputs_through_model``: the main entry point that orchestrates
      input setup, logging toggle, forward pass, output marking, and postprocessing
"""

import contextlib
import random
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from torch import get_default_dtype, nn

from .. import _state
from .._capture_state_helpers import CompiledCapturePrep, prepare_compiled_capture
from .._runnable_seam import runnable_trace_state
from ..backends import (
    TORCH_BACKEND_NAME,
    BackendName,
    BackendUnsupportedError,
    CaptureBackend,
    get_backend_spec,
    resolve_backend_spec,
)
from ..fastlog._halt import HaltSignal
from ..ir.container_registry import ModelSite, Phase, Role, walk_container
from ..quantities import Bytes, Duration
from .config import InternalCaptureConfig
from .outcome import (
    CapturePhase,
    count_committed_ops,
    demote_outcome,
    outcome_for,
    safe_exception_str,
    set_capture_phase,
    settle_completed,
    settle_failed,
    settle_halted,
)
from .peak_memory import peak_rss_bytes, process_rss_bytes, reset_peak_rss
from .session import (
    CaptureSession,
    attach_capture_events_session,
    attach_legacy_capture_session,
    detach_capture_session,
)
from .stop import StopDirective, evaluate_halt_stop

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
from ..data_classes._lookup_keys import _give_user_feedback_about_lookup_key
from ..utils.display import _timed_phase, _vprint
from ..utils.rng import (
    host_rng_advanced,
    log_current_rng_states,
    set_rng_from_saved_states,
    snapshot_host_rng,
)

_ACTIVE_CAPTURE_BACKEND: CaptureBackend | None = None

_AUTO_SEED_ENTROPY = random.Random()
"""Private entropy stream for ``random_seed=None`` capture seed picks (R57).

Seeded once from OS entropy at import. Drawing the auto seed from the user's
global ``random`` engine either advanced that stream past the capture's
restore bracket (breaking byte-exact RNG neutrality for a seeded host
process) or, if the draw were bracketed too, made consecutive auto-seeded
captures reuse one identical seed (silently correlating dropout patterns
across runs). A private stream preserves both guarantees; the chosen seed is
always disclosed on ``trace.random_seed``.
"""


def _cleanup_forward_memory_once(
    trace: "Trace",
    backend: CaptureBackend,
    session: CaptureSession | None,
) -> None:
    """Run the legacy forward-memory teardown through the active session.

    Parameters
    ----------
    trace
        Legacy trace compatibility owner.
    backend
        Selected capture backend.
    session
        Stage-2 run owner, when the compatibility adapter was initialized.
    """

    if session is None:
        backend.cleanup_forward_memory(trace)
        return
    session.run_cleanup("forward_memory", lambda: backend.cleanup_forward_memory(trace))


def _structure_only_forward_boundary(trace: "Trace") -> "contextlib.AbstractContextManager[None]":
    """LAYER-2 backstop for structure-only captures (L7a memo sec 2.2).

    Inert ``nullcontext`` on the default path; for ``structure_only=True``
    sessions the torch-backend boundary classifies exceptions escaping the
    user forward by raising-frame provenance (typed meta-kernel /
    unenumerated-escape refusals; user exceptions propagate annotated). The
    lazy import keeps this backend-neutral module torch-light.
    """

    if bool(getattr(trace, "structure_only", False)):
        return _weightsfree_and_belt_boundary(trace)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _weightsfree_and_belt_boundary(trace: "Trace") -> "Iterator[None]":
    """Compose the admitted-meta scope around the structure-only belt.

    The admitted-meta scope (W1-CTX factory slot, W1-AC autocast shim, D19
    ambient-context absorption, W1 transparency activation) wraps OUTSIDE
    the belt boundary so belt refusals still classify while the scope's
    state restores on any exit path. No-op without a pending admission
    (real-substrate structure-only captures).
    """

    from ..backends.torch.structure_only_belt import structure_only_forward_boundary
    from ._weightsfree_admission import weightsfree_forward_scope

    with weightsfree_forward_scope(trace), structure_only_forward_boundary(trace):
        yield


@contextlib.contextmanager
def _forward_peak_memory_bracket(trace: "Trace", device: "object | None") -> "Iterator[None]":
    """Record forward-pass peak memory around the model forward call.

    Stores the peak on ``trace.forward_peak_memory`` and the backend label on
    ``trace.forward_memory_backend``. CUDA snapshots ``max_memory_allocated``
    around the forward WITHOUT ``reset_peak_memory_stats`` (R36-2): resetting
    clobbered the caller's process-wide high-water counter on every capture
    (the CPU branch below explicitly refuses the analogous clobber). When the
    forward pushes a new device peak the reported figure is that exact peak;
    a forward that fits under the pre-existing high-water mark legitimately
    reads ``0``, same contract as the CPU/MPS delta below. The figure covers
    the MODEL device only (single-device heuristic, R36 doc line): a
    model-parallel forward's peaks on other devices are not measured.

    CPU/MPS measure a process resident-set-size (or MPS allocator) delta, which
    captures torch's C++-allocated tensor buffers for sizeable models. That
    counter is coarse: for models small enough that the forward fits in
    already-resident heap headroom the delta legitimately rounds to ``0``.

    ``CaptureOptions(measure_python_peak_memory=True)`` additionally folds in the
    stdlib ``tracemalloc`` Python-allocation peak, which stays reliably positive
    for those small models. It is OFF by default because starting tracemalloc
    installs a CPython allocator hook that fires on every allocation made by
    every traced operation, costing 1.7x-2.5x total capture time on real CNNs and
    ViTs -- an unacceptable tax on the default path for one diagnostic scalar.

    When enabled, the Python peak is measured as a delta against a baseline
    snapshot taken at bracket entry (after ``reset_peak()`` when tracemalloc was
    already tracing). This keeps the value scoped to this bracket even when
    tracemalloc was started earlier by external tooling: the reset discards any
    unrelated historical high-water mark, and subtracting the entry-time baseline
    discards memory that is legitimately still live but unrelated to this forward
    pass (e.g. process-wide Python state already resident when the bracket was
    entered).

    Only the exhaustive and predicate primary passes record memory; the fast
    second pass re-runs the model and must not clobber the measured forward peak.
    Measurement never raises into the capture path.

    Parameters
    ----------
    trace:
        Trace receiving the forward memory metadata.
    device:
        Device the forward pass runs on.

    Yields
    ------
    None
        Context body in which the model forward executes.
    """

    device_type = getattr(device, "type", None)
    torch_module: Any = None
    if device_type in {"cuda", "mps"}:
        try:
            import torch as torch_module
        except ImportError:
            torch_module = None

    if device_type == "cuda" and torch_module is not None and torch_module.cuda.is_available():
        backend_label = "cuda"
        cuda_device = device
        peak_before = 0
        reserved_before = 0
        with contextlib.suppress(Exception):
            peak_before = int(torch_module.cuda.max_memory_allocated(cuda_device))
        with contextlib.suppress(Exception):
            reserved_before = int(torch_module.cuda.max_memory_reserved(cuda_device))
        try:
            yield
        finally:
            live_peak: int | None = None
            resident_peak: int | None = None
            with contextlib.suppress(Exception):
                peak_after = int(torch_module.cuda.max_memory_allocated(cuda_device))
                # New device peak -> the forward's exact peak. No new peak ->
                # the forward stayed under the pre-existing high-water mark
                # and the figure honestly reads 0 (see docstring, R36-2).
                live_peak = peak_after if peak_after > peak_before else 0
                trace.forward_peak_memory = Bytes(live_peak)
            with contextlib.suppress(Exception):
                reserved_after = int(torch_module.cuda.max_memory_reserved(cuda_device))
                resident_peak = reserved_after if reserved_after > reserved_before else 0
            trace.forward_memory_backend = backend_label
            # F20 peak PAIR (brainpipe D-7): live allocation vs resident
            # high-water are different physical quantities; a single number
            # cannot be both correct and portable. Session-time only.
            trace._forward_peak_memory_pair = {
                "live": live_peak,
                "resident": resident_peak,
                "backend": "cuda:allocated+reserved",
                "resident_basis": "prior_high_water_delta",
            }
        return

    if device_type == "mps" and torch_module is not None and hasattr(torch_module, "mps"):
        backend_label = "mps"
        before = int(torch_module.mps.current_allocated_memory())
    else:
        backend_label = "cpu"
        before = process_rss_bytes()
    # F20 peak PAIR (brainpipe D-7): scope the host resident high-water mark
    # to THIS capture when the platform allows it. Without the reset, VmHWM
    # is the process-lifetime maximum and legitimately reads 0 growth for
    # every capture after the first -- the memo's sweep-scale instrument
    # defect.
    rss_peak_scoped = reset_peak_rss()
    rss_before = before if backend_label == "cpu" else process_rss_bytes()

    # Opt-in only: the tracemalloc allocator hook is the single most expensive
    # thing in a default CPU capture, so the default path never touches
    # tracemalloc at all -- not even `is_tracing()` / `reset_peak()`, which would
    # also clobber an external profiler's own high-water mark.
    tracemalloc_module: Any = None
    tracemalloc_started_here = False
    traced_baseline = 0
    if getattr(trace, "measure_python_peak_memory", False):
        import tracemalloc

        tracemalloc_module = tracemalloc
        tracemalloc_started_here = not tracemalloc.is_tracing()
        if tracemalloc_started_here:
            with contextlib.suppress(Exception):
                tracemalloc.start()
        else:
            # tracemalloc was already tracing when this bracket was entered (external
            # tooling, a pytest memory-leak plugin, or a leftover start elsewhere in the
            # process). Reset its high-water mark before yielding so the peak reported
            # below is scoped to this forward pass instead of an arbitrary earlier,
            # unrelated high-water mark from before this bracket ran.
            with contextlib.suppress(Exception):
                tracemalloc.reset_peak()
        # Snapshot the currently-live traced size as a baseline. When tracemalloc was
        # already running, this "current" size can itself be sizeable (e.g. process-wide
        # Python-level state that happens to already be resident, such as this same
        # trace() call's own one-time model-preparation work that ran moments earlier,
        # just before this bracket). reset_peak() alone only discards *historical* peaks
        # reached before the bracket; it cannot lower "current". Subtracting this
        # baseline from the post-yield peak below isolates the delta genuinely
        # introduced by the forward pass, mirroring the RSS-delta measurement used for
        # the CPU/MPS path just below.
        if tracemalloc.is_tracing():
            with contextlib.suppress(Exception):
                traced_baseline, _peak_at_entry = tracemalloc.get_traced_memory()
    try:
        yield
    finally:
        traced_peak = 0
        if tracemalloc_module is not None and tracemalloc_module.is_tracing():
            with contextlib.suppress(Exception):
                _current, traced_peak = tracemalloc_module.get_traced_memory()
                traced_peak = max(0, traced_peak - traced_baseline)
            if tracemalloc_started_here:
                with contextlib.suppress(Exception):
                    tracemalloc_module.stop()
        # Both readers can raise on a hostile host (`mps.current_allocated_memory()`
        # on a degraded MPS build, `psutil.Process().memory_info()` with AccessDenied /
        # NoSuchProcess inside a restricted container -- `_process_rss_bytes` only
        # catches ImportError). They were the only two statements in this finally
        # outside a suppress, so a failure there replaced the user's in-flight forward
        # exception with a psutil/MPS error AND skipped both field writes, contrary to
        # the docstring's "measurement never raises into the capture path". Default to
        # a zero delta: an unmeasurable host reports no measured growth, never a lie.
        after = before
        with contextlib.suppress(Exception):
            if backend_label == "mps" and torch_module is not None:
                after = int(torch_module.mps.current_allocated_memory())
            else:
                after = process_rss_bytes()
        rss_delta = max(0, after - before)
        trace.forward_memory_backend = backend_label
        trace.forward_peak_memory = Bytes(max(rss_delta, int(traced_peak)))
        # F20 peak PAIR (brainpipe D-7). ``live`` is a Python-allocation peak
        # and exists only when the tracemalloc opt-in paid for it; ``None``
        # means unmeasured, never zero. ``resident`` is the host high-water
        # growth over this bracket, per-capture-scoped when the VmHWM reset
        # succeeded (Linux) and a lifetime-max approximation otherwise.
        resident_growth: int | None = None
        with contextlib.suppress(Exception):
            peak_rss_after = peak_rss_bytes()
            if peak_rss_after > 0 and rss_before > 0:
                resident_growth = max(0, peak_rss_after - rss_before)
        # ``resident_basis`` must disclose WHY ``resident`` is ``None`` rather than
        # silently repeating whatever ``rss_peak_scoped`` says: the pre-forward RSS
        # baseline (``rss_before``, read via ``psutil``) is 0 whenever psutil is not
        # installed, which zeroes ``resident_growth`` REGARDLESS of whether the VmHWM
        # reset succeeded -- reporting "per_capture"/"process_lifetime" there would
        # claim a scoped-or-unscoped MEASUREMENT that never happened. "unavailable"
        # is the typed disclosure for that case (see ``psutil_available``); it is
        # reported whenever ``resident`` could not be computed, from any cause.
        if resident_growth is None:
            resident_basis = "unavailable"
        else:
            resident_basis = "per_capture" if rss_peak_scoped else "process_lifetime"
        trace._forward_peak_memory_pair = {
            "live": int(traced_peak) if tracemalloc_module is not None else None,
            "resident": resident_growth,
            "backend": ("mps:allocated+rss" if backend_label == "mps" else "cpu:maxlive+rss"),
            "resident_basis": resident_basis,
        }


def _backend_name_for_trace(trace: "Trace") -> BackendName:
    """Return the backend name recorded on a trace.

    Parameters
    ----------
    trace:
        Trace whose backend should be used for shared capture orchestration.

    Returns
    -------
    BackendName
        Backend name stored on the trace, defaulting to the legacy torch name
        for old or partially constructed trace objects.
    """

    return cast(BackendName, getattr(trace, "backend", "torch"))


def _capture_backend_from_registry(
    backend_name: BackendName,
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> CaptureBackend:
    """Resolve a trace execution backend through the public backend registry.

    Parameters
    ----------
    backend_name:
        Explicit backend name recorded on the trace.
    model:
        Model or callable being captured.
    input_args:
        Public positional inputs.
    input_kwargs:
        Public keyword inputs.

    Returns
    -------
    CaptureBackend
        Lower-level Protocol adapter owned by the resolved backend spec.
    """

    spec = resolve_backend_spec(backend_name, model, input_args, input_kwargs)
    if spec.capture_backend is None:
        raise BackendUnsupportedError(
            f"backend={spec.name!r} does not expose a shared capture Protocol adapter."
        )
    return spec.capture_backend()


def _clear_saved_activation_dedup_caches(trace: "Trace") -> None:
    """Release per-pass saved-activation dedup caches.

    Parameters
    ----------
    trace:
        Trace whose per-pass cache state should be cleared.

    Returns
    -------
    None
        Mutates trace-owned cache dictionaries.
    """

    for cache_name in ("_out_identity_cache", "_out_hash_cache"):
        cache = getattr(trace, cache_name, None)
        if isinstance(cache, dict):
            cache.clear()
    wrapper_ws = trace.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            registry.clear_live_state()


def _run_predicate_forward_with_root_frame(
    trace: "Trace",
    backend: CaptureBackend,
    model: object,
    input_args: tuple[Any, ...] | list[Any],
    input_kwargs: dict[Any, Any],
    model_device: object | None,
) -> Any:
    """Run predicate capture through the shared root module-frame boundary.

    Parameters
    ----------
    trace
        Active predicate-mode trace.
    backend
        Backend adapter owning module-frame stack operations.
    model
        Model being captured.
    input_args
        Normalized model positional inputs.
    input_kwargs
        Normalized model keyword inputs.
    model_device
        Device used for forward peak-memory measurement.

    Returns
    -------
    Any
        Raw model output.
    """

    from ..capture.predicates import _is_halt_only_capture, _module_capture_spec
    from ..capture.projections import (
        _build_record_context,
        append_projected_event,
        get_active_recording_state,
    )
    from ..fastlog.types import CaptureSpec, ModuleStackFrame

    state = get_active_recording_state()
    root_frame = ModuleStackFrame(
        address="",
        module_type=type(model).__name__,
        module_id=id(model),
        pass_index=1,
    )
    skipped_spec = CaptureSpec(save_out=False, save_metadata=False)
    backend.push_existing_module_frame(trace, state.module_stack, root_frame)
    state.event_index += 1
    enter_ctx = _build_record_context(
        kind="module_enter",
        op_log_or_op_data={
            "label": "root:enter:1",
            "address": "",
            "module_type": type(model).__name__,
            "module_pass_index": root_frame.pass_index,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=time.time() - trace.capture_start_time,
        include_source_events=state.options.include_source_events,
        sample_id=state.sample_id,
    )
    halt_only = _is_halt_only_capture(state.options)
    try:
        if halt_only:
            evaluate_halt_stop(trace, enter_ctx, state.options)
        else:
            enter_spec = _module_capture_spec(state.options)
            append_projected_event(
                trace,
                enter_ctx,
                enter_spec,
                predicate_matched=enter_spec.save_out or enter_spec.save_metadata,
            )
            evaluate_halt_stop(trace, enter_ctx, state.options)
    except HaltSignal:
        raise
    except Exception as exc:
        state.handle_predicate_exception(enter_ctx, exc)
        if not halt_only:
            append_projected_event(
                trace,
                enter_ctx,
                skipped_spec,
                predicate_matched=False,
            )
    finally:
        if not halt_only:
            state.append_context(enter_ctx)
    outputs = None
    try:
        with _timed_phase(trace, "dispatch:forward_model"):
            with _forward_peak_memory_bracket(trace, model_device):
                with backend.inference_context(trace):
                    outputs = cast(Callable[..., Any], model)(*input_args, **input_kwargs)
    finally:
        active_model_exc = sys.exc_info()[1]
        state.event_index += 1
        exit_ctx = _build_record_context(
            kind="module_exit",
            op_log_or_op_data={
                "label": "root:exit:1",
                "address": "",
                "module_type": type(model).__name__,
                "module_pass_index": root_frame.pass_index,
            },
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index,
            step_index=None,
            time_since_pass_start=time.time() - trace.capture_start_time,
            include_source_events=state.options.include_source_events,
            sample_id=state.sample_id,
        )
        try:
            if halt_only:
                evaluate_halt_stop(trace, exit_ctx, state.options, frontier_output=outputs)
            else:
                exit_spec = _module_capture_spec(state.options)
                append_projected_event(
                    trace,
                    exit_ctx,
                    exit_spec,
                    predicate_matched=exit_spec.save_out or exit_spec.save_metadata,
                )
                evaluate_halt_stop(trace, exit_ctx, state.options, frontier_output=outputs)
        except HaltSignal:
            if active_model_exc is None:
                raise
        except Exception as exc:
            if active_model_exc is None:
                state.handle_predicate_exception(exit_ctx, exc)
            else:
                state.add_predicate_failure(exit_ctx, exc)
            if not halt_only:
                if active_model_exc is None or not any(
                    event.raw_index == exit_ctx.event_index
                    for event in trace.capture_events.op_events
                ):
                    append_projected_event(
                        trace,
                        exit_ctx,
                        skipped_spec,
                        predicate_matched=False,
                    )
        finally:
            if not halt_only:
                state.append_context(exit_ctx)
            backend.pop_module_frame(trace, state.module_stack, root_frame)
    return outputs


def save_new_outs(
    self: "Trace",
    model: object,
    input_args: Any | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] = "all",
    grad_layers_to_save: str | list[Any] | None = "all",
    random_seed: int | None = None,
    backward_ready: bool | None = None,
    _run_until_plan: Any | None = None,
) -> None:
    """Re-run the model with new inputs, saving refreshed outs.

    This is the public API for refreshing outs without rebuilding the
    computational graph.  Much faster than ``trace`` because all
    metadata (graph structure, labels, module context) was captured in the
    original exhaustive pass and is reused here.

    The refresh assumes the computational graph is identical to the original
    pass. The refresh projector validates the captured graph and raises
    ``ValueError`` when dynamic control flow changes it.

    Parameters

    ----------
        model: Model for which to save outs.
        input_args: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        layers_to_save: List of layers to save, using any valid lookup keys.
        grad_layers_to_save: List of layers whose grads should be saved.
        random_seed: Which random seed to use for deterministic reproduction.
        backward_ready: Optional replay override. ``None`` inherits the existing
            model log settings; explicit values temporarily override saved
            tensor detachment for the whole replay.

    Returns

    -------
        Nothing; mutates ``self`` in place with new out values.
    """
    if backward_ready is not None:
        model_detach_saved_activations = self.detach_saved_activations
        model_train_mode = getattr(self, "backward_ready", False)
        layer_detach_saved_activations = {
            layer_log_entry: layer_log_entry.detach_saved_activations for layer_log_entry in self
        }
        target_detach_saved_activations = False if backward_ready else self.detach_saved_activations
        try:
            self.detach_saved_activations = target_detach_saved_activations
            self.backward_ready = backward_ready
            for layer_log_entry in layer_detach_saved_activations:
                layer_log_entry.detach_saved_activations = target_detach_saved_activations
            save_new_outs(
                self,
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                layers_to_save=layers_to_save,
                grad_layers_to_save=grad_layers_to_save,
                random_seed=random_seed,
                backward_ready=None,
                _run_until_plan=_run_until_plan,
            )
        finally:
            self.detach_saved_activations = model_detach_saved_activations
            self.backward_ready = model_train_mode
            for layer_log_entry, detach_saved_activations in layer_detach_saved_activations.items():
                layer_log_entry.detach_saved_activations = detach_saved_activations
        return

    from ..user_funcs import _run_model_and_save_specified_outs
    from .projectors import RefreshProjector

    save_grads_policy = getattr(self, "save_grads", None)
    layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)
    grad_layer_nums_to_save = _get_op_nums_from_user_labels(self, grad_layers_to_save)
    refresh_seed = self.random_seed if random_seed is None else random_seed
    resolved_layer_nums: tuple[int, ...] | None = None
    if layer_nums_to_save != "all":
        expanded_layer_nums = set(cast(list[int], layer_nums_to_save))
        for output_label in self.output_layers:
            output = self.layer_dict_all_keys[output_label]
            expanded_layer_nums.update(
                self.layer_dict_all_keys[parent].raw_index for parent in output.parents
            )
        resolved_layer_nums = tuple(sorted(expanded_layer_nums))
    refreshed = _run_model_and_save_specified_outs(
        model=cast(nn.Module, model),
        input_args=input_args,
        input_kwargs=input_kwargs or {},
        layers_to_save="all" if resolved_layer_nums is None else "none",
        output_device=getattr(self, "output_device", "same"),
        activation_transform=getattr(self, "activation_transform", None),
        grad_transform=getattr(self, "grad_transform", None),
        save_raw_activations=getattr(self, "save_raw_activations", True),
        save_raw_gradients=getattr(self, "save_raw_gradients", True),
        save_mode=getattr(self, "save_mode", "copy"),
        capture_tensor_grad_hooks=getattr(self, "capture_tensor_grad_hooks", True),
        keep_orphans=getattr(self, "keep_orphans", False),
        mark_layer_depths=getattr(self, "mark_layer_depths", False),
        detach_saved_activations=getattr(self, "detach_saved_activations", False),
        save_arg_values=getattr(self, "save_arg_values", False),
        save_grads=save_grads_policy not in (None, False),
        grads_to_save=grad_layers_to_save,
        random_seed=refresh_seed,
        num_context_lines=getattr(self, "num_context_lines", 7),
        optimizer=getattr(self, "_optimizer", None),
        save_code_context=getattr(self, "save_code_context", False),
        save_rng_states=getattr(self, "save_rng_states", False),
        recurrence_detection=getattr(self, "recurrence_detection", True),
        verbose=getattr(self, "verbose", False),
        backward_ready=getattr(self, "backward_ready", False),
        inference_only=getattr(self, "inference_only", False),
        # A refresh retains payloads like any capture; dropping the configured
        # budget here silently rebudgeted every refresh/run() forward at the
        # default "auto" (or left a save_budget=None session budgeted).
        save_budget=getattr(self, "save_budget", "auto"),
        # F2: a refresh re-arms the nonfinite tripwire. The historical refresh
        # forwarded only inference_only, so a raise_on_nan capture silently
        # lost its abort policy on every refreshed forward.
        raise_on_nan=bool(getattr(self, "raise_on_nan", False)),
        output_transform=getattr(self, "_output_transform", None),
        save_raw_output=getattr(self, "save_raw_output", "small"),
        retain_output_parents_for_layers_to_save=True,
        _resolved_layer_nums_to_save=resolved_layer_nums,
        _resolved_grad_layer_nums_to_save=(
            grad_layer_nums_to_save
            if grad_layer_nums_to_save == "all"
            else tuple(cast(list[int], grad_layer_nums_to_save))
        ),
        _refresh_projection_capture=True,
        # L4 2.3 live until=: the run-installed halt latch rides the EXISTING
        # halt= surface of the internal refresh capture (one forwarded argument;
        # the driver's halt arm settles the throwaway HALTED and returns the
        # partial normally into the projection flow below).
        halt_predicate=None if _run_until_plan is None else _run_until_plan.halt_predicate,
    )
    projected_layer_nums = (
        "all" if layer_nums_to_save == "all" else tuple(cast(list[int], layer_nums_to_save))
    )
    projected_grad_layer_nums = (
        "all"
        if grad_layer_nums_to_save == "all"
        else tuple(cast(list[int], grad_layer_nums_to_save))
    )
    if _run_until_plan is not None and _run_until_plan.fired:
        # L4 2.3 truncated refresh: the latch fired and the internal capture
        # settled HALTED. The projection is PREFIX-SCOPED (all projector
        # tripwires at full strength on the executed prefix; a prefix mismatch
        # refuses exactly like a full mismatch), and BOTH post-return feedback
        # writes are suppressed with their inherited authority explicitly
        # neutralized -- never left to absence:
        #   (1) output-losslessness: the fork INHERITED the source's positive
        #       stamp via the fork builder's runnable copy pass, and the halt
        #       frontier's own proof attests the truncated frontier tensor,
        #       never the full-forward output -- so the :672-style copy is
        #       SKIPPED and the inherited proof is CLEARED to None (the shipped
        #       fail-closed state; the live reconstructor fails closed).
        #   (2) replay-arg completeness: a truncated refresh did not witness
        #       complete replay arg/version data over a partial forward, and the
        #       field initializes True at construction -- so it is SET FALSE
        #       explicitly, never merely skipped.
        RefreshProjector(
            self,
            projected_layer_nums,
            projected_grad_layer_nums,
        ).project_prefix(refreshed, _run_until_plan)
        self._runnable.output_losslessness = None
        self._replay_arg_version_data_complete = False
        return
    RefreshProjector(
        self,
        projected_layer_nums,
        projected_grad_layer_nums,
    ).project(refreshed)
    # r39 corr2_5: copy the FRESH refresh forward's output-losslessness proof onto the
    # projected fork. A changed input may select a different return-container KIND than the
    # original capture, so the live provider must gate its bare-tensor fast path on the fresh
    # proof (``bare_tensor_root``), not the stale capture-time one. Missing/malformed fresh
    # proof leaves the field absent -> the live reconstructor fails closed (not faithful).
    self._runnable.output_losslessness = refreshed._runnable.output_losslessness
    if self.save_arg_values:
        self._replay_arg_version_data_complete = True


def _select_raw_indexes_by_retention_predicate(
    self: "Trace", predicate: "Callable[[Any], bool]"
) -> list[int]:
    """Evaluate a bare retention predicate per finalized op, strict bool.

    Bare retention predicates (the ``save_grads=`` callable spelling) are
    evaluated per finalized op with a layer-like ctx and a strict-bool return
    -- mirroring the halt-slot bool contract so a truthy tensor cannot
    silently select everything.

    Parameters
    ----------
    self:
        Finished trace whose ``layer_list`` supplies the candidate ops.
    predicate:
        User callable evaluated per op record.

    Returns
    -------
    list[int]
        Sorted unique raw indexes of the ops the predicate selected.

    Raises
    ------
    PredicateError
        If the predicate returns a non-bool (code ``predicate_return_invalid``).
    """

    from ..fastlog.exceptions import PredicateError

    selected_raw_indexes: set[int] = set()
    for layer_entry in getattr(self, "layer_list", []):
        decision = predicate(layer_entry)
        if not isinstance(decision, bool):
            raise PredicateError(
                "save_grads predicate must return bool. "
                "Remedy: return True or False from the save_grads predicate.",
                ctx=layer_entry,
                result=decision,
                code="predicate_return_invalid",
            )
        if decision:
            selected_raw_indexes.add(layer_entry.raw_index)
    return sorted(selected_raw_indexes)


def _get_op_nums_from_user_labels(
    self: "Trace", which_layers: str | list[str | int] | None
) -> list[int] | str:
    """Resolve user-provided layer identifiers to internal raw_index values.

    Supports exact key match, substring match across all lookup keys, and the
    special sentinel ``"all"`` (which ops through as-is).  Returns sorted
    unique raw operation numbers for refresh projection.
    """
    if which_layers == "all":
        return which_layers
    elif which_layers in [None, "none", "None", "NONE", []]:
        return []

    from ..intervention.selectors import BaseSelector

    if isinstance(which_layers, BaseSelector):
        return sorted(
            {
                site.raw_index
                for site in self.resolve_sites(
                    which_layers,
                    strict=False,
                    max_fanout=max(1, len(getattr(self, "layer_list", []))),
                )
            }
        )

    if callable(which_layers):
        return _select_raw_indexes_by_retention_predicate(self, which_layers)

    if not isinstance(which_layers, list):
        which_layers = [which_layers]  # type: ignore[list-item]
    raw_layer_nums_to_save: set[int] = set()
    for layer_key in which_layers:
        if isinstance(layer_key, BaseSelector):
            raw_layer_nums_to_save.update(
                site.raw_index
                for site in self.resolve_sites(
                    layer_key,
                    strict=False,
                    max_fanout=max(1, len(getattr(self, "layer_list", []))),
                )
            )
            continue
        if isinstance(layer_key, str) and ":" not in layer_key:
            matching_layer_passes = [
                layer_entry
                for layer_entry in getattr(self, "layer_list", [])
                if layer_key in {layer_entry.layer_label, layer_entry.layer_label_short}
            ]
            if matching_layer_passes:
                raw_layer_nums_to_save.update(
                    layer_entry.raw_index for layer_entry in matching_layer_passes
                )
                continue
        if layer_key in self._lookup_keys_to_layer_num_dict:
            raw_layer_nums_to_save.add(self._lookup_keys_to_layer_num_dict[layer_key])
            continue

        keys_with_substr = [key for key in self.layer_dict_all_keys if str(layer_key) in str(key)]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_layer_nums_to_save.add(self.layer_dict_all_keys[key].raw_index)
            continue

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_layer_nums_to_save = sorted(raw_layer_nums_to_save)  # type: ignore[assignment]
    return raw_layer_nums_to_save  # type: ignore[return-value]


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


def _register_model_input_container_snapshots(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Register top-level model input containers before forward invocation.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    capability = get_backend_spec(
        str(_backend_name_for_trace(trace))
    ).capabilities.input_container_structure
    if capability == "none":
        return
    registry = trace._wrapper_runtime_ws.container_registry
    first_spec = None
    for index, arg in enumerate(input_args):
        result = walk_container(arg, role=Role.MODEL_INPUT, capability=capability)
        if result is None:
            continue
        if first_spec is None:
            first_spec = result.spec
        registry.register_snapshot(
            arg,
            site=ModelSite(model_ref="self:1", position=("arg", index)),
            role=Role.MODEL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
        registry.register_snapshot(
            arg,
            site=ModelSite(model_ref="self:1", position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in input_kwargs.items():
        result = walk_container(value, role=Role.MODEL_INPUT, capability=capability)
        if result is None:
            continue
        if first_spec is None:
            first_spec = result.spec
        registry.register_snapshot(
            value,
            site=ModelSite(model_ref="self:1", position=("kwarg", key)),
            role=Role.MODEL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
        registry.register_snapshot(
            value,
            site=ModelSite(model_ref="self:1", position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=0,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    if first_spec is not None:
        trace.__dict__["input_structure"] = first_spec


_OPAQUE_INPUT_LEAF = object()
"""Sentinel marking a non-tensor input subtree that cannot be witnessed.

Recorded in place of the children under a mapping key that is not representable
in the frozen literal grammar. The runnable producer treats it as an opaque
(value-free) leaf, so the run honestly downgrades to ``UNVERIFIABLE`` instead of
silently skipping the subtree.
"""


def _record_runnable_input_literal_leaves(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Stash capture-time non-tensor model-input leaves for runnable honesty.

    A sparse runnable descriptor replays the *recorded taken-path* DAG, which is
    only valid for the recorded inputs. Non-tensor Python inputs
    (``bool``/``int``/``float``/``str``/``None`` literal leaves) can steer
    Python-level control flow that TorchLens never observes as an op, so a
    changed non-tensor input can silently make the recorded path wrong. TorchLens
    binds only tensor input leaves at run time, so without this record a changed
    non-tensor input is invisible and the run would falsely report a verified,
    attested -- but numerically wrong -- result.

    This records the model-boundary non-tensor leaves (site position, container
    path, and immutable literal value) so the sparse producer can witness them
    and the runnable executor can diverge on a changed non-tensor input instead
    of silently replaying the recorded path. It runs only when replay templates
    are captured (the runnable prerequisite), touches no tensors, and stores an
    in-memory list consumed by the producer at save time.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not bool(getattr(trace, "intervention_ready", False)):
        return

    from torchlens._input_walk import tagged_mapping_key_component, walk_input_boundary
    from torchlens._io.runnable import EMPTY_CONTAINER_PATH_MARKER

    leaves: list[tuple[object, tuple[str | int, ...], Any]] = []

    def _walk_site(position: object, value: Any) -> None:
        """Record one boundary site's non-tensor leaves through the shared traversal.

        Container dispatch is single-sourced in ``torchlens._input_walk`` (r65
        Cluster Y): this walker only declares WHAT it records. A mapping child under
        a literal-grammar key is recorded at its tagged path (bool keys stay distinct
        from equal-valued int keys in the leaf-path set); a child under a
        NON-representable key (enum, object, bytes, ...) cannot be
        re-derived at run time, so the whole subtree is recorded as one OPAQUE marker
        leaf -- downgrading witness coverage to UNVERIFIABLE rather than silently
        dropping it (a silently skipped leaf under an exotic key is the false-VERIFIED
        money bug this walker exists to prevent). An EMPTY container adds no child
        leaf, so it is witnessed by a synthetic marker leaf carrying its KIND at
        ``(*path, EMPTY_CONTAINER_PATH_MARKER)`` so an added/removed/kind-changed
        empty container (which can steer ``'flag' in d`` / ``if not lst`` control
        flow) diverges instead of silently replaying the recorded path. r42 corr1_2:
        dataclass containers descend by DECLARED FIELD with the same leaf vocabulary,
        so a tensor-only dataclass input records ZERO opaque leaves (stays fully
        witnessable -> VERIFIED) while a genuinely-opaque field still surfaces.
        """

        walk_input_boundary(
            value,
            (),
            key_component=tagged_mapping_key_component,
            on_leaf=lambda leaf, p: leaves.append((position, p, leaf)),
            on_empty_container=lambda kind, p: leaves.append(
                (position, (*p, EMPTY_CONTAINER_PATH_MARKER), kind)
            ),
            on_opaque_key_subtree=lambda _child, p: leaves.append(
                (position, p, _OPAQUE_INPUT_LEAF)
            ),
        )

    for index, arg in enumerate(input_args):
        _walk_site(("arg", index), arg)
    for key, value in input_kwargs.items():
        _walk_site(("kwarg", key), value)

    if leaves:
        runnable_trace_state(trace).input_nontensor_leaves = tuple(leaves)


def _record_runnable_input_structure(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Stash ONE per-site input-boundary structure snapshot for runnable honesty (r67 C2).

    One traversal per normalized model-input site (the snapshot spine in
    ``torchlens._input_walk``) records the complete site set/arity and every nested node
    -- kind, exact ``(module, qualname)`` type, declared child schema, ordered
    type-strict-encoded mapping keys, registered aux, and the instance-state proof. The
    runnable producer persists the records as REQUIRED structure facts (positive proof:
    a site without a snapshot, or a snapshot carrying a refusal, refuses the runnable
    save); the executor re-derives the runtime snapshot with the SAME function and
    diverges on any node mismatch. Analysis captures are unaffected.
    """

    if not bool(getattr(trace, "intervention_ready", False)):
        return

    from torchlens._input_walk import snapshot_input_boundary

    snapshots: list[dict[str, Any]] = []
    for index, arg in enumerate(input_args):
        record = snapshot_input_boundary(arg)
        record["position"] = ["arg", index]
        snapshots.append(record)
    for key, value in input_kwargs.items():
        record = snapshot_input_boundary(value)
        record["position"] = ["kwarg", str(key)]
        snapshots.append(record)
    runnable_trace_state(trace).input_structure = tuple(snapshots)


def _record_runnable_input_tensor_sites(
    trace: "Trace",
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
) -> None:
    """Index model-input TENSOR leaves by object identity for metadata-read witnessing.

    A Python-level metadata predicate read on a model input (``x.is_contiguous()`` /
    ``x.stride()`` / ``x.requires_grad``) can steer control flow that TorchLens never
    observes as an op: the input contract checks only shape+dtype, so a same-shape
    runtime input differing in layout or grad flag would silently replay the wrong
    recorded path. The completeness-witness scoped patch observes such reads during
    the runnable forward; this map lets it attribute a read RECEIVER back to its
    model-boundary site (position, container path) so the producer can witness the
    read fact and the executor can diverge on a mismatched runtime input. Keys are
    ``id(tensor)`` -- stable for the forward's duration because ``input_args`` /
    ``input_kwargs`` hold strong references until the capture completes. It runs only
    for intervention-ready captures and stores no tensors.

    Parameters
    ----------
    trace:
        Active trace.
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Normalized keyword model inputs.
    """

    if not bool(getattr(trace, "intervention_ready", False)):
        return

    from torchlens._input_walk import raw_mapping_key_component, walk_input_boundary

    sites: dict[int, tuple[object, tuple[str | int, ...]]] = {}
    # (tensor, site) leaves so the completeness witness can additionally index model-input
    # leaves by STORAGE identity (r31): a metadata read routed through a ``.data`` / ``.detach()``
    # alias shares the leaf's storage but is a distinct object the id map above misses.
    tensor_leaves: list[tuple[Any, tuple[object, tuple[str | int, ...]]]] = []

    def _walk_site(position: object, value: Any) -> None:
        """Index one boundary site's tensor leaves by identity through the shared traversal.

        Container dispatch is single-sourced in ``torchlens._input_walk`` (r65
        Cluster Y), so this walker descends EXACTLY the container set the literal
        walker descends -- including dataclasses (the r64 Finding-1 false-VERIFIED: a
        missing dataclass branch here recorded no metadata witness for
        ``box.x.is_contiguous()`` on a dataclass input field, so a same-value
        non-contiguous twin replayed the captured branch as VERIFIED). Mapping
        children are indexed under EVERY key with the RAW key component (the declared
        dual vocabulary, residual R6): a fact site whose path carries a
        non-representable key simply fails literal encoding at witness time and is
        dropped, and the literal-leaf walker independently records such a subtree as
        an OPAQUE leaf that downgrades the run to UNVERIFIABLE -- so a dropped
        metadata fact under an exotic key can never yield a false VERIFIED.
        """

        def _index_tensor(tensor: Any, path: tuple[Any, ...]) -> None:
            """Record one tensor leaf's identity-keyed site."""

            site = (position, path)
            sites[id(tensor)] = site
            tensor_leaves.append((tensor, site))

        walk_input_boundary(
            value, (), key_component=raw_mapping_key_component, on_tensor=_index_tensor
        )

    for index, arg in enumerate(input_args):
        _walk_site(("arg", index), arg)
    for key, value in input_kwargs.items():
        _walk_site(("kwarg", key), value)

    if sites:
        runnable_trace_state(trace).input_tensor_sites = sites
        from ..backends.torch.completeness_witness import record_runnable_input_storage_sites

        record_runnable_input_storage_sites(trace, tensor_leaves)


def _record_runnable_module_training_modes(trace: "Trace", model: Any) -> None:
    """Stash the capture-time per-module ``training`` mode for runnable honesty.

    ``self.training`` is module state that is NOT part of the ``state_dict`` and is not a
    model input, yet it steers mode-sensitive ops (BatchNorm running-stats vs batch-stats,
    Dropout on/off). The runnable VERIFIED oracle is a *fresh instance in the captured mode*
    on the given inputs, so the captured mode is DECLARED state the replay reproduces.
    Recording it (per submodule -- submodules can differ) lets the producer declare the mode
    as a witness fact; a mode-sensitive op replayed without a recorded mode fact is downgraded
    to UNVERIFIABLE (fail closed). It runs for every capture (D18 widened the former
    intervention-ready gate so the live refresh projector's mode-claim belt holds on the
    default path; the widening is verdict-inert on the sparse side because the runnable
    producer refuses non-intervention-ready captures outright), touches no tensors, and
    stores an in-memory map consumed by the producer at save time and by the projector belt.

    Parameters
    ----------
    trace:
        Active trace.
    model:
        The prepared source model whose per-module ``training`` flags are recorded.
    """

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    modes: dict[str, bool] = {}
    try:
        for name, module in named_modules():
            address = name or "self"
            modes[address] = bool(getattr(module, "training", False))
    except (AttributeError, TypeError):
        return
    if modes:
        runnable_trace_state(trace).module_training_modes = modes


#: Session-time semantic-output scratch (B1-02). Written at capture entry
#: (``user_funcs.py``) and consumed ONLY by ``decode_outputs_for_trace`` on the
#: normal forward-return arm. Two of the four pin LIVE USER OBJECTS -- an HF
#: tokenizer (``bridge/hf.py``) and a model-derived metadata key -- so a copy
#: surviving onto a settled product is both a retention leak and a privacy leak:
#: plain ``pickle``/``torch.save`` of the Trace serializes the tokenizer's
#: vocab/merges into an artifact the user believes is a graph. Declared
#: ``FieldPolicy.DROP`` on ``Trace`` and dropped on EVERY settlement path.
_SEMANTIC_OUTPUT_TRANSIENT_FIELDS: tuple[str, ...] = (
    "_output_style",
    "_output_head",
    "_output_tokenizer",
    "_semantic_output_metadata",
)


def _drop_semantic_output_transients(self: "Trace") -> None:
    """Drop the session-time semantic-output scratch from one trace.

    Idempotent, and safe on every arm: the sole consumer
    (``decode_outputs_for_trace``) runs on the normal forward-return arm
    strictly before this, and the halted/failed arms never decode at all.

    Parameters
    ----------
    self:
        Trace whose semantic-output scratch should be discarded.

    Returns
    -------
    None. Mutates ``self.__dict__``.
    """

    for attr_name in _SEMANTIC_OUTPUT_TRANSIENT_FIELDS:
        self.__dict__.pop(attr_name, None)


def _publish_streamed_bundle_at_settlement(self: "Trace") -> None:
    """Publish a staged streamed bundle with its just-settled outcome.

    WT1 A-IV item 18 (lane A08): postprocess step 18 STAGES the streamed
    bundle and this seam -- immediately after ``settle_completed`` /
    ``settle_halted`` -- performs the tmp->final publish, injecting the
    settled capture-outcome attestation into ``metadata.pkl`` (the same
    ``_capture_outcome`` key ordinary saves persist). A capture that settles
    FAILED never reaches here: the propagating exception hits the streaming
    abort handler in ``user_funcs.py`` and the temp bundle is marked PARTIAL,
    never published. No-op when nothing is staged (non-streamed captures, the
    deferred grad-streaming mode, the postprocess-free fastlog arm).
    """

    writer = self.__dict__.get("_out_writer")
    if writer is None or not getattr(writer, "staged_for_settlement", False):
        return
    settled = outcome_for(self)
    writer.publish_staged(None if settled is None else settled.to_payload())
    self._out_writer = None


def _scrub_failed_capture_transients(self: "Trace") -> None:
    """Failure-axis twin of the success arms' transient drops (R11/R32).

    The success arms pop ``_output_attribution_input_tensors`` (live USER
    INPUT tensors) and postprocess replaces the mutable ``capture_events``
    alias with the payload-free ``_capture_events`` home. A failed or
    interrupted forward reached neither, so the trace escaping on
    ``exc.partial_log`` pickled the user's input tensors through an
    undeclared attribute and retained every activation payload and grad_fn
    handle of the failed run (GB-class on real models).

    Partial diagnostics stay intact: ``PartialTrace.from_trace`` materialized
    the raw layers during backend cleanup, strictly before this scrub, and
    sidecar release keeps the structural event facts.

    Predicate (fastlog) captures keep their event buffer untouched apart from
    the trace-side alias pop: the buffer may be OWNED by the live Recorder
    (a shared object accumulating prior passes), and the failed-pass snapshot
    (``_failed_fastlog_capture_events``) already isolated the failing pass.

    Parameters
    ----------
    self:
        Trace settled FAILED whose capture transients should be discarded.

    Returns
    -------
    None. Mutates ``self.__dict__``.
    """

    from ..utils.tensor_utils import synchronize_pending_cpu_async_copies

    # A failed forward may leave in-flight cpu_async D2H copies whose
    # pinned host buffers escape on ``exc.partial_log`` (R36-1): fence them
    # before anything reads the partial's payloads.
    synchronize_pending_cpu_async_copies()
    self.__dict__.pop("_output_attribution_input_tensors", None)
    # The aborted-nonfinite frontier stash is a live activation tensor; a
    # failed capture that never ran the prefix finalization must not retain
    # it past settlement (same R11/R32 class as the attribution inputs).
    self.__dict__.pop("_nonfinite_frontier_out", None)
    events = self.__dict__.pop("capture_events", None)
    if events is None:
        return
    if getattr(self, "capture_mode", None) == "predicate":
        return
    if hasattr(events, "release_runtime_sidecars"):
        events.release_runtime_sidecars()
        # The trace is the sole strong owner of its (sidecar-released) event
        # stream, mirroring the postprocess success seam.
        self.__dict__["_capture_events"] = events


def _extract_and_mark_outputs(
    self: "Trace",
    outputs: Any,
    backend: CaptureBackend | None = None,
) -> tuple[list[Any], list[str]]:
    """Extract output tensors from model outputs through the active backend.

    Called AFTER the forward pass completes (outside ``active_logging``). The
    backend marks each output tensor's graph entry as ``is_output_parent=True``
    so postprocessing can identify them.

    Parameters
    ----------
    self:
        Active trace.
    outputs:
        Raw model output object returned by the captured forward pass.
    backend:
        Active capture backend. When omitted, the backend is loaded from the
        trace backend name through the registry.

    Returns
    -------
    tuple[list[Any], list[str]]
        Output tensors and output tensor addresses.
    """
    if backend is None:
        spec = get_backend_spec(str(_backend_name_for_trace(self)))
        if spec.capture_backend is None:
            raise BackendUnsupportedError(
                f"backend={spec.name!r} does not expose a shared capture Protocol adapter."
            )
        backend = spec.capture_backend()
    output_tensors, output_tensor_addresses = backend.extract_and_mark_outputs(
        self,
        outputs,
    )
    return list(output_tensors), output_tensor_addresses


def _settle_interrupted_halted_arm(
    trace: "Trace",
    capture_session: Any,
    interrupt_exc: BaseException,
    halt_exc: HaltSignal,
) -> None:
    """Stamp an interrupt that escaped the halted arm, never masking it.

    Mirrors the outer ``except BaseException`` arm's guarded settlement: the
    stamp is FAILED/interrupted with the halt boundary disclosed, and an
    ordinary settlement/scrub failure attaches as a note instead of replacing
    the unwinding KeyboardInterrupt/SystemExit (B8-23 discipline).
    """

    try:
        settle_failed(
            trace,
            capture_session,
            interrupt_exc,
            interrupted=True,
            settlement_note=(
                "interrupted during halted finalization after halt at "
                f"{getattr(halt_exc, 'reason', '')!r}"
            ),
        )
        _scrub_failed_capture_transients(trace)
    except Exception as settle_exc:
        note = (
            "TorchLens settlement/scrub also failed while handling this "
            f"interrupt: {type(settle_exc).__name__}: {safe_exception_str(settle_exc)}"
        )
        add_note = getattr(interrupt_exc, "add_note", None)
        if add_note is not None:
            add_note(note)
        else:  # Python 3.10: no PEP 678 notes -- surface via warning.
            warnings.warn(note, RuntimeWarning, stacklevel=2)


def _finalize_halted_trace(
    self: "Trace",
    backend: CaptureBackend,
    halt_exc: HaltSignal,
    model: object,
    input_tensors: list[Any],
    postprocess: bool,
) -> Any | None:
    """Finalize a predicate trace that stopped at a halt frontier.

    Parameters
    ----------
    self:
        Active trace.
    backend:
        Active capture backend.
    halt_exc:
        Halt signal raised by the predicate layer.
    model:
        Model whose session metadata should be cleaned up.
    input_tensors:
        Input tensors tagged for the current pass.
    postprocess:
        Whether to run the standard postprocess pipeline.

    Returns
    -------
    Any | None
        Halt-frontier output object when available.
    """

    # Settlement phase marker: everything from halted cleanup through frontier
    # recovery and output extraction is the FINALIZE failure class; the halted
    # postprocess below is POSTPROCESS. The two classes are never conflated.
    set_capture_phase(self, CapturePhase.FINALIZE)
    # F3a: recover the frontier BEFORE model-session cleanup. The cleanup
    # strips TorchLens metadata from model-owned tensors, so the reverse scan
    # must read the raw entries while their capture-time state is intact.
    frontier_output = halt_exc.frontier_output
    if frontier_output is None:
        raw_layer_dict = self._raw_graph_ws.raw_layer_dict
        for event in reversed(getattr(self.capture_events, "op_events", ())):
            entry = raw_layer_dict.get(event.label_raw)
            if entry is not None and getattr(entry, "out", None) is not None:
                frontier_output = entry.out
                break
    backend.cleanup_model_session(self, (model, input_tensors))
    if frontier_output is None:
        raise RuntimeError(
            "trace(halt=...) could not identify a tensor frontier for the halted partial graph."
        ) from halt_exc

    self.halted = True
    self.halt_reason = halt_exc.reason
    self.halt_frontier = halt_exc.reason
    self.raw_output = None
    if not postprocess:
        self.__dict__.pop("_output_attribution_input_tensors", None)
        self.capture_end_time = time.time()
        return frontier_output

    output_tensors, output_tensor_addresses = _extract_and_mark_outputs(
        self,
        frontier_output,
        backend,
    )
    # Mirror the completed paths: the transient attribution inputs are consumed
    # by output extraction and must never survive onto the finished product
    # (an unpopped copy blocked halted analysis saves via PORTABLE_STATE_SPEC).
    self.__dict__.pop("_output_attribution_input_tensors", None)
    _vprint(self, f"Postprocessing halted graph at {self.halt_frontier!r}...")
    set_capture_phase(self, CapturePhase.POSTPROCESS)
    self._postprocess(output_tensors, output_tensor_addresses)
    return frontier_output


def run_and_log_inputs_through_model(
    self: "Trace",
    model: object,
    input_args: Any | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[str | int] | None = "all",
    grad_layers_to_save: str | list[str | int] | None = "all",
    random_seed: int | None = None,
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
    if random_seed is None:
        # R57 (neutrality half): the auto-seed pick draws from a PRIVATE
        # entropy stream, never the user's global ``random`` engine. Drawing
        # from the global stream either leaked one ``randint`` advance past
        # the restore bracket below (capture not byte-neutral to a seeded
        # host process) or, if bracketed, made every ``random_seed=None``
        # capture reuse the identical seed (correlated dropout across runs).
        # The private stream keeps both properties: byte-exact global-engine
        # neutrality AND fresh seeds per capture.
        random_seed = _AUTO_SEED_ENTROPY.randint(1, 4294967294)
    self.random_seed = random_seed  # type: ignore[assignment]
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
        from ..visualization.code_panel import make_weak_model_ref

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
        backend.seed_rng(self, random_seed)
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

        from ..ir import CaptureEvents

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
            random_seed=random_seed,
            postprocess=postprocess,
            stop=self._stop_directive,
        )
        capture_session = attach_legacy_capture_session(
            self,
            backend_token=backend,
            backend_name=str(_backend_name_for_trace(self)),
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
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
                from ..utils._torch_compat import (
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
                                from ..utils.rng import host_nondeterminism_monitor

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

        # F6 boundary checkpoint: a "normal" forward return with the
        # stop-request latch set means user code swallowed the control signal
        # (halt or nonfinite abort) in a broad except. The capture must never
        # be blessed COMPLETE; the typed error settles FAILED through the
        # normal failure arm. Covers tl.trace and every Recorder pass.
        swallowed_stop = self.__dict__.get("_stop_requested")
        if swallowed_stop is not None:
            from .outcome import StopSignalSwallowedError

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
        set_capture_phase(self, CapturePhase.FINALIZE)
        from ..data_classes._nonfinite import drain_pending_nonfinite
        from ..utils.tensor_utils import synchronize_pending_cpu_async_copies

        # Fence every cpu_async D2H copy recorded this forward BEFORE any
        # host-side consumer (finalize, postprocess digests, ``op.out``,
        # ``tl.save`` serialization) can observe partial bytes (R36-1).
        synchronize_pending_cpu_async_copies()
        # Settle deferred track_nonfinite device flags here, after the forward
        # is complete, so the record costs one batch of scalar reads instead of
        # a per-op device synchronization (the flags' kernels are long done).
        drain_pending_nonfinite(self)
        backend.finalize_forward_session(self, self._raw_graph_ws)

        output_transform = getattr(self, "_output_transform", None)
        self.raw_output = output_transform(outputs) if output_transform is not None else None
        from ..autoroute._builtin_output import decode_outputs_for_trace

        decode_outputs_for_trace(
            self,
            outputs,
            output_style=getattr(self, "_output_style", None),
            output_head=getattr(self, "_output_head", None),
        )
        # Tight window on the normal arm: drop immediately after the only
        # consumer. The outer ``finally`` repeats this for the arms that never
        # reach here (halt, failure, interrupt) -- see B1-02.
        _drop_semantic_output_transients(self)

        self.forward_duration = Duration(
            time.time() - self.capture_start_time - self.setup_duration
        )
        _vprint(
            self,
            f"Forward pass complete ({self.forward_duration:.2f s}, "
            f"{len(self.capture_events.op_events)} raw operations)",
        )

        if not postprocess:
            # Extract/mark output tensors BEFORE cleanup, mirroring the
            # postprocess=True branch below. cleanup_model_session() strips
            # TorchLens tensor metadata from every model-owned tensor
            # (buffers included, via _undecorate_model_tensors); extracting
            # afterward would let output-attribution race against that wipe.
            # Callers that skip postprocess (fastlog Recorder) read these
            # scratch results back off the trace and pop them immediately.
            output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(
                self, outputs
            )
            self._fastlog_output_tensors = list(output_tensors_any)
            self._fastlog_output_tensor_addresses = output_tensor_addresses
            capture_session.snapshot_recording_projection(
                self,
                output_tensors=list(output_tensors_any),
                output_tensor_addresses=output_tensor_addresses,
            )
            self._fastlog_captured_run_core = capture_session.seal()
            self.__dict__.pop("_output_attribution_input_tensors", None)
            backend.cleanup_model_session(self, (model, input_tensors, (input_args, input_kwargs)))
            self.capture_end_time = time.time()
            self.__dict__.pop("_capture_producer_policy", None)
            settle_completed(self, capture_session)
            return outputs

        output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(
            self, outputs
        )
        output_tensors = list(output_tensors_any)
        self.__dict__.pop("_output_attribution_input_tensors", None)

        backend.cleanup_model_session(self, (model, input_tensors, (input_args, input_kwargs)))
        _vprint(self, f"Postprocessing {len(self.capture_events.op_events)} operations...")
        set_capture_phase(self, CapturePhase.POSTPROCESS)
        self._postprocess(output_tensors, output_tensor_addresses)
        self.__dict__.pop("_capture_producer_policy", None)
        settle_completed(self, capture_session)
        _publish_streamed_bundle_at_settlement(self)
        return outputs

    except HaltSignal as halt_exc:
        compiled_unwrap_exception = sys.exc_info()
        options = getattr(self, "_predicate_save_options", None)
        if (
            options is not None
            and getattr(options, "halt", None) is not None
            and getattr(self, "_halt_returns_partial_trace", False)
        ):
            try:
                halted_output = _finalize_halted_trace(
                    self,
                    backend,
                    halt_exc,
                    model,
                    input_tensors,
                    postprocess,
                )
            except Exception as secondary_exc:
                # Halted-finalization secondary failure: the capture settles
                # FAILED (FINALIZE or POSTPROCESS per the finalizer's phase
                # markers) with a mandatory disclosure note; the secondary
                # exception propagates with the HaltSignal chained, exactly
                # as before settlement existed.
                self.__dict__.pop("_capture_producer_policy", None)
                settle_failed(
                    self,
                    capture_session,
                    secondary_exc,
                    settlement_note=(
                        "halted finalization failed after halt at "
                        f"{getattr(halt_exc, 'reason', '')!r}"
                    ),
                )
                _scrub_failed_capture_transients(self)
                raise
            except BaseException as halt_interrupt_exc:
                # Halted-arm interrupt: ``except Exception`` above cannot see a
                # KeyboardInterrupt/SystemExit, which used to escape with NO
                # settlement stamp -- the product read UNKNOWN only through the
                # fail-closed no-sidecar default instead of a settled record.
                # Stamp FAILED/interrupted like the outer BaseException arm.
                self.__dict__.pop("_capture_producer_policy", None)
                _settle_interrupted_halted_arm(self, capture_session, halt_interrupt_exc, halt_exc)
                raise
            self.__dict__.pop("_capture_producer_policy", None)
            settle_halted(
                self,
                capture_session,
                halt_exc,
                finalize_partial=True,
                postprocess_ran=postprocess,
            )
            _publish_streamed_bundle_at_settlement(self)
            return halted_output
        try:
            # Same double-fault fence as the failed-forward arm below: a raising
            # seal must not skip the model-session teardown.
            try:
                if capture_session is not None and not postprocess:
                    capture_session.snapshot_recording_projection(self)
                    self._fastlog_captured_run_core = capture_session.seal()
            finally:
                backend.cleanup_halted_forward_session(
                    self, (model, input_tensors, (input_args, input_kwargs))
                )
        except Exception as secondary_exc:
            self.__dict__.pop("_capture_producer_policy", None)
            settle_failed(
                self,
                capture_session,
                secondary_exc,
                settlement_note=(
                    f"halted cleanup failed after halt at {getattr(halt_exc, 'reason', '')!r}"
                ),
            )
            _scrub_failed_capture_transients(self)
            raise
        except BaseException as halt_interrupt_exc:
            # Same halted-arm interrupt stamp as the finalize path above: the
            # seal/cleanup seam's ``except Exception`` cannot see an interrupt,
            # which used to escape unsettled.
            self.__dict__.pop("_capture_producer_policy", None)
            _settle_interrupted_halted_arm(self, capture_session, halt_interrupt_exc, halt_exc)
            raise
        self.__dict__.pop("_capture_producer_policy", None)
        settle_halted(
            self,
            capture_session,
            halt_exc,
            finalize_partial=False,
            postprocess_ran=False,
        )
        raise

    except Exception as e:
        compiled_unwrap_exception = sys.exc_info()
        # Boundary facts snapshot eagerly at cause time: cleanup below may pop
        # the event stream (or double-fault on already-popped workspaces), and
        # the stamp in ``finally`` must still carry the committed-op count.
        committed_ops = count_committed_ops(self)
        # Aborted-nonfinite label unification (observe item 1): postprocess
        # the committed prefix with the offending tensor seeded as the output
        # frontier, so every public surface reads FINAL labels through the
        # step-8 identity map instead of leaking raw spellings. Failure-safe
        # by contract: any finalization failure attaches as secondary
        # evidence on ``e`` and this arm proceeds exactly as before.
        if postprocess:
            from ._nonfinite_prefix import maybe_finalize_nonfinite_prefix

            maybe_finalize_nonfinite_prefix(self, backend, e, model, input_tensors)
        try:
            # The seal runs FIRST (it reads live capture state that cleanup
            # strips) but must not be able to SKIP cleanup: a raising seal used
            # to bypass requires_grad restore, tl_* metadata stripping, buffer
            # tracker uninstall and end_label_session, leaving the user's model
            # permanently altered by a failed capture.
            try:
                try:
                    if capture_session is not None and not postprocess:
                        capture_session.snapshot_recording_projection(self)
                        self._fastlog_captured_run_core = capture_session.seal()
                finally:
                    backend.cleanup_failed_forward_session(
                        self, (model, input_tensors, (input_args, input_kwargs)), e
                    )
                self.__dict__.pop("_capture_producer_policy", None)
            except Exception as cleanup_exc:
                # grind-r6 b1 R06 (sol MED, probe): the PRIMARY user error must
                # propagate. An ordinary seal/cleanup double-fault used to
                # escape INSTEAD of ``raise e`` -- the settled CaptureOutcome
                # named the primary while the escaping exception was the
                # secondary and carried no partial_log. Mirror the interrupt
                # arm: attach the secondary as a note and re-raise the primary
                # below. A BaseException secondary (Ctrl-C during cleanup)
                # keeps escaping -- interrupts always win (B8-23 doctrine).
                note = (
                    "TorchLens failed-forward cleanup also failed while handling "
                    f"this error: {type(cleanup_exc).__name__}: "
                    f"{safe_exception_str(cleanup_exc)}"
                )
                add_note = getattr(e, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:  # Python 3.10: no PEP 678 notes -- surface via warning.
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
        finally:
            # Guaranteed settlement: a cleanup double-fault still stamps the
            # terminal outcome before the (original or secondary) exception
            # escapes; exception identity/chaining is byte-identical to the
            # pre-settlement arms. The transient scrub runs after settlement
            # (R11/R32: the escaping partial must not pin live input tensors
            # or the payload-bearing event stream).
            settle_failed(self, capture_session, e, n_ops_committed=committed_ops)
            _scrub_failed_capture_transients(self)
        raise e

    except BaseException as interrupt_exc:
        # ``except Exception`` above handles ordinary failed-forward diagnostics,
        # but user code may raise e.g. KeyboardInterrupt or a custom BaseException.
        # The torch session forces gradient-capable parameters to require grads, so
        # its teardown must run before re-raising any such escape.
        compiled_unwrap_exception = sys.exc_info()
        committed_ops = count_committed_ops(self)
        try:
            try:
                backend.cleanup_model_session(
                    self, (model, input_tensors, (input_args, input_kwargs))
                )
            except Exception as cleanup_exc:
                # B8-23: the PRIMARY control-flow exception (KeyboardInterrupt /
                # SystemExit) must propagate. Letting an ordinary cleanup
                # Exception escape here demoted the KI to ``__context__``, and
                # a caller's ``except Exception`` retry loop swallowed Ctrl-C
                # outright. Attach the cleanup failure instead of raising it.
                note = (
                    "TorchLens model-session cleanup also failed while handling "
                    f"this interrupt: {type(cleanup_exc).__name__}: "
                    f"{safe_exception_str(cleanup_exc)}"
                )
                add_note = getattr(interrupt_exc, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:  # Python 3.10: no PEP 678 notes -- surface via warning.
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
            self.__dict__.pop("_capture_producer_policy", None)
        finally:
            try:
                settle_failed(
                    self,
                    capture_session,
                    interrupt_exc,
                    interrupted=True,
                    n_ops_committed=committed_ops,
                )
                _scrub_failed_capture_transients(self)
            except Exception as settle_exc:
                # B8-23 applies here too: an ordinary settlement/scrub failure
                # inside this ``finally`` would replace the unwinding
                # KeyboardInterrupt/SystemExit (demoting it to __context__),
                # letting a caller's ``except Exception`` swallow Ctrl-C.
                # Attach the failure to the interrupt instead; an unsettled
                # outcome reads UNKNOWN (most restrictive), never blessed.
                note = (
                    "TorchLens settlement/scrub also failed while handling "
                    f"this interrupt: {type(settle_exc).__name__}: "
                    f"{safe_exception_str(settle_exc)}"
                )
                add_note = getattr(interrupt_exc, "add_note", None)
                if add_note is not None:
                    add_note(note)
                else:  # Python 3.10: no PEP 678 notes -- surface via warning.
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
        raise

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
