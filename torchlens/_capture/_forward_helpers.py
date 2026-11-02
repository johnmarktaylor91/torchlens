"""Forward-pass helpers for the RUN stage: memory bracket, backend lookup and predicate forward.

Moved verbatim from ``torchlens._capture.trace`` by the capture seam.
"""

import contextlib
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

from torchlens import _state
from torchlens._capture import peak_memory
from torchlens.backends.registry import BackendUnsupportedError

from .._quantities import Bytes
from ..backends.registry import (
    BackendName,
    CaptureBackend,
    resolve_backend_spec,
)
from ..recording._halt import StopSignal
from .peak_memory import process_rss_bytes, record_forward_peak, reset_peak_rss
from .session import (
    CaptureSession,
)
from .stop import evaluate_halt_stop

if TYPE_CHECKING:
    from .._capture_state_helpers import CompiledCapturePrep
    from .._data_classes.trace import Trace
from .._utils.display import _timed_phase


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

    if bool(trace._options_structure_only):
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

    Stores the peak on ``trace.cost.forward_peak_memory`` and the backend label on
    ``trace.provenance.forward_memory_backend``. CUDA snapshots ``max_memory_allocated``
    around the forward WITHOUT ``reset_peak_memory_stats`` (R36-2): resetting
    clobbered the caller's process-wide high-water counter on every capture
    (the CPU branch below explicitly refuses the analogous clobber). When the
    forward pushes a new device peak the reported figure is that exact peak;
    a forward that fits under the pre-existing high-water mark legitimately
    reads ``0``, same contract as the CPU/MPS delta below. The figure covers
    the MODEL device only (single-device heuristic, R36 doc line): a
    model-parallel forward's peaks on other devices are not measured.

    CPU measures only under ``CaptureOptions(measure_host_peak_memory=True)``
    (the resident high-water reset is process-wide; otherwise ``None`` with a
    reason); MPS measures its allocator delta. A measured CPU peak of 0 is
    likewise recorded as ``None`` with a reason.

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
                trace._cost_forward_peak_memory = Bytes(live_peak)
                trace._cost_forward_peak_memory_reason = None  # the pair moves together
            with contextlib.suppress(Exception):
                reserved_after = int(torch_module.cuda.max_memory_reserved(cuda_device))
                resident_peak = reserved_after if reserved_after > reserved_before else 0
            trace._provenance_forward_memory_backend = backend_label
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
    # F20 peak PAIR (brainpipe D-7): scope the host resident high-water mark to
    # THIS capture, but only under the measure_host_peak_memory opt-in:
    # the reset is process-wide and also resets the caller's ru_maxrss.
    host_requested = bool(getattr(trace, "measure_host_peak_memory", False))
    rss_peak_scoped = host_requested and reset_peak_rss()
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
        trace._provenance_forward_memory_backend = backend_label
        # F20 peak PAIR (brainpipe D-7). ``live`` is a Python-allocation peak
        # and exists only when the tracemalloc opt-in paid for it; ``None``
        # means unmeasured, never zero. ``resident`` is the host high-water
        # growth over this bracket; ``resident_basis`` discloses its scope or
        # why it is ``None`` (not requested, or an unreadable baseline).
        resident_growth, resident_basis = peak_memory.resident_growth_and_basis(
            rss_before, bool(rss_peak_scoped), host_requested
        )
        measured = peak_memory.host_forward_peak(
            backend_label, rss_delta, int(traced_peak), resident_growth, resident_basis
        )
        record_forward_peak(trace, backend_label, measured, tracemalloc_module, host_requested)
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

    return cast(BackendName, trace._provenance_backend)


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

    from ..recording.types import CaptureSpec, ModuleStackFrame
    from .predicates import _is_halt_only_capture, _module_capture_spec
    from .projections import (
        _build_record_context,
        append_projected_event,
        get_active_recording_state,
    )

    state = get_active_recording_state()
    root_frame = ModuleStackFrame(
        address="",
        module_type=type(model).__name__,
        module_id=id(model),
        pass_number=1,
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
            "module_pass_number": root_frame.pass_number,
        },
        module_stack=state.module_stack,
        history=tuple(state.history),
        op_counts=state.op_counts,
        pass_index=state.pass_index,
        event_index=state.event_index,
        step_index=None,
        time_since_pass_start=time.time() - trace._provenance_start_time,
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
    except StopSignal:
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
        with _timed_phase(trace, "dispatch_forward_model"):
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
                "module_pass_number": root_frame.pass_number,
            },
            module_stack=state.module_stack,
            history=tuple(state.history),
            op_counts=state.op_counts,
            pass_index=state.pass_index,
            event_index=state.event_index,
            step_index=None,
            time_since_pass_start=time.time() - trace._provenance_start_time,
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
        except StopSignal:
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


def _warn_unlogged_dynamo_region(
    self: "Trace", compiled_prep: "CompiledCapturePrep | None"
) -> None:
    """Disclose compiled callables whose interiors the capture cannot log."""

    sites = compiled_prep.sites if compiled_prep is not None else ()
    # Under an active ``force_eager`` stance (torch >= 2.6) the inventoried
    # compiled callables run their original eager Python and their interiors
    # ARE logged; the ceiling below is the honest pre-2.6 fallback.
    if not sites or (compiled_prep is not None and compiled_prep.force_eager_stance):
        return
    self._raw_dynamo_region_detected = True
    self._raw_transform_escape_detected = True
    _state._dynamo_warning_emitted = True
    warnings.warn(
        "TorchLens detected a torch.compile (Dynamo) region on the captured model "
        f"at {', '.join(sites)}. Operations that run inside the "
        "compiled region are not logged: on a cold compile the tensors there are "
        "data-free FakeTensors, while a warm-cache execution can bypass Python "
        "wrappers entirely. The returned Trace contains only operations that ran "
        "OUTSIDE the compiled region. Use the eager callable during capture if you "
        "need its interior logged (on torch >= 2.6, TorchLens instead runs compiled "
        "callables eagerly via torch.compiler.set_stance and this gap does not "
        "arise).",
        UserWarning,
        stacklevel=4,
    )


def _stamp_host_rng_verdict(self: "Trace", global_advanced: bool, channels: Any) -> None:
    """Stamp the host-RNG witness verdict after the monitor tore down."""

    if channels is not None:
        # r65 CLUSTER Z stamping split: a torch RNG ``replayable_read`` (the
        # ``initial_seed`` family, a host scalar fully determined by the
        # capture seed) sets CONSUMED without poisoning the capture seed, so
        # a run at the capture seed stays verified while any other/absent
        # seed ceilings; ceiling ``channels`` alone decide UNREPLAYABLE.
        self._runnable.host_rng_consumed = (
            global_advanced or bool(channels.channels) or bool(channels.replayable_reads)
        )
        self._runnable.host_rng_unreplayable = bool(channels.channels)
        self._runnable.host_rng_channels = tuple(sorted(channels.channels))
        self._runnable.host_rng_replayable_reads = tuple(sorted(channels.replayable_reads))
        self._runnable.rng_monitor_uncertain = bool(channels.uncertain)
        # r39 CLASS A: name the offending threads / coverage failure so the
        # INCOMPLETE ceiling's readiness diagnostic is actionable.
        self._runnable.rng_monitor_uncertain_detail = tuple(channels.uncertain_detail)
    else:
        # Global engines are still bracketed (cheap); the channel verdict was
        # never observed, so it must read as UNKNOWABLE, never as no-consumption.
        self._runnable.host_rng_consumed = global_advanced
        self._runnable.rng_monitor_uncertain = True
        self._runnable.rng_monitor_uncertain_detail = ("monitor_not_armed",)
