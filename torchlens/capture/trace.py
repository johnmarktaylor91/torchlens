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
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from torch import nn

from ..backends import (
    BackendName,
    BackendUnsupportedError,
    CaptureBackend,
    get_backend_spec,
    resolve_backend_spec,
)
from ..fastlog._halt import HaltSignal
from ..ir.container_registry import ModelSite, Phase, Role, walk_container
from ..quantities import Bytes, Duration
from .._capture_state_helpers import unwrap_compiled_submodules
from .config import InternalCaptureConfig
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
from ..utils.rng import host_rng_advanced, snapshot_host_rng

_ACTIVE_CAPTURE_BACKEND: CaptureBackend | None = None


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


def _process_rss_bytes() -> int:
    """Return the current process resident-set size in bytes, or 0 if unavailable.

    Used as a coarse host-memory proxy for the CPU forward-pass peak. psutil is an
    optional dependency; absence degrades to 0 rather than raising.

    Returns
    -------
    int
        Resident-set size in bytes, or 0 when psutil is unavailable.
    """

    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


@contextlib.contextmanager
def _forward_peak_memory_bracket(trace: "Trace", device: "object | None") -> "Iterator[None]":
    """Record forward-pass peak memory around the model forward call.

    Stores the peak on ``trace.forward_peak_memory`` and the backend label on
    ``trace.forward_memory_backend``. CUDA reports the true device-side peak via
    ``max_memory_allocated`` after ``reset_peak_memory_stats``. CPU/MPS use the
    larger of (a) a process resident-set-size delta -- which captures torch's
    C++-allocated tensor buffers for sizeable models -- and (b) the stdlib
    ``tracemalloc`` Python-allocation peak, measured as a delta against a
    baseline snapshot taken at bracket entry (after ``reset_peak()`` when
    tracemalloc was already tracing). This keeps the value scoped to this
    bracket even when tracemalloc was started earlier by external tooling: the
    reset discards any unrelated historical high-water mark, and subtracting
    the entry-time baseline discards memory that is legitimately still live
    but unrelated to this forward pass (e.g. process-wide Python state already
    resident when the bracket was entered). tracemalloc stays reliably
    positive for small models where RSS granularity rounds the delta to zero.

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
        with contextlib.suppress(Exception):
            torch_module.cuda.reset_peak_memory_stats(cuda_device)
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                trace.forward_peak_memory = Bytes(
                    max(0, int(torch_module.cuda.max_memory_allocated(cuda_device)))
                )
            trace.forward_memory_backend = backend_label
        return

    if device_type == "mps" and torch_module is not None and hasattr(torch_module, "mps"):
        backend_label = "mps"
        before = int(torch_module.mps.current_allocated_memory())
    else:
        backend_label = "cpu"
        before = _process_rss_bytes()

    import tracemalloc

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
    traced_baseline = 0
    if tracemalloc.is_tracing():
        with contextlib.suppress(Exception):
            traced_baseline, _peak_at_entry = tracemalloc.get_traced_memory()
    try:
        yield
    finally:
        traced_peak = 0
        if tracemalloc.is_tracing():
            with contextlib.suppress(Exception):
                _current, traced_peak = tracemalloc.get_traced_memory()
                traced_peak = max(0, traced_peak - traced_baseline)
            if tracemalloc_started_here:
                with contextlib.suppress(Exception):
                    tracemalloc.stop()
        if backend_label == "mps" and torch_module is not None:
            after = int(torch_module.mps.current_allocated_memory())
        else:
            after = _process_rss_bytes()
        rss_delta = max(0, after - before)
        trace.forward_memory_backend = backend_label
        trace.forward_peak_memory = Bytes(max(rss_delta, int(traced_peak)))


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
    build_state = trace.__dict__.get("_build_state")
    if build_state is not None:
        registry = getattr(build_state, "container_registry", None)
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

    from ..capture.predicates import _evaluate_keep_module, _is_halt_only_capture
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
            enter_spec = _evaluate_keep_module(enter_ctx, state.options)
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
                exit_spec = _evaluate_keep_module(exit_ctx, state.options)
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
    )
    projected_layer_nums = (
        "all" if layer_nums_to_save == "all" else tuple(cast(list[int], layer_nums_to_save))
    )
    projected_grad_layer_nums = (
        "all"
        if grad_layer_nums_to_save == "all"
        else tuple(cast(list[int], grad_layer_nums_to_save))
    )
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
    self.__dict__["_runnable_output_losslessness"] = refreshed.__dict__.get(
        "_runnable_output_losslessness"
    )
    if self.save_arg_values:
        self._replay_arg_version_data_complete = True


def _get_op_nums_from_user_labels(
    self: "Trace", which_layers: str | list[str | int] | None
) -> list[int] | str:
    """Resolve user-provided layer identifiers to internal raw_index values.

    Supports exact key match, substring match across all lookup keys, and the
    special sentinel ``"all"`` (which ops through as-is).  Returns sorted
    unique raw operation numbers for refresh projection.
    """
    if which_layers == "all":
        return which_layers  # type: ignore[return-value]
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
            raw_layer_nums_to_save.add(self._lookup_keys_to_layer_num_dict[layer_key])  # type: ignore[index]
            continue

        keys_with_substr = [key for key in self.layer_dict_all_keys if str(layer_key) in str(key)]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_layer_nums_to_save.add(self.layer_dict_all_keys[key].raw_index)
            continue

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_layer_nums_to_save = sorted(list(raw_layer_nums_to_save))  # type: ignore[assignment]
    return raw_layer_nums_to_save  # type: ignore[return-value]


def _fetch_label_move_input_tensors(
    input_args: list[Any],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
    model_device: object,
) -> tuple[list[Any], list[str]]:
    """Delegate input tensor movement and source labeling to the active backend.

    Parameters
    ----------
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
        None,
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
    registry = trace._ensure_build_state().container_registry
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

    from collections.abc import Mapping as _Mapping

    import torch as _torch

    from torchlens._io.runnable import (
        EMPTY_CONTAINER_PATH_MARKER,
        _UnsupportedLiteralError,
        _encode_literal_key,
        empty_container_kind,
        input_path_key_component,
    )

    leaves: list[tuple[object, tuple[str | int, ...], Any]] = []

    def _walk(position: object, value: Any, path: tuple[str | int, ...]) -> None:
        """Descend one boundary value, recording every non-tensor leaf.

        A mapping child is descended under *every* key type. When a key is
        representable in the frozen literal grammar (``bool``/``int``/``float``/
        ``str``/``None`` or a safe scalar tuple) the encodable key becomes the
        container-path component so a changed leaf beneath it can be witnessed
        and diverged upon. When a key is *not* representable (enum, object,
        non-finite float, ...) no leaf below it can be re-derived at run time, so
        the whole child subtree is recorded as one OPAQUE marker leaf. That
        downgrades witness coverage to UNVERIFIABLE rather than silently dropping
        the subtree -- a silently skipped leaf under an exotic key is the
        false-VERIFIED money bug this walker exists to prevent.

        An EMPTY container adds no child leaf, so it is witnessed by a synthetic
        marker leaf carrying its KIND at ``(*path, EMPTY_CONTAINER_PATH_MARKER)``
        so an added/removed/kind-changed empty container (which can steer
        ``'flag' in d`` / ``if not lst`` control flow) diverges instead of
        silently replaying the recorded path. A BOOL mapping key is tagged so it
        stays distinct from the equal-valued int key in the leaf-path set.
        """

        if isinstance(value, _torch.Tensor):
            return
        kind = empty_container_kind(value)
        if kind is not None:
            leaves.append((position, (*path, EMPTY_CONTAINER_PATH_MARKER), kind))
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in value._fields:
                _walk(position, getattr(value, name), (*path, str(name)))
            return
        if isinstance(value, _Mapping):
            for key, child in value.items():
                try:
                    _encode_literal_key(key)
                except _UnsupportedLiteralError:
                    leaves.append((position, path, _OPAQUE_INPUT_LEAF))
                    continue
                _walk(position, child, (*path, input_path_key_component(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(position, child, (*path, index))
            return
        leaves.append((position, path, value))

    for index, arg in enumerate(input_args):
        _walk(("arg", index), arg, ())
    for key, value in input_kwargs.items():
        _walk(("kwarg", key), value, ())

    if leaves:
        trace.__dict__["_runnable_input_nontensor_leaves"] = tuple(leaves)


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

    from collections.abc import Mapping as _Mapping

    import torch as _torch

    sites: dict[int, tuple[object, tuple[str | int, ...]]] = {}
    # (tensor, site) leaves so the completeness witness can additionally index model-input
    # leaves by STORAGE identity (r31): a metadata read routed through a ``.data`` / ``.detach()``
    # alias shares the leaf's storage but is a distinct object the id map above misses.
    tensor_leaves: list[tuple[Any, tuple[object, tuple[str | int, ...]]]] = []

    def _walk(position: object, value: Any, path: tuple[str | int, ...]) -> None:
        """Descend one boundary value, indexing every tensor leaf by identity.

        Mapping children are indexed under EVERY key type: a fact site whose path
        carries a non-representable key simply fails literal encoding at witness time
        and is dropped, and the literal-leaf walker independently records such a
        subtree as an OPAQUE leaf that downgrades the run to UNVERIFIABLE -- so a
        dropped metadata fact under an exotic key can never yield a false VERIFIED.
        """

        if isinstance(value, _torch.Tensor):
            site = (position, path)
            sites[id(value)] = site
            tensor_leaves.append((value, site))
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in value._fields:
                _walk(position, getattr(value, name), (*path, str(name)))
            return
        if isinstance(value, _Mapping):
            for key, child in value.items():
                _walk(position, child, (*path, key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(position, child, (*path, index))
            return

    for index, arg in enumerate(input_args):
        _walk(("arg", index), arg, ())
    for key, value in input_kwargs.items():
        _walk(("kwarg", key), value, ())

    if sites:
        trace.__dict__["_runnable_input_tensor_sites"] = sites
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
    to UNVERIFIABLE (fail closed). It runs only for intervention-ready captures, touches no
    tensors, and stores an in-memory map consumed by the producer at save time.

    Parameters
    ----------
    trace:
        Active trace.
    model:
        The prepared source model whose per-module ``training`` flags are recorded.
    """

    if not bool(getattr(trace, "intervention_ready", False)):
        return
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
        trace.__dict__["_runnable_module_training_modes"] = modes


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

    backend.cleanup_model_session(self, (model, input_tensors))
    frontier_output = halt_exc.frontier_output
    if frontier_output is None:
        raw_layer_dict = getattr(self, "_raw_layer_dict", {})
        for event in reversed(getattr(self.capture_events, "op_events", ())):
            entry = raw_layer_dict.get(event.label_raw)
            if entry is not None and getattr(entry, "out", None) is not None:
                frontier_output = entry.out
                break
    if frontier_output is None:
        raise RuntimeError(
            "trace(halt=...) could not identify a tensor frontier for the halted partial graph."
        ) from halt_exc

    self.halted = True
    self.halt_reason = halt_exc.reason
    self.halt_frontier = halt_exc.reason
    self.raw_output = None
    if not postprocess:
        self.capture_end_time = time.time()
        return frontier_output

    output_tensors, output_tensor_addresses = _extract_and_mark_outputs(
        self,
        frontier_output,
        backend,
    )
    _vprint(self, f"Postprocessing halted graph at {self.halt_frontier!r}...")
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
        random_seed = random.randint(1, 4294967294)
    self.random_seed = random_seed  # type: ignore[assignment]
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

    backend.seed_rng(self, random_seed)
    input_args, input_kwargs, input_arg_names, model_device = backend.setup_inputs_and_device(
        self,
        model,
        input_args,
        input_kwargs,
    )

    self.capture_start_time = time.time()
    input_tensors: list[Any] = []
    capture_session: CaptureSession | None = None
    capture_events: object | None = None
    compiled_unwrap_exception: tuple[
        type[BaseException] | None, BaseException | None, TracebackType | None
    ] = (None, None, None)
    compiled_unwrap_context = (
        unwrap_compiled_submodules(model)
        if isinstance(model, nn.Module)
        else contextlib.nullcontext()
    )
    compiled_unwrap_context.__enter__()

    try:
        global _ACTIVE_CAPTURE_BACKEND
        previous_capture_backend = _ACTIVE_CAPTURE_BACKEND
        _ACTIVE_CAPTURE_BACKEND = backend
        try:
            (
                input_tensors_any,
                input_tensor_addresses,
            ) = _fetch_label_move_input_tensors(
                input_args,
                input_arg_names,
                input_kwargs,
                model_device,
            )
        finally:
            _ACTIVE_CAPTURE_BACKEND = previous_capture_backend
        input_tensors = list(input_tensors_any)
        self._input_tensor_addresses = list(input_tensor_addresses)
        self._output_attribution_input_tensors = input_tensors

        # RNG state snapshot for deterministic explicit refreshes and legacy
        # two-pass consistency (#58).
        if self.capture_mode == "exhaustive":
            self._pre_forward_rng_states = backend.snapshot_rng(self)  # type: ignore[attr-defined]

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
            from .._runnable_state import snapshot_capture_state, snapshot_state_alias_topology

            # r37 corr2-4: the live bound-state alias topology (object identity,
            # storage overlap) must be captured BEFORE ``snapshot_capture_state``'s
            # clones erase it; the runnable producer refuses unsupported topologies
            # at save and reproduces identity groups from this record.
            self._runnable_state_alias_topology = snapshot_state_alias_topology(model)
            self._runnable_capture_state = snapshot_capture_state(model)

        # Turn on the logging toggle and run the forward pass.
        # Inside this context, every decorated torch function will log its
        # inputs/outputs.  Source tensors (model inputs) are logged explicitly
        # before invoking the model; all subsequent operations are captured
        # automatically by the decorated wrappers.
        _vprint(self, f"Running {self.capture_mode} forward pass...")
        with backend.active_logging(self):
            for i, t in enumerate(input_tensors):
                backend.log_source_tensor(self, t, "input", input_tensor_addresses[i])
            _register_model_input_container_snapshots(self, input_args, input_kwargs)
            _record_runnable_input_literal_leaves(self, input_args, input_kwargs)
            _record_runnable_input_tensor_sites(self, input_args, input_kwargs)
            _record_runnable_module_training_modes(self, model)
            if bool(getattr(self, "intervention_ready", False)):
                # r35 decision E: capture the ambient backend execution context the
                # forward is about to run under (defaults, matmul precision,
                # determinism, TF32/cuDNN flags, SDP toggles) so the sparse runnable
                # descriptor can restore it explicitly at replay.
                from ..utils._torch_compat import snapshot_ambient_execution_context

                self._runnable_capture_ambient = snapshot_ambient_execution_context()

            if self.capture_mode == "predicate":
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
                            from ..utils.rng import host_nondeterminism_monitor

                            _host_rng_before = snapshot_host_rng()
                            with host_nondeterminism_monitor(model) as _rng_channels:
                                outputs = cast(Callable[..., Any], model)(
                                    *input_args, **input_kwargs
                                )
                            _global_advanced = host_rng_advanced(
                                _host_rng_before, snapshot_host_rng()
                            )
                            self._runnable_host_rng_consumed = _global_advanced or bool(
                                _rng_channels.channels
                            )
                            self._runnable_host_rng_unreplayable = bool(_rng_channels.channels)
                            self._runnable_host_rng_channels = tuple(sorted(_rng_channels.channels))
                            self._runnable_rng_monitor_uncertain = bool(_rng_channels.uncertain)
                            # r39 CLASS A: name the offending threads / coverage failure so
                            # the INCOMPLETE ceiling's readiness diagnostic is actionable.
                            self._runnable_rng_monitor_uncertain_detail = tuple(
                                _rng_channels.uncertain_detail
                            )

        backend.finalize_forward_session(self)

        output_transform = getattr(self, "_output_transform", None)
        self.raw_output = output_transform(outputs) if output_transform is not None else None
        from ..autoroute._builtin_output import decode_outputs_for_trace

        decode_outputs_for_trace(
            self,
            outputs,
            output_style=getattr(self, "_output_style", None),
            output_head=getattr(self, "_output_head", None),
        )
        for attr_name in (
            "_output_style",
            "_output_head",
            "_output_tokenizer",
            "_semantic_output_metadata",
        ):
            self.__dict__.pop(attr_name, None)

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
            capture_session.transition("complete")
            return outputs

        output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(
            self, outputs
        )
        output_tensors = list(output_tensors_any)
        self.__dict__.pop("_output_attribution_input_tensors", None)

        backend.cleanup_model_session(self, (model, input_tensors, (input_args, input_kwargs)))
        _vprint(self, f"Postprocessing {len(self.capture_events.op_events)} operations...")
        self._postprocess(output_tensors, output_tensor_addresses)
        self.__dict__.pop("_capture_producer_policy", None)
        capture_session.transition("complete")
        return outputs

    except HaltSignal as halt_exc:
        compiled_unwrap_exception = sys.exc_info()
        options = getattr(self, "_predicate_save_options", None)
        if (
            options is not None
            and getattr(options, "halt", None) is not None
            and getattr(self, "_halt_returns_partial_trace", False)
        ):
            halted_output = _finalize_halted_trace(
                self,
                backend,
                halt_exc,
                model,
                input_tensors,
                postprocess,
            )
            self.__dict__.pop("_capture_producer_policy", None)
            if capture_session is not None:
                capture_session.transition(
                    "halted",
                )
            return halted_output
        if capture_session is not None and not postprocess:
            capture_session.snapshot_recording_projection(self)
            self._fastlog_captured_run_core = capture_session.seal()
        backend.cleanup_halted_forward_session(
            self, (model, input_tensors, (input_args, input_kwargs))
        )
        self.__dict__.pop("_capture_producer_policy", None)
        if capture_session is not None:
            capture_session.transition("halted")
        raise

    except Exception as e:
        compiled_unwrap_exception = sys.exc_info()
        if capture_session is not None and not postprocess:
            capture_session.snapshot_recording_projection(self)
            self._fastlog_captured_run_core = capture_session.seal()
        backend.cleanup_failed_forward_session(
            self, (model, input_tensors, (input_args, input_kwargs)), e
        )
        self.__dict__.pop("_capture_producer_policy", None)
        if capture_session is not None:
            capture_session.transition(
                "failed",
            )
        raise e

    except BaseException:
        # ``except Exception`` above handles ordinary failed-forward diagnostics,
        # but user code may raise e.g. KeyboardInterrupt or a custom BaseException.
        # The torch session forces gradient-capable parameters to require grads, so
        # its teardown must run before re-raising any such escape.
        compiled_unwrap_exception = sys.exc_info()
        backend.cleanup_model_session(self, (model, input_tensors, (input_args, input_kwargs)))
        self.__dict__.pop("_capture_producer_policy", None)
        if capture_session is not None:
            capture_session.transition("failed")
        raise

    finally:
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
            compiled_unwrap_context.__exit__(*compiled_unwrap_exception)
