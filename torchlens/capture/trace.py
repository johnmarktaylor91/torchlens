"""Forward-pass orchestration: runs the model, manages logging state, and saves outs.

This module implements the two-pass architecture that TorchLens uses to extract
model outs:

1. **Exhaustive pass** (``capture_mode="exhaustive"``): Runs the model once,
   capturing every tensor operation's full metadata (shapes, dtypes, FLOPs,
   parent-child relationships, module context, etc.) into Op entries.
   This builds the complete computational graph.

2. **Fast pass** (``capture_mode="fast"``): Re-runs the model using the graph
   structure from the exhaustive pass, only saving new out values.
   Much faster because it skips all metadata collection.  Used by
   ``save_new_outs()`` to refresh outs for new inputs without
   rebuilding the entire graph.

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

import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .. import _state
from ..backends import (
    BackendName,
    BackendUnsupportedError,
    CaptureBackend,
    get_backend_spec,
    resolve_backend_spec,
)
from ..fastlog._halt import HaltSignal
from ..quantities import Bytes, Duration

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
from ..utils.rng import set_random_seed, log_current_rng_states, set_rng_from_saved_states
from ..utils.tensor_utils import _is_cuda_available
from ..data_classes._lookup_keys import _give_user_feedback_about_lookup_key
from ..utils.display import _timed_phase, _vprint

_ACTIVE_CAPTURE_BACKEND: CaptureBackend | None = None


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


def save_new_outs(
    self: "Trace",
    model: nn.Module,
    input_args: torch.Tensor | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] = "all",
    grad_layers_to_save: str | list[Any] | None = "all",
    random_seed: int | None = None,
    backward_ready: bool | None = None,
) -> None:
    """Re-run the model with new inputs, saving only outs (fast pass).

    This is the public API for refreshing outs without rebuilding the
    computational graph.  Much faster than ``trace`` because all
    metadata (graph structure, labels, module context) was captured in the
    original exhaustive pass and is reused here.

    The fast pass assumes the computational graph is identical to the exhaustive
    pass.  If the model has dynamic control flow that changes between inputs,
    the counter-alignment checks in ``log_function_output_tensors_fast`` will
    detect the mismatch and raise ``ValueError``.

    Args:
        model: Model for which to save outs.
        input_args: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        layers_to_save: List of layers to save, using any valid lookup keys.
        grad_layers_to_save: List of layers whose grads should be saved.
        random_seed: Which random seed to use for deterministic reproduction.
        backward_ready: Optional replay override. ``None`` inherits the existing
            model log settings; explicit values temporarily override saved
            tensor detachment for the whole replay.

    Returns:
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

    # Switch to fast mode: reuse graph structure, only capture new outs.
    self.capture_mode = "fast"
    self._in_exhaustive_pass = False

    # Clear all existing outs from the previous pass.
    for layer_log_entry in self:
        layer_log_entry._internal_set("out", None)
        layer_log_entry._internal_set("transformed_out", None)
        layer_log_entry.transformed_out_shape = None
        layer_log_entry.transformed_out_dtype = None
        layer_log_entry.transformed_activation_memory = None
        layer_log_entry.has_saved_activation = False
        layer_log_entry.has_grad = False
        layer_log_entry._internal_set("grad", None)
        layer_log_entry._internal_set("transformed_grad", None)
        layer_log_entry.transformed_grad_shape = None
        layer_log_entry.transformed_grad_dtype = None
        layer_log_entry.transformed_gradient_memory = None
        layer_log_entry.has_out_variations = False
        layer_log_entry.out_versions_by_child = {}
    self._replay_arg_version_data_complete = False

    # Reset per-pass bookkeeping fields.  Graph-level totals (total_activation_memory,
    # num_tensors) are NOT reset — they describe the static graph structure.
    self._saved_grad_labels = set()
    self.has_gradients = False
    self.num_saved_ops = 0
    self.saved_activation_memory = Bytes(0)
    self.total_gradient_memory = Bytes(0)
    self.saved_gradient_memory = Bytes(0)
    self.func_calls_duration = Duration(0)  # #87: reset timing
    # Reset counters so fast-pass operations align 1:1 with exhaustive-pass labels.
    # Counter alignment is the mechanism that lets the fast pass verify the graph
    # hasn't changed: same counter value → same raw label → same operation.
    self._layer_counter = 0
    self._raw_layer_type_counter = defaultdict(lambda: 0)
    # #97: clear stale internal lookup caches.  User-facing dicts
    # (layer_dict_all_keys, _lookup_keys_to_layer_num_dict) are NOT cleared
    # because _get_op_nums_from_user_labels needs them for layers_to_save lookup
    # before the new forward pass populates them.  They're rebuilt in postprocessing.
    if hasattr(self, "_tensor_num_to_lookup_keys_dict"):
        self._tensor_num_to_lookup_keys_dict.clear()

    # Now run and log the new inputs.
    _vprint(self, "Running fast pass (saving requested outs)")
    self._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed
    )
    if self.save_arg_values:
        self._replay_arg_version_data_complete = True


def _get_op_nums_from_user_labels(
    self: "Trace", which_layers: str | list[str | int] | None
) -> list[int] | str:
    """Resolve user-provided layer identifiers to internal raw_index values.

    Supports exact key match, substring match across all lookup keys, and the
    special sentinel ``"all"`` (which ops through as-is).  Returns sorted
    unique tensor numbers so the fast pass can check membership efficiently.
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

    if type(which_layers) != list:
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


def _extract_and_mark_outputs(
    self: "Trace",
    outputs: Any,
    backend: CaptureBackend | None = None,
) -> tuple[list[torch.Tensor], list[str]]:
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
    tuple[list[torch.Tensor], list[str]]
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
    return cast(list[torch.Tensor], output_tensors), output_tensor_addresses


def _finalize_halted_trace(
    self: "Trace",
    backend: CaptureBackend,
    halt_exc: HaltSignal,
    model: nn.Module,
    input_tensors: list[torch.Tensor],
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
    model: nn.Module,
    input_args: torch.Tensor | list[Any],
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
      5. Capture/restore RNG state for fast-pass reproducibility.
      6. Prepare model (one-time decoration + per-session hooks).
      7. Enter ``active_logging()`` context — toggles ``_state._logging_enabled``.
      8. Log source tensors (inputs), then run ``model(*args, **kwargs)``.
      9. Exit logging context, extract/mark outputs, clean up, postprocess.

    RNG ordering constraint: ``set_random_seed`` and ``log_current_rng_states``
    are called BEFORE ``active_logging()`` because entering the logging context
    may trigger decorated operations (e.g., module hooks) that consume RNG state.
    The fast pass restores the same pre-forward RNG state so that stochastic
    layers (dropout, etc.) produce identical graph structure.
    """
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    self.random_seed = random_seed  # type: ignore[assignment]
    set_random_seed(random_seed)

    if self.capture_mode == "predicate":
        self._layer_nums_to_save = []
        self._grad_op_nums_to_save = []
    else:
        self._layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)  # type: ignore[assignment]
        self._grad_op_nums_to_save = _get_op_nums_from_user_labels(self, grad_layers_to_save)

    # In fast mode, output layers' out are derived from their parents
    # (see postprocess_fast).  If the user requested a subset of layers, we must
    # also include output-layer parents so their outs are available (#46).
    layer_nums_to_save = cast(Any, self._layer_nums_to_save)
    if layer_nums_to_save != "all" and self._tracing_finished:
        output_parent_nums = set()
        for output_label in self.output_layers:
            output_entry = self[output_label]
            for parent_label in output_entry.parents:
                parent_entry = self[parent_label]
                output_parent_nums.add(parent_entry.raw_index)
        if output_parent_nums:
            combined = set(layer_nums_to_save) | output_parent_nums
            self._layer_nums_to_save = sorted(combined)

    backend = _capture_backend_from_registry(
        _backend_name_for_trace(self),
        model,
        input_args,
        input_kwargs,
    )
    input_args, input_kwargs, input_arg_names, model_device = backend.setup_inputs_and_device(
        self,
        model,
        input_args,
        input_kwargs,
    )

    self.capture_start_time = time.time()
    input_tensors: list[torch.Tensor] = []

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
        input_tensors = cast(list[torch.Tensor], input_tensors_any)
        self._input_tensor_addresses = list(input_tensor_addresses)

        # RNG state snapshot/restore for two-pass consistency (#58).
        # Exhaustive pass: snapshot state BEFORE forward so fast pass can replay.
        # Fast pass: restore the snapshot so dropout masks, etc. are identical,
        # ensuring the same computational graph (counter alignment depends on this).
        if self.capture_mode == "exhaustive":
            self._pre_forward_rng_states = log_current_rng_states()  # type: ignore[attr-defined]
        elif self.capture_mode == "fast" and hasattr(self, "_pre_forward_rng_states"):
            set_rng_from_saved_states(self._pre_forward_rng_states)

        from ..ir import CaptureEvents

        self.capture_events = CaptureEvents()

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

        # Turn on the logging toggle and run the forward pass.
        # Inside this context, every decorated torch function will log its
        # inputs/outputs.  Source tensors (model inputs) are logged explicitly
        # before invoking the model; all subsequent operations are captured
        # automatically by the decorated wrappers.
        _vprint(self, f"Running {self.capture_mode} forward pass...")
        with backend.active_logging(self):
            for i, t in enumerate(input_tensors):
                backend.log_source_tensor(self, t, "input", input_tensor_addresses[i])

            if self.capture_mode == "predicate":
                from ..capture.predicates import (
                    _evaluate_halt,
                    _evaluate_keep_module,
                    _is_halt_only_capture,
                )
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
                backend.push_existing_module_frame(self, state.module_stack, root_frame)
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
                    time_since_pass_start=time.time() - self.capture_start_time,
                    include_source_events=state.options.include_source_events,
                    sample_id=state.sample_id,
                )
                halt_only = _is_halt_only_capture(state.options)
                try:
                    if halt_only:
                        _evaluate_halt(enter_ctx, state.options)
                    else:
                        enter_spec = _evaluate_keep_module(enter_ctx, state.options)
                        append_projected_event(
                            self,
                            enter_ctx,
                            enter_spec,
                            predicate_matched=enter_spec.save_out or enter_spec.save_metadata,
                        )
                        _evaluate_halt(enter_ctx, state.options)
                except HaltSignal:
                    raise
                except Exception as exc:
                    state.handle_predicate_exception(enter_ctx, exc)
                    if not halt_only:
                        append_projected_event(
                            self,
                            enter_ctx,
                            skipped_spec,
                            predicate_matched=False,
                        )
                finally:
                    if not halt_only:
                        state.append_context(enter_ctx)
                outputs = None
                try:
                    with _timed_phase(self, "dispatch:forward_model"):
                        outputs = model(*input_args, **input_kwargs)
                finally:
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
                        time_since_pass_start=time.time() - self.capture_start_time,
                        include_source_events=state.options.include_source_events,
                        sample_id=state.sample_id,
                    )
                    try:
                        if halt_only:
                            _evaluate_halt(exit_ctx, state.options, frontier_output=outputs)
                        else:
                            exit_spec = _evaluate_keep_module(exit_ctx, state.options)
                            append_projected_event(
                                self,
                                exit_ctx,
                                exit_spec,
                                predicate_matched=exit_spec.save_out or exit_spec.save_metadata,
                            )
                            _evaluate_halt(exit_ctx, state.options, frontier_output=outputs)
                    except HaltSignal:
                        raise
                    except Exception as exc:
                        state.handle_predicate_exception(exit_ctx, exc)
                        if not halt_only:
                            append_projected_event(
                                self,
                                exit_ctx,
                                skipped_spec,
                                predicate_matched=False,
                            )
                    finally:
                        if not halt_only:
                            state.append_context(exit_ctx)
                        backend.pop_module_frame(self, state.module_stack, root_frame)
            else:
                with _timed_phase(self, "dispatch:forward_model"):
                    outputs = model(*input_args, **input_kwargs)

        backend.finalize_forward_session(self)

        output_transform = getattr(self, "_output_transform", None)
        self.raw_output = output_transform(outputs) if output_transform is not None else None

        self.forward_duration = Duration(
            time.time() - self.capture_start_time - self.setup_duration
        )
        _vprint(
            self,
            f"Forward pass complete ({self.forward_duration:.2f s}, "
            f"{len(self.capture_events.op_events)} raw operations)",
        )

        if not postprocess:
            backend.cleanup_model_session(self, (model, input_tensors))
            self.capture_end_time = time.time()
            return outputs

        output_tensors_any, output_tensor_addresses = backend.extract_and_mark_outputs(
            self, outputs
        )
        output_tensors = cast(list[torch.Tensor], output_tensors_any)

        backend.cleanup_model_session(self, (model, input_tensors))
        _vprint(self, f"Postprocessing {len(self.capture_events.op_events)} operations...")
        self._postprocess(output_tensors, output_tensor_addresses)
        return outputs

    except HaltSignal as halt_exc:
        options = getattr(self, "_predicate_save_options", None)
        if (
            options is not None
            and getattr(options, "halt", None) is not None
            and getattr(self, "_halt_returns_partial_trace", False)
        ):
            return _finalize_halted_trace(
                self,
                backend,
                halt_exc,
                model,
                input_tensors,
                postprocess,
            )
        backend.cleanup_halted_forward_session(self, (model, input_tensors))
        raise

    except Exception as e:
        backend.cleanup_failed_forward_session(self, (model, input_tensors), e)
        raise e

    finally:
        _clear_saved_activation_dedup_caches(self)
        # Release input tensor references so GC can reclaim CUDA memory.
        # Gated behind cached cuda.is_available() so CPU-only runs don't pay
        # the CUDA driver / NVML probe cost (per profiling audit 2026-04-27
        # finding #4).
        input_tensors = None  # type: ignore[assignment]
        if _is_cuda_available():
            torch.cuda.empty_cache()
