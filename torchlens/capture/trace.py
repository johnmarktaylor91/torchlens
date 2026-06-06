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

import inspect
import dataclasses
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from .. import _state
from ..backends import CaptureBackend
from ..backends.torch._tl import clear_meta, get_tensor_label
from ..backends.torch.backend import TorchBackend
from ..fastlog._halt import HaltSignal
from ..quantities import Bytes, Duration

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
from ..utils.introspection import get_vars_of_type_from_obj, nested_assign
from ..utils.rng import set_random_seed, log_current_rng_states, set_rng_from_saved_states
from ..utils.arg_handling import safe_copy_args, safe_copy_kwargs, normalize_input_args
from ..utils.tensor_utils import _is_cuda_available
from ..backends.torch.sources import log_source_tensor
from ..backends.torch.ops import _walk_output_tensors_with_paths
from ..data_classes._lookup_keys import _give_user_feedback_about_lookup_key
from ..utils.display import _vprint, _vtimed

_TORCH_BACKEND: CaptureBackend = TorchBackend()


def save_new_outs(
    self: "Trace",
    model: nn.Module,
    input_args: torch.Tensor | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[Any] = "all",
    gradients_to_save: str | list[Any] | None = "all",
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
        gradients_to_save: List of layers whose grads should be saved.
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
                gradients_to_save=gradients_to_save,
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
        # Note: out_versions_by_child is cleared and NOT rebuilt in fast pass.
        # This is a known limitation (#93): validation should not be run after
        # save_new_outs since child tensor variations aren't recaptured.
        layer_log_entry.out_versions_by_child = {}

    # Reset per-pass bookkeeping fields.  Graph-level totals (total_activation_memory,
    # num_tensors) are NOT reset — they describe the static graph structure.
    self._saved_grads_set = set()
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
        model, input_args, input_kwargs, layers_to_save, gradients_to_save, random_seed
    )


def _get_input_arg_names(model: nn.Module, input_args: list[Any]) -> list[str]:
    """Extract parameter names from the model's forward() signature for the given input args.

    Inspects the forward method's argspec, strips 'self', and generates synthetic
    names for any *args overflow positions.
    """
    spec = inspect.getfullargspec(model.forward)
    input_arg_names = list(spec.args)
    if "self" in input_arg_names:
        input_arg_names.remove("self")
    input_arg_names = input_arg_names[0 : len(input_args)]
    # Handle *args: generate synthetic names for uncovered positions
    if len(input_arg_names) < len(input_args) and spec.varargs is not None:
        for i in range(len(input_arg_names), len(input_args)):
            input_arg_names.append(f"{spec.varargs}_{i}")
    return input_arg_names


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
    model_device: str,
) -> tuple[list[torch.Tensor], list[str]]:
    """Extract all tensors from input args/kwargs, move to model device, and build addresses.

    Handles nested structures (lists, tuples, dicts) up to ``search_depth=5``.
    Each tensor gets a hierarchical address string like ``"input.x"`` or
    ``"input.x.0.nested"`` that is stored as its ``io_role``.

    Returns:
        (input_tensors, input_tensor_addresses): flat lists of tensors and
        their corresponding address strings.
    """
    input_arg_tensors = [
        get_vars_of_type_from_obj(arg, torch.Tensor, search_depth=5, return_addresses=True)
        for arg in input_args
    ]
    input_kwarg_tensors = [
        get_vars_of_type_from_obj(kwarg, torch.Tensor, search_depth=5, return_addresses=True)
        for kwarg in input_kwargs.values()
    ]
    # Move each tensor to model device.  Tuples must be temporarily converted
    # to lists for item assignment, then converted back to preserve type.
    for arg_idx, arg in enumerate(input_args):
        was_tuple = isinstance(arg, tuple)
        if was_tuple:
            input_args[arg_idx] = list(arg)
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_arg_tensors[arg_idx]):
            moved_tensor = tensor.to(model_device)
            input_arg_tensors[arg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_args[arg_idx] = moved_tensor
            else:
                nested_assign(input_args[arg_idx], addr_full, moved_tensor)
        if was_tuple and isinstance(input_args[arg_idx], list):
            input_args[arg_idx] = tuple(input_args[arg_idx])

    for kwarg_idx, (key, val) in enumerate(input_kwargs.items()):
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_kwarg_tensors[kwarg_idx]):
            moved_tensor = tensor.to(model_device)
            input_kwarg_tensors[kwarg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_kwargs[key] = moved_tensor
            else:
                nested_assign(input_kwargs[key], addr_full, moved_tensor)

    # Build flat lists of (tensor, address) for both positional and keyword args.
    # Address format: "input.<argname>" or "input.<argname>.<nested_path>"
    input_tensors = []
    input_tensor_addresses = []
    for arg_idx, arg_tensors in enumerate(input_arg_tensors):
        for tensor, addr, addr_full in arg_tensors:
            input_tensors.append(tensor)
            tensor_addr = f"input.{input_arg_names[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    for arg_idx, kwarg_tensors in enumerate(input_kwarg_tensors):
        for tensor, addr, addr_full in kwarg_tensors:
            input_tensors.append(tensor)
            tensor_addr = f"input.{list(input_kwargs.keys())[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    return input_tensors, input_tensor_addresses


def _setup_inputs_and_device(
    model: nn.Module,
    input_args: torch.Tensor | list[Any],
    input_kwargs: dict[Any, Any] | None,
) -> tuple[list[Any], dict[Any, Any], list[str], str]:
    """Normalize inputs, detect model device, copy args, and extract input arg names.

    This is the single place where user-provided inputs are transformed into
    the canonical internal form:
      1. Unwrap DataParallel to get the underlying module.
      2. ``normalize_input_args``: resolve the tuple-vs-single-arg ambiguity
         by inspecting the model's forward() signature.
      3. ``safe_copy_args/kwargs``: clone tensors so in-place device moves
         (in ``_fetch_label_move_input_tensors``) don't mutate the caller's data.
      4. Detect model device from first param or buffer (for auto-moving inputs).

    Returns:
        (input_args, input_kwargs, input_arg_names, model_device)
    """
    if type(model) == nn.DataParallel:
        model = model.module

    # Resolve ambiguity: is [tensor_a, tensor_b] two args or one list-arg?
    # normalize_input_args checks the model's forward() signature to decide.
    input_args = normalize_input_args(input_args, model)

    if not input_kwargs:
        input_kwargs = {}

    # Detect device from first param or buffer; fall back to CPU for param-free models.
    first_param = next(model.parameters(), None)
    first_buffer = next(model.buffers(), None)
    if first_param is not None:
        model_device = first_param.device
    elif first_buffer is not None:
        model_device = first_buffer.device
    else:
        model_device = "cpu"  # type: ignore[assignment]

    # Clone tensors to protect user's originals from in-place device moves.
    input_args = safe_copy_args(input_args)
    input_arg_names = _get_input_arg_names(model, input_args)
    input_kwargs = safe_copy_kwargs(input_kwargs)

    return input_args, input_kwargs, input_arg_names, model_device  # type: ignore[return-value]


def _extract_and_mark_outputs(
    self: "Trace",
    outputs: Any,
) -> tuple[list[torch.Tensor], list[str]]:
    """Extract output tensors from model outputs, deduplicate, and mark in Trace.

    Called AFTER the forward pass completes (outside ``active_logging``), so
    operations here don't trigger logging.  Marks each output tensor's graph
    entry as ``is_output_parent=True`` so postprocessing can identify them.

    Deduplication is by address string (not tensor identity) to handle cases
    where the same tensor appears at multiple output positions.

    Returns:
        (output_tensors, output_tensor_addresses)
    """
    if getattr(self, "intervention_ready", False):
        output_tensors_w_addresses_all = [
            (tensor, _container_path_to_address(path), None)
            for tensor, path, container_spec in _walk_output_tensors_with_paths(outputs)
        ]
        output_specs_by_raw_label = {}
        for tensor, path, container_spec in _walk_output_tensors_with_paths(outputs):
            _label_raw = get_tensor_label(tensor)
            if _label_raw is not None:
                output_specs_by_raw_label[_label_raw] = (
                    path,
                    container_spec,
                )
        setattr(self, "_output_container_specs_by_raw_label", output_specs_by_raw_label)
    else:
        output_tensors_w_addresses_all = get_vars_of_type_from_obj(
            outputs,
            torch.Tensor,
            search_depth=5,
            return_addresses=True,
            allow_repeats=True,
        )
    # Remove duplicate addresses (same tensor at multiple output positions).
    addresses_seen = set()
    output_tensors_w_addresses = []
    for entry in output_tensors_w_addresses_all:
        if entry[1] in addresses_seen:
            continue
        output_tensors_w_addresses.append(entry)
        addresses_seen.add(entry[1])

    output_tensors = [t for t, _, _ in output_tensors_w_addresses]
    output_tensor_addresses = [addr for _, addr, _ in output_tensors_w_addresses]

    for t in output_tensors:
        # Only record output_layers during exhaustive pass; fast pass reuses the list.
        # Defensive: user-injected output tensors (raw register_forward_hook
        # returning a fresh tensor, intervention API replacements that don't
        # propagate metadata, etc.) lack _tl labels. Skip them rather than
        # crashing — they aren't in our graph but the experiment can continue.
        _label_raw = get_tensor_label(t)
        if _label_raw is None:
            continue
        if self.capture_mode == "exhaustive":
            self.output_layers.append(_label_raw)
            event = self.capture_events.op_event_by_label_raw.get(_label_raw)
            if event is not None:
                updated_event = dataclasses.replace(event, is_output_parent=True)
                self.capture_events.op_event_by_label_raw[_label_raw] = updated_event
                for index, existing_event in enumerate(self.capture_events.op_events):
                    if existing_event.label_raw == _label_raw:
                        self.capture_events.op_events[index] = updated_event
                        break

    return output_tensors, output_tensor_addresses


def _container_path_to_address(path: tuple[Any, ...]) -> str:
    """Convert an output path tuple to TorchLens' display address string.

    Parameters
    ----------
    path
        Path components from path-aware output traversal.

    Returns
    -------
    str
        Dot-separated output address suffix.
    """

    parts: list[str] = []
    for component in path:
        if hasattr(component, "index"):
            parts.append(str(component.index))
        elif hasattr(component, "key"):
            parts.append(str(component.key))
        elif hasattr(component, "name"):
            parts.append(str(component.name))
        else:
            parts.append(str(component))
    return ".".join(parts)


def run_and_log_inputs_through_model(
    self: "Trace",
    model: nn.Module,
    input_args: torch.Tensor | list[Any],
    input_kwargs: dict[Any, Any] | None = None,
    layers_to_save: str | list[str | int] | None = "all",
    gradients_to_save: str | list[str | int] | None = "all",
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
        self._grad_layer_nums_to_save = []
    else:
        self._layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)  # type: ignore[assignment]
        self._grad_layer_nums_to_save = _get_op_nums_from_user_labels(self, gradients_to_save)

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

    input_args, input_kwargs, input_arg_names, model_device = _setup_inputs_and_device(
        model,
        input_args,
        input_kwargs,
    )

    self.capture_start_time = time.time()
    input_tensors: list[torch.Tensor] = []

    try:
        (
            input_tensors,
            input_tensor_addresses,
        ) = _fetch_label_move_input_tensors(input_args, input_arg_names, input_kwargs, model_device)
        self._input_tensor_addresses = list(input_tensor_addresses)

        # RNG state snapshot/restore for two-pass consistency (#58).
        # Exhaustive pass: snapshot state BEFORE forward so fast pass can replay.
        # Fast pass: restore the snapshot so dropout masks, etc. are identical,
        # ensuring the same computational graph (counter alignment depends on this).
        if self.capture_mode == "exhaustive":
            self._pre_forward_rng_states = log_current_rng_states()  # type: ignore[attr-defined]
        elif self.capture_mode == "fast" and hasattr(self, "_pre_forward_rng_states"):
            set_rng_from_saved_states(self._pre_forward_rng_states)

        backend = _TORCH_BACKEND
        from ..ir import CaptureEvents

        self.capture_events = CaptureEvents()

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
                log_source_tensor(self, t, "input", input_tensor_addresses[i])

            if self.capture_mode == "predicate":
                from ..backends.torch import module_stack as _mstack
                from ..capture.predicates import _evaluate_keep_module
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
                _mstack.push_existing_frame(state.module_stack, root_frame)
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
                try:
                    enter_spec = _evaluate_keep_module(enter_ctx, state.options)
                    append_projected_event(
                        self,
                        enter_ctx,
                        enter_spec,
                        predicate_matched=enter_spec.save_out or enter_spec.save_metadata,
                    )
                except HaltSignal:
                    raise
                except Exception as exc:
                    state.handle_predicate_exception(enter_ctx, exc)
                    append_projected_event(
                        self,
                        enter_ctx,
                        skipped_spec,
                        predicate_matched=False,
                    )
                finally:
                    state.append_context(enter_ctx)
                try:
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
                        exit_spec = _evaluate_keep_module(exit_ctx, state.options)
                        append_projected_event(
                            self,
                            exit_ctx,
                            exit_spec,
                            predicate_matched=exit_spec.save_out or exit_spec.save_metadata,
                        )
                    except HaltSignal:
                        raise
                    except Exception as exc:
                        state.handle_predicate_exception(exit_ctx, exc)
                        append_projected_event(
                            self,
                            exit_ctx,
                            skipped_spec,
                            predicate_matched=False,
                        )
                    finally:
                        state.append_context(exit_ctx)
                        _mstack.pop_frame(state.module_stack, root_frame)
            else:
                outputs = model(*input_args, **input_kwargs)

        from ..backends.torch.buffer_writes import reconcile_buffer_writes

        reconcile_buffer_writes(self)

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

        output_tensors, output_tensor_addresses = _extract_and_mark_outputs(self, outputs)

        backend.cleanup_model_session(self, (model, input_tensors))
        _vprint(self, f"Postprocessing {len(self.capture_events.op_events)} operations...")
        self._postprocess(output_tensors, output_tensor_addresses)
        return outputs

    except HaltSignal:
        _TORCH_BACKEND.cleanup_model_session(self, (model, input_tensors))
        raw_layer_dict = getattr(self, "_raw_layer_dict", {})
        for label in list(raw_layer_dict.keys()):
            entry = raw_layer_dict.get(label)
            if entry is not None and hasattr(entry, "out") and entry.out is not None:
                clear_meta(entry.out)
        raise

    except Exception as e:
        # active_logging's __exit__ already turned off the toggle.
        # Clean up model session state and strip TorchLens metadata from any
        # partially-constructed tensor entries to avoid stale references (#110).
        from ..partial import PartialTrace

        try:
            e.partial_log = PartialTrace.from_trace(self, e)  # type: ignore[attr-defined]
        except Exception:
            pass
        _TORCH_BACKEND.cleanup_model_session(self, (model, input_tensors))
        raw_layer_dict = getattr(self, "_raw_layer_dict", {})
        for label in list(raw_layer_dict.keys()):
            entry = raw_layer_dict.get(label)
            if entry is not None and hasattr(entry, "out") and entry.out is not None:
                clear_meta(entry.out)
        print(
            "************\nFeature extraction failed; returning model and environment to normal\n*************"
        )
        raise e

    finally:
        # Release input tensor references so GC can reclaim CUDA memory.
        # Gated behind cached cuda.is_available() so CPU-only runs don't pay
        # the CUDA driver / NVML probe cost (per profiling audit 2026-04-27
        # finding #4).
        input_tensors = None  # type: ignore[assignment]
        if _is_cuda_available():
            torch.cuda.empty_cache()
