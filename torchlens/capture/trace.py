"""Forward-pass orchestration: runs the model, manages logging state, and saves activations.

This module implements the two-pass architecture that TorchLens uses to extract
model activations:

1. **Exhaustive pass** (``logging_mode="exhaustive"``): Runs the model once,
   capturing every tensor operation's full metadata (shapes, dtypes, FLOPs,
   parent-child relationships, module context, etc.) into LayerPassLog entries.
   This builds the complete computational graph.

2. **Fast pass** (``logging_mode="fast"``): Re-runs the model using the graph
   structure from the exhaustive pass, only saving new activation values.
   Much faster because it skips all metadata collection.  Used by
   ``save_new_activations()`` to refresh activations for new inputs without
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
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch
from torch import nn

from .. import _state
from ..decoration.model_prep import (
    _ensure_model_prepared,
    _prepare_model_session,
    _cleanup_model_session,
)

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog
from ..utils.introspection import get_vars_of_type_from_obj, nested_assign
from ..utils.rng import set_random_seed, log_current_rng_states, set_rng_from_saved_states
from ..utils.arg_handling import safe_copy_args, safe_copy_kwargs, normalize_input_args
from .source_tensors import log_source_tensor
from ..data_classes._lookup_keys import _give_user_feedback_about_lookup_key
from ..utils.display import _vprint, _vtimed


def save_new_activations(
    self: "ModelLog",
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Union[str, List] = "all",
    gradients_to_save: Union[str, List, None] = "all",
    random_seed: Optional[int] = None,
    train_mode: bool | None = None,
) -> None:
    """Re-run the model with new inputs, saving only activations (fast pass).

    This is the public API for refreshing activations without rebuilding the
    computational graph.  Much faster than ``log_forward_pass`` because all
    metadata (graph structure, labels, module context) was captured in the
    original exhaustive pass and is reused here.

    The fast pass assumes the computational graph is identical to the exhaustive
    pass.  If the model has dynamic control flow that changes between inputs,
    the counter-alignment checks in ``log_function_output_tensors_fast`` will
    detect the mismatch and raise ``ValueError``.

    Args:
        model: Model for which to save activations.
        input_args: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        layers_to_save: List of layers to save, using any valid lookup keys.
        gradients_to_save: List of layers whose gradients should be saved.
        random_seed: Which random seed to use for deterministic reproduction.
        train_mode: Optional replay override. ``None`` inherits the existing
            model log settings; explicit values temporarily override saved
            tensor detachment for the whole replay.

    Returns:
        Nothing; mutates ``self`` in place with new activation values.
    """
    if train_mode is not None:
        model_detach_saved_tensors = self.detach_saved_tensors
        model_train_mode = getattr(self, "train_mode", False)
        layer_detach_saved_tensors = {
            layer_log_entry: layer_log_entry.detach_saved_tensor for layer_log_entry in self
        }
        target_detach_saved_tensors = False if train_mode else self.detach_saved_tensors
        try:
            self.detach_saved_tensors = target_detach_saved_tensors
            self.train_mode = train_mode
            for layer_log_entry in layer_detach_saved_tensors:
                layer_log_entry.detach_saved_tensor = target_detach_saved_tensors
            save_new_activations(
                self,
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                layers_to_save=layers_to_save,
                gradients_to_save=gradients_to_save,
                random_seed=random_seed,
                train_mode=None,
            )
        finally:
            self.detach_saved_tensors = model_detach_saved_tensors
            self.train_mode = model_train_mode
            for layer_log_entry, detach_saved_tensor in layer_detach_saved_tensors.items():
                layer_log_entry.detach_saved_tensor = detach_saved_tensor
        return

    # Switch to fast mode: reuse graph structure, only capture new activations.
    self.logging_mode = "fast"
    self._in_exhaustive_pass = False

    # Clear all existing activations from the previous pass.
    for layer_log_entry in self:
        layer_log_entry.activation = None
        layer_log_entry.has_saved_activations = False
        layer_log_entry.has_gradient = False
        layer_log_entry.gradient = None
        layer_log_entry.has_child_tensor_variations = False
        # Note: children_tensor_versions is cleared and NOT rebuilt in fast pass.
        # This is a known limitation (#93): validation should not be run after
        # save_new_activations since child tensor variations aren't recaptured.
        layer_log_entry.children_tensor_versions = {}

    # Reset per-pass bookkeeping fields.  Graph-level totals (total_activation_memory,
    # num_tensors_total) are NOT reset — they describe the static graph structure.
    self.layers_with_saved_activations = []
    self.layers_with_saved_gradients = []
    self._saved_gradients_set = set()
    self.has_gradients = False
    self.unlogged_layers = []
    self.num_tensors_saved = 0
    self.saved_activation_memory = 0
    self.time_function_calls = 0  # #87: reset timing
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
    if hasattr(self, "_unsaved_layers_lookup_keys"):
        self._unsaved_layers_lookup_keys.clear()

    # Remove zombie entries: raw labels that weren't mapped to final labels (#75).
    # These arise from operations that were later pruned from the graph (e.g.,
    # orphan removal).  If left, the fast pass would try to look them up and crash.
    zombie_labels = [
        lbl
        for lbl in list(self._raw_layer_labels_list)
        if lbl not in self._raw_to_final_layer_labels
    ]
    for label in zombie_labels:
        self._raw_layer_labels_list.remove(label)
        self._raw_layer_dict.pop(label, None)

    # Now run and log the new inputs.
    _vprint(self, "Running fast pass (saving requested activations)")
    self._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, gradients_to_save, random_seed
    )


def _get_input_arg_names(model, input_args) -> List[str]:
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
    self: "ModelLog", which_layers: Union[str, List[Union[str, int]]]
) -> Union[List[int], str]:
    """Resolve user-provided layer identifiers to internal creation_order values.

    Supports exact key match, substring match across all lookup keys, and the
    special sentinel ``"all"`` (which passes through as-is).  Returns sorted
    unique tensor numbers so the fast pass can check membership efficiently.
    """
    if which_layers == "all":
        return which_layers  # type: ignore[return-value]
    elif which_layers in [None, "none", "None", "NONE", []]:
        return []

    if type(which_layers) != list:
        which_layers = [which_layers]  # type: ignore[list-item]
    raw_layer_nums_to_save = set()
    for layer_key in which_layers:
        if layer_key in self._lookup_keys_to_layer_num_dict:
            raw_layer_nums_to_save.add(self._lookup_keys_to_layer_num_dict[layer_key])  # type: ignore[index]
            continue

        keys_with_substr = [key for key in self.layer_dict_all_keys if str(layer_key) in str(key)]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_layer_nums_to_save.add(self.layer_dict_all_keys[key].creation_order)
            continue

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_layer_nums_to_save = sorted(list(raw_layer_nums_to_save))  # type: ignore[assignment]
    return raw_layer_nums_to_save  # type: ignore[return-value]


def _fetch_label_move_input_tensors(
    input_args: List[Any],
    input_arg_names: List[str],
    input_kwargs: Dict,
    model_device: str,
) -> Tuple[List[torch.Tensor], List[str]]:
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
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Optional[Dict[Any, Any]],
) -> Tuple[List[Any], Dict[Any, Any], List[str], str]:
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
    self: "ModelLog",
    outputs: Any,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Extract output tensors from model outputs, deduplicate, and mark in ModelLog.

    Called AFTER the forward pass completes (outside ``active_logging``), so
    operations here don't trigger logging.  Marks each output tensor's graph
    entry as ``feeds_output=True`` so postprocessing can identify them.

    Deduplication is by address string (not tensor identity) to handle cases
    where the same tensor appears at multiple output positions.

    Returns:
        (output_tensors, output_tensor_addresses)
    """
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
        if self.logging_mode == "exhaustive":
            self.output_layers.append(t.tl_tensor_label_raw)
        self._raw_layer_dict[t.tl_tensor_label_raw].feeds_output = True

    return output_tensors, output_tensor_addresses


def run_and_log_inputs_through_model(
    self: "ModelLog",
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
    gradients_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
    random_seed: Optional[int] = None,
) -> None:
    """Core orchestration: run a forward pass and log everything into ModelLog.

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
    self.random_seed_used = random_seed  # type: ignore[assignment]
    set_random_seed(random_seed)

    self._layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)  # type: ignore[assignment, arg-type]
    self._gradient_layer_nums_to_save = _get_op_nums_from_user_labels(self, gradients_to_save)  # type: ignore[assignment, arg-type]

    # In fast mode, output layers' activation are derived from their parents
    # (see postprocess_fast).  If the user requested a subset of layers, we must
    # also include output-layer parents so their activations are available (#46).
    if self._layer_nums_to_save != "all" and self._pass_finished:
        output_parent_nums = set()
        for output_label in self.output_layers:
            output_entry = self[output_label]
            for parent_label in output_entry.parent_layers:
                parent_entry = self[parent_label]
                output_parent_nums.add(parent_entry.creation_order)
        if output_parent_nums:
            combined = set(self._layer_nums_to_save) | output_parent_nums
            self._layer_nums_to_save = sorted(combined)

    input_args, input_kwargs, input_arg_names, model_device = _setup_inputs_and_device(
        model,
        input_args,
        input_kwargs,
    )

    self.pass_start_time = time.time()
    input_tensors: List[torch.Tensor] = []

    try:
        (
            input_tensors,
            input_tensor_addresses,
        ) = _fetch_label_move_input_tensors(input_args, input_arg_names, input_kwargs, model_device)

        # RNG state snapshot/restore for two-pass consistency (#58).
        # Exhaustive pass: snapshot state BEFORE forward so fast pass can replay.
        # Fast pass: restore the snapshot so dropout masks, etc. are identical,
        # ensuring the same computational graph (counter alignment depends on this).
        if self.logging_mode == "exhaustive":
            self._pre_forward_rng_states = log_current_rng_states()  # type: ignore[attr-defined]
        elif self.logging_mode == "fast" and hasattr(self, "_pre_forward_rng_states"):
            set_rng_from_saved_states(self._pre_forward_rng_states)

        # One-time model preparation + incremental sys.modules crawl
        _ensure_model_prepared(model)

        # Per-session model preparation
        _prepare_model_session(self, model, self._optimizer)
        self.time_setup = time.time() - self.pass_start_time
        _vprint(self, f"Model prepared ({self.time_setup:.2f}s)")

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
        _vprint(self, f"Running {self.logging_mode} forward pass...")
        with _state.active_logging(self):
            for i, t in enumerate(input_tensors):
                log_source_tensor(self, t, "input", input_tensor_addresses[i])

            outputs = model(*input_args, **input_kwargs)

        self.time_forward_pass = time.time() - self.pass_start_time - self.time_setup
        _vprint(
            self,
            f"Forward pass complete ({self.time_forward_pass:.2f}s, "
            f"{len(self._raw_layer_dict)} raw operations)",
        )

        output_tensors, output_tensor_addresses = _extract_and_mark_outputs(self, outputs)

        _cleanup_model_session(model, input_tensors)
        _vprint(self, f"Postprocessing {len(self._raw_layer_dict)} operations...")
        self._postprocess(output_tensors, output_tensor_addresses)

    except Exception as e:
        # active_logging's __exit__ already turned off the toggle.
        # Clean up model session state and strip tl_ attributes from any
        # partially-constructed tensor entries to avoid stale references (#110).
        _cleanup_model_session(model, input_tensors)
        for label in list(self._raw_layer_dict.keys()):
            entry = self._raw_layer_dict.get(label)
            if entry is not None and hasattr(entry, "activation") and entry.activation is not None:
                for attr in ("tl_tensor_label_raw",):
                    if hasattr(entry.activation, attr):
                        try:
                            delattr(entry.activation, attr)
                        except Exception:
                            pass
        print(
            "************\nFeature extraction failed; returning model and environment to normal\n*************"
        )
        raise e

    finally:
        # Release input tensor references so GC can reclaim CUDA memory.
        input_tensors = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
