"""Forward-pass orchestration: runs the model, manages logging state, and saves activations."""

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
from ..data_classes.interface import _give_user_feedback_about_lookup_key


def save_new_activations(
    self: "ModelLog",
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Union[str, List] = "all",
    random_seed: Optional[int] = None,
):
    """Saves activations to a new input to the model, replacing existing saved activations.
    This will be much faster than the initial call to log_forward_pass (since all the of the metadata has
    already been saved), so if you wish to save the activations to many different inputs for a given model
    this is the function you should use. The one caveat is that this function assumes that the computational
    graph will be the same for the new input; if the model involves a dynamic computational graph that can change
    across inputs, and this graph changes for the new input, then this function will throw an error. In that case,
    you'll have to do a new call to log_forward_pass to log the new graph.

    Args:
        model: Model for which to save activations
        input_args: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        layers_to_save: List of layers to save, using any valid lookup keys
        random_seed: Which random seed to use
    Returns:
        Nothing, but now the ModelLog object will have saved activations for the new input.
    """
    self.logging_mode = "fast"

    # Go through and clear all existing activations.
    for layer_log_entry in self:
        layer_log_entry.tensor_contents = None
        layer_log_entry.has_saved_activations = False
        layer_log_entry.has_saved_grad = False
        layer_log_entry.grad_contents = None
        layer_log_entry.has_child_tensor_variations = False
        # Note: children_tensor_versions is cleared and NOT rebuilt in fast pass.
        # This is a known limitation (#93): validation should not be run after
        # save_new_activations since child tensor variations aren't recaptured.
        layer_log_entry.children_tensor_versions = {}

    # Reset relevant fields.
    self.layers_with_saved_activations = []
    self.layers_with_saved_gradients = []
    self._saved_gradients_set = set()
    self.has_saved_gradients = False
    self.unlogged_layers = []
    self.num_tensors_saved = 0
    self.tensor_fsize_saved = 0
    self.elapsed_time_function_calls = 0  # #87: reset timing
    # Note: tensor_fsize_total and num_tensors_total are NOT reset here —
    # they represent total graph size which doesn't change between fast passes.
    self._layer_counter = 0
    self._raw_layer_type_counter = defaultdict(lambda: 0)
    # #97: clear stale lookup caches (only internal caches, NOT user-facing dicts)
    # Note: layer_dict_all_keys and _lookup_keys_to_layer_num_dict are NOT cleared
    # here because _get_op_nums_from_user_labels needs them for layers_to_save lookup
    # before the new forward pass populates them. They're rebuilt in postprocessing.
    if hasattr(self, "_tensor_num_to_lookup_keys_dict"):
        self._tensor_num_to_lookup_keys_dict.clear()
    if hasattr(self, "_unsaved_layers_lookup_keys"):
        self._unsaved_layers_lookup_keys.clear()

    # Remove zombie entries: raw labels that weren't mapped to final labels (#75)
    zombie_labels = [
        lbl
        for lbl in list(self._raw_layer_labels_list)
        if lbl not in self._raw_to_final_layer_labels
    ]
    for label in zombie_labels:
        self._raw_layer_labels_list.remove(label)
        self._raw_layer_dict.pop(label, None)

    # Now run and log the new inputs.
    self._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, random_seed
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
    """Given list of user layer labels, returns the original tensor numbers for those labels."""
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
                raw_layer_nums_to_save.add(self.layer_dict_all_keys[key].realtime_tensor_num)
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
    """Fetches input tensors, gets their addresses, and moves them to the model device."""
    input_arg_tensors = [
        get_vars_of_type_from_obj(arg, torch.Tensor, search_depth=5, return_addresses=True)
        for arg in input_args
    ]
    input_kwarg_tensors = [
        get_vars_of_type_from_obj(kwarg, torch.Tensor, search_depth=5, return_addresses=True)
        for kwarg in input_kwargs.values()
    ]
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

    Returns:
        (input_args, input_kwargs, input_arg_names, model_device)
    """
    if type(model) == nn.DataParallel:
        model = model.module

    input_args = normalize_input_args(input_args, model)

    if not input_kwargs:
        input_kwargs = {}

    first_param = next(model.parameters(), None)
    first_buffer = next(model.buffers(), None)
    if first_param is not None:
        model_device = first_param.device
    elif first_buffer is not None:
        model_device = first_buffer.device
    else:
        model_device = "cpu"  # type: ignore[assignment]

    input_args = safe_copy_args(input_args)
    input_arg_names = _get_input_arg_names(model, input_args)
    input_kwargs = safe_copy_kwargs(input_kwargs)

    return input_args, input_kwargs, input_arg_names, model_device  # type: ignore[return-value]


def _extract_and_mark_outputs(
    self: "ModelLog",
    outputs: Any,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Extract output tensors from model outputs, deduplicate, and mark in ModelLog.

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
    # Remove duplicate addresses
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
        if self.logging_mode == "exhaustive":
            self.output_layers.append(t.tl_tensor_label_raw)
        self._raw_layer_dict[t.tl_tensor_label_raw].is_output_parent = True

    return output_tensors, output_tensor_addresses


def run_and_log_inputs_through_model(
    self: "ModelLog",
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Optional[Dict[Any, Any]] = None,
    layers_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
    random_seed: Optional[int] = None,
) -> None:
    """Runs input through model and logs it in ModelLog.

    Uses toggle-gated decoration: torch functions are already decorated
    at import time.  ``active_logging()`` turns the toggle on for the
    forward pass, ``_cleanup_model_session()`` cleans up session state.
    """
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    self.random_seed_used = random_seed  # type: ignore[assignment]
    set_random_seed(random_seed)

    self._layer_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)  # type: ignore[assignment, arg-type]

    # In fast mode, also save parents of output layers so output tensor_contents
    # can be populated in postprocess_fast (issue #46).
    if self._layer_nums_to_save != "all" and self._pass_finished:
        output_parent_nums = set()
        for output_label in self.output_layers:
            output_entry = self[output_label]
            for parent_label in output_entry.parent_layers:
                parent_entry = self[parent_label]
                output_parent_nums.add(parent_entry.realtime_tensor_num)
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

        # Capture/restore RNG state so the fast pass reproduces the same
        # stochastic graph as the exhaustive pass (issue #58).
        if self.logging_mode == "exhaustive":
            self._pre_forward_rng_states = log_current_rng_states()  # type: ignore[attr-defined]
        elif self.logging_mode == "fast" and hasattr(self, "_pre_forward_rng_states"):
            set_rng_from_saved_states(self._pre_forward_rng_states)

        # One-time model preparation + incremental sys.modules crawl
        _ensure_model_prepared(model)

        # Per-session model preparation
        _prepare_model_session(self, model, self._optimizer)
        self.elapsed_time_setup = time.time() - self.pass_start_time

        # Activate logging toggle and run forward pass
        with _state.active_logging(self):
            for i, t in enumerate(input_tensors):
                log_source_tensor(self, t, "input", input_tensor_addresses[i])

            outputs = model(*input_args, **input_kwargs)

        self.elapsed_time_forward_pass = (
            time.time() - self.pass_start_time - self.elapsed_time_setup
        )

        output_tensors, output_tensor_addresses = _extract_and_mark_outputs(self, outputs)

        _cleanup_model_session(model, input_tensors)
        self._postprocess(output_tensors, output_tensor_addresses)

    except Exception as e:
        # active_logging's finally already turned off the toggle.
        # Clean up model state and any decorated output tensors (#110).
        _cleanup_model_session(model, input_tensors)
        for label in list(self._raw_layer_dict.keys()):
            entry = self._raw_layer_dict.get(label)
            if (
                entry is not None
                and hasattr(entry, "tensor_contents")
                and entry.tensor_contents is not None
            ):
                for attr in ("tl_tensor_label_raw",):
                    if hasattr(entry.tensor_contents, attr):
                        try:
                            delattr(entry.tensor_contents, attr)
                        except Exception:
                            pass
        print(
            "************\nFeature extraction failed; returning model and environment to normal\n*************"
        )
        raise e

    finally:
        input_tensors = None  # type: ignore[assignment]
        torch.cuda.empty_cache()
