import inspect
import random
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch
from torch import nn

from . import _state
from .model_funcs import _ensure_model_prepared, _prepare_model_session, _cleanup_model_session

if TYPE_CHECKING:
    from .data_classes.model_log import ModelLog
from .helper_funcs import (
    get_vars_of_type_from_obj,
    set_random_seed,
    log_current_rng_states,
    set_rng_from_saved_states,
    nested_assign,
    safe_copy_args,
    safe_copy_kwargs,
    normalize_input_args,
)
from .logging_funcs import log_source_tensor
from .interface import _give_user_feedback_about_lookup_key


def _get_input_arg_names(model, input_args):
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
        return which_layers
    elif which_layers in [None, "none", "None", "NONE", []]:
        return []

    if type(which_layers) != list:
        which_layers = [which_layers]
    raw_tensor_nums_to_save = set()
    for layer_key in which_layers:
        if layer_key in self._lookup_keys_to_tensor_num_dict:
            raw_tensor_nums_to_save.add(self._lookup_keys_to_tensor_num_dict[layer_key])
            continue

        keys_with_substr = [key for key in self.layer_dict_all_keys if str(layer_key) in str(key)]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_tensor_nums_to_save.add(self.layer_dict_all_keys[key].realtime_tensor_num)
            continue

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
    return raw_tensor_nums_to_save


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
    for a, arg in enumerate(input_args):
        was_tuple = isinstance(arg, tuple)
        if was_tuple:
            input_args[a] = list(arg)
        for i, (t, addr, addr_full) in enumerate(input_arg_tensors[a]):
            t_moved = t.to(model_device)
            input_arg_tensors[a][i] = (t_moved, addr, addr_full)
            if not addr_full:
                input_args[a] = t_moved
            else:
                nested_assign(input_args[a], addr_full, t_moved)
        if was_tuple and isinstance(input_args[a], list):
            input_args[a] = tuple(input_args[a])

    for k, (key, val) in enumerate(input_kwargs.items()):
        for i, (t, addr, addr_full) in enumerate(input_kwarg_tensors[k]):
            t_moved = t.to(model_device)
            input_kwarg_tensors[k][i] = (t_moved, addr, addr_full)
            if not addr_full:
                input_kwargs[key] = t_moved
            else:
                nested_assign(input_kwargs[key], addr_full, t_moved)

    input_tensors = []
    input_tensor_addresses = []
    for a, arg_tensors in enumerate(input_arg_tensors):
        for t, addr, addr_full in arg_tensors:
            input_tensors.append(t)
            tensor_addr = f"input.{input_arg_names[a]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    for a, kwarg_tensors in enumerate(input_kwarg_tensors):
        for t, addr, addr_full in kwarg_tensors:
            input_tensors.append(t)
            tensor_addr = f"input.{list(input_kwargs.keys())[a]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    return input_tensors, input_tensor_addresses


def run_and_log_inputs_through_model(
    self: "ModelLog",
    model: nn.Module,
    input_args: Union[torch.Tensor, List[Any]],
    input_kwargs: Dict[Any, Any] = None,
    layers_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
    random_seed: Optional[int] = None,
):
    """Runs input through model and logs it in ModelLog.

    Uses toggle-gated decoration: torch functions are already decorated
    at import time.  ``active_logging()`` turns the toggle on for the
    forward pass, ``_cleanup_model_session()`` cleans up session state.
    """
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    self.random_seed_used = random_seed
    set_random_seed(random_seed)

    self._tensor_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)

    # In fast mode, also save parents of output layers so output tensor_contents
    # can be populated in postprocess_fast (issue #46).
    if self._tensor_nums_to_save != "all" and self._pass_finished:
        output_parent_nums = set()
        for output_label in self.output_layers:
            output_entry = self[output_label]
            for parent_label in output_entry.parent_layers:
                parent_entry = self[parent_label]
                output_parent_nums.add(parent_entry.realtime_tensor_num)
        if output_parent_nums:
            combined = set(self._tensor_nums_to_save) | output_parent_nums
            self._tensor_nums_to_save = sorted(combined)

    if type(model) == nn.DataParallel:
        model = model.module

    input_args = normalize_input_args(input_args, model)

    if not input_kwargs:
        input_kwargs = {}

    if len(list(model.parameters())) > 0:
        model_device = next(iter(model.parameters())).device
    elif len(list(model.buffers())) > 0:
        model_device = next(iter(model.buffers())).device
    else:
        model_device = "cpu"

    input_args = safe_copy_args(input_args)
    input_arg_names = _get_input_arg_names(model, input_args)
    input_kwargs = safe_copy_kwargs(input_kwargs)

    self.pass_start_time = time.time()
    input_tensors = []

    try:
        (
            input_tensors,
            input_tensor_addresses,
        ) = _fetch_label_move_input_tensors(input_args, input_arg_names, input_kwargs, model_device)

        # Capture/restore RNG state so the fast pass reproduces the same
        # stochastic graph as the exhaustive pass (issue #58).
        if self.logging_mode == "exhaustive":
            self._pre_forward_rng_states = log_current_rng_states()
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

        output_tensors_w_addresses_all = get_vars_of_type_from_obj(
            outputs,
            torch.Tensor,
            search_depth=5,
            return_addresses=True,
            allow_repeats=True,
        )
        # Remove duplicate addresses
        addresses_used = []
        output_tensors_w_addresses = []
        for entry in output_tensors_w_addresses_all:
            if entry[1] in addresses_used:
                continue
            output_tensors_w_addresses.append(entry)
            addresses_used.append(entry[1])

        output_tensors = [t for t, _, _ in output_tensors_w_addresses]
        output_tensor_addresses = [addr for _, addr, _ in output_tensors_w_addresses]

        for t in output_tensors:
            if self.logging_mode == "exhaustive":
                self.output_layers.append(t.tl_tensor_label_raw)
            self._raw_tensor_dict[t.tl_tensor_label_raw].is_output_parent = True

        _cleanup_model_session(model, input_tensors)
        self._postprocess(output_tensors, output_tensor_addresses)

    except Exception as e:
        # active_logging's finally already turned off the toggle.
        # Just clean up model state.
        _cleanup_model_session(model, input_tensors)
        print(
            "************\nFeature extraction failed; returning model and environment to normal\n*************"
        )
        raise e

    finally:
        input_tensors = None
        torch.cuda.empty_cache()
