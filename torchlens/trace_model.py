import copy
import inspect
import random
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch
from torch import nn

if TYPE_CHECKING:
    from .model_history import ModelHistory
from .decorate_torch import undecorate_pytorch
from .helper_funcs import get_vars_of_type_from_obj, set_random_seed, nested_assign
from .logging_funcs import log_source_tensor
from .interface import _give_user_feedback_about_lookup_key


def _get_input_arg_names(model, input_args):
    input_arg_names = inspect.getfullargspec(model.forward).args
    if "self" in input_arg_names:
        input_arg_names.remove("self")
    input_arg_names = input_arg_names[0: len(input_args)]
    return input_arg_names


def _get_op_nums_from_user_labels(
        self: "ModelHistory", which_layers: Union[str, List[Union[str, int]]]
) -> List[int]:
    """Given list of user layer labels, returns the original tensor numbers for those labels (i.e.,
    the numbers that were generated on the fly during the forward pass, such that they can be
    saved on a subsequent pass). Raises an error if the user's labels don't correspond to any layers.

    Args:
        which_layers: List of layers to include, using any indexing desired: either the layer label,
        the module label, or the ordinal position of the layer. If a layer has multiple passes and
        none is specified, will return all of them.

    Returns:
        Ordered, unique list of raw tensor numbers associated with the specified layers.
    """
    if which_layers == "all":
        return which_layers
    elif which_layers in [None, "none", "None", "NONE", []]:
        return []

    if type(which_layers) != list:
        which_layers = [which_layers]
    which_layers = [
        layer.lower() if (type(layer) == str) else layer for layer in which_layers
    ]

    raw_tensor_nums_to_save = set()
    for layer_key in which_layers:
        # First check if it matches a lookup key. If so, use that.
        if layer_key in self._lookup_keys_to_tensor_num_dict:
            raw_tensor_nums_to_save.add(
                self._lookup_keys_to_tensor_num_dict[layer_key]
            )
            continue

        # If not, pull out all layers for which the key is a substring.
        keys_with_substr = [
            key for key in self.layer_dict_all_keys if layer_key in str(key)
        ]
        if len(keys_with_substr) > 0:
            for key in keys_with_substr:
                raw_tensor_nums_to_save.add(
                    self.layer_dict_all_keys[key].realtime_tensor_num
                )
            continue

        # If no luck, try to at least point user in right direction:

        _give_user_feedback_about_lookup_key(self, layer_key, "query_multiple")

    raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
    return raw_tensor_nums_to_save


def _fetch_label_move_input_tensors(
        input_args: List[Any],
        input_arg_names: List[str],
        input_kwargs: Dict,
        model_device: str,
) -> Tuple[List[torch.Tensor], List[str]]:
    """Fetches input tensors, gets their addresses, and moves them to the model device.

    Args:
        input_args: input arguments
        input_arg_names: name of input arguments
        input_kwargs: input keyword arguments
        model_device: model device

    Returns:
        input tensors and their addresses
    """
    input_arg_tensors = [
        get_vars_of_type_from_obj(
            arg, torch.Tensor, search_depth=5, return_addresses=True
        )
        for arg in input_args
    ]
    input_kwarg_tensors = [
        get_vars_of_type_from_obj(
            kwarg, torch.Tensor, search_depth=5, return_addresses=True
        )
        for kwarg in input_kwargs.values()
    ]
    for a, arg in enumerate(input_args):
        for i, (t, addr, addr_full) in enumerate(input_arg_tensors[a]):
            t_moved = t.to(model_device)
            input_arg_tensors[a][i] = (t_moved, addr, addr_full)
            if not addr_full:
                input_args[a] = t_moved
            else:
                nested_assign(input_args[a], addr_full, t_moved)

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
        self: "ModelHistory",
        model: nn.Module,
        input_args: Union[torch.Tensor, List[Any]],
        input_kwargs: Dict[Any, Any] = None,
        layers_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
        random_seed: Optional[int] = None,
):
    """Runs input through model and logs it in ModelHistory.

    Args:
        model: Model for which to save activations
        input_args: Either a single tensor input to the model, or list of input arguments.
        input_kwargs: Dict of keyword arguments to the model.
        layers_to_save: List of tensor numbers to save
        random_seed: Which random seed to use
    Returns:
        Nothing, but now the ModelHistory object will have saved activations for the new input.
    """
    if random_seed is None:  # set random seed
        random_seed = random.randint(1, 4294967294)
    self.random_seed_used = random_seed
    set_random_seed(random_seed)

    self._tensor_nums_to_save = _get_op_nums_from_user_labels(self, layers_to_save)

    if type(input_args) is tuple:
        input_args = list(input_args)
    elif (type(input_args) not in [list, tuple]) and (input_args is not None):
        input_args = [input_args]

    if not input_args:
        input_args = []

    if not input_kwargs:
        input_kwargs = {}

    if (
            type(model) == nn.DataParallel
    ):  # Unwrap model from DataParallel if relevant:
        model = model.module

    if (
            len(list(model.parameters())) > 0
    ):  # Get the model device by looking at the parameters:
        model_device = next(iter(model.parameters())).device
    else:
        model_device = "cpu"

    input_args = [copy.deepcopy(arg) for arg in input_args]
    input_arg_names = _get_input_arg_names(model, input_args)
    input_kwargs = {key: copy.deepcopy(val) for key, val in input_kwargs.items()}

    self.pass_start_time = time.time()
    module_orig_forward_funcs = {}
    orig_func_defs = []

    try:
        (
            input_tensors,
            input_tensor_addresses,
        ) = _fetch_label_move_input_tensors(
            input_args, input_arg_names, input_kwargs, model_device
        )
        buffer_tensors = list(model.buffers())
        tensors_to_decorate = input_tensors + buffer_tensors
        decorated_func_mapper = self._decorate_pytorch(torch, orig_func_defs)
        self._track_tensors = True
        for i, t in enumerate(input_tensors):
            log_source_tensor(self, t, "input", input_tensor_addresses[i])
        self._prepare_model(model, module_orig_forward_funcs, decorated_func_mapper)
        self.elapsed_time_setup = time.time() - self.pass_start_time
        outputs = model(*input_args, **input_kwargs)
        self.elapsed_time_forward_pass = (
                time.time() - self.pass_start_time - self.elapsed_time_setup
        )
        self._track_tensors = False
        output_tensors_w_addresses_all = get_vars_of_type_from_obj(
            outputs,
            torch.Tensor,
            search_depth=5,
            return_addresses=True,
            allow_repeats=True,
        )
        # Remove duplicate addresses TODO: make this match the validation procedure so they work the same way
        addresses_used = []
        output_tensors_w_addresses = []
        for entry in output_tensors_w_addresses_all:
            if entry[1] in addresses_used:
                continue
            output_tensors_w_addresses.append(entry)
            addresses_used.append(entry[1])

        output_tensors = [t for t, _, _ in output_tensors_w_addresses]
        output_tensor_addresses = [
            addr for _, addr, _ in output_tensors_w_addresses
        ]

        for t in output_tensors:
            self.output_layers.append(t.tl_tensor_label_raw)
            self._raw_tensor_dict[t.tl_tensor_label_raw].is_output_parent = True
        tensors_to_undecorate = tensors_to_decorate + output_tensors
        undecorate_pytorch(torch, orig_func_defs, tensors_to_undecorate)
        self._cleanup_model(model, module_orig_forward_funcs, decorated_func_mapper)
        self._postprocess(output_tensors, output_tensor_addresses)
        decorated_func_mapper.clear()

    except (
            Exception
    ) as e:  # if anything fails, make sure everything gets cleaned up
        undecorate_pytorch(torch, orig_func_defs, input_tensors)
        self._cleanup_model(model, module_orig_forward_funcs, decorated_func_mapper)
        print(
            "************\nFeature extraction failed; returning model and environment to normal\n*************"
        )
        raise e

    finally:  # do garbage collection no matter what
        if 'input_args' in globals():
            del input_args
        if 'input_kwargs' in globals():
            del input_kwargs
        if 'input_tensors' in globals():
            del input_tensors
        if 'output_tensors' in globals():
            del output_tensors
        if 'outputs' in globals():
            del outputs
        torch.cuda.empty_cache()
