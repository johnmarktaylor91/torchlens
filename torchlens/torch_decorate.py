# This module has all functions relating to decorating PyTorch so tensor operations can be logged.

import time
import types
from functools import wraps
from typing import Callable, Dict, List, Tuple

import torch

from torchlens.constants import ORIG_TORCH_FUNCS
from torchlens.helper_funcs import get_vars_of_type_from_obj, identity, log_current_rng_states, make_random_barcode, \
    nested_getattr, print_override, safe_copy
from torchlens.model_history import ModelHistory

print_funcs = ['__repr__', '__str__', '_str']
funcs_not_to_log = ['cpu', 'cuda', 'numpy', '__array__']


def torch_func_decorator(func: Callable,
                         model_history: ModelHistory):
    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Initial bookkeeping; check if it's a special function, organize the arguments.
        model_history.current_function_call_barcode = 0
        func_name = func.__name__
        if (func_name in funcs_not_to_log) or not model_history.track_tensors:
            out = func(*args, **kwargs)
            return out
        all_args = list(args) + list(kwargs.values())
        arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)
        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = print_override(args[0], func_name)
            return out

        # Copy the args and kwargs in case they change in-place:
        arg_copies = tuple([safe_copy(arg) for arg in args])
        kwarg_copies = {k: safe_copy(v) for k, v in kwargs.items()}

        # Call the function, tracking the timing, rng states, and whether it's a nested function
        func_call_barcode = make_random_barcode()
        model_history.current_function_call_barcode = func_call_barcode
        start_time = time.time()
        func_rng_states = log_current_rng_states()
        out_orig = func(*args, **kwargs)
        func_time_elapsed = time.time() - start_time
        is_bottom_level_func = (model_history.current_function_call_barcode == func_call_barcode)

        if func_name in ['__setitem__', 'zero_', '__delitem__']:
            out_orig = args[0]

        # Log all output tensors
        output_tensors = get_vars_of_type_from_obj(out_orig,
                                                   which_type=torch.Tensor,
                                                   subclass_exceptions=[torch.nn.Parameter])

        if len(output_tensors) > 0:
            model_history.log_function_output_tensors(func,
                                                      args,
                                                      kwargs,
                                                      arg_copies,
                                                      kwarg_copies,
                                                      out_orig,
                                                      func_time_elapsed,
                                                      func_rng_states,
                                                      is_bottom_level_func)
        return out_orig

    return wrapped_func


def decorate_pytorch(torch_module: types.ModuleType,
                     tensors_to_mutate: List[torch.Tensor],
                     orig_func_defs: List[Tuple],
                     model_history: ModelHistory) -> Dict[Callable, Callable]:
    """Mutates all PyTorch functions (TEMPORARILY!) to save the outputs of any functions
    that return Tensors, along with marking them with metadata. Returns a list of tuples that
    save the current state of the functions, such that they can be restored when done.

    args:
        torch_module: The top-level torch module (i.e., from "import torch").
            This is supplied as an argument on the off-chance that the user has imported torch
            and done their own monkey-patching.
        tensors_to_mutate: A list of tensors that will be mutated (since any tensors created
            before calling the torch mutation function will not be mutated).
        orig_func_defs: Supply a list from outside to guarantee it can be cleaned up properly.
        tensor_record: A list to which the outputs of the functions will be appended.

    returns:
        List of tuples consisting of [namespace, func_name, orig_func], sufficient
        to return torch to normal when finished, and also a dict mapping mutated functions to original functions.
    """

    # Do a pass to save the original func defs.
    collect_orig_func_defs(torch_module, orig_func_defs)
    decorated_func_mapper = {}

    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        if getattr(orig_func, '__name__', False) == 'wrapped_func':
            continue
        new_func = torch_func_decorator(orig_func, model_history)
        try:
            setattr(local_func_namespace, func_name, new_func)
        except (AttributeError, TypeError) as _:
            pass
        new_func.tl_is_decorated_function = True
        decorated_func_mapper[new_func] = orig_func
        decorated_func_mapper[orig_func] = new_func
        if 'torch.Tensor' in namespace_name:
            decorate_tensors(tensors_to_mutate,
                             namespace_name,
                             func_name,
                             model_history)

    # Bolt on the identity function
    new_identity = torch_func_decorator(identity, model_history)
    torch.identity = new_identity

    return decorated_func_mapper


def decorate_tensors(tensors_to_decorate: List[torch.Tensor],
                     namespace_name: str,
                     func_name: str,
                     model_history: ModelHistory):
    """Decorates a list of tensors such that they can be tracked.

    Args:
        tensors_to_decorate: list of tensors to decorate
        namespace_name: namespace in which the function to decorate is located
        func_name: name of the function to decorate
        model_history: ModelHistory object tracking the tensor operations
    """
    namespace_name_notensor = namespace_name.replace('torch.Tensor.', '').replace('torch.Tensor', '')
    for t in tensors_to_decorate:
        try:
            local_tensor_namespace = nested_getattr(t, namespace_name_notensor)
            orig_tensor_func = getattr(local_tensor_namespace, func_name)
            new_tensor_func = torch_func_decorator(orig_tensor_func, model_history)
            setattr(local_tensor_namespace, func_name, new_tensor_func)
        except (AttributeError, TypeError, RuntimeError) as _:
            pass


def undecorate_pytorch(torch_module,
                       orig_func_defs: List[Tuple]):
    """
    Returns all PyTorch functions back to the definitions they had when mutate_pytorch was called.
    This is done for the output tensors and history_dict too to avoid ugliness.

    args:
        torch_module: The torch module object.
        orig_func_defs: List of tuples consisting of [namespace_name, func_name, orig_func], sufficient
            to regenerate the original functions.
    """
    for namespace_name, func_name, orig_func in orig_func_defs:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        try:
            setattr(local_func_namespace, func_name, orig_func)
        except (AttributeError, TypeError) as _:
            continue
    delattr(torch, 'identity')


def undecorate_tensor(t, device: str = 'cpu'):
    """Convenience function to replace the tensor with an unmutated version of itself, keeping the same data.

    Args:
        t: tensor or parameter object
        device: device to move the tensor to

    Returns:
        Unmutated tensor.
    """
    if type(t) == torch.Tensor:
        new_t = safe_copy(t)
    elif type(t) == torch.nn.Parameter:
        new_t = torch.nn.Parameter(safe_copy(t))
    else:
        new_t = t
    for attr in dir(new_t):
        if attr.startswith('tl_'):
            delattr(new_t, attr)
    new_t = new_t.to(device)
    return new_t


def collect_orig_func_defs(torch_module: types.ModuleType,
                           orig_func_defs: List[Tuple], ):
    """Collects the original torch function definitions, so they can be restored after the logging is done.

    Args:
        torch_module: The top-level torch module
        orig_func_defs: List of tuples keeping track of the original function definitions
    """
    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        orig_func_defs.append((namespace_name, func_name, orig_func))
