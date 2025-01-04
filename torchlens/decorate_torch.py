import inspect
import time
import types
import warnings
from functools import wraps
from typing import Callable, Dict, List, TYPE_CHECKING, Tuple

import torch

from .constants import ORIG_TORCH_FUNCS
from .helper_funcs import (clean_to, get_vars_of_type_from_obj, identity, log_current_rng_states, make_random_barcode,
                           nested_getattr, print_override, safe_copy)
from .logging_funcs import log_function_output_tensors, log_source_tensor

if TYPE_CHECKING:
    from model_history import ModelHistory

funcs_not_to_log = ["numpy", "__array__", "size", "dim"]
print_funcs = ["__repr__", "__str__", "_str"]


def torch_func_decorator(self, func: Callable):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        # Initial bookkeeping; check if it's a special function, organize the arguments.
        self.current_function_call_barcode = 0
        func_name = func.__name__
        if (
                (func_name in funcs_not_to_log)
                or (not self._track_tensors)
                or self._pause_logging
        ):
            out = func(*args, **kwargs)
            return out
        all_args = list(args) + list(kwargs.values())
        arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)

        # Register any buffer tensors in the arguments.

        for t in arg_tensorlike:
            if hasattr(t, 'tl_buffer_address'):
                log_source_tensor(self, t, 'buffer', getattr(t, 'tl_buffer_address'))

        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = print_override(args[0], func_name)
            return out

        # Copy the args and kwargs in case they change in-place:
        if self.save_function_args:
            arg_copies = tuple([safe_copy(arg) for arg in args])
            kwarg_copies = {k: safe_copy(v) for k, v in kwargs.items()}
        else:
            arg_copies = args
            kwarg_copies = kwargs

        # Call the function, tracking the timing, rng states, and whether it's a nested function
        func_call_barcode = make_random_barcode()
        self.current_function_call_barcode = func_call_barcode
        start_time = time.time()
        func_rng_states = log_current_rng_states()
        out_orig = func(*args, **kwargs)
        func_time_elapsed = time.time() - start_time
        is_bottom_level_func = (
                self.current_function_call_barcode == func_call_barcode
        )

        if func_name in ["__setitem__", "zero_", "__delitem__"]:
            out_orig = args[0]

        # Log all output tensors
        output_tensors = get_vars_of_type_from_obj(
            out_orig,
            which_type=torch.Tensor,
            subclass_exceptions=[torch.nn.Parameter],
        )

        if len(output_tensors) > 0:
            log_function_output_tensors(
                self,
                func,
                args,
                kwargs,
                arg_copies,
                kwarg_copies,
                out_orig,
                func_time_elapsed,
                func_rng_states,
                is_bottom_level_func,
            )

        return out_orig

    return wrapped_func


def decorate_pytorch(
        self: "ModelHistory", torch_module: types.ModuleType, orig_func_defs: List[Tuple]
) -> Dict[Callable, Callable]:
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
        namespace_name_notorch = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        if func_name not in self.func_argnames:
            get_func_argnames(self, orig_func, func_name)
        if getattr(orig_func, "__name__", False) == "wrapped_func":
            continue
        new_func = torch_func_decorator(self, orig_func)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, new_func)
        except (AttributeError, TypeError) as _:
            pass
        new_func.tl_is_decorated_function = True
        decorated_func_mapper[new_func] = orig_func
        decorated_func_mapper[orig_func] = new_func

    # Bolt on the identity function
    new_identity = torch_func_decorator(self, identity)
    torch.identity = new_identity

    return decorated_func_mapper


def undecorate_pytorch(
        torch_module, orig_func_defs: List[Tuple], input_tensors: List[torch.Tensor]
):
    """
    Returns all PyTorch functions back to the definitions they had when mutate_pytorch was called.
    This is done for the output tensors and history_dict too to avoid ugliness. Also deletes
    the mutant versions of the functions to remove any references to old ModelHistory object.

    args:
        torch_module: The torch module object.
        orig_func_defs: List of tuples consisting of [namespace_name, func_name, orig_func], sufficient
            to regenerate the original functions.
        input_tensors: List of input tensors whose fucntions will be undecorated.
        decorated_func_mapper: Maps the decorated function to the original function
    """
    for namespace_name, func_name, orig_func in orig_func_defs:
        namespace_name_notorch = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decorated_func = getattr(local_func_namespace, func_name)
        del decorated_func
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, orig_func)
        except (AttributeError, TypeError) as _:
            continue
    delattr(torch, "identity")
    for input_tensor in input_tensors:
        if hasattr(input_tensor, "tl_tensor_label_raw"):
            delattr(input_tensor, "tl_tensor_label_raw")


def undecorate_tensor(t, device: str = "cpu"):
    """Convenience function to replace the tensor with an unmutated version of itself, keeping the same data.

    Args:
        t: tensor or parameter object
        device: device to move the tensor to

    Returns:
        Unmutated tensor.
    """
    if type(t) in [torch.Tensor, torch.nn.Parameter]:
        new_t = safe_copy(t)
    else:
        new_t = t
    del t
    for attr in dir(new_t):
        if attr.startswith("tl_"):
            delattr(new_t, attr)
    new_t = clean_to(new_t, device)
    return new_t


def collect_orig_func_defs(
        torch_module: types.ModuleType, orig_func_defs: List[Tuple]
):
    """Collects the original torch function definitions, so they can be restored after the logging is done.

    Args:
        torch_module: The top-level torch module
        orig_func_defs: List of tuples keeping track of the original function definitions
    """
    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        namespace_name_notorch = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        orig_func_defs.append((namespace_name, func_name, orig_func))


# TODO: hard-code some of the arg names; for example truediv, getitem, etc. Can crawl through and see what isn't working
def get_func_argnames(self, orig_func: Callable, func_name: str):
    """Attempts to get the argument names for a function, first by checking the signature, then
    by checking the documentation. Adds these names to func_argnames if it can find them,
    doesn't do anything if it can't."""
    try:
        argnames = list(inspect.signature(orig_func).parameters.keys())
        argnames = tuple([arg.replace('*', '') for arg in argnames if arg not in ['cls', 'self']])
        self.func_argnames[func_name] = argnames
        return
    except ValueError:
        pass

    docstring = orig_func.__doc__
    if (type(docstring) is not str) or (len(docstring) == 0):  # if docstring missing, skip it
        return

    open_ind, close_ind = docstring.find('('), docstring.find(')')
    argstring = docstring[open_ind + 1: close_ind]
    arg_list = argstring.split(',')
    arg_list = [arg.strip(' ') for arg in arg_list]
    argnames = []
    for arg in arg_list:
        argname = arg.split('=')[0]
        if argname in ['*', '/', '//', '']:
            continue
        argname = argname.replace('*', '')
        argnames.append(argname)
    argnames = tuple([arg for arg in argnames if arg not in ['self', 'cls']])
    self.func_argnames[func_name] = argnames
    return
