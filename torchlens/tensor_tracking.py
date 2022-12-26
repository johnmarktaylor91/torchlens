# TODO: this module will later be the one for handling all
# torch mutation stuff.

import __future__
import collections
import copy
import functools
import inspect
import time
import types
import warnings
from collections import OrderedDict, defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.overrides import get_ignored_functions, get_testing_overrides

from torchlens.helper_funcs import barcode_tensors_in_obj, get_marks_from_tensor_list, get_rng_states, \
    get_tensor_memory_amount, get_tensors_in_obj_with_mark, get_vars_of_type_from_obj, make_barcode, \
    mark_tensors_in_obj, tensor_in_obj_has_mark

clean_from_numpy = copy.deepcopy(torch.from_numpy)
clean_to_numpy = copy.deepcopy(torch.Tensor.__array__)
clean_clone = copy.deepcopy(torch.clone)
clean_new_tensor = copy.deepcopy(torch.tensor)

# Taken from https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions
# This branch will be for refactoring everything nicely.

print_funcs = ['__repr__', '__str__']
funcs_not_to_log = ['cpu', 'cuda', 'numpy', 'to']
ignored_funcs = [
    ('torch', 'load'),
    ('torch', 'as_tensor'),
    ('torch', 'from_numpy'),
    ('torch', 'tensor'),
    ('torch', 'arange'),
    ('torch', 'as_strided'),
    ('torch', 'bartlett_window'),
    ('torch', 'blackman_window'),
    ('torch', 'cudnn_affine_grid_generator'),
    ('torch', 'cudnn_batch_norm'),
    ('torch', 'cudnn_convolution'),
    ('torch', 'cudnn_convolution_transpose'),
    ('torch', 'cudnn_convolution_relu'),
    ('torch', 'cudnn_convolution_add_relu'),
    ('torch', 'cudnn_grid_sampler'),
    ('torch', 'cudnn_is_acceptable'),
    ('torch', 'eye'),
    ('torch.fft', 'fftfreq'),
    ('torch.fft', 'rfftfreq'),
    ('torch', 'from_file'),
    ('torch', 'full'),
    ('torch', 'fill_'),
    ('torch', 'hamming_window'),
    ('torch', 'hann_window'),
    ('torch', 'kaiser_window'),
    ('torch', 'linspace'),
    ('torch', 'logspace'),
    ('torch', 'mkldnn_adaptive_avg_pool2d'),
    ('torch', 'mkldnn_convolution'),
    ('torch', 'mkldnn_max_pool2d'),
    ('torch', 'mkldnn_max_pool3d'),
    ('torch', 'mkldnn_linear_backward_weights'),
    ('torch', 'normal'),
    ('torch', 'ones'),
    ('torch', 'rand'),
    ('torch', 'randn'),
    ('torch', 'randint'),
    ('torch', 'randperm'),
    ('torch', 'range'),
    ('torch', 'scalar_tensor'),
    ('torch', 'sparse_coo_tensor'),
    ('torch', '_sparse_csr_tensor'),
    ('torch', 'tril_indices'),
    ('torch', 'triu_indices'),
    ('torch', 'vander'),
    ('torch', 'zeros'),
    ('torch.nn.functional', 'upsample'),
    ('torch.nn.functional', 'upsample_bilinear'),
    ('torch.nn.functional', 'upsample_nearest'),
    ('torch.nn.functional', 'handle_torch_function'),
    ('torch.nn.functional', 'sigmoid'),
    ('torch.nn.functional', 'hardsigmoid'),
    ('torch.nn.functional', 'tanh'),
    ('torch.nn.init', 'calculate_gain'),
    ('torch.nn.init', 'uniform'),
    ('torch.nn.init', 'normal'),
    ('torch.nn.init', 'constant'),
    ('torch.nn.init', 'eye'),
    ('torch.nn.init', 'dirac'),
    ('torch.nn.init', 'xavier_uniform'),
    ('torch.nn.init', 'xavier_normal'),
    ('torch.nn.init', 'kaiming_uniform'),
    ('torch.nn.init', 'kaiming_normal'),
    ('torch.nn.init', 'orthogonal'),
    ('torch.nn.init', 'sparse'),
    ('torch.nn.functional', 'hardswish'),
    ('torch.Tensor', '__delitem__'),
    ('torch.Tensor', '__iter__'),
    ('torch.Tensor', '__init_subclass__'),
    ('torch.Tensor', '__torch_function__'),
    ('torch.Tensor', '__new__'),
    ('torch.Tensor', '__subclasshook__'),
    ('torch.Tensor', 'as_subclass'),
    ('torch.Tensor', 'reinforce'),
    ('torch.Tensor', 'new'),
    ('torch.Tensor', 'new_tensor'),
    ('torch.Tensor', 'new_empty'),
    ('torch.Tensor', 'new_empty_strided'),
    ('torch.Tensor', 'new_zeros'),
    ('torch.Tensor', 'new_ones'),
    ('torch.Tensor', 'new_full'),
    ('torch.Tensor', '_make_subclass'),
    ('torch.Tensor', 'solve'),
    ('torch.Tensor', 'unflatten')]


@functools.lru_cache(None)
def my_get_overridable_functions() -> Tuple[Dict[Any, List[Callable]], Dict[Callable, str]]:
    overridable_funcs = collections.defaultdict(list)
    index = {}
    func_names = []
    tested_namespaces = [
        ("torch", torch, torch.__all__ + dir(torch._C._VariableFunctions)),
        ("torch.functional", torch.functional, torch.functional.__all__),
        ("torch.nn.functional", torch.nn.functional, dir(torch.nn.functional)),
        ("torch.nn.init", torch.nn.init, dir(torch.nn.init)),
        ("torch.Tensor", torch.Tensor, dir(torch.Tensor)),
        ("torch.linalg", torch.linalg, dir(torch.linalg)),
        ("torch.fft", torch.fft, dir(torch.fft))
    ]
    if hasattr(torch, 'special'):
        tested_namespaces.append(("torch.special", torch.special, dir(torch.special)))
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            ignore = False
            # ignore private functions or functions that are deleted in torch.__init__
            if namespace is not torch.Tensor:
                if func_name.startswith('__'):
                    continue
                elif func_name.startswith('_'):
                    ignore = True
                elif func_name.endswith('_'):
                    ignore = True
                elif not func_name[0].islower():
                    ignore = True
                elif func_name == 'unique_dim':
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == '__weakref__':
                    continue
            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, __future__._Feature):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                index[func.__get__] = f"{namespace_str}.{func_name}.__get__"
                index[func.__set__] = f"{namespace_str}.{func_name}.__set__"
                if ignore:
                    continue
                if func.__get__ in get_ignored_functions():
                    msg = ("{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                           "but still has an explicit override")
                    assert func.__get__ not in get_testing_overrides(), msg.format(namespace, func.__name__)
                    continue
                else:
                    overridable_funcs[func].append(func.__get__)
                    func_names.append((f"{namespace_str}.{func_name}", "__get__"))
                    continue

            if not callable(func):
                continue

            index[func] = f"{namespace_str}.{func_name}"

            if ignore:
                continue

            # cannot be overriden by __torch_function__
            if func in get_ignored_functions():
                msg = ("{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                       "but still has an explicit override")
                assert func not in get_testing_overrides(), msg.format(namespace, func.__name__)
                continue
            overridable_funcs[namespace].append(func)
            func_names.append((f"{namespace_str}", func_name))
    return func_names


overridable_funcs = my_get_overridable_functions()
orig_torch_funcs = overridable_funcs + ignored_funcs


def print_override(t: torch.Tensor, func_name: str):
    """Overrides the __str__ and __repr__ methods of Tensor so as not to lead to any infinite recursion.

    Args:
        t: Tensor
        func_name: Either "__str__" or "__repr__"

    Returns:
        The string representation of the tensor.
    """
    n = np.array(t.data.cpu())
    np_str = getattr(n, func_name)()
    np_str = np_str.replace('array', 'tensor')
    np_str = np_str.replace('\n', '\n ')
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return np_str


def safe_copy(x):
    """Utility function to make a copy of a tensor or parameter when torch is in mutated mode, or just copy
    the thing if it's not a tensor.

    Args:
        x: Input

    Returns:
        Safely copied variant of the input with same values and same class, but different memory
    """
    if issubclass(type(x), (torch.Tensor, torch.nn.Parameter)):
        vals = clean_clone(x)
        if type(x) == torch.Tensor:
            return vals
        elif type(x) == torch.nn.Parameter:
            return torch.nn.Parameter(vals)
    else:
        return copy.copy(x)


def nested_getattr(obj: Any, attr: str) -> Any:
    """Helper function that takes in an object, and a string of attributes separated by '.' and recursively
    returns the attribute.

    Args:
        obj: Any object, e.g. "torch"
        attr: String specifying the nested attribute, e.g. "nn.functional"

    Returns:
        The attribute specified by the string.
    """
    if attr == '':
        return obj

    attributes = attr.split(".")
    for i, a in enumerate(attributes):
        if a in ['volatile', 'T']:  # avoid annoying warning; if there's more, make a list
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if i == 0:
                    out = getattr(obj, a)
                else:
                    out = getattr(out, a)
        else:
            if i == 0:
                out = getattr(obj, a)
            else:
                out = getattr(out, a)
    return out


def mutate_pytorch(torch_module: types.ModuleType,
                   tensors_to_mutate: List[torch.Tensor],  # TODO check if this is necessary or not.
                   orig_func_defs: List[Tuple],
                   history_log: Dict) -> Tuple[List[Tuple], Dict]:
    """Mutates all PyTorch functions (TEMPORARILY!) so as to save the outputs of any functions
    that return Tensors, along with marking them with metadata. Returns a list of tuples that
    save the current state of the functions, such that they can be restored when done.

    args:
        torch_module: The top-level torch module (i.e., from "import torch").
            This is supplied as an argument on the off-chance that the user has imported torch
            and done their own monkey-patching.
        tensors_to_mutate: A list of tensors that will be mutated (since any tensors created
            before calling the torch mutation function will not be mutated).
        orig_func_defs: Supply a list from outside so as to guarantee it can be cleaned up properly.
        tensor_record: A list to which the outputs of the functions will be appended.

    returns:
        List of tuples consisting of [namespace, func_name, orig_func], sufficient
        to return torch to normal when finished, and also a dict mapping mutated functions to original functions.
    """

    # Do a pass to save the original func defs.
    mutant_to_orig_funcs_dict = {}

    for namespace_name, func_name in orig_torch_funcs:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        # try:
        #    try:
        #        orig_func = copy.deepcopy(getattr(local_func_namespace, func_name))
        #    except:
        #        orig_func = copy.copy(getattr(local_func_namespace, func_name))
        # except:
        #    orig_func = getattr(local_func_namespace, func_name)
        orig_func_defs.append((namespace_name, func_name, orig_func))

    for namespace_name, func_name in orig_torch_funcs:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        if hasattr(orig_func, '__name__') and orig_func.__name__ == 'wrapped_func':
            continue
        new_func = torch_func_decorator(orig_func, history_log)
        new_func.tl_is_mutant_function = True
        mutant_to_orig_funcs_dict[new_func] = orig_func
        try:
            setattr(local_func_namespace, func_name, new_func)
        except (AttributeError, TypeError) as _:
            pass
        if 'torch.Tensor' in namespace_name:
            namespace_name_notensor = namespace_name.replace('torch.Tensor.', '').replace('torch.Tensor', '')
            for t in tensors_to_mutate:
                try:
                    local_tensor_namespace = nested_getattr(t, namespace_name_notensor)
                    orig_tensor_func = getattr(local_tensor_namespace, func_name)
                    new_tensor_func = torch_func_decorator(orig_tensor_func, history_log)
                    setattr(local_tensor_namespace, func_name, new_tensor_func)
                except (AttributeError, TypeError, RuntimeError) as _:
                    pass
    return orig_func_defs, mutant_to_orig_funcs_dict


def unmutate_pytorch(torch_module,
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


def update_tensor_containing_modules(t: torch.Tensor) -> List[str]:
    """Utility function that updates the containing modules of a Tensor by starting from the containing modules
    as of the last function call, then looks at the sequence of module transitions (in or out of a module) as of
    the last module it saw, and updates accordingly.

    Args:
        t: Tensor to check.

    Returns:
        List of the updated containing modules.
    """
    containing_modules = t.tl_function_call_modules_nested[:]
    thread_modules = t.tl_containing_modules_thread[:]
    for thread_module in thread_modules:
        if thread_module[0] == '+':
            containing_modules.append(thread_module[1:])
        elif thread_module[0] == '-':
            if thread_module[1:] in containing_modules:
                containing_modules.remove(thread_module[1:])
    return containing_modules


def get_parent_tensor_function_call_location(parent_tensors: List[torch.Tensor],
                                             args: Tuple[Any],
                                             kwargs: Dict[Any, Any]) -> Dict:
    """Utility function that takes in the parent tensors, the args, and kwargs, and returns a list of tuples
    specifying where in the args or kwargs the parent tensors enter in.

    Args:
        parent_tensors: List of parent tensors.
        args: Tuple of function args
        kwargs: Dict of function kwargs

    Returns:
        Dict that itself contains two dicts, one specifing which args are associated with parent tensors,
        and another specifing which kwargs are associated with parent tensors.
    """
    tensor_arg_positions = {}
    tensor_kwarg_positions = {}

    for parent_tensor in parent_tensors:
        if not hasattr(parent_tensor, 'tl_barcode'):
            continue
        for arg_position, arg in enumerate(args):
            if arg is parent_tensor:
                tensor_arg_positions[arg_position] = parent_tensor.tl_barcode
            elif type(arg) in [list, tuple]:
                for sub_arg_position, sub_arg in enumerate(arg):
                    if sub_arg is parent_tensor:
                        tensor_arg_positions[(arg_position, sub_arg_position)] = parent_tensor.tl_barcode
            elif type(arg) == dict:
                for key, value in arg.items():
                    if value is parent_tensor:
                        tensor_arg_positions[(arg_position, key)] = parent_tensor.tl_barcode
        for kwarg_key, kwarg_value in kwargs.items():
            if kwarg_value is parent_tensor:
                tensor_kwarg_positions[kwarg_key] = parent_tensor.tl_barcode
            elif type(kwarg_value) in [list, tuple]:
                for sub_arg_position, sub_arg in enumerate(kwarg_value):
                    if sub_arg is parent_tensor:
                        tensor_arg_positions[(kwarg_key, sub_arg_position)] = parent_tensor.tl_barcode
            elif type(kwarg_value) == dict:
                for key, value in kwarg_value.items():
                    if value is parent_tensor:
                        tensor_arg_positions[(kwarg_key, key)] = parent_tensor.tl_barcode
    tensor_all_arg_positions = {'args': tensor_arg_positions, 'kwargs': tensor_kwarg_positions}
    return tensor_all_arg_positions


def make_output_iterable(x):
    """Utility function to facilitate dealing with outputs:
    - If not a list, tuple, or dict, make it a list of length 1
    - If a dict, make it a list of the values
    - If a list or tuple, keep it.

    Args:
        x: Output of the function

    Returns:
        Iterable output
    """
    if type(x) in [tuple, list, set]:
        return x
    if issubclass(type(x), dict):
        return list(x.values())
    else:
        return [x]


def torch_func_decorator(func: Callable,
                         model_history: ModelHistory):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        tensor_log = model_history['tensor_log']
        model_history['current_function_call_barcode'] = 0
        func_name = func.__name__
        if (func_name in funcs_not_to_log) or not model_history['track_tensors']:
            out = func(*args, **kwargs)
            return out

        if (func_name in print_funcs) and (type(args[0]) in [torch.Tensor, torch.nn.parameter.Parameter]):
            out = print_override(args[0], func_name)
            return out
        all_args = list(args) + list(kwargs.values())
        non_tensor_args = [arg for arg in args if not issubclass(type(arg), torch.Tensor)]
        non_tensor_kwargs = {key: val for key, val in kwargs.items() if not issubclass(type(val), torch.Tensor)}
        arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.parameter.Parameter])
        parent_tensor_barcodes = get_marks_from_tensor_list(arg_tensors, 'tl_barcode')
        if len(parent_tensor_barcodes) > 0:
            has_parents = True
        else:
            has_parents = False

        # Figure out where the parent tensors are used in the function call args and kwargs.
        parent_tensor_arg_locations = get_parent_tensor_function_call_location(arg_tensors, args, kwargs)

        arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
        any_inputs_entered_module = tensor_in_obj_has_mark(arg_tensors, 'tl_entered_module', True)
        if any_inputs_entered_module:  # TODO make this a function
            # Use the tensor with the most nested containing module (in case of internally generated ones).
            max_input_module_nesting = 0
            most_nested_arg_tensor_index = None
            most_nested_containing_modules = []
            for i, t in enumerate(arg_tensors):
                if not hasattr(t, 'tl_containing_modules_nested'):
                    continue
                containing_modules = update_tensor_containing_modules(t)
                if len(containing_modules) > max_input_module_nesting:
                    max_input_module_nesting = len(containing_modules)
                    most_nested_arg_tensor_index = i
                    most_nested_containing_modules = containing_modules[:]
            reference_arg_tensor = arg_tensors[most_nested_arg_tensor_index]
            containing_module = reference_arg_tensor.tl_containing_module
            last_module_seen = reference_arg_tensor.tl_last_module_seen
            last_module_seen_address = reference_arg_tensor.tl_last_module_seen_address
            last_module_seen_entry_barcode = reference_arg_tensor.tl_last_module_seen_entry_barcode
            containing_modules = most_nested_containing_modules

        else:
            containing_modules = []
            containing_module = None
            last_module_seen = None
            last_module_seen_address = None
            last_module_seen_entry_barcode = None

        # Mark any parameters.

        parent_param_passes = {}
        total_params_size = 0
        for param in arg_parameters:
            if not hasattr(param, 'tl_param_barcode'):
                param_barcode = make_barcode()
                param.tl_param_barcode = param_barcode
                param.tl_pass_num = 1
            else:
                param_barcode = param.tl_param_barcode
                param.tl_pass_num += 1
            parent_param_passes[param_barcode] = param.tl_pass_num
            total_params_size += get_tensor_memory_amount(param.data)
        param_group_barcode = '_'.join(param.tl_param_barcode for param in arg_parameters)

        came_from_input = tensor_in_obj_has_mark(arg_tensors, 'tl_has_input_ancestor', True)
        if came_from_input:
            has_input_ancestor = True
        else:
            has_input_ancestor = False

        parent_internal_tensors = get_tensors_in_obj_with_mark(arg_tensors, 'tl_has_internal_ancestor', True)
        parent_internal_tensor_barcodes = get_marks_from_tensor_list(parent_internal_tensors, 'tl_barcode')

        internal_ancestors = set([])
        for internal_parent in parent_internal_tensors:
            internal_ancestors = internal_ancestors.union(internal_parent.tl_internal_ancestors)
        if len(internal_ancestors) > 0:
            has_internal_ancestor = True
        else:
            has_internal_ancestor = False

        arg_copies = [safe_copy(arg) for arg in args]
        kwarg_copies = {key: safe_copy(value) for key, value in kwargs.items()}
        func_call_barcode = make_barcode()
        model_history['current_function_call_barcode'] = func_call_barcode
        start_time = time.time()
        rng_states = get_rng_states()
        out_orig = func(*args, **kwargs)
        time_elapsed = time.time() - start_time
        if model_history['current_function_call_barcode'] == func_call_barcode:
            is_bottom_level_func = True
        else:
            is_bottom_level_func = False

        # TODO: Come up with more general logic for dealing with in-place functions:
        if func_name in ['__setitem__', 'zero_', '__delitem__']:
            out_orig = args[0]

        # Check if single boolean tensor:

        if (type(out_orig) == torch.Tensor) and (out_orig.dtype == torch.bool) and (out_orig.dim()) == 0:
            output_is_single_bool = True
            output_bool_val = out_orig.item()
        else:
            output_is_single_bool = False
            output_bool_val = None

        out_iter = make_output_iterable(out_orig)  # so we can iterate through it

        # Initial processing

        # Get func info

        # Get argument info

        # Get output info

        # Get module info

        for i, out in enumerate(out_iter):
            if type(out) == torch.Tensor and (not hasattr(out, 'tl_barcode') or
                                              out.grad_fn not in out.tl_gradfuncs or
                                              is_bottom_level_func):
                model_history.log_function_output_tensor(out, func, func_name, func_changes_input, func_time_elapsed,
                                                         func_rng_states, args, kwargs,
                                                         parent_tensor_barcodes, parent_tensor_arg_locs,
                                                         nontensor_args, nontensor_kwargs, parent_params,
                                                         parent_param_passes, input_ancestors,
                                                         internally_initialized_parents,
                                                         internally_initialized_ancestors, orig_ancestors,
                                                         is_part_of_iterable_output, iterable_output_index,
                                                         containing_modules_origin_nested,
                                                         containing_modules_final_nested,
                                                         module_passes_entered, module_passes_exited,
                                                         module_entry_exit_thread)
                model_history.log_function_output_tensor_func_info()
                model_history.log_function_output_tensor_param_info()
                model_history.log_function_output_tensor_graph_info()
                model_history.log_function_output_tensor_module_info()
                model_history.make_tensor_log_entry(out, t_args=[], t_kwargs={})

        return out_orig

    return wrapped_func
