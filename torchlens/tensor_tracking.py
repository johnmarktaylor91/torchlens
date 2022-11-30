import __future__
import collections
import copy
import functools
import inspect
import time
import types
from collections import OrderedDict, defaultdict
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

import numpy as np
import torch
from torch.overrides import get_ignored_functions, get_testing_overrides

from torchlens.helper_funcs import barcode_tensors_in_obj, get_marks_from_tensor_list, get_rng_states, \
    get_tensor_memory_amount, get_tensors_in_obj_with_mark, get_vars_of_type_from_obj, make_barcode, \
    mark_tensors_in_obj, tensor_in_obj_has_mark

clean_from_numpy = copy.deepcopy(torch.from_numpy)
clean_to_numpy = copy.deepcopy(torch.Tensor.__array__)
clean_clone = copy.deepcopy(torch.clone)

# Taken from https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions

print_funcs = ['__repr__', '__str__']
funcs_not_to_log = ['cpu', 'cuda', 'numpy', 'from_numpy', 'to']
ignored_funcs = [
    ('torch', 'load'),
    ('torch', 'parse_ir'),
    ('torch', 'parse_schema'),
    ('torch', 'parse_type_comment'),
    ('torch', 'set_anomaly_enabled'),
    ('torch', 'set_flush_denormal'),
    ('torch', 'set_num_interop_threads'),
    ('torch', 'set_num_threads'),
    ('torch', 'wait'),
    ('torch', 'as_tensor'),
    ('torch', 'from_numpy'),
    ('torch', 'get_device'),
    ('torch', 'tensor'),
    ('torch', 'default_generator'),
    ('torch', 'dtype'),
    ('torch', 'finfo'),
    ('torch', 'iinfo'),
    ('torch', 'memory_format'),
    ('torch', 'qscheme'),
    ('torch', 'set_grad_enabled'),
    ('torch', 'no_grad'),
    ('torch', 'enable_grad'),
    ('torch', 'inference_mode'),
    ('torch', 'is_inference_mode_enabled'),
    ('torch', 'layout'),
    ('torch', 'align_tensors'),
    ('torch', 'arange'),
    ('torch', 'as_strided'),
    ('torch', 'bartlett_window'),
    ('torch', 'blackman_window'),
    ('torch', 'broadcast_shapes'),
    ('torch', 'can_cast'),
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
    ('torch', 'promote_types'),
    ('torch', 'rand'),
    ('torch', 'randn'),
    ('torch', 'randint'),
    ('torch', 'randperm'),
    ('torch', 'range'),
    ('torch', 'result_type'),
    ('torch', 'scalar_tensor'),
    ('torch', 'sparse_coo_tensor'),
    ('torch', '_sparse_csr_tensor'),
    ('torch', 'tril_indices'),
    ('torch', 'triu_indices'),
    ('torch', 'vander'),
    ('torch', 'zeros'),
    ('torch._jit_internal', 'boolean_dispatch'),
    ('torch.nn.functional', 'assert_int_or_pair'),
    ('torch.nn.functional', 'upsample'),
    ('torch.nn.functional', 'upsample_bilinear'),
    ('torch.nn.functional', 'upsample_nearest'),
    ('torch.nn.functional', 'has_torch_function'),
    ('torch.nn.functional', 'has_torch_function_unary'),
    ('torch.nn.functional', 'has_torch_function_variadic'),
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
    ('torch', 'is_vulkan_available'),
    ('torch', 'unify_type_list'),
    ('torch', 'is_warn_always_enabled'),
    ('torch', 'set_warn_always'),
    ('torch', 'vitals_enabled'),
    ('torch', 'set_vital'),
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
    ('torch.Tensor', 'stride'),
    ('torch.Tensor', 'unflatten'),
    ('torch.Tensor', '_reduce_ex_internal')]


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
        ("torch.fft", torch.fft, dir(torch.fft)),
        ("torch.special", torch.special, dir(torch.special)),
    ]
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
        vals = clean_clone(x)  # torch.from_numpy(x.data.cpu().numpy().copy())
        if type(x) == torch.Tensor:
            return vals
        elif type(x) == torch.nn.Parameter:
            return torch.nn.Parameter(vals)
    else:
        return copy.copy(x)


def get_enclosing_frame():
    """Gets the frame of the caller of the caller of this function.

    Returns:
        The frame of the caller of the caller of this function.
    """
    return inspect.currentframe().f_back.f_back


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
        if a == 'volatile':  # avoid annoying warning; if there's more, make a list
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
    

def torch_func_decorator(func, history_dict):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        tensor_log = history_dict['tensor_log']
        func_name = func.__name__
        if (func_name in funcs_not_to_log) or not history_dict['track_tensors']:
            out = func(*args, **kwargs)
            return out

        if (func_name in print_funcs) and (type(args[0]) in [torch.Tensor, torch.nn.parameter.Parameter]):
            out = print_override(args[0], func_name)
            return out
        all_args = list(args) + list(kwargs.values())
        non_tensor_args = [arg for arg in args if not issubclass(type(arg), torch.Tensor)]
        non_tensor_kwargs = {key: val for key, val in kwargs.items() if not issubclass(type(val), torch.Tensor)}
        arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.parameter.Parameter])
        if len(arg_tensors) > 0:
            has_parents = True
        else:
            has_parents = False
        parent_tensor_barcodes = get_marks_from_tensor_list(arg_tensors, 'tl_barcode')

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

        internal_ancestors = []
        for internal_parent in parent_internal_tensors:
            internal_ancestors.extend(internal_parent.tl_internal_ancestors)
        if len(internal_ancestors) > 0:
            has_internal_ancestor = True
        else:
            has_internal_ancestor = False

        arg_copies = [safe_copy(arg) for arg in args]
        kwarg_copies = {key: safe_copy(value) for key, value in kwargs.items()}
        start_time = time.time()
        rng_states = get_rng_states()
        out_orig = func(*args, **kwargs)
        time_elapsed = time.time() - start_time

        out_iter = make_output_iterable(out_orig)  # so we can iterate through it
        for i, out in enumerate(out_iter):
            if type(out) == torch.Tensor:
                out.tl_history_dict = history_dict
                # If a new tensor, or has a new grad_fn, mark it with everything.
                if not hasattr(out, 'tl_barcode') or (out.grad_fn not in out.tl_gradfuncs):
                    out.tl_barcode = make_barcode()

                    # Update history_dict

                    history_dict['tensor_counter'] += 1
                    if not has_parents:
                        history_dict['internally_generated_tensors'].append(out.tl_barcode)
                    if len(arg_parameters) > 0:
                        history_dict['param_group_tensors'][param_group_barcode].append(out.tl_barcode)

                    # Update tensor_log
                    out.tl_tensor_num = history_dict['tensor_counter']
                    out.tl_tensor_shape = tuple(out.shape)
                    out.tl_tensor_dtype = out.dtype
                    out.tl_tensor_fsize = get_tensor_memory_amount(out)
                    out.tl_has_input_ancestor = has_input_ancestor
                    out.tl_is_model_output = False
                    out.tl_is_model_input = False
                    out.tl_parent_tensor_barcodes = parent_tensor_barcodes
                    out.tl_has_parents = has_parents
                    if (not has_parents) and (not out.tl_is_model_input):
                        out.tl_is_internally_generated = True
                        out.tl_has_internal_ancestor = True
                        out.tl_internal_ancestors = [out.tl_barcode]
                    else:
                        out.tl_is_internally_generated = False
                        out.tl_has_internal_ancestor = has_internal_ancestor
                        out.tl_internal_ancestors = internal_ancestors
                    out.tl_parent_internal_tensor_barcodes = parent_internal_tensor_barcodes
                    out.tl_funcs_applied = [func]
                    out.tl_funcs_applied_names = [func_name]
                    out.tl_parent_tensor_arg_locs = parent_tensor_arg_locations
                    if type(out_orig) in [list, tuple]:  # in case the function returned a tuple/list of tensors
                        out.tl_out_index = i
                    else:
                        out.tl_out_index = None
                    out.tl_func_time_elapsed = time_elapsed
                    out.tl_func_rng_states = rng_states
                    out.tl_gradfuncs = [out.grad_fn]
                    out.tl_gradfuncs_names = [type(out.grad_fn).__name__]
                    out.tl_parent_params = arg_parameters[:]
                    out.tl_parent_param_barcodes = [param.tl_param_barcode for param in arg_parameters]
                    out.tl_parent_params_shape = [tuple(param.shape) for param in arg_parameters]
                    out.tl_parent_param_passes = parent_param_passes.copy()
                    out.tl_params_memory_size = total_params_size
                    if len(arg_parameters) == 0:
                        out.tl_has_params = False
                        out.tl_pass_num = 1
                        out.tl_layer_barcode = out.tl_barcode
                    else:
                        out.tl_has_params = True
                        out.tl_pass_num = len(history_dict['param_group_tensors'][param_group_barcode])
                        out.tl_layer_barcode = param_group_barcode
                    out.tl_nontensor_args = non_tensor_args
                    out.tl_nontensor_kwargs = non_tensor_kwargs
                    out.tl_nontensor_all_args = non_tensor_args + list(non_tensor_kwargs.values())
                    out.tl_num_args = len(args)
                    out.tl_num_kwargs = len(kwargs)

                    # Module stuff.

                    out.tl_entered_module = any_inputs_entered_module
                    out.tl_containing_modules_nested = containing_modules[:]
                    out.tl_function_call_modules_nested = containing_modules[:]
                    out.tl_function_call_modules_nested_multfuncs = [containing_modules[:]]
                    out.tl_containing_modules_thread = []
                    out.tl_containing_module = containing_module
                    out.tl_last_module_seen = last_module_seen
                    out.tl_module_just_entered_address = None
                    out.tl_funcs_applied_modules = [last_module_seen_address]
                    out.tl_last_module_seen_address = last_module_seen_address
                    out.tl_last_module_seen_entry_barcode = last_module_seen_entry_barcode
                    out.tl_is_module_output = False
                    out.tl_modules_exited = []
                    out.tl_module_passes_exited = []
                    out.tl_is_bottom_level_module_output = False
                    out.tl_bottom_module_barcode = None

                else:  # This means that the function returned the same tensor (e.g., identity); just need to mark that.
                    out.tl_funcs_applied.append(func)
                    out.tl_funcs_applied_names.append(func_name)
                    out.tl_func_rng_states = rng_states
                    out.tl_parent_tensor_arg_locs = parent_tensor_arg_locations
                    out.tl_funcs_applied_modules.append(last_module_seen_address)
                    out.tl_function_call_modules_nested_multfuncs.append(out.tl_function_call_modules_nested[:])

                log_tensor_metadata(out)
                log_tensor_data(out, arg_copies, kwarg_copies, arg_parameters)

        return out_orig

    return wrapped_func


def initialize_history_dict(tensor_nums_to_save: Union[List[int], str],
                            tensor_nums_to_save_temporarily: List) -> Dict:
    """Convenience function for making the history dict. This will contain the tensor counter, which tensors to
    save, and the log of the tensors.

    Returns:
        Initialized history dict.
    """
    if tensor_nums_to_save_temporarily is None:
        tensor_nums_to_save_temporarily = []

    history_dict = OrderedDict()
    history_dict['tensor_counter'] = 0
    history_dict['tensor_nums_to_save'] = tensor_nums_to_save
    history_dict['tensor_nums_to_save_temporarily'] = tensor_nums_to_save_temporarily
    history_dict['input_tensors'] = []
    history_dict['internally_generated_tensors'] = []
    history_dict['output_tensors'] = []
    history_dict['param_group_tensors'] = defaultdict(list)  # tensors output by each param group
    history_dict['module_output_tensors'] = defaultdict(list)
    history_dict['module_dict'] = OrderedDict()
    history_dict['bottom_level_module_output_tensors'] = defaultdict(list)
    history_dict['tensor_log'] = defaultdict(lambda: OrderedDict())  # TODO replace with above function when it's done.
    return history_dict


def prepare_input_tensors(x: Any,
                          history_dict: Dict):
    """Prepares input tensors before feeding them into the network.

    Args:
        x: Network input.
        history_dict: The history dict object.

    Returns:
        None; tensors changed in place.
    """
    barcode_tensors_in_obj(x)
    mark_tensors_in_obj(x, 'has_input_ancestor', True)
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    for t in input_tensors:
        history_dict['tensor_counter'] += 1
        t.tl_history_dict = history_dict
        t.tl_barcode = make_barcode()
        t.tl_tensor_num = history_dict['tensor_counter']
        t.tl_tensor_shape = tuple(t.shape)
        t.tl_tensor_dtype = t.dtype
        t.tl_tensor_fsize = get_tensor_memory_amount(t)
        t.tl_has_input_ancestor = True
        t.tl_is_internally_generated = False
        t.tl_has_internal_ancestor = False
        t.tl_is_model_input = True
        t.tl_is_model_output = False
        t.tl_module_just_entered_address = None
        t.tl_parent_tensor_barcodes = []
        t.tl_has_parents = False
        t.tl_parent_internal_tensor_barcodes = []
        t.tl_internal_ancestors = []
        t.tl_funcs_applied = []
        t.tl_funcs_applied_names = []
        t.tl_funcs_applied_modules = []
        t.tl_func_rng_states = get_rng_states()
        t.tl_parent_tensor_arg_locs = {'args': {}, 'kwargs': {}}
        t.tl_gradfuncs = []
        t.tl_gradfuncs_names = []
        t.tl_num_args = 0
        t.tl_num_kwargs = 0
        t.tl_nontensor_args = []
        t.tl_nontensor_kwargs = {}
        t.tl_parent_params = []
        t.tl_parent_param_barcodes = []
        t.tl_parent_params_shape = []
        t.tl_parent_param_passes = {}
        t.tl_params_memory_size = 0
        t.tl_has_params = False
        t.tl_pass_num = 1
        t.tl_layer_barcode = t.tl_barcode

        # Module stuff.

        t.tl_entered_module = False
        t.tl_containing_modules_nested = []
        t.tl_containing_modules_thread = []
        t.tl_function_call_modules_nested = []
        t.tl_function_call_modules_nested_multfuncs = []
        t.tl_func_time_elapsed = 0
        t.tl_containing_module = None
        t.tl_containing_modules_nested = []
        t.tl_last_module_seen = None
        t.tl_last_module_seen_address = None
        t.tl_last_module_seen_entry_barcode = None
        t.tl_is_module_output = False
        t.tl_modules_exited = []
        t.tl_module_passes_exited = []
        t.tl_is_bottom_level_module_output = False
        t.tl_bottom_module_barcode = None
        t.tl_linked_bottom_module = None

        history_dict['input_tensors'].append(t.tl_barcode)
        log_tensor_metadata(t)
        log_tensor_data(t, [], {}, [])


def log_tensor_metadata(t: torch.Tensor):
    """Given a tensor, logs its metadata to the history_dict.

    Args:
        t: tensor to log
    """
    history_dict = t.tl_history_dict
    tensor_barcode = t.tl_barcode

    # Save any relevant fields.

    for field in dir(t):
        if not field.startswith('tl_'):  # tl is the keyword for marking relevant fields.
            continue
        field_stripped = field.removeprefix('tl_')
        history_dict['tensor_log'][tensor_barcode][field_stripped] = getattr(t, field)

    if t.tl_is_module_output:
        module_just_exited = t.tl_modules_exited[-1]
        if tensor_barcode not in history_dict['module_output_tensors'][module_just_exited]:
            history_dict['module_output_tensors'][module_just_exited].append(tensor_barcode)

        if all([t.tl_is_bottom_level_module_output,
                tensor_barcode not in history_dict['bottom_level_module_output_tensors'][module_just_exited],
                module_just_exited == t.tl_bottom_module_barcode]):
            history_dict['bottom_level_module_output_tensors'][module_just_exited].append(tensor_barcode)


def log_tensor_data(t: torch.Tensor,
                    t_args: Union[Tuple, List],
                    t_kwargs: Dict,
                    t_params: List[torch.nn.parameter.Parameter]):
    """Given a tensor, checks whether to log the data, and if so, logs the tensor, its args, kwargs, and params.

    Args:
        t: tensor to log
        t_args: args used to create the tensor
        t_kwargs: kwargs used to create the tensor
        t_params: params used to create the tensor.
        history_dict: The dictionary with the history of what's happened in the model.
    """
    history_dict = t.tl_history_dict
    tensor_barcode = t.tl_barcode
    tensor_num = t.tl_tensor_num

    if (history_dict['tensor_nums_to_save'] == 'all') or (tensor_num in history_dict['tensor_nums_to_save']) or \
            (tensor_num in history_dict['tensor_nums_to_save_temporarily']):
        # Get tensor contents
        history_dict['tensor_log'][tensor_barcode]['tensor_contents'] = safe_copy(t)

        # Get argument contents
        creation_args = []
        for arg in t_args:
            if issubclass(type(arg), (torch.Tensor, torch.nn.parameter.Parameter, torch.nn.Parameter)):
                creation_args.append(safe_copy(arg))
            else:
                creation_args.append(arg)
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = tuple(creation_args)

        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = {}
        for key, value in t_kwargs.items():
            if issubclass(type(value), (torch.Tensor, torch.nn.parameter.Parameter, torch.nn.Parameter)):
                history_dict['tensor_log'][tensor_barcode]['creation_kwargs'][key] = safe_copy(value)
            else:
                history_dict['tensor_log'][tensor_barcode]['creation_kwargs'][key] = value

        # Get parent parameters
        history_dict['tensor_log'][tensor_barcode]['parent_params'] = t_params
    else:
        history_dict['tensor_log'][tensor_barcode]['tensor_contents'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = []
        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = {}
        history_dict['tensor_log'][tensor_barcode]['parent_params'] = []
