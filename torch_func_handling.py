from functools import wraps

import numpy as np

from torch.overrides import *

import __future__

import collections
import functools
import types
from typing import Dict, List, Any, Callable, Tuple
import inspect

import torch

from tensor_tracking_funcs import log_tensor_data, log_tensor_metadata
from util_funcs import get_marks_from_tensor_list, get_tensor_memory_amount, get_vars_of_type_from_obj, make_barcode, \
    tensor_in_obj_has_mark

print_funcs = ['__repr__', '__str__']
funcs_not_to_log = ['cpu', 'cuda']

# Taken from https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions

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


# Copy-pasted with tweaks just to get the actual paths to each function rather than the functions themselves.
# TODO: Clean this up.
# https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions

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
    n = np.array(t.data)
    np_str = getattr(n, func_name)()
    np_str = np_str.replace('array', 'tensor')
    np_str = np_str.replace('\n', '\n ')
    if t.grad_fn is not None:
        grad_fn_str = f", grad_fn={type(t.grad_fn).__name__})"
        np_str = np_str[0:-1] + grad_fn_str
    elif t.requires_grad:
        np_str = np_str[0:-1] + ", requires_grad=True)"
    return np_str


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
        if i == 0:
            out = getattr(obj, a)
        else:
            out = getattr(out, a)
    return out


def mutate_pytorch(torch_module: types.ModuleType,
                   tensors_to_mutate: List[torch.Tensor],  # TODO check if this is necessary or not.
                   orig_func_defs: List[Tuple],
                   history_log: Dict) -> List[Tuple]:
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
        to return torch to normal when finished.
    """
    for namespace_name, func_name in orig_torch_funcs:
        namespace_name_notorch = namespace_name.replace('torch.', '')
        local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
        orig_func = getattr(local_func_namespace, func_name)
        if hasattr(orig_func, '__name__') and orig_func.__name__ == 'wrapped_func':
            continue
        orig_func_defs.append((namespace_name, func_name, orig_func))
        new_func = torch_func_decorator(orig_func, history_log)
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
    return orig_func_defs


def unmutate_pytorch(torch_module,
                     orig_func_defs: List[Tuple]):
    """
    Returns all PyTorch functions back to the definitions they had when mutate_pytorch was called.

    args:
        torch_module: The torch module object.
        tensors_to_fix: Any tensors that you want to restore their original behavior for.
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


def torch_func_decorator(func, history_dict):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        func_name = func.__name__
        if func_name in funcs_not_to_log:
            out = func(*args, **kwargs)
            return out

        if (func_name in print_funcs) and (type(args[0]) in [torch.Tensor, torch.nn.parameter.Parameter]):
            out = print_override(args[0], func_name)
            return out
        all_args = list(args) + list(kwargs.values())
        arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor)
        parent_tensor_barcodes = get_marks_from_tensor_list(arg_tensors, 'xray_barcode')
        arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
        # TODO when designing the graph cleanup, make sure any tensors created inside a module
        # are indicated as being inside the module (check their descendents)
        any_inputs_entered_module = tensor_in_obj_has_mark(arg_tensors, 'xray_entered_module', True)
        if any_inputs_entered_module:
            for t in arg_tensors:
                if not hasattr(t, 'xray_containing_modules_nested'):
                    continue
                if len(t.xray_containing_modules_nested) > 0:
                    containing_modules = t.xray_containing_modules_nested[:]
                    containing_module = t.xray_containing_module
                    last_module_seen = t.xray_last_module_seen
                    last_module_seen_address = t.xray_last_module_seen_address
                    break
        else:
            containing_modules = []
            containing_module = None
            last_module_seen = None
            last_module_seen_address = None

        # Mark any parameters.

        parent_param_passes = {}
        for param in arg_parameters:
            if not hasattr(param, 'xray_param_barcode'):
                param_barcode = make_barcode()  # TODO also clean up all params with cleanup func.
                param.xray_param_barcode = param_barcode
                param.xray_pass_num = 1
            else:
                param_barcode = param.xray_param_barcode
                param.xray_pass_num += 1
            parent_param_passes[param_barcode] = param.xray_pass_num

        came_from_input = tensor_in_obj_has_mark(arg_tensors, 'xray_origin', 'input')
        if came_from_input:
            origin = 'input'
        else:
            origin = 'internal'
        out = func(*args, **kwargs)
        if type(out) == torch.Tensor:
            out.xray_history_dict = history_dict
            # If a new tensor, or has a new grad_fn, mark it with everything.
            if not hasattr(out, 'xray_barcode') or (out.grad_fn not in out.xray_gradfuncs):
                history_dict['tensor_counter'] += 1
                out.xray_barcode = make_barcode()
                out.xray_tensor_num = history_dict['tensor_counter']
                out.xray_tensor_shape = out.shape
                out.xray_tensor_dtype = out.dtype
                out.xray_tensor_fsize = get_tensor_memory_amount(out)
                out.xray_origin = origin
                out.xray_is_model_output = False
                out.xray_is_model_input = False
                out.xray_parent_tensor_barcodes = parent_tensor_barcodes
                out.xray_funcs_applied = [func]
                out.xray_funcs_applied_names = [func_name]
                out.xray_gradfuncs = [out.grad_fn]
                out.xray_gradfuncs_names = [type(out.grad_fn).__name__]
                out.xray_parent_params = arg_parameters[:]
                out.xray_parent_param_passes = parent_param_passes.copy()

                # Module stuff.

                out.xray_entered_module = any_inputs_entered_module
                out.xray_containing_modules_nested = containing_modules
                out.xray_containing_module = containing_module
                out.xray_last_module_seen = last_module_seen
                out.xray_last_module_seen_address = last_module_seen_address
                out.xray_is_module_output = False
                out.xray_modules_exited = []
                out.xray_is_bottom_level_module_output = False
            else:  # This means that the function returned the same tensor (e.g., identity); just need to mark that.
                out.xray_funcs_applied.append(func)
                out.xray_funcs_applied_names.append(func_name)

            log_tensor_data(out, args, kwargs, arg_parameters)
            log_tensor_metadata(out)

        return out

    return wrapped_func
