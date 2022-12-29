# TODO: this module will later be the one for handling all
# torch mutation stuff.

import __future__
import collections
import copy
import functools
import time
import types
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.overrides import get_ignored_functions, get_testing_overrides

import helper_funcs as hf
from helper_funcs import safe_copy
from torchlens.model_history import ModelHistory
from torchlens.helper_funcs import get_marks_from_tensor_list, get_rng_states, \
    get_tensors_in_obj_with_mark, get_vars_of_type_from_obj, make_random_barcode, \
    identity, is_iterable, make_var_iterable

print_funcs = ['__repr__', '__str__', '_str']
funcs_not_to_log = ['cpu', 'cuda', 'numpy', 'to', '__array__']

# Taken from https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions
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
def my_get_overridable_functions() -> List:
    overridable_funcs = collections.defaultdict(list)
    index = {}
    func_names = []
    tested_namespaces = [
        ("torch", torch, torch.__all__),
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
orig_torch_funcs = overridable_funcs + ignored_funcs + [('', 'identity')]


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


def decorate_pytorch(torch_module: types.ModuleType,
                     tensors_to_mutate: List[torch.Tensor],  # TODO check if this is necessary or not.
                     orig_func_defs: List[Tuple],
                     model_history: ModelHistory) -> Tuple[List[Tuple], Dict]:
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
        new_func = torch_func_decorator(orig_func, model_history)
        new_func.tl_is_decorated_function = True
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
                    new_tensor_func = torch_func_decorator(orig_tensor_func, model_history)
                    setattr(local_tensor_namespace, func_name, new_tensor_func)
                except (AttributeError, TypeError, RuntimeError) as _:
                    pass

    # Bolt on the identity function
    new_identity = torch_func_decorator(identity, model_history)
    torch.identity = new_identity
    return orig_func_defs, mutant_to_orig_funcs_dict


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


def find_arg_positions_for_single_parent(parent_tensor: torch.Tensor,
                                         arg_type: str,
                                         arg_struct: Union[List, Tuple, Dict],
                                         tensor_all_arg_positions: Dict):
    """Helper function that finds where a single parent tensor is used in either the args or kwargs of a function,
    and updates a dict that tracks this information.

    Args:
        parent_tensor: Parent tensor
        arg_type: 'args' or 'kwargs'
        arg_struct: args or kwargs
        tensor_all_arg_positions: dict tracking where the tensors are used
    """
    iterfunc_dict = {'args': enumerate,
                     'kwargs': lambda x: x.items(),
                     list: enumerate,
                     tuple: enumerate,
                     dict: lambda x: x.items()}
    iterfunc = iterfunc_dict[arg_type]

    for arg_key, arg in iterfunc(arg_struct):
        if arg is parent_tensor:
            tensor_all_arg_positions[arg_type][arg_key] = parent_tensor.tl_tensor_label_raw
        elif type(arg) in [list, tuple, dict]:
            iterfunc2 = iterfunc_dict[type(arg)]
            for sub_arg_key, sub_arg in iterfunc2(arg):
                if sub_arg is parent_tensor:
                    tensor_all_arg_positions[arg_type][(arg_key, sub_arg_key)] = parent_tensor.tl_tensor_label_raw


def get_parent_tensor_function_call_location(parent_tensors: List[torch.Tensor],
                                             args: Tuple[Any],
                                             kwargs: Dict[Any, Any]) -> Dict:
    """Utility function that takes in the parent tensors, the args, and kwargs, and returns a dict specifying
    where in the function call the parent tensors were used.

    Args:
        parent_tensors: List of parent tensors.
        args: Tuple of function args
        kwargs: Dict of function kwargs

    Returns:
        Dict that itself contains two dicts, one specifing which args are associated with parent tensors,
        and another specifing which kwargs are associated with parent tensors.
    """
    tensor_all_arg_positions = {'args': {}, 'kwargs': {}}

    for parent_tensor in parent_tensors:
        if not hasattr(parent_tensor, 'tl_barcode'):
            continue
        for arg_type in ['args', 'kwargs']:
            if arg_type == 'args':
                arg_struct = args
            elif arg_type == 'kwargs':
                arg_struct = kwargs
            find_arg_positions_for_single_parent(parent_tensor, arg_type, arg_struct, tensor_all_arg_positions)

    return tensor_all_arg_positions


def update_tensor_containing_modules(t: torch.Tensor) -> List[str]:
    """Utility function that updates the containing modules of a Tensor by starting from the containing modules
    as of the last function call, then looks at the sequence of module transitions (in or out of a module) as of
    the last module it saw, and updates accordingly.

    Args:
        t: Tensor to check.

    Returns:
        List of the updated containing modules.
    """
    containing_modules = t.tl_containing_modules_origin_nested[:]
    thread_modules = t.tl_module_entry_exit_thread[:]
    for thread_module in thread_modules:
        if thread_module[0] == '+':
            containing_modules.append(thread_module[1:])
        elif (thread_module[0] == '-') and (thread_module[1:] in containing_modules):
            containing_modules.remove(thread_module[1:])
    return containing_modules


def get_input_module_info(arg_tensors):
    """Utility function to extract information about module entry/exit from input tensors.

    Args:
        arg_tensors: List of input tensors

    Returns:
        Variables with module entry/exit information
    """
    max_input_module_nesting = 0
    most_nested_containing_modules = []
    for t in arg_tensors:
        if not hasattr(t, 'tl_containing_modules_origin_nested'):
            continue
        containing_modules = update_tensor_containing_modules(t)
        if len(containing_modules) > max_input_module_nesting:
            max_input_module_nesting = len(containing_modules)
            most_nested_containing_modules = containing_modules[:]
    return most_nested_containing_modules


def get_ancestors_from_parents(arg_tensors: List[torch.Tensor]) -> Tuple[set[str], set[str]]:
    """Utility function to get the ancestors of a tensor based on those of its parent tensors.

    Args:
        arg_tensors: list of parent tensors

    Returns:
        List of input ancestors and internally initialized ancestors.
    """
    input_ancestors = set()
    internally_initialized_ancestors = set()
    for arg_tensor in arg_tensors:
        if hasattr(arg_tensor, 'tl_input_ancestors'):
            input_ancestors.update(arg_tensor.tl_input_ancestors)
        if hasattr(arg_tensor, 'tl_internally_initialized_ancestors'):
            internally_initialized_ancestors.update(arg_tensor.tl_internally_initialized_ancestors)
    return input_ancestors, internally_initialized_ancestors


def process_parent_param_passes(arg_parameters: List[torch.nn.Parameter]) -> Dict[str, int]:
    """Utility function to mark the parameters with barcodes, and log which pass they're on.

    Args:
        arg_parameters: List of arg parameters

    Returns:

    """
    parent_param_passes = {}
    for param in arg_parameters:
        if not hasattr(param, 'tl_param_barcode'):
            param_barcode = make_random_barcode()
            param.tl_param_barcode = param_barcode
            param.tl_pass_num = 1
        else:
            param_barcode = param.tl_param_barcode
            param.tl_pass_num += 1
        parent_param_passes[param_barcode] = param.tl_pass_num
    return parent_param_passes


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
        non_tensor_args = [arg for arg in args if not issubclass(type(arg), torch.Tensor)]
        non_tensor_kwargs = {key: val for key, val in kwargs.items() if not issubclass(type(val), torch.Tensor)}
        arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.parameter.Parameter])
        arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)
        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = hf.print_override(args[0], func_name)
            return out

        parent_tensor_labels = get_marks_from_tensor_list(arg_tensors, 'tl_tensor_label_raw')
        parent_tensor_arg_locs = get_parent_tensor_function_call_location(arg_tensors, args, kwargs)

        # Handle modules:

        containing_modules_origin_nested = get_input_module_info(arg_tensors)

        # Handle parameters:

        arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
        parent_param_passes = process_parent_param_passes(arg_parameters)

        # Handle ancestry:

        input_ancestors, internally_initialized_ancestors = get_ancestors_from_parents(arg_tensors)
        parent_internal_tensors = get_tensors_in_obj_with_mark(arg_tensors,
                                                               'tl_has_internally_initialialized_ancestor',
                                                               True)
        parent_internal_tensor_labels = get_marks_from_tensor_list(parent_internal_tensors, 'tl_tensor_label_raw')

        # Copy the arguments.

        arg_copies = [hf.safe_copy(arg) for arg in args]
        kwarg_copies = {key: hf.safe_copy(value) for key, value in kwargs.items()}

        # Call the function.

        func_call_barcode = make_random_barcode()
        model_history.current_function_call_barcode = func_call_barcode
        start_time = time.time()
        func_rng_states = get_rng_states()
        out_orig = func(*args, **kwargs)
        func_time_elapsed = time.time() - start_time

        if model_history.current_function_call_barcode == func_call_barcode:
            is_bottom_level_func = True
        else:
            is_bottom_level_func = False

        if func_name in ['__setitem__', 'zero_', '__delitem__']:
            out_orig = args[0]

        out_iter = make_var_iterable(out_orig)  # so we can iterate through it
        if type(out_orig) in [list, tuple, dict, set]:
            is_part_of_iterable_output = True
        else:
            is_part_of_iterable_output = False

        for i, out in enumerate(out_iter):
            if not (type(out) == torch.Tensor and (not hasattr(out, 'tl_tensor_label_raw') or
                                                   out.grad_fn != out.tl_gradfunc or
                                                   is_bottom_level_func)):
                continue
            if hasattr(out, 'tl_gradfunc') and (out.grad_fn == out.tl_gradfunc):
                func_changes_input = False
            else:
                func_changes_input = True
            model_history.log_function_output_tensor_func_info(out, args, kwargs, func, func_name,
                                                               func_changes_input, func_time_elapsed,
                                                               func_rng_states,
                                                               non_tensor_args, non_tensor_kwargs,
                                                               is_part_of_iterable_output, i)
            model_history.log_function_output_tensor_param_info(out, arg_parameters, parent_param_passes)
            model_history.log_function_output_tensor_graph_info(out, parent_tensor_labels, parent_tensor_arg_locs,
                                                                input_ancestors, parent_internal_tensor_labels,
                                                                internally_initialized_ancestors)
            model_history.log_function_output_tensor_module_info(out, containing_modules_origin_nested)
            model_history.make_tensor_log_entry(out, t_args=arg_copies, t_kwargs=kwarg_copies)
        return out_orig

    return wrapped_func


def undecorate_tensor(t):
    """Convenience function to replace the tensor with an unmutated version of itself, keeping the same data.

    Args:
        t: tensor or parameter object

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
    return new_t
