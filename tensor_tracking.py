from collections import OrderedDict, defaultdict
from functools import wraps
from typing import Dict, List, Tuple, Union

import torch

from torch_func_handling import print_funcs, print_override
from xray_utils import get_marks_from_tensor_list, get_tensor_memory_amount, get_vars_of_type_from_obj, make_barcode, \
    tensor_in_obj_has_mark


def initialize_tensor_log_entry() -> Dict:
    """Utility function for making ordered dict of tensor fields, mostly for display convenience.

    Returns:
        Ordered Dictionary of tensor fields.
    """
    pass  # TODO, fill it out once we know what to log.


def initialize_history_dict(tensor_nums_to_save: List[int]) -> Dict:
    """Convenience function for making the history dict. This will contain the tensor counter, which tensors to
    save, and the log of the tensors.

    Returns:
        Initialized history dict.
    """
    history_dict = OrderedDict()
    history_dict['tensor_counter'] = 0
    history_dict['tensors_nums_to_save'] = []
    history_dict['tensor_log'] = defaultdict(lambda: OrderedDict())  # TODO replace with above function when it's done.
    return history_dict


def log_tensor_metadata(t: torch.Tensor):
    """Given a tensor, logs its metadata to the history_dict.

    Args:
        t: tensor to log
    """
    history_dict = t.xray_history_dict
    tensor_barcode = t.xray_barcode

    # Save any relevant fields.

    for field in dir(t):
        if not field.startswith('xray_'):  # xray is the keyword for marking relevant fields.
            continue
        field_stripped = field.strip('xray_')
        history_dict['tensor_log'][tensor_barcode][field_stripped] = getattr(t, field)


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
    history_dict = t.xray_history_dict
    tensor_barcode = t.xray_barcode
    tensor_num = t.xray_tensor_num

    if tensor_num in history_dict['tensors_nums_to_save']:
        history_dict['tensor_log'][tensor_barcode]['tensor'] = t.detach().cpu().numpy()
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = t_args
        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = t_kwargs
        history_dict['tensor_log'][tensor_barcode]['creation_params'] = t_params
    else:
        history_dict['tensor_log'][tensor_barcode]['tensor'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_params'] = None


def torch_func_decorator(func, history_dict):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        func_name = func.__name__
        if (func_name in print_funcs) and (type(args[0]) == torch.Tensor):
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
                    containing_modules = t.xray_containing_modules_nested
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
        out_shape = out.shape
        out_dtype = out.dtype
        out_fsize = get_tensor_memory_amount(out)
        if type(out) == torch.Tensor:
            out.xray_history_dict = history_dict
            if not hasattr(out, 'xray_barcode'):  # If a new tensor, mark it up with everything.
                history_dict['tensor_counter'] += 1
                out.xray_barcode = make_barcode()
                out.xray_tensor_num = history_dict['tensor_counter']
                out.xray_tensor_shape = out_shape
                out.xray_tensor_dtype = out_dtype
                out.xray_tensor_fsize = out_fsize
                out.xray_origin = origin
                out.xray_parent_tensor_barcodes = parent_tensor_barcodes
                out.xray_funcs_applied = [func_name]
                out.xray_parent_params = arg_parameters
                out.xray_parent_param_passes = parent_param_passes

                # Module stuff.

                out.xray_entered_module = any_inputs_entered_module
                out.xray_containing_modules_nested = containing_modules
                out.xray_containing_module = containing_module
                out.xray_last_module_seen = last_module_seen
                out.xray_last_module_seen_address = last_module_seen_address
                out.is_module_output = False
                out.is_bottom_level_module_output = False
            else:  # This means that the function returned the same tensor (e.g., identity); just need to mark that.
                out.xray_funcs_applied.append(func_name)

            log_tensor_metadata(out)
            log_tensor_data(out, args, kwargs, arg_parameters)

        return out

    return wrapped_func
