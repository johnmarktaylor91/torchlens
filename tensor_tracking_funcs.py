from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Tuple, Union

from util_funcs import barcode_tensors_in_obj, mark_tensors_in_obj, get_vars_of_type_from_obj, make_barcode, \
    get_tensor_memory_amount

import torch


def initialize_tensor_log_entry() -> Dict:
    """Utility function for making ordered dict of tensor fields, mostly for display convenience.

    Returns:
        Ordered Dictionary of tensor fields.
    """
    pass  # TODO, fill it out once we know what to log.


def initialize_history_dict(tensor_nums_to_save: Union[List[int], str]) -> Dict:
    """Convenience function for making the history dict. This will contain the tensor counter, which tensors to
    save, and the log of the tensors. TODO: give it the input so it can save these tensors.

    Returns:
        Initialized history dict.
    """
    history_dict = OrderedDict()
    history_dict['tensor_counter'] = 0
    history_dict['tensor_nums_to_save'] = tensor_nums_to_save
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
    mark_tensors_in_obj(x, 'xray_origin', 'input')
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    for t in input_tensors:
        history_dict['tensor_counter'] += 1
        t.xray_history_dict = history_dict
        t.xray_barcode = make_barcode()
        t.xray_tensor_num = history_dict['tensor_counter']
        t.xray_tensor_shape = t.shape
        t.xray_tensor_dtype = t.dtype
        t.xray_tensor_fsize = get_tensor_memory_amount(t)
        t.xray_origin = 'input'
        t.xray_parent_tensor_barcodes = []
        t.xray_funcs_applied = []
        t.xray_funcs_applied_names = []
        t.xray_parent_params = []
        t.xray_parent_param_passes = {}

        # Module stuff.

        t.xray_entered_module = False
        t.xray_containing_modules_nested = []
        t.xray_containing_module = None
        t.xray_last_module_seen = None
        t.xray_last_module_seen_address = None
        t.xray_is_module_output = False
        t.xray_modules_exited = []
        t.xray_is_bottom_level_module_output = False

        log_tensor_metadata(t)
        log_tensor_data(t, [], {}, [])


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

    if (history_dict['tensor_nums_to_save'] == 'all') or (tensor_num in history_dict['tensor_nums_to_save']):
        history_dict['tensor_log'][tensor_barcode]['tensor_contents'] = t.cpu().data
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = t_args
        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = t_kwargs
        history_dict['tensor_log'][tensor_barcode]['parent_params'] = t_params
    else:
        history_dict['tensor_log'][tensor_barcode]['tensor_contents'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_args'] = None
        history_dict['tensor_log'][tensor_barcode]['creation_kwargs'] = None
        history_dict['tensor_log'][tensor_barcode]['parent_params'] = None
