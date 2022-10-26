from collections import OrderedDict
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from tensor_tracking_funcs import log_tensor_metadata, initialize_history_dict, prepare_input_tensors
from torch_func_handling import mutate_pytorch, unmutate_pytorch
from util_funcs import get_vars_of_type_from_obj, text_num_split, set_random_seed, barcode_tensors_in_obj


# TODO think about best way to get non-leaf module activations if desired by user. Maybe just some kind of checker
# when it's generating the module-level graph. Shouldn't be hard, it can just look at the address and that of
# the successor. Remaining steps: make sure the on-the-fly log saves everything, then code up the
# function to clean it up and make the graph, then just the functions to make the graph and make the output pretty.
# Then finally a testing function; this will require saving enough information to "pick up" the forward pass
# where it left off, with everything except the actual inputs.

def get_module_from_address(model: nn.Module, address: str) -> nn.Module:
    """Given a model and an address to a layer, returns the module at that address.
    The address gives nested instructions for going from the top-level module to the desired module,
    in the format 'level1.level2.level3', etc. If a level is a string, then it'll be
    indexed by looking it up as an attribute; if an integer greater than or equal to zero,
    it'll be indexed like a list. For example, 'classifier.1' will first grab
    the 'classifier' attribute of the network, and then go into the second element of that list.

    Args:
        model: PyTorch model.
        address: String address.

    Returns:
        The module at the given address.
    """
    address = address.split('.')
    module = model
    for a in address:
        if (a.isdigit()) and (int(a) >= 0):
            module = module[int(a)]
        else:
            module = getattr(module, a)
    return module


def make_layer_name(layer_type: str,
                    layer_type_num: int,
                    layer_num: int,
                    pass_num: Optional[int] = None) -> str:
    """Makes a string name for a layer, given its type, type number, layer number, and pass number."""
    layer_name = f"{layer_type}{layer_type_num}_{layer_num}"
    if pass_num is not None:
        layer_name += f":{pass_num}"
    return layer_name


def parse_layer_name(layer_name: str) -> OrderedDict:
    """Given layer name, decomposes it into the relevant values.

    Args:
        layer_name: Name of the layer in format {layer_type}{layer_type_num}_{layer_num}:{pass_num},
            with pass_num optional. For example, conv4_9:2 is the second pass through the 4th convolutional
            layer, which is the 9th layer overall.
    Returns:
        Dict with layer_type, layer_type_num, layer_num, and pass_num if it's there.
    """
    layer_dict = OrderedDict()
    if ':' in layer_name:
        layer_name, pass_num = layer_name.split(':')
        layer_dict['pass_num'] = int(pass_num)
    else:
        pass_num = None

    type_label, layer_num = layer_name.split('_')
    layer_type, layer_type_num = text_num_split(type_label)
    layer_dict['layer_type'] = layer_type
    layer_dict['layer_type_num'] = layer_type_num
    layer_dict['layer_num'] = layer_num
    if pass_num:
        layer_dict['pass_num'] = pass_num
    return layer_dict


def get_bottom_level_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Recursively crawls through a given model, and returns a dict of the bottom-level
    modules, with keys corresponding to their address, and values corresponding to the
    modules themselves.

    Args:
        model: PyTorch model.

    Returns:
        Dict of bottom-level modules.
    """
    module_stack = [('', model)]
    module_dict = OrderedDict()
    while len(module_stack) > 0:
        module_tuple = module_stack.pop(0)
        module_address, module = module_tuple
        module_children = list(module.named_children())
        if len(module_children) == 0:
            module_dict[module_address] = module
        else:
            children_to_add = []
            for name, child in module_children:
                child_address = f"{module_address}.{name}" if module_address else name
                children_to_add.append((child_address, child))
            module_stack = children_to_add + module_stack
    return module_dict


def get_all_submodules(model: nn.Module) -> List[nn.Module]:
    """Recursively gets list of all submodules for given module, no matter their level in the
    hierarchy; this includes the model itself.

    Args:
        model: PyTorch model.

    Returns:
        List of all submodules.
    """
    submodules = [model]
    for module in model.children():
        submodules.append(module)
        submodules += get_all_submodules(module)
    return submodules


def hook_bottom_level_modules(model: nn.Module,
                              hook_fns: List[Callable],
                              prehook_fns: List[Callable],
                              hook_handles: List) -> List:
    """Hook all bottom-level modules in a model with given function and return
    list of hook handles (so as to easily clear later).

    Args:
        model: PyTorch model.
        hook_fns: List of hook functions to add.
        prehook_fns: List of pre-hook functions to add.
        hook_handles: Pre-allocated list for storing the hook handles.

    Returns:
        List of tuples (module_pointer, hook_handle) for each module.
    """
    bottom_level_modules = get_bottom_level_modules(model)
    for module_address, module in bottom_level_modules.items():
        for hook_fn in hook_fns:
            hook_handle = module.register_forward_hook(hook_fn)
            hook_handles.append(hook_handle)
        for prehook_fn in prehook_fns:
            hook_handle = module.register_forward_pre_hook(prehook_fn)
            hook_handles.append(hook_handle)
    return hook_handles


def module_pre_hook(module: nn.Module,
                    input_: tuple):
    """Pre-hook to attach to the modules: it marks the tensors as currently being inside a module, and
    indicates which module it is.

    Args:
        module: PyTorch module.
        input_: The input.

    Returns:
        The input, now marked with information about the module it's entering.
    """
    module_address = module.xray_module_address
    module.xray_module_pass_num += 1
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    for t in input_tensors:
        t.xray_containing_modules_nested.append(module_address)
        t.xray_containing_module = module_address
        t.xray_entered_module = True
        t.xray_last_module_seen_address = module_address
        t.xray_last_module_seen = module

        log_tensor_metadata(t)  # Update tensor log with this new information.


def module_post_hook(module: nn.Module,
                     input_,
                     output_):
    """Hook to run after the module is executed: it marks the tensors as no longer being inside a module,
    and indicates which module it is.

    Args:
        module: The module.
        input_: The input.
        output_: The output.

    Returns:
        Nothing, but records all relevant data.
    """

    module_address = module.xray_module_address
    output_tensors = get_vars_of_type_from_obj(output_, torch.Tensor)
    for t in output_tensors:
        t.xray_is_module_output = True
        t.xray_modules_exited.append(module_address)
        t.xray_containing_modules_nested.pop()  # remove the last module address.
        if module.xray_is_bottom_level_module:
            t.xray_is_bottom_level_module_output = True
        else:
            t.xray_is_bottom_level_module_output = False

        if len(t.xray_containing_modules_nested) == 0:
            t.xray_entered_module = False
            t.xray_containing_module = None
        else:
            t.xray_containing_module = t.xray_containing_modules_nested[-1]

        log_tensor_metadata(t)  # Update tensor log with this new information.

    return output_


def prepare_model(model: nn.Module,
                  hook_handles: List,
                  mode: str = 'modules_only') -> List:
    """Adds annotations and hooks to the model.

    Args:
        model: Model to prepare.
        hook_handles: Pre-allocated list to store the hooks so they can be cleared even if execution fails.
        mode: Either 'modules_only' for just the modules, or 'exhaustive' for all function calls.

    Returns:
        Model with hooks and attributes added.
    """
    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    module_stack = [('', model)]  # list of tuples (name, module)
    while len(module_stack) > 0:
        address, module = module_stack.pop()
        module_children = list(module.named_children())
        # Annotate the children with the full address.
        for child_name, child_module in module_children:
            child_module.xray_module_address = f"{address}.{child_name}" if address != '' else child_name
        module_stack = module_children + module_stack

        if module == model:  # don't tag the model itself.
            continue

        if len(module_children) == 0:
            is_bottom_level_module = True
        else:
            is_bottom_level_module = False

        module.xray_module_type = str(type(module).__name__).lower()
        module.xray_is_bottom_level_module = is_bottom_level_module
        module.xray_module_pass_num = 0

        # Add hooks.

        hook_handles.append(module.register_forward_pre_hook(module_pre_hook))
        hook_handles.append(module.register_forward_hook(module_post_hook))


def clear_hooks(hook_handles: List):
    """Takes in a list of tuples (module, hook_handle), and clears the hook at that
    handle for each module.

    Args:
        hook_handles: List of tuples (module, hook_handle)

    Returns:
        Nothing.
    """
    for hook_handle in hook_handles:
        hook_handle.remove()


def clear_model_keyword_attributes(model: nn.Module, attribute_keyword: str = 'xray'):
    """Recursively clears the given attribute from all modules in the model.
    TODO: have it clear the params too.

    Args:
        model: PyTorch model.
        attribute_keyword: Any attribute with this keyword will be cleared.

    Returns:
        Nothing.
    """
    for module in get_all_submodules(model):
        for attribute_name in dir(module):
            if attribute_keyword in attribute_name:
                delattr(module, attribute_name)


def cleanup_model(model: nn.Module, hook_handles: List) -> nn.Module:
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).
    TODO: cleanup params too.

    Args:
        model: PyTorch model.
        hook_handles: List of hooks.

    Returns:
        Original version of the model.
    """
    clear_hooks(hook_handles)
    clear_model_keyword_attributes(model, attribute_keyword='xray')
    return model


def run_model_and_save_specified_activations(model: nn.Module,
                                             x: Any,
                                             mode: str,
                                             tensor_nums_to_save: Optional[Union[str, List[int]]] = None,
                                             random_seed: Optional[int] = None) -> Dict:
    """Internal function that runs the given input through the given model, and saves the
    specified activations, as given by the internal tensor numbers (these will not be visible to the user;
    they will be generated from the nicer human-readable names and then fed in). The output
    will be nicely formatted.

    Args:
        model: PyTorch model.
        x: Input to the model.
        mode: Either 'modules_only' or 'exhaustive'
        tensor_nums_to_save: List of tensor numbers to save.
        mode: either 'modules_only' or 'exhaustive'.
        random_seed: Which random seed to use.

    Returns:
        history_dict
    """
    if random_seed is not None:
        set_random_seed(random_seed)
    x = copy.deepcopy(x)
    input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
    if tensor_nums_to_save is None:
        tensor_nums_to_save = []
    hook_handles = []
    prepare_model(model, hook_handles)
    history_dict = initialize_history_dict(tensor_nums_to_save)
    orig_func_defs = []
    try:  # TODO: replace with context manager.
        orig_func_defs = mutate_pytorch(torch, input_tensors, orig_func_defs, history_dict)
        prepare_input_tensors(x, history_dict)
        outputs = model(x)
        # mark_tensors_in_obj(outputs, 'xray_is_model_output', True)

    except Exception as e:
        print("Feature extraction failed; returning model and environment to normal")
        unmutate_pytorch(torch, orig_func_defs)
        model = cleanup_model(model, hook_handles)
        raise e

    # history_dict = postprocess_history_dict(history_dict, mode)
    return history_dict
