import copy
import random
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from torchlens.graph_handling import postprocess_history_dict, unmutate_tensor
from torchlens.helper_funcs import get_vars_of_type_from_obj, make_barcode, mark_tensors_in_obj, set_random_seed, \
    text_num_split
from torchlens.tensor_tracking import initialize_history_dict, log_tensor_metadata, mutate_pytorch, \
    prepare_input_tensors, \
    unmutate_pytorch, update_tensor_containing_modules, register_buffer_tensor


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


def get_all_submodules(model: nn.Module,
                       is_top_level_model: bool = True) -> List[nn.Module]:
    """Recursively gets list of all submodules for given module, no matter their level in the
    hierarchy; this includes the model itself.

    Args:
        model: PyTorch model.
        is_top_level_model: Whether it's the top-level model; just for the recursive logic of it.

    Returns:
        List of all submodules.
    """
    submodules = []
    if is_top_level_model:
        submodules.append(model)
    for module in model.children():
        submodules.append(module)
        submodules += get_all_submodules(module, is_top_level_model=False)
    return submodules


def hook_bottom_level_modules(model: nn.Module,
                              hook_fns: List[Callable],
                              prehook_fns: List[Callable],
                              hook_handles: List) -> List:
    """Hook all bottom-level modules in a model with given function and return
    list of hook handles (to easily clear later).

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
    module_address = module.tl_module_address
    module_entry_barcode = (module_address, make_barcode())
    module.tl_module_entry_barcodes.append(module_entry_barcode)
    module.tl_module_pass_num += 1
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    for t in input_tensors:
        t.tl_containing_modules_nested.append(module_entry_barcode)
        t.tl_containing_module = module_entry_barcode
        t.tl_entered_module = True
        t.tl_last_module_seen_address = module_address
        t.tl_last_module_seen_entry_barcode = module_entry_barcode
        t.tl_last_module_seen = module
        t.tl_module_just_entered_address = module_address
        t.tl_containing_modules_thread.append(('+', module_entry_barcode[0], module_entry_barcode[1]))

        log_tensor_metadata(t)  # Update tensor log with this new information.


def module_post_hook(module: nn.Module,
                     input_,
                     output_):
    """Hook to run after the module is executed: it marks the tensors as no longer being inside a module,
    indicates which module it is, and add

    Args:
        module: The module.
        input_: The input.
        output_: The output.

    Returns:
        Nothing, but records all relevant data.
    """

    module_address = module.tl_module_address
    module_pass_num = module.tl_module_pass_num
    module_entry_barcode = module.tl_module_entry_barcodes.pop()
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    output_tensors = get_vars_of_type_from_obj(output_, torch.Tensor)
    for t in output_tensors:
        tensor_log = t.tl_history_dict['tensor_log']
        t.tl_is_module_output = True

        # If it's not a bottom-level module output, check if it is one and tag it accordingly if so.

        if not t.tl_is_bottom_level_module_output and (len(t.tl_modules_exited) == 0):
            is_bottom_level_module_output = True
            for parent_barcode in t.tl_parent_tensor_barcodes:
                parent_tensor = tensor_log[parent_barcode]
                if parent_tensor['module_just_entered_address'] != module_address:
                    is_bottom_level_module_output = False
                    break
                if all([not parent_tensor['entered_module'], not parent_tensor['has_input_ancestor']]):
                    is_bottom_level_module_output = False
                    break

            if is_bottom_level_module_output:
                t.tl_is_bottom_level_module_output = True
                t.tl_bottom_module_barcode = module_address
                t.tl_bottom_module_type = type(module).__name__
                t.tl_bottom_module_pass_num = module_pass_num

        if module.tl_module_type.lower() == 'identity':
            t.tl_funcs_applied.append(lambda x: x)
            t.tl_funcs_applied_names.append('identity')
            t.tl_funcs_applied_modules.append(module_address)
            t.tl_function_call_modules_nested = update_tensor_containing_modules(t)
            t.tl_function_call_modules_nested_multfuncs.append(t.tl_function_call_modules_nested[:])
            t.tl_containing_modules_thread = []

        if len(t.tl_containing_modules_nested) > 0:
            t.tl_containing_modules_nested.pop()  # remove the last module address.
        t.tl_modules_exited.append(module_address)
        t.tl_module_passes_exited.append((module_address, module_pass_num))

        t.tl_containing_modules_thread.append(('-', module_entry_barcode[0], module_entry_barcode[1]))
        t.tl_last_module_seen_entry_barcode = module_entry_barcode

        if len(t.tl_containing_modules_nested) == 0:
            t.tl_entered_module = False
            t.tl_containing_module = None
        else:
            t.tl_containing_module = t.tl_containing_modules_nested[-1]

        log_tensor_metadata(t)  # Update tensor log with this new information.

    for t in input_tensors:  # Now that module is finished, dial back the threads of all input tensors.
        input_module_thread = t.tl_containing_modules_thread[:]
        if ('+', module_entry_barcode[0], module_entry_barcode[1]) in input_module_thread[::-1]:
            module_entry_ix = input_module_thread.index(('+', module_entry_barcode[0], module_entry_barcode[1]))
            t.tl_containing_modules_thread = t.tl_containing_modules_thread[:module_entry_ix]

    return output_


def prepare_model(model: nn.Module,
                  history_dict: Dict,
                  hook_handles: List,
                  mode: str = 'modules_only') -> List:
    """Adds annotations and hooks to the model.

    Args:
        model: Model to prepare.
        history_dict: Dictionary with the tensor history.
        hook_handles: Pre-allocated list to store the hooks so they can be cleared even if execution fails.
        device: Which device, 'cpu' or 'cuda', to put the model parameters.
        mode: Either 'modules_only' for just the modules, or 'exhaustive' for all function calls.

    Returns:
        Model with hooks and attributes added.
    """
    # TODO keep running list of modules that were converted to dataparallel so they can be changed back.

    history_dict['model_name'] = str(type(model).__name__)

    if mode not in ['modules_only', 'exhaustive']:
        raise ValueError("Mode must be either 'modules_only' or 'exhaustive'.")

    module_stack = [('', model)]  # list of tuples (name, module)
    model.tl_module_address = ''

    while len(module_stack) > 0:
        parent_address, module = module_stack.pop()
        module_children = list(module.named_children())

        # Annotate the children with the full address.
        for c, (child_name, child_module) in enumerate(module_children):
            child_address = f"{parent_address}.{child_name}" if parent_address != '' else child_name
            child_module.tl_module_address = child_address
            history_dict['module_dict'][child_module.tl_module_address] = child_module
            module_children[c] = (child_address, child_module)
        module_stack = module_children + module_stack

        if module == model:  # don't tag the model itself.
            continue

        if len(module_children) == 0:
            is_bottom_level_module = True
        else:
            is_bottom_level_module = False

        module.tl_module_type = str(type(module).__name__).lower()
        module.tl_is_bottom_level_module = is_bottom_level_module
        module.tl_module_pass_num = 0
        module.tl_module_entry_barcodes = []

        # Add hooks.

        hook_handles.append(module.register_forward_pre_hook(module_pre_hook))
        hook_handles.append(module.register_forward_hook(module_post_hook))

    # Mark all parameters with requires_grad = True, and mark what they were before so they can be restored on cleanup.
    for param in model.parameters():
        param.tl_requires_grad = param.requires_grad
        param.requires_grad = True

    prepare_buffer_tensors(model, history_dict)


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


def restore_model_attributes(model: nn.Module, attribute_keyword: str = 'tl'):
    """Recursively clears the given attribute from all modules in the model.

    Args:
        model: PyTorch model.
        attribute_keyword: Any attribute with this keyword will be cleared.

    Returns:
        Nothing.
    """
    for module in get_all_submodules(model):
        for attribute_name in dir(module):
            if attribute_name.startswith(attribute_keyword):
                delattr(module, attribute_name)

    for param in model.parameters():
        param.requires_grad = param.tl_requires_grad
        for attribute_name in dir(param):
            if attribute_name.startswith(attribute_keyword):
                delattr(param, attribute_name)


def unmutate_model_tensors(model: nn.Module):
    """Goes through a model and all its submodules, and unmutates any tensor attributes. Normally just clearing
    parameters would have done this, but some module types (e.g., batchnorm) contain attributes that are tensors,
    but not parameteres.

    Args:
        model: PyTorch model

    Returns:
        PyTorch model with unmutated versions of all tensor attributes.
    """
    submodules = get_all_submodules(model)
    for submodule in submodules:
        for attribute_name in dir(submodule):
            attribute = getattr(submodule, attribute_name)
            if issubclass(type(attribute), torch.Tensor):
                setattr(submodule, attribute_name, unmutate_tensor(attribute))


def prepare_buffer_tensors(model: nn.Module,
                           history_dict: Dict):
    """Goes through a model and all its submodules, and prepares any "buffer" tensors: tensors
    attached to the module that aren't model parameters.

    Args:
        model: PyTorch model
        history_dict: Dict of history

    Returns:
        PyTorch model with all buffer tensors prepared and ready to track.
    """
    submodules = get_all_submodules(model)
    for submodule in submodules:
        for attribute_name in dir(submodule):
            attribute = getattr(submodule, attribute_name)
            if issubclass(type(attribute), torch.Tensor) and not issubclass(type(attribute), torch.nn.Parameter):
                if submodule.tl_module_address == '':
                    buffer_address = attribute_name
                else:
                    buffer_address = submodule.tl_module_address + '.' + attribute_name
                register_buffer_tensor(attribute, buffer_address, history_dict)


def cleanup_model(model: nn.Module, hook_handles: List) -> nn.Module:
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

    Args:
        model: PyTorch model.
        hook_handles: List of hooks.

    Returns:
        Original version of the model.
    """
    clear_hooks(hook_handles)
    restore_model_attributes(model, attribute_keyword='tl')
    unmutate_model_tensors(model)
    return model


def run_model_and_save_specified_activations(model: nn.Module,
                                             x: Any,
                                             tensor_nums_to_save: Optional[Union[str, List[int]]] = None,
                                             tensor_nums_to_save_temporarily: Optional[Union[str, List[int]]] = None,
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
        tensor_nums_to_save_temporarily: List of tensor nums to save temporarily, but not at the end (for identity funcs)
        mode: either 'modules_only' or 'exhaustive'.
        device: which device to put the model and inputs on
        random_seed: Which random seed to use.

    Returns:
        history_dict
    """
    model.requires_grad = True
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)

    x = copy.deepcopy(x)
    x.requires_grad = True  # have it require_grad so as to autopopulate the grad_fns to help with book-keeping
    if tensor_nums_to_save is None:
        tensor_nums_to_save = []
    hook_handles = []
    history_dict = initialize_history_dict(tensor_nums_to_save, tensor_nums_to_save_temporarily)
    history_dict['random_seed'] = random_seed
    orig_func_defs = []
    try:  # TODO: replace with context manager.
        if type(model) == nn.DataParallel:
            model = model.module  # TODO: note this so it can be changed back later.
            model_is_dataparallel = True
        else:
            model_is_dataparallel = False
        if len(list(model.parameters())) > 0:
            model_device = next(iter(model.parameters())).device
        else:
            model_device = 'cpu'
        x = x.to(model_device)
        input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
        prepare_model(model, history_dict, hook_handles)
        orig_func_defs, mutant_to_orig_funcs_dict = mutate_pytorch(torch,
                                                                   input_tensors,
                                                                   orig_func_defs,
                                                                   history_dict)
        history_dict['mutant_to_orig_funcs_dict'] = mutant_to_orig_funcs_dict
        history_dict['track_tensors'] = True
        prepare_input_tensors(x, history_dict)
        start_time = time.time()
        outputs = model(x)
        history_dict['track_tensors'] = False
        history_dict['elapsed_time'] = time.time() - start_time
        output_tensors = get_vars_of_type_from_obj(outputs, torch.Tensor)
        for t in output_tensors:
            history_dict['output_tensors'].append(t.tl_barcode)
            mark_tensors_in_obj(t, 'tl_is_model_output', True)
            log_tensor_metadata(t)
        if model_is_dataparallel:
            model = nn.DataParallel(model)
        unmutate_pytorch(torch, orig_func_defs)
        history_dict = postprocess_history_dict(history_dict)
        model = cleanup_model(model, hook_handles)
        del output_tensors
        del x
        return history_dict

    except Exception as e:
        print("************\nFeature extraction failed; returning model and environment to normal\n*************")
        unmutate_pytorch(torch, orig_func_defs)
        model = cleanup_model(model, hook_handles)
        raise e
