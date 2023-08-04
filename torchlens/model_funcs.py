import copy
from functools import wraps
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from helper_funcs import move_input_tensors_to_device
from torchlens.helper_funcs import get_vars_of_type_from_obj, remove_attributes_starting_with_str, set_random_seed
from torchlens.model_history import ModelHistory
from torchlens.torch_decorate import decorate_pytorch, undecorate_pytorch, undecorate_tensor


def run_model_and_save_specified_activations(model: nn.Module,
                                             input_args: Union[torch.Tensor, List[Any]],
                                             input_kwargs: Dict[Any, Any],
                                             tensor_nums_to_save: Optional[Union[str, List[int]]] = 'all',
                                             keep_unsaved_layers: bool = True,
                                             output_device: str = 'same',
                                             activation_postfunc: Optional[Callable] = None,
                                             mark_input_output_distances: bool = False,
                                             detach_saved_tensors: bool = False,
                                             save_function_args: bool = False,
                                             save_gradients: bool = False,
                                             random_seed: Optional[int] = None) -> ModelHistory:
    """Internal function that runs the given input through the given model, and saves the
    specified activations, as given by the tensor numbers (these will not be visible to the user;
    they will be generated from the nicer human-readable names and then fed in).

    Args:
        model: PyTorch model.
        input_args: Input arguments to the model's forward pass: either a single tensor, or a list of arguments.
        input_kwargs: Keyword arguments to the model's forward pass.
        tensor_nums_to_save: List of tensor numbers to save.
        keep_unsaved_layers: Whether to keep layers in the ModelHistory log if they don't have saved activations.
        output_device: device where saved tensors will be stored: either 'same' to keep unchanged, or
            'cpu' or 'cuda' to move to cpu or cuda.
        activation_postfunc: Function to apply to activations before saving them (e.g., any averaging)
        mark_input_output_distances: Whether to compute the distance of each layer from the input or output.
            This is computationally expensive for large networks, so it is off by default.
        detach_saved_tensors: whether to detach the saved tensors, so they remain attached to the computational graph
        save_function_args: whether to save the arguments to each function
        save_gradients: whether to save gradients from any subsequent backward pass
        random_seed: Which random seed to use.

    Returns:
        ModelHistory object with full log of the forward pass
    """
    model_name = str(type(model).__name__)
    model_history = ModelHistory(model_name,
                                 output_device,
                                 activation_postfunc,
                                 keep_unsaved_layers,
                                 save_function_args,
                                 save_gradients,
                                 detach_saved_tensors,
                                 mark_input_output_distances)
    model_history.run_and_log_inputs_through_model(model,
                                                   input_args,
                                                   input_kwargs,
                                                   tensor_nums_to_save,
                                                   random_seed)
    return model_history


def prepare_model(model: nn.Module,
                  module_orig_forward_funcs: Dict,
                  model_history: ModelHistory,
                  decorated_func_mapper: Dict[Callable, Callable]):
    """Adds annotations and hooks to the model, and decorates any functions in the model.

    Args:
        model: Model to prepare.
        module_orig_forward_funcs: Dict with the original forward funcs for each submodule
        model_history: ModelHistory object logging the forward pass
        decorated_func_mapper: Dictionary mapping decorated functions to original functions, so they can be restored

    Returns:
        Model with hooks and attributes added.
    """
    model_history.model_name = str(type(model).__name__)
    model.tl_module_address = ''
    model.tl_source_model_history = model_history

    module_stack = [(model, '')]  # list of tuples (name, module)

    while len(module_stack) > 0:
        module, parent_address = module_stack.pop()
        module_children = list(module.named_children())

        # Decorate any torch functions in the model:
        for func_name, func in module.__dict__.items():
            if (func_name[0:2] == '__') or (not callable(func)) or (func not in decorated_func_mapper):
                continue
            module.__dict__[func_name] = decorated_func_mapper[func]

        # Annotate the children with the full address.
        for c, (child_name, child_module) in enumerate(module_children):
            child_address = f"{parent_address}.{child_name}" if parent_address != '' else child_name
            child_module.tl_module_address = child_address
            module_children[c] = (child_module, child_address)
        module_stack = module_children + module_stack

        if module == model:  # don't tag the model itself.
            continue

        module.tl_source_model_history = model_history
        module.tl_module_type = str(type(module).__name__)
        model_history.module_types[module.tl_module_address] = module.tl_module_type
        module.tl_module_pass_num = 0
        module.tl_module_pass_labels = []
        module.tl_tensors_entered_labels = []
        module.tl_tensors_exited_labels = []

        # Add decorators.

        if hasattr(module, 'forward'):
            module_orig_forward_funcs[module] = module.forward
            module.forward = module_forward_decorator(module.forward, module, model_history)
            module.forward.tl_forward_call_is_decorated = True

    # Mark all parameters with requires_grad = True, and mark what they were before, so they can be restored on cleanup.
    for param in model.parameters():
        param.tl_requires_grad = param.requires_grad
        param.requires_grad = True

    # And prepare any buffer tensors.
    prepare_buffer_tensors(model, model_history)


def prepare_buffer_tensors(model: nn.Module,
                           model_history: ModelHistory):
    """Goes through a model and all its submodules, and prepares any "buffer" tensors: tensors
    attached to the module that aren't model parameters.

    Args:
        model: PyTorch model
        model_history: the ModelHistory object logging the forward pass

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
                model_history.log_source_tensor_exhaustive(attribute, 'buffer', buffer_address)


def module_forward_decorator(orig_forward: Callable,
                             module: nn.Module,
                             model_history: ModelHistory) -> Callable:
    @wraps(orig_forward)
    def decorated_forward(*args, **kwargs):

        # "Pre-hook" operations:
        module_address = module.tl_module_address
        module.tl_module_pass_num += 1
        module_pass_label = (module_address, module.tl_module_pass_num)
        module.tl_module_pass_labels.append(module_pass_label)
        input_tensors = get_vars_of_type_from_obj([args, kwargs], torch.Tensor)
        for t in input_tensors:
            tensor_entry = model_history[t.tl_tensor_label_raw]
            module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
            tensor_entry.modules_entered.append(module_address)
            tensor_entry.module_passes_entered.append(module_pass_label)
            tensor_entry.is_submodule_input = True
            tensor_entry.module_entry_exit_thread_output.append(('+', module_pass_label[0], module_pass_label[1]))

        # The function call
        out = orig_forward(*args, **kwargs)

        # "Post-hook" operations:
        module_address = module.tl_module_address
        module_pass_num = module.tl_module_pass_num
        module_entry_label = module.tl_module_pass_labels.pop()
        output_tensors = get_vars_of_type_from_obj(out, torch.Tensor)
        for t in output_tensors:
            if module.tl_module_type.lower() == 'identity':  # if identity module, run the function for bookkeeping
                t = getattr(torch, 'identity')(t)
            tensor_entry = model_history[t.tl_tensor_label_raw]
            tensor_entry.is_submodule_output = True
            tensor_entry.is_bottom_level_submodule_output = log_whether_exited_submodule_is_bottom_level(t, module)
            tensor_entry.modules_exited.append(module_address)
            tensor_entry.module_passes_exited.append((module_address, module_pass_num))
            tensor_entry.module_entry_exit_thread_output.append(('-', module_entry_label[0], module_entry_label[1]))
            module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)

        for t in input_tensors:  # Now that module is finished, roll back the threads of all input tensors.
            tensor_entry = model_history[t.tl_tensor_label_raw]
            input_module_thread = tensor_entry.module_entry_exit_thread_output[:]
            if ('+', module_entry_label[0], module_entry_label[1]) in input_module_thread:
                module_entry_ix = input_module_thread.index(('+', module_entry_label[0], module_entry_label[1]))
                tensor_entry.module_entry_exit_thread_output = tensor_entry.module_entry_exit_thread_output[
                                                               :module_entry_ix]

        return out

    return decorated_forward


def log_whether_exited_submodule_is_bottom_level(t: torch.Tensor,
                                                 submodule: nn.Module):
    """Checks whether the submodule that a tensor is leaving is a "bottom-level" submodule;
    that is, that only one tensor operation happened inside the submodule.

    Args:
        t: the tensor leaving the module
        submodule: the module that the tensor is leaving

    Returns:
        Whether the tensor operation is bottom level.
    """
    model_history = getattr(submodule, 'tl_source_model_history')
    tensor_entry = model_history[getattr(t, 'tl_tensor_label_raw')]
    submodule_address = submodule.tl_module_address

    if tensor_entry.is_bottom_level_submodule_output:
        return True

    # If it was initialized inside the model and nothing entered the module, it's bottom-level.
    if tensor_entry.initialized_inside_model and len(submodule.tl_tensors_entered_labels) == 0:
        tensor_entry.is_bottom_level_submodule_output = True
        tensor_entry.bottom_level_submodule_pass_exited = (submodule_address, submodule.tl_module_pass_num)
        return True

    # Else, all parents must have entered the submodule for it to be a bottom-level submodule.
    for parent_label in tensor_entry.parent_layers:
        parent_tensor = model_history[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != submodule_address):
            tensor_entry.is_bottom_level_submodule_output = False
            return False

    # If it survived the above tests, it's a bottom-level submodule.
    tensor_entry.is_bottom_level_submodule_output = True
    tensor_entry.bottom_level_submodule_pass_exited = (submodule_address, submodule.tl_module_pass_num)
    return True


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


def cleanup_model(model: nn.Module,
                  module_orig_forward_funcs: Dict[nn.Module, Callable],
                  model_device: str,
                  decorated_func_mapper: Dict[Callable, Callable]):
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

    Args:
        model: PyTorch model.
        model_device: Device the model is stored on
        module_orig_forward_funcs: Dict containing the original, undecorated forward pass functions for each submodule
        decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs

    Returns:
        Original version of the model.
    """
    submodules = get_all_submodules(model, is_top_level_model=True)
    for submodule in submodules:
        if submodule == model:
            continue
        submodule.forward = module_orig_forward_funcs[submodule]
    restore_model_attributes(model, decorated_func_mapper=decorated_func_mapper, attribute_keyword='tl')
    undecorate_model_tensors(model, model_device)


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


def restore_module_attributes(module: nn.Module,
                              decorated_func_mapper: Dict[Callable, Callable],
                              attribute_keyword: str = 'tl'):
    for attribute_name in dir(module):
        if attribute_name.startswith(attribute_keyword):
            delattr(module, attribute_name)
            continue
        attr = getattr(module, attribute_name)
        if isinstance(attr, Callable) and (attr in decorated_func_mapper) and (attribute_name[0:2] != '__'):
            setattr(module, attribute_name, decorated_func_mapper[attr])


def restore_model_attributes(model: nn.Module,
                             decorated_func_mapper: Dict[Callable, Callable],
                             attribute_keyword: str = 'tl'):
    """Recursively clears the given attribute from all modules in the model.

    Args:
        model: PyTorch model.
        decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs
        attribute_keyword: Any attribute with this keyword will be cleared.

    Returns:
        Nothing.
    """
    for module in get_all_submodules(model):
        restore_module_attributes(module,
                                  decorated_func_mapper=decorated_func_mapper,
                                  attribute_keyword=attribute_keyword)

    for param in model.parameters():
        param.requires_grad = getattr(param, 'tl_requires_grad')
        for attribute_name in dir(param):
            if attribute_name.startswith(attribute_keyword):
                delattr(param, attribute_name)


def undecorate_model_tensors(model: nn.Module, model_device: str):
    """Goes through a model and all its submodules, and unmutates any tensor attributes. Normally just clearing
    parameters would have done this, but some module types (e.g., batchnorm) contain attributes that are tensors,
    but not parameters.

    Args:
        model: PyTorch model
        model_device: device the model is stored on

    Returns:
        PyTorch model with unmutated versions of all tensor attributes.
    """
    submodules = get_all_submodules(model)
    for submodule in submodules:
        for attribute_name in dir(submodule):
            attribute = getattr(submodule, attribute_name)
            if issubclass(type(attribute), torch.Tensor):
                if not issubclass(type(attribute), torch.nn.Parameter):
                    setattr(submodule, attribute_name, undecorate_tensor(attribute, model_device))
                else:
                    remove_attributes_starting_with_str(attribute, 'tl_')
