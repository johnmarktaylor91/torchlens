import copy
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from torchlens.graph_handling import postprocess_history_dict
from torch_decorate import undecorate_tensor
from torchlens.helper_funcs import get_vars_of_type_from_obj, mark_tensors_in_obj, set_random_seed
from torchlens.torch_decorate import decorate_pytorch, undecorate_pytorch, update_tensor_containing_modules
from torchlens.model_history import ModelHistory


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


def module_pre_hook(module: nn.Module,
                    input_: tuple):
    """Pre-hook to attach to the modules to log the fact that the tensor entered this module.

    Args:
        module: PyTorch module.
        input_: The input.

    Returns:
        The input, now marked with information about the module it's entering.
    """
    module_address = module.tl_module_address
    module.tl_module_pass_num += 1
    module_pass_label = (module_address, module.tl_module_pass_num)
    module.tl_module_pass_labels.append(module_pass_label)
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    for t in input_tensors:
        model_history = t.tl_source_model_history
        module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
        t.tl_modules_entered.append(module_address)
        t.tl_module_passes_entered.append(module_pass_label)
        t.tl_is_submodule_input = True
        t.tl_module_entry_exit_thread.append(('+', module_pass_label[0], module_pass_label[1]))
        model_history.update_tensor_log_entry(t)  # Update tensor log with this new information.


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
    model_history = t.tl_source_model_history
    submodule_address = submodule.tl_module_address

    if t.tl_is_bottom_level_submodule_output:
        return True

    # If it was initialized inside the model and nothing entered the module, it's bottom-level.
    if t.tl_initialized_inside_model and len(submodule.tl_tensors_entered_labels) == 0:
        t.tl_is_bottom_level_submodule_output = True
        return True

    # Else, all parents must have entered the submodule for it to be a bottom-level submodule.
    for parent_label in t.tl_parent_tensors:
        parent_tensor = model_history[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != submodule_address):
            t.tl_is_bottom_level_submodule_output = False
            return False

    # If it survived the above tests, it's a bottom-level submodule.
    t.tl_is_bottom_level_submodule_output = True
    return True


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
    module_entry_label = module.tl_module_pass_labels.pop()
    input_tensors = get_vars_of_type_from_obj(input_, torch.Tensor)
    output_tensors = get_vars_of_type_from_obj(output_, torch.Tensor)
    for t in output_tensors:
        model_history = t.tl_source_model_history
        t.tl_is_submodule_output = True

        if module.tl_module_type.lower() == 'identity':  # if identity module, run the function for bookkeeping
            t = torch.identity(t)

        t.tl_is_bottom_level_submodule_output = log_whether_exited_submodule_is_bottom_level(t, module)
        t.tl_modules_exited.append(module_address)
        t.tl_module_passes_exited.append((module_address, module_pass_num))
        t.tl_module_entry_exit_thread.append(('-', module_entry_label[0], module_entry_label[1]))
        module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)
        model_history.update_tensor_log_entry(t)  # Update tensor log with this new information.

    for t in input_tensors:  # Now that module is finished, dial back the threads of all input tensors.
        input_module_thread = t.tl_module_entry_exit_thread[:]
        if ('+', module_entry_label[0], module_entry_label[1]) in input_module_thread:
            module_entry_ix = input_module_thread.index(('+', module_entry_label[0], module_entry_label[1]))
            t.tl_module_entry_exit_thread = t.tl_module_entry_exit_thread[:module_entry_ix]

    return output_


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
                model_history.log_source_tensor(attribute, 'buffer', buffer_address)


def prepare_model(model: nn.Module, hook_handles: List, model_history: ModelHistory):
    """Adds annotations and hooks to the model.

    Args:
        model: Model to prepare.
        hook_handles: Pre-allocated list to store the hooks, so they can be cleared even if execution fails.
        model_history: ModelHistory object logging the forward pass

    Returns:
        Model with hooks and attributes added.
    """
    model_history.model_name = str(type(model).__name__)
    model.tl_module_address = ''

    module_stack = [(model, '')]  # list of tuples (name, module)
    while len(module_stack) > 0:
        module, parent_address = module_stack.pop()
        module_children = list(module.named_children())

        # Annotate the children with the full address.
        for c, (child_name, child_module) in enumerate(module_children):
            child_address = f"{parent_address}.{child_name}" if parent_address != '' else child_name
            child_module.tl_module_address = child_address
            module_children[c] = (child_module, child_address)
        module_stack = module_children + module_stack

        if module == model:  # don't tag the model itself.
            continue

        module.tl_module_type = str(type(module).__name__)
        module.tl_module_pass_num = 0
        module.tl_module_pass_labels = []
        module.tl_tensors_entered_labels = []
        module.tl_tensors_exited_labels = []

        # Add hooks.

        hook_handles.append(module.register_forward_pre_hook(module_pre_hook))
        hook_handles.append(module.register_forward_hook(module_post_hook))

    # Mark all parameters with requires_grad = True, and mark what they were before, so they can be restored on cleanup.
    for param in model.parameters():
        param.tl_requires_grad = param.requires_grad
        param.requires_grad = True

    # And prepare any buffer tensors.
    prepare_buffer_tensors(model, model_history)


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


def undecorate_model_tensors(model: nn.Module):
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
                setattr(submodule, attribute_name, undecorate_tensor(attribute))


def cleanup_model(model: nn.Module, hook_handles: List):
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
    undecorate_model_tensors(model)


def run_model_and_save_specified_activations(model: nn.Module,
                                             x: Any,
                                             tensor_nums_to_save: Optional[Union[str, List[int]]] = 'all',
                                             random_seed: Optional[int] = None) -> ModelHistory:
    """Internal function that runs the given input through the given model, and saves the
    specified activations, as given by the tensor numbers (these will not be visible to the user;
    they will be generated from the nicer human-readable names and then fed in).

    Args:
        model: PyTorch model.
        x: Input to the model.
        tensor_nums_to_save: List of tensor numbers to save.
        random_seed: Which random seed to use.

    Returns:
        history_dict
    """
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)

    x = copy.deepcopy(x)
    hook_handles = []
    model_name = str(type(model).__name__)
    model_history = ModelHistory(model_name, random_seed, tensor_nums_to_save)
    orig_func_defs = []
    try:  # TODO: replace with context manager.
        if type(model) == nn.DataParallel:
            model = model.module
            model_is_dataparallel = True
        else:
            model_is_dataparallel = False
        if len(list(model.parameters())) > 0:
            model_device = next(iter(model.parameters())).device
        else:
            model_device = 'cpu'
        input_tensors = get_vars_of_type_from_obj(x, torch.Tensor)
        prepare_model(model, hook_handles, model_history)
        orig_func_defs, mutant_to_orig_funcs_dict = decorate_pytorch(torch,
                                                                     input_tensors,
                                                                     orig_func_defs,
                                                                     model_history)
        model_history.mutant_to_orig_funcs_dict = mutant_to_orig_funcs_dict
        model_history.track_tensors = True
        for t in input_tensors:
            t.requires_grad = True
            t = t.to(model_device)
            model_history.log_source_tensor(t, 'input')
        start_time = time.time()
        outputs = model(x)
        model_history.elapsed_time = time.time() - start_time
        model_history.track_tensors = False
        output_tensors = get_vars_of_type_from_obj(outputs, torch.Tensor)
        for t in output_tensors:
            model_history.output_tensors.append(t.tl_tensor_label_raw)
            t.tl_is_output_tensor = True
            model_history.update_tensor_log_entry(t)
        if model_is_dataparallel:
            model = nn.DataParallel(model)
        undecorate_pytorch(torch, orig_func_defs)
        cleanup_model(model, hook_handles)
        model_history.postprocess()
        del output_tensors
        del x
        return model_history

    except Exception as e:
        print("************\nFeature extraction failed; returning model and environment to normal\n*************")
        undecorate_pytorch(torch, orig_func_defs)
        cleanup_model(model, hook_handles)
        raise e
