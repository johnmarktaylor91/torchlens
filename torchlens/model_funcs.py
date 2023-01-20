import copy
import random
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from torchlens.helper_funcs import get_vars_of_type_from_obj, remove_attributes_starting_with_str, set_random_seed
from torchlens.model_history import ModelHistory
from torchlens.torch_decorate import decorate_pytorch, undecorate_pytorch, undecorate_tensor


def run_model_and_save_specified_activations(model: nn.Module,
                                             input_args: Union[torch.Tensor, List[Any]],
                                             input_kwargs: Dict[Any, Any],
                                             tensor_nums_to_save: Optional[Union[str, List[int]]] = 'all',
                                             mark_input_output_distances: bool = False,
                                             random_seed: Optional[int] = None) -> ModelHistory:
    """Internal function that runs the given input through the given model, and saves the
    specified activations, as given by the tensor numbers (these will not be visible to the user;
    they will be generated from the nicer human-readable names and then fed in).

    Args:
        model: PyTorch model.
        input_args: Input arguments to the model's forward pass: either a single tensor, or a list of arguments.
        input_kwargs: Keyword arguments to the model's forward pass.
        tensor_nums_to_save: List of tensor numbers to save.
        mark_input_output_distances: Whether to compute the distance of each layer from the input or output.
            This is computationally expensive for large networks, so it is off by default.
        random_seed: Which random seed to use.

    Returns:
        ModelHistory object with full log of the forward pass
    """
    if type(input_args) == torch.Tensor:
        input_args = [input_args]

    if random_seed is None:  # set random seed
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)

    if type(model) == nn.DataParallel:  # Unwrap model from DataParallel if relevant:
        model = model.module

    if len(list(model.parameters())) > 0:  # Get the model device by looking at the parameters:
        model_device = next(iter(model.parameters())).device
    else:
        model_device = 'cpu'

    input_args = copy.deepcopy(input_args)
    model_name = str(type(model).__name__)
    model_history = ModelHistory(model_name, random_seed, tensor_nums_to_save)
    model_history.pass_start_time = time.time()

    hook_handles = []
    orig_func_defs = []

    try:
        input_args = move_input_tensors_to_device(input_args, model_device)
        input_kwargs = move_input_tensors_to_device(input_kwargs, model_device)
        input_arg_tensors = get_vars_of_type_from_obj(input_args, torch.Tensor)
        input_kwarg_tensors = get_vars_of_type_from_obj(input_kwargs, torch.Tensor)
        input_tensors = input_arg_tensors + input_kwarg_tensors
        decorated_func_mapper = decorate_pytorch(torch,
                                                 input_tensors,
                                                 orig_func_defs,
                                                 model_history)
        model_history.track_tensors = True
        for t in input_tensors:
            if 'int' not in str(t.dtype):
                t.requires_grad = True
            model_history.log_source_tensor(t, 'input')
        prepare_model(model, hook_handles, model_history, decorated_func_mapper)
        outputs = model(*input_args, **input_kwargs)
        model_history.track_tensors = False
        output_tensors = get_vars_of_type_from_obj(outputs, torch.Tensor)
        for t in output_tensors:
            model_history.output_layers.append(t.tl_tensor_label_raw)
            model_history[t.tl_tensor_label_raw].is_output_layer = True
        undecorate_pytorch(torch, orig_func_defs)
        cleanup_model(model, hook_handles, model_device, decorated_func_mapper)
        model_history.postprocess(mark_input_output_distances)
        return model_history

    except Exception as e:  # if anything fails, make sure everything gets cleaned up
        undecorate_pytorch(torch, orig_func_defs)
        cleanup_model(model, hook_handles, model_device, decorated_func_mapper)
        print("************\nFeature extraction failed; returning model and environment to normal\n*************")
        raise e

    finally:  # do garbage collection no matter what
        del input_args
        del output_tensors
        del outputs
        torch.cuda.empty_cache()


def move_input_tensors_to_device(x: Any,
                                 device: str):
    """Moves all tensors in the input to the given device.

    Args:
        x: Input to the model.
        device: Device to move the tensors to.
    """
    if type(x) == list:
        x = [t.to(device) for t in x]
    elif type(x) == dict:
        for k in x.keys():
            x[k] = x[k].to(device)
    elif type(x) == torch.Tensor:
        x = x.to(device)
    return x


def prepare_model(model: nn.Module,
                  hook_handles: List,
                  model_history: ModelHistory,
                  decorated_func_mapper: Dict[Callable, Callable]):
    """Adds annotations and hooks to the model, and decorates any functions in the model.

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

        module.tl_module_type = str(type(module).__name__)
        model_history.module_types[module.tl_module_address] = module.tl_module_type
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
        tensor_entry = model_history[t.tl_tensor_label_raw]
        module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
        tensor_entry.modules_entered.append(module_address)
        tensor_entry.module_passes_entered.append(module_pass_label)
        tensor_entry.is_submodule_input = True
        tensor_entry.module_entry_exit_thread.append(('+', module_pass_label[0], module_pass_label[1]))


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
        if module.tl_module_type.lower() == 'identity':  # if identity module, run the function for bookkeeping
            t = getattr(torch, 'identity')(t)

        model_history = t.tl_source_model_history
        tensor_entry = model_history[t.tl_tensor_label_raw]
        tensor_entry.is_submodule_output = True
        tensor_entry.is_bottom_level_submodule_output = log_whether_exited_submodule_is_bottom_level(t, module)
        tensor_entry.modules_exited.append(module_address)
        tensor_entry.module_passes_exited.append((module_address, module_pass_num))
        tensor_entry.module_entry_exit_thread.append(('-', module_entry_label[0], module_entry_label[1]))
        module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)

    for t in input_tensors:  # Now that module is finished, roll back the threads of all input tensors.
        model_history = t.tl_source_model_history
        tensor_entry = model_history[t.tl_tensor_label_raw]
        input_module_thread = tensor_entry.module_entry_exit_thread[:]
        if ('+', module_entry_label[0], module_entry_label[1]) in input_module_thread:
            module_entry_ix = input_module_thread.index(('+', module_entry_label[0], module_entry_label[1]))
            tensor_entry.module_entry_exit_thread = tensor_entry.module_entry_exit_thread[:module_entry_ix]

    return output_


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
    model_history = getattr(t, 'tl_source_model_history')
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
                  hook_handles: List,
                  model_device: str,
                  decorated_func_mapper: Dict[Callable, Callable]):
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

    Args:
        model: PyTorch model.
        hook_handles: List of hooks.
        model_device: Device the model is stored on
        decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs

    Returns:
        Original version of the model.
    """
    clear_hooks(hook_handles)
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
