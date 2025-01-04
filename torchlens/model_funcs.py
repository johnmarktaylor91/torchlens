import warnings
from functools import wraps
from typing import Callable, Dict, List, TYPE_CHECKING

import torch
from torch import nn

from .helper_funcs import get_vars_of_type_from_obj, iter_accessible_attributes, remove_attributes_starting_with_str
from .logging_funcs import log_source_tensor

if TYPE_CHECKING:
    from .model_history import ModelHistory


def prepare_model(
        model_log: "ModelHistory",
        model: nn.Module,
        module_orig_forward_funcs: Dict,
        decorated_func_mapper: Dict[Callable, Callable],
):
    """Adds annotations and hooks to the model, and decorates any functions in the model.

    Args:
        model: Model to prepare.
        module_orig_forward_funcs: Dict with the original forward funcs for each submodule
        decorated_func_mapper: Dictionary mapping decorated functions to original functions, so they can be restored

    Returns:
        Model with hooks and attributes added.
    """
    model_log.model_name = str(type(model).__name__)
    model.tl_module_address = ""
    model.tl_source_model_history = model_log

    module_stack = [(model, "")]  # list of tuples (name, module)

    while len(module_stack) > 0:
        module, parent_address = module_stack.pop()
        module_children = list(module.named_children())

        # Decorate any torch functions in the model:
        for func_name, func in module.__dict__.items():
            if (
                    (func_name[0:2] == "__")
                    or (not callable(func))
                    or (func not in decorated_func_mapper)
            ):
                continue
            module.__dict__[func_name] = decorated_func_mapper[func]

        # Annotate the children with the full address.
        for c, (child_name, child_module) in enumerate(module_children):
            child_address = (
                f"{parent_address}.{child_name}"
                if parent_address != ""
                else child_name
            )
            child_module.tl_module_address = child_address
            module_children[c] = (child_module, child_address)
        module_stack = module_children + module_stack

        if module == model:  # don't tag the model itself.
            continue

        module.tl_source_model_history = model_log
        module.tl_module_type = str(type(module).__name__)
        model_log.module_types[module.tl_module_address] = module.tl_module_type
        module.tl_module_pass_num = 0
        module.tl_module_pass_labels = []
        module.tl_tensors_entered_labels = []
        module.tl_tensors_exited_labels = []

        # Add decorators.

        if hasattr(module, "forward") and not hasattr(
                module.forward, "tl_forward_call_is_decorated"
        ):
            module_orig_forward_funcs[module] = module.forward
            module.forward = module_forward_decorator(model_log, module.forward, module)
            module.forward.tl_forward_call_is_decorated = True

    # Mark all parameters with requires_grad = True, and mark what they were before, so they can be restored on cleanup.
    for param in model.parameters():
        param.tl_requires_grad = param.requires_grad
        param.requires_grad = True

    # And prepare any buffer tensors.
    prepare_buffer_tensors(model_log, model)


def prepare_buffer_tensors(model_log, model: nn.Module):
    """Goes through a model and all its submodules, and prepares any "buffer" tensors: tensors
    attached to the module that aren't model parameters.

    Args:
        model: PyTorch model

    Returns:
        PyTorch model with all buffer tensors prepared and ready to track.
    """
    submodules = get_all_submodules(model)
    for submodule in submodules:
        attr_list = list(submodule.named_buffers()) + list(iter_accessible_attributes(submodule))
        for attribute_name, attribute in attr_list:
            if issubclass(type(attribute), torch.Tensor) and not issubclass(
                    type(attribute), torch.nn.Parameter
            ) and not hasattr(attribute, 'tl_buffer_address'):
                if submodule.tl_module_address == "":
                    buffer_address = attribute_name
                else:
                    buffer_address = (
                            submodule.tl_module_address + "." + attribute_name
                    )
                setattr(attribute, 'tl_buffer_address', buffer_address)


def module_forward_decorator(
        model_log, orig_forward: Callable, module: nn.Module
) -> Callable:
    @wraps(orig_forward)
    def decorated_forward(*args, **kwargs):
        if model_log.logging_mode == "fast":  # do bare minimum for logging.
            out = orig_forward(*args, **kwargs)
            output_tensors = get_vars_of_type_from_obj(
                out, torch.Tensor, search_depth=4
            )
            for t in output_tensors:
                # if identity module, run the function for bookkeeping
                if module.tl_module_type.lower() == "identity":
                    t = getattr(torch, "identity")(t)
            return out

        # "Pre-hook" operations:
        module_address = module.tl_module_address
        module.tl_module_pass_num += 1
        module_pass_label = (module_address, module.tl_module_pass_num)
        module.tl_module_pass_labels.append(module_pass_label)
        input_tensors = get_vars_of_type_from_obj(
            [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=4
        )
        input_tensor_labels = set()
        for t in input_tensors:
            if (not hasattr(t, 'tl_tensor_label_raw')) and hasattr(t, 'tl_buffer_address'):
                log_source_tensor(model_log, t, 'buffer', getattr(t, 'tl_buffer_address'))
            tensor_entry = model_log._raw_tensor_dict[t.tl_tensor_label_raw]
            input_tensor_labels.add(t.tl_tensor_label_raw)
            module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
            tensor_entry.modules_entered.append(module_address)
            tensor_entry.module_passes_entered.append(module_pass_label)
            tensor_entry.is_submodule_input = True
            for arg_key, arg_val in list(enumerate(args)) + list(kwargs.items()):
                if arg_val is t:
                    tensor_entry.modules_entered_argnames[
                        f"{module_pass_label[0]}:{module_pass_label[1]}"].append(arg_key)
                    model_log.module_layer_argnames[(f"{module_pass_label[0]}:"
                                                     f"{module_pass_label[1]}")].append(
                        (t.tl_tensor_label_raw, arg_key))
            tensor_entry.module_entry_exit_thread_output.append(
                ("+", module_pass_label[0], module_pass_label[1])
            )

        # Check the buffers.
        for buffer_name, buffer_tensor in module.named_buffers():
            if hasattr(buffer_tensor, 'tl_buffer_address'):
                continue
            if module.tl_module_address == '':
                buffer_address = buffer_name
            else:
                buffer_address = f"{module.tl_module_address}.{buffer_name}"
            buffer_tensor.tl_buffer_address = buffer_address
            buffer_tensor.tl_buffer_parent = getattr(buffer_tensor, 'tl_tensor_label_raw')
            delattr(buffer_tensor, 'tl_tensor_label_raw')

        # The function call
        out = orig_forward(*args, **kwargs)

        # "Post-hook" operations:
        module_address = module.tl_module_address
        module_pass_num = module.tl_module_pass_num
        module_entry_label = module.tl_module_pass_labels.pop()
        output_tensors = get_vars_of_type_from_obj(
            out, torch.Tensor, search_depth=4
        )
        for t in output_tensors:
            # if identity module or tensor unchanged, run the identity function for bookkeeping
            if (module.tl_module_type.lower() == "identity") or (
                    t.tl_tensor_label_raw in input_tensor_labels
            ):
                t = getattr(torch, "identity")(t)
            tensor_entry = model_log._raw_tensor_dict[t.tl_tensor_label_raw]
            tensor_entry.is_submodule_output = True
            tensor_entry.is_bottom_level_submodule_output = (
                log_whether_exited_submodule_is_bottom_level(model_log, t, module)
            )
            tensor_entry.modules_exited.append(module_address)
            tensor_entry.module_passes_exited.append(
                (module_address, module_pass_num)
            )
            tensor_entry.module_entry_exit_thread_output.append(
                ("-", module_entry_label[0], module_entry_label[1])
            )
            module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)

        for (
                t
        ) in (
                input_tensors
        ):  # Now that module is finished, roll back the threads of all input tensors.
            tensor_entry = model_log._raw_tensor_dict[t.tl_tensor_label_raw]
            input_module_thread = tensor_entry.module_entry_exit_thread_output[:]
            if (
                    "+",
                    module_entry_label[0],
                    module_entry_label[1],
            ) in input_module_thread:
                module_entry_ix = input_module_thread.index(
                    ("+", module_entry_label[0], module_entry_label[1])
                )
                tensor_entry.module_entry_exit_thread_output = (
                    tensor_entry.module_entry_exit_thread_output[:module_entry_ix]
                )

        return out

    return decorated_forward


def log_whether_exited_submodule_is_bottom_level(
        model_log, t: torch.Tensor, submodule: nn.Module
):
    """Checks whether the submodule that a tensor is leaving is a "bottom-level" submodule;
    that is, that only one tensor operation happened inside the submodule.

    Args:
        t: the tensor leaving the module
        submodule: the module that the tensor is leaving

    Returns:
        Whether the tensor operation is bottom level.
    """
    tensor_entry = model_log._raw_tensor_dict[getattr(t, "tl_tensor_label_raw")]
    submodule_address = submodule.tl_module_address

    if tensor_entry.is_bottom_level_submodule_output:
        return True

    # If it was initialized inside the model and nothing entered the module, it's bottom-level.
    if (
            tensor_entry.initialized_inside_model
            and len(submodule.tl_tensors_entered_labels) == 0
    ):
        tensor_entry.is_bottom_level_submodule_output = True
        tensor_entry.bottom_level_submodule_pass_exited = (
            submodule_address,
            submodule.tl_module_pass_num,
        )
        return True

    # Else, all parents must have entered the submodule for it to be a bottom-level submodule.
    for parent_label in tensor_entry.parent_layers:
        parent_tensor = model_log[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (
                parent_modules_entered[-1] != submodule_address
        ):
            tensor_entry.is_bottom_level_submodule_output = False
            return False

    # If it survived the above tests, it's a bottom-level submodule.
    tensor_entry.is_bottom_level_submodule_output = True
    tensor_entry.bottom_level_submodule_pass_exited = (
        submodule_address,
        submodule.tl_module_pass_num,
    )
    return True


def get_all_submodules(
        model: nn.Module, is_top_level_model: bool = True
) -> List[nn.Module]:
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
        if module not in submodules:
            submodules.append(module)
        submodules += get_all_submodules(module, is_top_level_model=False)
    return submodules


def cleanup_model(
        model_log,
        model: nn.Module,
        module_orig_forward_funcs: Dict[nn.Module, Callable],
        decorated_func_mapper: Dict[Callable, Callable],
):
    """Reverses all temporary changes to the model (namely, the forward hooks and added
    model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

    Args:
        model: PyTorch model.
        module_orig_forward_funcs: Dict containing the original, undecorated forward pass functions for
            each submodule
        decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs

    Returns:
        Original version of the model.
    """
    submodules = get_all_submodules(model, is_top_level_model=True)
    for submodule in submodules:
        if submodule == model:
            continue
        submodule.forward = module_orig_forward_funcs[submodule]
    restore_model_attributes(
        model, decorated_func_mapper=decorated_func_mapper, attribute_keyword="tl"
    )
    undecorate_model_tensors(model)


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


def restore_module_attributes(
        module: nn.Module,
        decorated_func_mapper: Dict[Callable, Callable],
        attribute_keyword: str = "tl",
):
    def del_attrs_with_prefix(module, attribute_name):
        if attribute_name.startswith(attribute_keyword):
            delattr(module, attribute_name)
            return True

    for attribute_name, attr in iter_accessible_attributes(module, short_circuit=del_attrs_with_prefix):
        if (
                isinstance(attr, Callable)
                and (attr in decorated_func_mapper)
                and (attribute_name[0:2] != "__")
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(module, attribute_name, decorated_func_mapper[attr])


def restore_model_attributes(
        model: nn.Module,
        decorated_func_mapper: Dict[Callable, Callable],
        attribute_keyword: str = "tl",
):
    """Recursively clears the given attribute from all modules in the model.

    Args:
        model: PyTorch model.
        decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs
        attribute_keyword: Any attribute with this keyword will be cleared.

    Returns:
        Nothing.
    """
    for module in get_all_submodules(model):
        restore_module_attributes(
            module,
            decorated_func_mapper=decorated_func_mapper,
            attribute_keyword=attribute_keyword,
        )

    for param in model.parameters():
        if hasattr(param, "tl_requires_grad"):
            param.requires_grad = getattr(param, "tl_requires_grad")
            delattr(param, "tl_requires_grad")


def undecorate_model_tensors(model: nn.Module):
    """Goes through a model and all its submodules, and unmutates any tensor attributes. Normally just clearing
    parameters would have done this, but some module types (e.g., batchnorm) contain attributes that are tensors,
    but not parameters.

    Args:
        model: PyTorch model

    Returns:
        PyTorch model with unmutated versions of all tensor attributes.
    """
    submodules = get_all_submodules(model)
    for submodule in submodules:
        for attribute_name, attribute in iter_accessible_attributes(submodule):
            if issubclass(type(attribute), torch.Tensor):
                if not issubclass(type(attribute), torch.nn.Parameter) and hasattr(
                        attribute, "tl_tensor_label_raw"
                ):
                    delattr(attribute, "tl_tensor_label_raw")
                    if hasattr(attribute, 'tl_buffer_address'):
                        delattr(attribute, "tl_buffer_address")
                    if hasattr(attribute, 'tl_buffer_parent'):
                        delattr(attribute, "tl_buffer_parent")
                else:
                    remove_attributes_starting_with_str(attribute, "tl_")
            elif type(attribute) in [list, tuple, set]:
                for item in attribute:
                    if issubclass(type(item), torch.Tensor) and hasattr(
                            item, "tl_tensor_label_raw"
                    ):
                        delattr(item, "tl_tensor_label_raw")
                        if hasattr(item, 'tl_buffer_address'):
                            delattr(item, "tl_buffer_address")
                        if hasattr(item, 'tl_buffer_parent'):
                            delattr(item, "tl_buffer_parent")
            elif type(attribute) == dict:
                for key, val in attribute.items():
                    if issubclass(type(val), torch.Tensor) and hasattr(
                            val, "tl_tensor_label_raw"
                    ):
                        delattr(val, "tl_tensor_label_raw")
                        if hasattr(val, 'tl_buffer_address'):
                            delattr(val, "tl_buffer_address")
                        if hasattr(val, 'tl_buffer_parent'):
                            delattr(val, "tl_buffer_parent")
