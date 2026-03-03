"""Model preparation: attaches metadata attributes and forward hooks to nn.Modules before logging."""

import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, TYPE_CHECKING

import torch
from torch import nn

from .. import _state
from ..data_classes import ParamAccessor, ParamLog
from ..utils.tensor_utils import get_tensor_memory_amount
from ..utils.introspection import get_vars_of_type_from_obj, iter_accessible_attributes
from ..utils.display import human_readable_size
from ..utils.hashing import make_random_barcode
from ..capture.source_tensors import log_source_tensor

# Cache class-level module metadata (shared across instances of the same class).
# Cleared at the start of each session in _prepare_model_session.
_module_class_metadata_cache: Dict[type, dict] = {}

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

# Session-scoped attributes that are set per-call and removed after.
# IMPORTANT: Do NOT add tl_module_address or tl_module_type here — those are
# permanent (cached across sessions in _prepare_model_once).
_SESSION_MODULE_ATTRS = [
    "tl_source_model_log",
    "tl_module_pass_num",
    "tl_module_pass_labels",
    "tl_tensors_entered_labels",
    "tl_tensors_exited_labels",
]

# Session-scoped attributes on parameters
_SESSION_PARAM_ATTRS = [
    "tl_requires_grad",
    "tl_param_barcode",
    "tl_param_address",
    "tl_pass_num",
]


# ---------------------------------------------------------------------------
# Shared module traversal
# ---------------------------------------------------------------------------


def _traverse_model_modules(model: nn.Module, visitor_fn) -> None:
    """DFS over all modules in a model, calling ``visitor_fn`` for each.

    Args:
        model: Root module.
        visitor_fn: Called as ``visitor_fn(module, address, named_children, is_root)``
            for every module. ``named_children`` is ``list(module.named_children())``.
    """
    module_stack = [(model, "")]
    while module_stack:
        module, address = module_stack.pop()
        named_children = list(module.named_children())
        child_entries = []
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_entries.append((child_module, child_address))
        module_stack = child_entries + module_stack
        visitor_fn(module, address, named_children, module is model)


# ---------------------------------------------------------------------------
# One-time model preparation (cached in _state._prepared_models)
# ---------------------------------------------------------------------------


def _prepare_model_once(model: nn.Module) -> None:
    """Prepare a model for permanent forward decoration.

    Runs once per model instance.  Assigns ``tl_module_address`` and
    ``tl_module_type`` to all submodules, wraps ``module.forward`` with
    ``module_forward_decorator``, and replaces any original torch functions
    in module ``__dict__`` with decorated versions.

    Results are cached in ``_state._prepared_models`` (WeakSet).
    """
    if model in _state._prepared_models:
        return

    model.tl_module_address = ""

    def _visit_once(module, address, named_children, is_root):
        # Replace any original torch functions stored in module __dict__
        for func_name, func in list(module.__dict__.items()):
            if func_name.startswith("__") or not callable(func):
                continue
            if id(func) in _state._orig_to_decorated:
                module.__dict__[func_name] = _state._orig_to_decorated[id(func)]

        # Annotate children with full address
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_module.tl_module_address = child_address

        if is_root:
            return

        module.tl_module_type = str(type(module).__name__)

        # Wrap forward if not already wrapped
        if hasattr(module, "forward") and not hasattr(
            module.forward, "tl_forward_call_is_decorated"
        ):
            module.forward = module_forward_decorator(module.forward, module)
            module.forward.tl_forward_call_is_decorated = True

    _traverse_model_modules(model, _visit_once)
    _state._prepared_models.add(model)


# ---------------------------------------------------------------------------
# Per-session model preparation
# ---------------------------------------------------------------------------


def _prepare_model_session(
    model_log: "ModelLog",
    model: nn.Module,
    optimizer=None,
) -> None:
    """Per-session model preparation.

    Sets session-scoped attributes (``tl_source_model_log``, pass counters),
    captures module metadata, forces ``requires_grad=True`` on all params,
    creates ParamLog objects, and tags buffer tensors.
    """
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    model_log.model_name = str(type(model).__name__)
    model.tl_source_model_log = model_log

    _seen_module_ids = {}

    def _visit_session(module, address, named_children, is_root):
        _capture_module_metadata(
            model_log,
            module,
            address,
            named_children,
            _seen_module_ids,
            is_root=is_root,
        )
        if is_root:
            return
        module.tl_source_model_log = model_log
        model_log.module_types[module.tl_module_address] = module.tl_module_type
        module.tl_module_pass_num = 0
        module.tl_module_pass_labels = []
        module.tl_tensors_entered_labels = []
        module.tl_tensors_exited_labels = []

    _traverse_model_modules(model, _visit_session)
    _create_session_param_logs(model_log, model, optimizer)
    prepare_buffer_tensors(model_log, model)


def _create_session_param_logs(model_log: "ModelLog", model: nn.Module, optimizer=None) -> None:
    """Create ParamLog objects for all parameters and force requires_grad=True."""
    optimized_param_ids = set()
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimized_param_ids.add(id(p))

    param_logs = {}
    for address, param in model.named_parameters():
        param.tl_requires_grad = param.requires_grad
        param.requires_grad = True

        barcode = make_random_barcode()
        param.tl_param_barcode = barcode
        param.tl_param_address = address
        param.tl_pass_num = 0

        parts = address.rsplit(".", 1)
        module_address = parts[0] if len(parts) > 1 else ""
        param_name = parts[-1]

        module = model
        if module_address:
            for attr in module_address.split("."):
                module = getattr(module, attr)
        module_type = type(module).__name__

        param_fsize = get_tensor_memory_amount(param)
        param_log = ParamLog(
            address=address,
            name=param_name,
            shape=tuple(param.shape),
            dtype=param.dtype,
            num_params=param.numel(),
            fsize=param_fsize,
            fsize_nice=human_readable_size(param_fsize),
            trainable=param.tl_requires_grad,
            module_address=module_address,
            module_type=module_type,
            barcode=barcode,
            has_optimizer=id(param) in optimized_param_ids if optimizer is not None else None,
        )
        param_log._param_ref = param
        param_logs[address] = param_log

    model_log.param_logs = ParamAccessor(param_logs)


# ---------------------------------------------------------------------------
# Module metadata capture (unchanged from before)
# ---------------------------------------------------------------------------


def _get_class_metadata(module_class: type) -> dict:
    """Return class-level metadata for a module class, cached across instances."""
    cached = _module_class_metadata_cache.get(module_class)
    if cached is not None:
        return cached

    meta = {}
    meta["module_class_name"] = module_class.__name__

    try:
        meta["source_file"] = inspect.getfile(module_class)
    except (TypeError, OSError):
        meta["source_file"] = None
    try:
        _, line = inspect.getsourcelines(module_class)
        meta["source_line"] = line
    except (TypeError, OSError):
        meta["source_line"] = None

    meta["class_docstring"] = module_class.__doc__

    try:
        meta["init_signature"] = str(inspect.signature(module_class.__init__))
    except (ValueError, TypeError):
        meta["init_signature"] = None
    meta["init_docstring"] = getattr(module_class.__init__, "__doc__", None)

    try:
        meta["forward_signature"] = str(inspect.signature(module_class.forward))
    except (ValueError, TypeError):
        meta["forward_signature"] = None
    meta["forward_docstring"] = getattr(module_class.forward, "__doc__", None)

    _module_class_metadata_cache[module_class] = meta
    return meta


def _capture_module_metadata(
    model_log: "ModelLog",
    module: nn.Module,
    parent_address: str,
    module_children: list,
    seen_module_ids: dict,
    is_root: bool = False,
) -> None:
    """Capture live module metadata during prepare_model(), while tl_* attrs are still present on modules.

    Must be called before the later cleanup step that removes tl_* session attributes.
    """
    address = "self" if is_root else parent_address

    module_id = id(module)
    if module_id in seen_module_ids:
        primary = seen_module_ids[module_id]
        if primary in model_log._module_metadata:
            model_log._module_metadata[primary]["all_addresses"].append(address)
        return
    seen_module_ids[module_id] = address

    # Start from cached class-level metadata (shallow copy — lists/dicts are replaced below)
    class_meta = _get_class_metadata(type(module))
    meta = dict(class_meta)
    meta["all_addresses"] = [address]

    # Per-instance forward override (rare — e.g. user assigned forward on instance before prep)
    if "forward" in module.__dict__:
        forward_func = module.__dict__["forward"]
        try:
            meta["forward_signature"] = str(inspect.signature(forward_func))
        except (ValueError, TypeError):
            pass
        doc = getattr(forward_func, "__doc__", None)
        if doc is not None:
            meta["forward_docstring"] = doc

    # Instance-specific fields
    meta["has_forward_hooks"] = bool(module._forward_hooks)
    meta["has_backward_hooks"] = bool(module._backward_hooks)
    meta["training_mode"] = module.training

    child_addresses = []
    for child_name, _ in module_children:
        if is_root:
            child_addresses.append(child_name)
        else:
            child_addresses.append(f"{parent_address}.{child_name}")
    meta["address_children"] = child_addresses

    _pytorch_internal = {
        "_parameters",
        "_buffers",
        "_modules",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_forward_hooks",
        "_forward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
        "_non_persistent_buffers_set",
        "training",
        "T_destination",
        "dump_patches",
        "call_super_init",
    }
    extra_attrs = {}
    user_methods = []
    nn_module_attrs = set(dir(nn.Module))
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        if attr_name.startswith("tl_"):
            continue
        if attr_name in _pytorch_internal:
            continue
        try:
            val = getattr(module, attr_name)
        except Exception:
            continue
        if callable(val):
            if attr_name not in nn_module_attrs:
                user_methods.append(attr_name)
        else:
            if attr_name not in nn_module_attrs:
                extra_attrs[attr_name] = val
    meta["extra_attributes"] = extra_attrs
    meta["methods"] = user_methods

    model_log._module_metadata[address] = meta


# ---------------------------------------------------------------------------
# Buffer tensor preparation
# ---------------------------------------------------------------------------


def prepare_buffer_tensors(model_log, model: nn.Module):
    """Tags buffer tensors with tl_buffer_address."""
    submodules = get_all_submodules(model)
    for submodule in submodules:
        attr_list = list(submodule.named_buffers()) + list(iter_accessible_attributes(submodule))
        for attribute_name, attribute in attr_list:
            if (
                issubclass(type(attribute), torch.Tensor)
                and not issubclass(type(attribute), torch.nn.Parameter)
                and not hasattr(attribute, "tl_buffer_address")
            ):
                if submodule.tl_module_address == "":
                    buffer_address = attribute_name
                else:
                    buffer_address = submodule.tl_module_address + "." + attribute_name
                setattr(attribute, "tl_buffer_address", buffer_address)


# ---------------------------------------------------------------------------
# Module forward decorator — reads model_log from _state
# ---------------------------------------------------------------------------


def _tag_untagged_buffers(module: nn.Module) -> None:
    """Tag any buffers that lack a ``tl_buffer_address`` attribute."""
    for buffer_name, buffer_tensor in module.named_buffers():
        if hasattr(buffer_tensor, "tl_buffer_address"):
            continue
        if module.tl_module_address == "":
            buffer_address = buffer_name
        else:
            buffer_address = f"{module.tl_module_address}.{buffer_name}"
        buffer_tensor.tl_buffer_address = buffer_address
        if hasattr(buffer_tensor, "tl_tensor_label_raw"):
            buffer_tensor.tl_buffer_parent = buffer_tensor.tl_tensor_label_raw
            delattr(buffer_tensor, "tl_tensor_label_raw")


def _handle_module_entry(model_log, module, args, kwargs):
    """Pre-forward: track input tensors, increment pass counters, tag buffers.

    Returns:
        Tuple of (input_tensor_labels, input_tensor_labels_at_entry) needed by the exit handler.
    """
    module_address = module.tl_module_address
    model_log.module_training_modes[module_address] = module.training
    module.tl_module_pass_num += 1
    module_pass_label = (module_address, module.tl_module_pass_num)
    module.tl_module_pass_labels.append(module_pass_label)

    # Capture forward args/kwargs for this pass (consumed by _build_module_logs)
    model_log._module_forward_args[(module_address, module.tl_module_pass_num)] = (args, kwargs)

    input_tensors = get_vars_of_type_from_obj(
        [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
    )
    input_tensor_labels = set()
    input_tensor_labels_at_entry = []
    for t in input_tensors:
        if (not hasattr(t, "tl_tensor_label_raw")) and hasattr(t, "tl_buffer_address"):
            log_source_tensor(model_log, t, "buffer", getattr(t, "tl_buffer_address"))
        tensor_entry = model_log._raw_tensor_dict[t.tl_tensor_label_raw]
        input_tensor_labels.add(t.tl_tensor_label_raw)
        module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
        tensor_entry.modules_entered.append(module_address)
        tensor_entry.module_passes_entered.append(module_pass_label)
        tensor_entry.is_submodule_input = True
        for arg_key, arg_val in list(enumerate(args)) + list(kwargs.items()):
            if arg_val is t:
                tensor_entry.modules_entered_argnames[
                    f"{module_pass_label[0]}:{module_pass_label[1]}"
                ].append(arg_key)
                model_log.module_layer_argnames[
                    (f"{module_pass_label[0]}:{module_pass_label[1]}")
                ].append((t.tl_tensor_label_raw, arg_key))
        tensor_entry.module_entry_exit_thread_output.append(
            ("+", module_pass_label[0], module_pass_label[1])
        )
        input_tensor_labels_at_entry.append(t.tl_tensor_label_raw)

    _tag_untagged_buffers(module)
    return input_tensor_labels, input_tensor_labels_at_entry


def _handle_module_exit(model_log, module, out, input_tensor_labels, input_tensor_labels_at_entry):
    """Post-forward: log output tensors, update module pass labels, trim entry threads."""
    module_address = module.tl_module_address
    module_pass_num = module.tl_module_pass_num
    module_entry_label = module.tl_module_pass_labels.pop()
    output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
    for t in output_tensors:
        if (module.tl_module_type.lower() == "identity") or (
            t.tl_tensor_label_raw in input_tensor_labels
        ):
            t = torch.identity(t)
        tensor_entry = model_log._raw_tensor_dict[t.tl_tensor_label_raw]
        tensor_entry.is_submodule_output = True
        tensor_entry.is_bottom_level_submodule_output = _is_bottom_level_submodule_exit(
            model_log, t, module
        )
        tensor_entry.modules_exited.append(module_address)
        tensor_entry.module_passes_exited.append((module_address, module_pass_num))
        tensor_entry.module_entry_exit_thread_output.append(
            ("-", module_entry_label[0], module_entry_label[1])
        )
        module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)

    for entry_label in input_tensor_labels_at_entry:
        tensor_entry = model_log._raw_tensor_dict[entry_label]
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


def module_forward_decorator(orig_forward: Callable, module: nn.Module) -> Callable:
    """Toggle-gated forward wrapper.  Closes over ``module`` (stable instance)
    but reads ``model_log`` from ``_state._active_model_log`` (global lookup).
    """

    @wraps(orig_forward)
    def decorated_forward(*args, **kwargs):
        # Fast path: if logging is off, pass through immediately.
        if not _state._logging_enabled or _state._active_model_log is None:
            return orig_forward(*args, **kwargs)

        model_log = _state._active_model_log

        if model_log.logging_mode == "fast":
            out = orig_forward(*args, **kwargs)
            input_tensors_fast = get_vars_of_type_from_obj(
                [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
            )
            input_tensor_labels = {
                t.tl_tensor_label_raw
                for t in input_tensors_fast
                if hasattr(t, "tl_tensor_label_raw")
            }
            output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
            for t in output_tensors:
                if (module.tl_module_type.lower() == "identity") or (
                    hasattr(t, "tl_tensor_label_raw")
                    and t.tl_tensor_label_raw in input_tensor_labels
                ):
                    t = torch.identity(t)
            return out

        # Exhaustive mode: entry → forward → exit
        input_tensor_labels, input_tensor_labels_at_entry = _handle_module_entry(
            model_log, module, args, kwargs
        )
        out = orig_forward(*args, **kwargs)
        _handle_module_exit(
            model_log, module, out, input_tensor_labels, input_tensor_labels_at_entry
        )
        return out

    return decorated_forward


# ---------------------------------------------------------------------------
# Helper: _is_bottom_level_submodule_exit
# ---------------------------------------------------------------------------


def _is_bottom_level_submodule_exit(model_log, t: torch.Tensor, submodule: nn.Module) -> bool:
    """Checks whether the submodule that a tensor is leaving is a "bottom-level" submodule.

    A "bottom-level" submodule is a leaf submodule that contains no further sub-submodules
    (i.e., it has no children in the module hierarchy).
    """
    tensor_entry = model_log._raw_tensor_dict[getattr(t, "tl_tensor_label_raw")]
    submodule_address = submodule.tl_module_address

    if tensor_entry.is_bottom_level_submodule_output:
        return True

    if tensor_entry.initialized_inside_model and len(submodule.tl_tensors_entered_labels) == 0:
        tensor_entry.is_bottom_level_submodule_output = True
        tensor_entry.bottom_level_submodule_pass_exited = (
            submodule_address,
            submodule.tl_module_pass_num,
        )
        return True

    for parent_label in tensor_entry.parent_layers:
        parent_tensor = model_log[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != submodule_address):
            tensor_entry.is_bottom_level_submodule_output = False
            return False

    tensor_entry.is_bottom_level_submodule_output = True
    tensor_entry.bottom_level_submodule_pass_exited = (
        submodule_address,
        submodule.tl_module_pass_num,
    )
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_all_submodules(model: nn.Module, is_top_level_model: bool = True) -> List[nn.Module]:
    """Recursively gets list of all submodules for given module."""
    submodules = []
    if is_top_level_model:
        submodules.append(model)
    for module in model.children():
        if module not in submodules:
            submodules.append(module)
        submodules += get_all_submodules(module, is_top_level_model=False)
    return submodules


def clear_hooks(hook_handles: List):
    """Clears a list of hook handles."""
    for hook_handle in hook_handles:
        hook_handle.remove()


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


def _cleanup_model_session(model: nn.Module, input_tensors=None) -> None:
    """Clean up session-specific attributes from a model.

    Restores ``requires_grad`` on all parameters, removes session-scoped
    ``tl_*`` attributes from modules, and removes ``tl_tensor_label_raw``
    from input/buffer tensors.

    Does NOT remove permanent attributes (``tl_module_address``,
    ``tl_module_type``) or restore original forward functions.
    """
    # Restore requires_grad on parameters
    for param in model.parameters():
        if hasattr(param, "tl_requires_grad"):
            param.requires_grad = param.tl_requires_grad

    # Remove session-scoped param attributes
    for param in model.parameters():
        for attr_name in _SESSION_PARAM_ATTRS:
            if hasattr(param, attr_name):
                try:
                    delattr(param, attr_name)
                except (AttributeError, RuntimeError):
                    pass

    # Remove session-scoped module attributes
    for module in model.modules():
        for attr_name in _SESSION_MODULE_ATTRS:
            if hasattr(module, attr_name):
                try:
                    delattr(module, attr_name)
                except (AttributeError, RuntimeError):
                    pass

    # Clean tensor labels from model tensors (buffers, etc.)
    _undecorate_model_tensors(model)

    # Clean tensor labels from input tensors
    if input_tensors:
        for t in input_tensors:
            if hasattr(t, "tl_tensor_label_raw"):
                delattr(t, "tl_tensor_label_raw")


def _undecorate_model_tensors(model: nn.Module) -> None:
    """Remove tl_tensor_label_raw and tl_buffer_* from non-parameter tensors in the model."""
    submodules = get_all_submodules(model)
    for submodule in submodules:
        for attribute_name, attribute in iter_accessible_attributes(submodule):
            if issubclass(type(attribute), torch.Tensor):
                if not issubclass(type(attribute), torch.nn.Parameter):
                    for tl_attr in ("tl_tensor_label_raw", "tl_buffer_address", "tl_buffer_parent"):
                        if hasattr(attribute, tl_attr):
                            delattr(attribute, tl_attr)
            elif type(attribute) in [list, tuple, set]:
                for item in attribute:
                    if issubclass(type(item), torch.Tensor):
                        for tl_attr in (
                            "tl_tensor_label_raw",
                            "tl_buffer_address",
                            "tl_buffer_parent",
                        ):
                            if hasattr(item, tl_attr):
                                delattr(item, tl_attr)
            elif type(attribute) == dict:
                for key, val in attribute.items():
                    if issubclass(type(val), torch.Tensor):
                        for tl_attr in (
                            "tl_tensor_label_raw",
                            "tl_buffer_address",
                            "tl_buffer_parent",
                        ):
                            if hasattr(val, tl_attr):
                                delattr(val, tl_attr)


# ---------------------------------------------------------------------------
# Ensure model is prepared (one-time + incremental crawl)
# ---------------------------------------------------------------------------


def _ensure_model_prepared(model: nn.Module) -> None:
    """One-time torch decoration + model preparation + incremental sys.modules crawl.

    On first call, decorates all torch functions permanently via
    ``decorate_all_once()``.  Subsequent calls are no-ops for decoration.
    """
    from .torch_funcs import decorate_all_once, patch_detached_references, patch_model_instance

    decorate_all_once()  # idempotent — no-op after first call
    _prepare_model_once(model)
    patch_detached_references()  # incremental — only new modules
    patch_model_instance(model)  # level 4 — model instance attrs


# Keep old names as aliases for backward compatibility during transition
prepare_model = _prepare_model_session
cleanup_model = _cleanup_model_session
