"""Model preparation: two-phase setup of nn.Modules for activation logging.

This module implements the model preparation pipeline, which has two phases:

**Phase 1 — One-time preparation** (``_prepare_model_once``):
  Runs once per model instance (cached in ``_state._prepared_models`` WeakSet).
  Assigns permanent attributes (``tl_module_address``, ``tl_module_type``) and
  wraps each submodule's ``forward`` with ``module_forward_decorator``. These
  survive across multiple ``log_forward_pass`` calls.

**Phase 2 — Per-session preparation** (``_prepare_model_session``):
  Runs on every ``log_forward_pass`` call. Populates session-scoped tracking
  dicts on ModelLog (pass counters, entered/exited labels), captures module
  metadata, forces ``requires_grad=True`` on all parameters (needed for
  ``grad_fn`` metadata), creates ``ParamLog`` objects, and tags buffer tensors.
  Session data is GC'd with the ModelLog — no per-module cleanup needed.

The ``module_forward_decorator`` wraps each submodule's ``forward`` with a
toggle-gated wrapper that reads ``model_log`` from ``_state._active_model_log``
(not closed-over). In exhaustive mode, it calls ``_handle_module_entry`` /
``_handle_module_exit`` to track which tensors enter and exit each module. In
fast mode, it skips entry/exit tracking entirely but still handles
``nn.Identity`` and pass-through detection via ``torch.identity``.
"""

import inspect
import itertools
import warnings
from collections import deque
from functools import wraps
from typing import Callable, Dict, List, TYPE_CHECKING

import torch
from torch import nn

from .. import _state
from ..data_classes import ParamAccessor, ParamLog
from ..utils.tensor_utils import get_tensor_memory_amount
from ..utils.introspection import get_vars_of_type_from_obj, iter_accessible_attributes
from ..utils.hashing import make_random_barcode
from ..capture.source_tensors import log_source_tensor

# Cache class-level module metadata (inspect.getsourcelines, inspect.signature, etc.)
# shared across instances of the same class type. Cleared at the start of each
# session in _prepare_model_session to avoid stale data from reloaded modules.
_module_class_metadata_cache: Dict[type, dict] = {}

# Pre-computed set of nn.Module attribute names (from MRO). Used to filter out
# inherited methods/attrs when scanning for user-defined extras. Computed once
# at import time — nn.Module's interface is stable within a process.
_NN_MODULE_ATTRS = set(dir(nn.Module))

# PyTorch internal instance attributes to skip when scanning for user-defined
# extras. Module-level constant to avoid recreating per-module.
_PYTORCH_INTERNAL = frozenset(
    {
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
)

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

# Session-scoped attributes set on parameters during _create_session_param_logs.
_SESSION_PARAM_ATTRS = [
    "tl_requires_grad",  # original requires_grad (restored on cleanup)
    "tl_param_barcode",  # unique id for param-sharing detection
    "tl_param_address",  # dotted address (e.g. "encoder.weight")
    "tl_pass_num",  # how many times this param has been used
]


# ---------------------------------------------------------------------------
# Shared module traversal
# ---------------------------------------------------------------------------


def _traverse_model_modules(model: nn.Module, visitor_fn) -> None:
    """DFS over all modules in a model, calling ``visitor_fn`` for each.

    Visits parent before children (pre-order). The visitor receives the module,
    its dotted address, its named children list, and whether it is the root.

    Args:
        model: Root module.
        visitor_fn: Called as ``visitor_fn(module, address, named_children, is_root)``
            for every module. ``named_children`` is ``list(module.named_children())``.
    """
    module_stack: deque = deque([(model, "")])
    while module_stack:
        module, address = module_stack.popleft()
        named_children = list(module.named_children())
        child_entries = []
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_entries.append((child_module, child_address))
        # Prepend children to front of deque for DFS pre-order traversal.
        # extendleft reverses, so we reverse child_entries first to maintain order.
        for entry in reversed(child_entries):
            module_stack.appendleft(entry)
        visitor_fn(module, address, named_children, module is model)


# ---------------------------------------------------------------------------
# One-time model preparation (cached in _state._prepared_models)
# ---------------------------------------------------------------------------


def _prepare_model_once(model: nn.Module) -> None:
    """Phase 1: Permanent one-time model preparation.

    Runs once per model instance (idempotent via ``_state._prepared_models``
    WeakSet). Performs three tasks for each submodule:

    1. **Patches instance-level torch function refs** — If the user stored
       ``self.act = torch.relu`` in ``__init__``, that reference predates
       decoration. We replace it here (same as ``patch_model_instance`` but
       done during the DFS so children are caught too).

    2. **Assigns permanent attributes** — ``tl_module_address`` (dotted path
       like ``"encoder.layer.0.attention"``) and ``tl_module_type`` (class name).
       These survive across sessions.

    3. **Wraps ``forward``** — Replaces ``module.forward`` with
       ``module_forward_decorator(module.forward, module)``. The wrapper is
       toggle-gated: no-op when logging is off, full entry/exit tracking when on.
       The ``tl_forward_call_is_decorated`` sentinel prevents double-wrapping.

    The root module is skipped for type annotation and forward wrapping because
    its forward is called directly by ``log_forward_pass`` with its own
    entry/exit handling.
    """
    if model in _state._prepared_models:
        return

    model.tl_module_address = ""  # type: ignore[assignment]

    def _visit_once(module, address, named_children, is_root):
        # Replace any original torch functions stored as instance attributes
        # (e.g. self.act = torch.relu assigned before decoration).
        for func_name, func in list(module.__dict__.items()):
            if func_name.startswith("__") or not callable(func):
                continue
            if id(func) in _state._orig_to_decorated:
                module.__dict__[func_name] = _state._orig_to_decorated[id(func)]

        # Annotate children with their full dotted address path.
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_module.tl_module_address = child_address

        # Root module is handled separately by log_forward_pass.
        if is_root:
            return

        module.tl_module_type = str(type(module).__name__)

        # Wrap forward with toggle-gated decorator (idempotent via sentinel).
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
    """Phase 2: Per-session model preparation, called on every ``log_forward_pass``.

    Performs setup that must be fresh for each logging session:

    1. Clears metadata caches (class metadata, dir cache).
    2. Captures module metadata (source file, signatures, hooks, etc.) into
       ``model_log._module_metadata``.
    3. Sets session-scoped ``tl_*`` attributes on each submodule (pass counters,
       tensor entry/exit tracking lists, back-reference to model_log).
    4. Creates ``ParamLog`` objects and forces ``requires_grad=True`` on all
       parameters (needed so ``grad_fn`` chain is available for metadata).
    5. Tags buffer tensors with ``tl_buffer_address``.

    All session-scoped state is cleaned up by ``_cleanup_model_session``.
    """
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    model_log.model_name = str(type(model).__name__)

    # Track seen module ids to detect shared modules (same module at multiple addresses).
    _seen_module_ids: Dict[int, str] = {}

    # Use model.modules() + cached tl_module_address from phase 1, avoiding a
    # second full DFS with string concatenation and list(named_children()) calls.
    for module in model.modules():
        is_root = module is model
        address = getattr(module, "tl_module_address", "")
        named_children = list(module.named_children())
        _capture_module_metadata(
            model_log,
            module,
            address,
            named_children,
            _seen_module_ids,
            is_root=is_root,
        )
        if not is_root:
            model_log._module_build_data["module_types"][module.tl_module_address] = (
                module.tl_module_type
            )
            # Session-scoped tracking in ModelLog dicts (keyed by id(module)).
            mod_id = id(module)
            model_log._mod_pass_num[mod_id] = 0
            model_log._mod_pass_labels[mod_id] = []
            model_log._mod_entered[mod_id] = []
            model_log._mod_exited[mod_id] = []
    if model_log.logging_mode != "predicate":
        _create_session_param_logs(model_log, model, optimizer)
    prepare_buffer_tensors(model_log, model)


def _create_session_param_logs(model_log: "ModelLog", model: nn.Module, optimizer=None) -> None:
    """Create ``ParamLog`` objects for all parameters and force ``requires_grad=True``.

    ``requires_grad`` is forced True so that ``grad_fn`` metadata is available
    on all intermediate tensors during the forward pass. The original value is
    saved to ``tl_requires_grad`` and restored during ``_cleanup_model_session``.
    """
    optimized_param_ids = set()
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimized_param_ids.add(id(p))

    param_logs = {}
    seen_param_ids: set = set()
    for module in model.modules():
        module_address = getattr(module, "tl_module_address", "")
        module_type = type(module).__name__
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            # Shared parameters: only create one ParamLog per unique tensor.
            pid = id(param)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)

            address = f"{module_address}.{param_name}" if module_address else param_name

            # Save original requires_grad before forcing True.
            param.tl_requires_grad = param.requires_grad  # type: ignore[attr-defined]
            param.requires_grad = True

            barcode = make_random_barcode()
            param.tl_param_barcode = barcode  # type: ignore[attr-defined]
            param.tl_param_address = address  # type: ignore[attr-defined]
            param.tl_pass_num = 0  # type: ignore[attr-defined]

            param_fsize = get_tensor_memory_amount(param)
            param_log = ParamLog(
                address=address,
                name=param_name,
                shape=tuple(param.shape),
                dtype=param.dtype,
                num_params=param.numel(),
                memory=param_fsize,
                trainable=param.tl_requires_grad,  # type: ignore[attr-defined]
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


def _get_class_metadata(module_class: type, save_source_context: bool = False) -> dict:
    """Return class-level metadata for a module class, cached across instances.

    When ``save_source_context`` is False (default), skips expensive
    ``inspect.getsourcelines`` and ``inspect.signature`` calls. Only class
    name and docstrings (already in memory) are captured.

    When True, also fetches source file/line and signatures. Cached per class
    type to avoid redundant filesystem reads.
    """
    cached = _module_class_metadata_cache.get(module_class)
    if cached is not None:
        return cached

    meta: Dict[str, object] = {}
    meta["module_class_name"] = module_class.__name__
    meta["class_docstring"] = module_class.__doc__

    # Cache user-defined methods from class __dict__ (same for all instances of this class).
    user_methods = []
    for attr_name in module_class.__dict__:
        if attr_name.startswith("_") or attr_name.startswith("tl_"):
            continue
        if attr_name in _PYTORCH_INTERNAL or attr_name in _NN_MODULE_ATTRS:
            continue
        val = module_class.__dict__[attr_name]
        if callable(val):
            user_methods.append(attr_name)
    meta["user_methods"] = user_methods

    if save_source_context:
        try:
            meta["source_file"] = inspect.getfile(module_class)
        except (TypeError, OSError):
            meta["source_file"] = None
        try:
            _, line = inspect.getsourcelines(module_class)
            meta["source_line"] = line
        except (TypeError, OSError):
            meta["source_line"] = None

        init_method = getattr(module_class, "__init__", None)
        try:
            meta["init_signature"] = (
                str(inspect.signature(init_method)) if init_method is not None else None
            )
        except (ValueError, TypeError):
            meta["init_signature"] = None
        meta["init_docstring"] = getattr(init_method, "__doc__", None)

        forward_method = getattr(module_class, "forward", None)
        try:
            meta["forward_signature"] = (
                str(inspect.signature(forward_method)) if forward_method is not None else None
            )
        except (ValueError, TypeError):
            meta["forward_signature"] = None
        meta["forward_docstring"] = getattr(forward_method, "__doc__", None)
    else:
        meta["source_file"] = None
        meta["source_line"] = None
        meta["init_signature"] = None
        meta["init_docstring"] = None
        meta["forward_signature"] = None
        meta["forward_docstring"] = None

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
    """Capture live module metadata during ``_prepare_model_session``.

    Records source file/line, signatures, docstrings, hooks, training mode,
    child addresses, user-defined attributes/methods, and more. Must be called
    while ``tl_*`` attrs are present on modules (before cleanup).

    **Shared module handling**: If the same module object appears at multiple
    addresses (weight sharing), subsequent encounters just append to
    ``all_addresses`` of the primary entry rather than creating duplicates.
    """
    address = "self" if is_root else parent_address

    # Shared module detection: if we've already seen this module object,
    # just record the additional address and skip full metadata capture.
    module_id = id(module)
    if module_id in seen_module_ids:
        primary = seen_module_ids[module_id]
        if primary in model_log._module_metadata:
            model_log._module_metadata[primary]["all_addresses"].append(address)
        return
    seen_module_ids[module_id] = address

    # Start from cached class-level metadata. dict() creates a shallow copy;
    # mutable fields (all_addresses, extra_attributes, methods) are replaced
    # below with fresh instances per module, so no cross-contamination.
    save_source = getattr(model_log, "save_source_context", False)
    class_meta = _get_class_metadata(type(module), save_source_context=save_source)
    meta = dict(class_meta)
    meta["all_addresses"] = [address]

    # Per-instance forward override — rare, but handles cases where user
    # assigned a custom forward directly on the instance before preparation.
    if save_source and "forward" in module.__dict__:
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
    meta["is_training"] = module.training

    child_addresses = []
    for child_name, _ in module_children:
        if is_root:
            child_addresses.append(child_name)
        else:
            child_addresses.append(f"{parent_address}.{child_name}")
    meta["address_children"] = child_addresses

    extra_attrs = {}
    # Scan instance __dict__ for user-defined non-callable attrs (e.g. fc1, act).
    # Much faster than dir(module) which walks the full MRO.
    for attr_name, val in module.__dict__.items():
        if attr_name.startswith("_") or attr_name.startswith("tl_"):
            continue
        if attr_name in _PYTORCH_INTERNAL or attr_name in _NN_MODULE_ATTRS:
            continue
        if not callable(val):
            extra_attrs[attr_name] = val
    meta["extra_attributes"] = extra_attrs
    # User-defined methods are cached per class type in _get_class_metadata.
    meta["methods"] = class_meta["user_methods"]

    model_log._module_metadata[address] = meta


# ---------------------------------------------------------------------------
# Buffer tensor preparation
# ---------------------------------------------------------------------------


def prepare_buffer_tensors(model_log, model: nn.Module):
    """Tag buffer tensors with ``tl_buffer_address`` for later identification.

    Buffers are non-parameter tensors registered via ``register_buffer()`` or
    stored as plain tensor attributes. They are tagged here so that when a
    buffer first appears as an argument to a wrapped torch function, the
    interceptor can call ``log_source_tensor`` with the correct address.

    Uses ``named_buffers()`` for registered buffers and ``__dict__`` scan for
    plain tensor attributes (faster than ``iter_accessible_attributes`` which
    walks the MRO via ``dir()``). Tracks tagged tensor ids in
    ``_state._tagged_buffer_ids`` for fast cleanup.
    """
    _state._tagged_buffer_ids.clear()
    for submodule in model.modules():
        module_addr = getattr(submodule, "tl_module_address", "")
        # Scan registered buffers
        for buf_name, buf_tensor in submodule.named_buffers(recurse=False):
            if (
                isinstance(buf_tensor, torch.Tensor)
                and not isinstance(buf_tensor, torch.nn.Parameter)
                and not hasattr(buf_tensor, "tl_buffer_address")
            ):
                buffer_address = f"{module_addr}.{buf_name}" if module_addr else buf_name
                try:
                    buf_tensor.tl_buffer_address = buffer_address  # type: ignore[attr-defined]
                    _state._tagged_buffer_ids.add(id(buf_tensor))
                except Exception:
                    pass
        # Scan __dict__ for plain tensor attributes (not registered as buffers/params)
        for attr_name, attr_val in submodule.__dict__.items():
            if attr_name.startswith("_") or attr_name.startswith("tl_"):
                continue
            if (
                isinstance(attr_val, torch.Tensor)
                and not isinstance(attr_val, torch.nn.Parameter)
                and not hasattr(attr_val, "tl_buffer_address")
            ):
                buffer_address = f"{module_addr}.{attr_name}" if module_addr else attr_name
                try:
                    attr_val.tl_buffer_address = buffer_address  # type: ignore[attr-defined]
                    _state._tagged_buffer_ids.add(id(attr_val))
                except Exception:
                    pass
            elif isinstance(attr_val, (list, tuple)):
                for i, item in enumerate(attr_val):
                    if (
                        isinstance(item, torch.Tensor)
                        and not isinstance(item, torch.nn.Parameter)
                        and not hasattr(item, "tl_buffer_address")
                    ):
                        item_addr = (
                            f"{module_addr}.{attr_name}.{i}" if module_addr else f"{attr_name}.{i}"
                        )
                        try:
                            item.tl_buffer_address = item_addr  # type: ignore[attr-defined]
                            _state._tagged_buffer_ids.add(id(item))
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# Module forward decorator — reads model_log from _state
# ---------------------------------------------------------------------------


def _tag_untagged_buffers(module: nn.Module) -> None:
    """Tag any buffers that lack a ``tl_buffer_address`` attribute.

    Called during ``_handle_module_entry`` to catch buffers that were created
    dynamically (e.g. in ``forward()``) after the initial ``prepare_buffer_tensors``
    scan. If a buffer already has a ``tl_tensor_label_raw`` from being logged as
    an intermediate tensor, that label is moved to ``tl_buffer_parent`` and cleared
    so the buffer gets a fresh source-tensor entry on next use.
    """
    for buffer_name, buffer_tensor in module.named_buffers():
        if hasattr(buffer_tensor, "tl_buffer_address"):
            continue
        if module.tl_module_address == "":
            buffer_address = buffer_name
        else:
            buffer_address = f"{module.tl_module_address}.{buffer_name}"
        buffer_tensor.tl_buffer_address = buffer_address  # type: ignore[attr-defined]
        # If this buffer was already logged as an intermediate tensor, save the
        # old label as parent and reset so it gets a proper buffer source entry.
        if hasattr(buffer_tensor, "tl_tensor_label_raw"):
            buffer_tensor.tl_buffer_parent = buffer_tensor.tl_tensor_label_raw  # type: ignore[attr-defined]
            delattr(buffer_tensor, "tl_tensor_label_raw")


def _handle_module_entry(model_log, module, args, kwargs):
    """Pre-forward bookkeeping for exhaustive mode.

    Called immediately before ``orig_forward(*args, **kwargs)`` in the
    ``module_forward_decorator``. Performs:

    1. Increments the module's pass counter (supports multi-pass modules like
       recurrent layers).
    2. Pushes a ``(address, pass_num)`` label onto the module's pass stack
       (popped by ``_handle_module_exit``).
    3. Finds all input tensors and annotates their layer entries with module
       entry metadata (``modules_entered``, ``is_submodule_input``, etc.).
    4. Records which argument position each tensor occupies (for arg-name
       reconstruction in ``_build_module_logs``).
    5. Appends a ``("+", address, pass_num)`` marker to each input tensor's
       ``module_entry_exit_thread_output`` (a chronological log of module
       boundaries that tensors cross).
    6. Tags any dynamically-created buffers that weren't caught during
       ``prepare_buffer_tensors``.

    Args:
        model_log: The active ModelLog.
        module: The nn.Module about to execute.
        args: Positional arguments to forward.
        kwargs: Keyword arguments to forward.

    Returns:
        Tuple of ``(input_tensor_labels, input_tensor_labels_at_entry)`` —
        needed by ``_handle_module_exit`` to trim the entry thread.
    """
    module_address = module.tl_module_address
    mod_id = id(module)
    model_log._module_build_data["module_training_modes"][module_address] = module.training
    model_log._mod_pass_num[mod_id] += 1
    module_pass_num = model_log._mod_pass_num[mod_id]
    module_pass_label = (module_address, module_pass_num)
    # Push onto stack — popped by _handle_module_exit (or exception handler).
    model_log._mod_pass_labels[mod_id].append(module_pass_label)

    # Stash forward args for later use by _build_module_logs.
    model_log._module_forward_args[(module_address, module_pass_num)] = (args, kwargs)

    # Find all tensor arguments (excluding Parameters, which are source tensors).
    input_tensors = get_vars_of_type_from_obj(
        [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
    )
    input_tensor_labels = set()
    input_tensor_labels_at_entry = []
    for t in input_tensors:
        # Lazily register buffer tensors that haven't been logged yet.
        if (not hasattr(t, "tl_tensor_label_raw")) and hasattr(t, "tl_buffer_address"):
            log_source_tensor(model_log, t, "buffer", getattr(t, "tl_buffer_address"))
        if not hasattr(t, "tl_tensor_label_raw"):
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        layer_entry = model_log._raw_layer_dict[t.tl_tensor_label_raw]
        input_tensor_labels.add(t.tl_tensor_label_raw)
        model_log._mod_entered[mod_id].append(t.tl_tensor_label_raw)
        layer_entry.modules_entered.append(module_address)
        layer_entry.module_passes_entered.append(module_pass_label)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                layer_entry.modules_entered_argnames[
                    f"{module_pass_label[0]}:{module_pass_label[1]}"
                ].append(arg_key)
                model_log._module_build_data["module_layer_argnames"][
                    (f"{module_pass_label[0]}:{module_pass_label[1]}")
                ].append((t.tl_tensor_label_raw, arg_key))
        # Record module entry in chronological thread (matched by "-" on exit).
        layer_entry.module_entry_exit_thread_output.append(
            ("+", module_pass_label[0], module_pass_label[1])
        )
        input_tensor_labels_at_entry.append(t.tl_tensor_label_raw)

    # Catch buffers created dynamically (e.g. in forward()) after initial scan.
    _tag_untagged_buffers(module)
    return input_tensor_labels, input_tensor_labels_at_entry


def _handle_module_exit(model_log, module, out, input_tensor_labels, input_tensor_labels_at_entry):
    """Post-forward bookkeeping for exhaustive mode.

    Called immediately after ``orig_forward()`` returns in the
    ``module_forward_decorator``. Performs:

    1. Pops the module pass label from the stack (pushed by ``_handle_module_entry``).
    2. For each output tensor:
       - If it's an ``nn.Identity`` pass-through or the same tensor that entered,
         wraps it with ``torch.identity()`` to create a distinct log entry at the
         module boundary.
       - Marks the tensor's layer entry as a submodule output.
       - Checks if this is a bottom-level (leaf) submodule exit.
       - Appends ``("-", address, pass_num)`` to the chronological thread.
    3. Trims the ``module_entry_exit_thread_output`` of input tensors — removes
       the ``"+"`` entry marker and everything after it, since the module boundary
       is now closed. This prevents stale entry markers from confusing downstream
       module-nesting analysis.
    """
    module_address = module.tl_module_address
    mod_id = id(module)
    module_pass_num = model_log._mod_pass_num[mod_id]
    module_entry_label = model_log._mod_pass_labels[mod_id].pop()
    output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
    for t in output_tensors:
        # nn.Identity modules and pass-through tensors (output is same object
        # as input) need _decorated_identity() to create a distinct log entry
        # so the graph correctly shows the module boundary.
        if (module.tl_module_type.lower() == "identity") or (
            hasattr(t, "tl_tensor_label_raw") and t.tl_tensor_label_raw in input_tensor_labels
        ):
            t = _state._decorated_identity(t)
        layer_entry = model_log._raw_layer_dict[t.tl_tensor_label_raw]
        layer_entry.is_submodule_output = True
        layer_entry.is_leaf_module_output = _is_bottom_level_submodule_exit(model_log, t, module)
        layer_entry.modules_exited.append(module_address)
        layer_entry.module_passes_exited.append((module_address, module_pass_num))
        # Record module exit in chronological thread (matches "+" from entry).
        layer_entry.module_entry_exit_thread_output.append(
            ("-", module_entry_label[0], module_entry_label[1])
        )
        model_log._mod_exited[mod_id].append(t.tl_tensor_label_raw)

    # Trim entry threads on input tensors: remove the "+" marker and
    # everything after it. The module boundary is now closed, so input
    # tensors' threads should not carry forward this module's context.
    for entry_label in input_tensor_labels_at_entry:
        layer_entry = model_log._raw_layer_dict[entry_label]
        input_module_thread = layer_entry.module_entry_exit_thread_output[:]
        if (
            "+",
            module_entry_label[0],
            module_entry_label[1],
        ) in input_module_thread:
            module_entry_ix = input_module_thread.index(
                ("+", module_entry_label[0], module_entry_label[1])
            )
            layer_entry.module_entry_exit_thread_output = (
                layer_entry.module_entry_exit_thread_output[:module_entry_ix]
            )


def module_forward_decorator(orig_forward: Callable, module: nn.Module) -> Callable:
    """Toggle-gated forward wrapper for an nn.Module's ``forward`` method.

    **Closure design**: Closes over ``module`` (a stable instance reference) but
    reads ``model_log`` from ``_state._active_model_log`` at call time. This is
    necessary because the same wrapper persists across multiple ``log_forward_pass``
    calls with different ModelLog instances.

    **Three execution modes**:

    1. **Logging off** (``_state._logging_enabled is False``): Pass through to
       ``orig_forward`` with zero overhead beyond one bool check. This is the
       normal production path.

    2. **Fast mode** (``model_log.logging_mode == "fast"``): Runs ``orig_forward``
       first, then handles ``nn.Identity`` and pass-through detection ONLY.
       Skips ``_handle_module_entry``/``_handle_module_exit`` entirely — the fast
       path doesn't track module nesting metadata. The ``torch.identity()`` call
       for nn.Identity/pass-through is still needed to keep tensor counters
       aligned with the exhaustive pass (which already ran and established the
       counter sequence).

    3. **Exhaustive mode**: Full entry/exit bookkeeping via ``_handle_module_entry``
       and ``_handle_module_exit``. Wrapped in try/except for **exception safety**:
       if ``orig_forward`` raises, the module pass label is popped from the stack
       to prevent state corruption in subsequent calls (#122).

    Args:
        orig_forward: The original ``module.forward`` method.
        module: The nn.Module instance (stable across sessions).

    Returns:
        The decorated forward function.
    """

    @wraps(orig_forward)
    def decorated_forward(*args, **kwargs):
        # ---- Toggle gate: near-zero overhead when logging is off ----
        if not _state._logging_enabled or _state._active_model_log is None:
            return orig_forward(*args, **kwargs)

        model_log = _state._active_model_log

        # ---- Fast mode: skip module entry/exit tracking ----
        # Only nn.Identity and pass-through detection is needed to keep
        # tensor counters aligned with the exhaustive pass.
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
                # Force _decorated_identity() for nn.Identity modules and pass-throughs
                # to create a new tensor entry, matching what exhaustive mode does.
                if (module.tl_module_type.lower() == "identity") or (
                    hasattr(t, "tl_tensor_label_raw")
                    and t.tl_tensor_label_raw in input_tensor_labels
                ):
                    t = _state._decorated_identity(t)
            return out

        if model_log.logging_mode == "predicate":
            from ..fastlog._predicate import _evaluate_keep_module
            from ..fastlog._record_context import _build_record_context
            from ..fastlog._state import get_active_recording_state
            from ..fastlog.types import ActivationRecord, ModuleStackFrame

            state = get_active_recording_state()
            mod_id = id(module)
            model_log._mod_pass_num[mod_id] += 1
            frame = ModuleStackFrame(
                module_address=module.tl_module_address,
                module_type=module.tl_module_type,
                module_id=mod_id,
                pass_index=model_log._mod_pass_num[mod_id],
            )
            state.module_stack.append(frame)
            state.event_index += 1
            enter_ctx = _build_record_context(
                kind="module_enter",
                layer_pass_log_or_op_data={
                    "label": f"{frame.module_address}:enter:{frame.pass_index}",
                    "module_address": frame.module_address,
                    "module_type": frame.module_type,
                    "module_pass_index": frame.pass_index,
                },
                module_stack=state.module_stack,
                history=tuple(state.history),
                op_counts=state.op_counts,
                pass_index=state.pass_index,
                event_index=state.event_index,
                op_index=None,
                time_since_pass_start=0.0,
                include_source_events=state.options.include_source_events,
                sample_id=state.sample_id,
            )
            try:
                enter_spec = _evaluate_keep_module(enter_ctx, state.options)
                if enter_spec.save_activation or enter_spec.save_metadata:
                    state.add_record(ActivationRecord(ctx=enter_ctx, spec=enter_spec))
            except Exception as exc:
                state.handle_predicate_exception(enter_ctx, exc)
            finally:
                state.append_context(enter_ctx)
            try:
                return orig_forward(*args, **kwargs)
            finally:
                state.event_index += 1
                exit_ctx = _build_record_context(
                    kind="module_exit",
                    layer_pass_log_or_op_data={
                        "label": f"{frame.module_address}:exit:{frame.pass_index}",
                        "module_address": frame.module_address,
                        "module_type": frame.module_type,
                        "module_pass_index": frame.pass_index,
                    },
                    module_stack=state.module_stack,
                    history=tuple(state.history),
                    op_counts=state.op_counts,
                    pass_index=state.pass_index,
                    event_index=state.event_index,
                    op_index=None,
                    time_since_pass_start=0.0,
                    include_source_events=state.options.include_source_events,
                    sample_id=state.sample_id,
                )
                try:
                    exit_spec = _evaluate_keep_module(exit_ctx, state.options)
                    if exit_spec.save_activation or exit_spec.save_metadata:
                        state.add_record(ActivationRecord(ctx=exit_ctx, spec=exit_spec))
                except Exception as exc:
                    state.handle_predicate_exception(exit_ctx, exc)
                finally:
                    state.append_context(exit_ctx)
                    state.module_stack.pop()

        # ---- Exhaustive mode: full entry → forward → exit ----
        input_tensor_labels, input_tensor_labels_at_entry = _handle_module_entry(
            model_log, module, args, kwargs
        )
        try:
            out = orig_forward(*args, **kwargs)
        except Exception:
            # Exception safety: pop module pass label to keep the stack
            # consistent, preventing corruption in subsequent forward calls (#122).
            mod_id = id(module)
            pass_labels = model_log._mod_pass_labels.get(mod_id)
            if pass_labels:
                pass_labels.pop()
            raise
        _handle_module_exit(
            model_log, module, out, input_tensor_labels, input_tensor_labels_at_entry
        )
        return out

    return decorated_forward


# ---------------------------------------------------------------------------
# Helper: _is_bottom_level_submodule_exit
# ---------------------------------------------------------------------------


def _is_bottom_level_submodule_exit(model_log, t: torch.Tensor, submodule: nn.Module) -> bool:
    """Check whether this tensor is exiting a "bottom-level" (leaf) submodule.

    A bottom-level submodule is one where the tensor's computation happened
    entirely within it — none of the tensor's parents were computed in a
    deeper submodule. This is used for visualization (bottom-level exits
    get special rendering) and module nesting analysis.

    **Three cases**:
    1. Already marked — return True immediately (cached from prior check).
    2. Tensor was initialized inside the model (e.g. a buffer) and no tensors
       entered this submodule — it's a leaf-generated tensor.
    3. All parent tensors' most recent ``modules_entered`` is this submodule —
       the computation stayed within this module.
    """
    layer_entry = model_log._raw_layer_dict[getattr(t, "tl_tensor_label_raw")]
    submodule_address = submodule.tl_module_address
    sub_id = id(submodule)

    # Case 1: already determined in a prior call.
    if layer_entry.is_leaf_module_output:
        return True

    # Case 2: tensor originated inside the model with no inputs entering
    # this submodule (e.g. a buffer-derived tensor in a leaf module).
    if layer_entry.is_internally_initialized and len(model_log._mod_entered[sub_id]) == 0:
        layer_entry.is_leaf_module_output = True
        layer_entry.leaf_module_pass = (
            submodule_address,
            model_log._mod_pass_num[sub_id],
        )
        return True

    # Case 3: check that ALL parent tensors last entered this exact submodule.
    # If any parent was last seen in a different (deeper) submodule, this is
    # NOT a bottom-level exit.
    for parent_label in layer_entry.parent_layers:
        parent_tensor = model_log[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != submodule_address):
            layer_entry.is_leaf_module_output = False
            return False

    layer_entry.is_leaf_module_output = True
    layer_entry.leaf_module_pass = (
        submodule_address,
        model_log._mod_pass_num[sub_id],
    )
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_all_submodules(model: nn.Module, is_top_level_model: bool = True) -> List[nn.Module]:
    """Return all modules reachable from ``model`` (including itself when top-level).

    Uses ``model.modules()`` which handles shared-module deduplication
    internally via ``id()`` checks.
    """
    return list(model.modules())


def clear_hooks(hook_handles: List):
    """Clears a list of hook handles."""
    for hook_handle in hook_handles:
        hook_handle.remove()


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


def _cleanup_model_session(model: nn.Module, input_tensors=None) -> None:
    """Clean up session-specific state after a ``log_forward_pass`` call.

    Restores ``requires_grad`` to its original value on all parameters,
    removes all session-scoped ``tl_*`` attributes from modules and params,
    and strips ``tl_tensor_label_raw`` / ``tl_buffer_address`` from tensors.

    **Does NOT** remove permanent attributes (``tl_module_address``,
    ``tl_module_type``) or unwrap ``module.forward`` — those persist for
    the lifetime of the model instance.
    """
    # Restore requires_grad and remove session-scoped param attributes
    for param in model.parameters():
        if hasattr(param, "tl_requires_grad"):
            param.requires_grad = param.tl_requires_grad
        for attr_name in _SESSION_PARAM_ATTRS:
            if hasattr(param, attr_name):
                try:
                    delattr(param, attr_name)
                except (AttributeError, RuntimeError):
                    pass

    # Session-scoped module tracking data lives in ModelLog dicts (not on
    # modules), so no per-module cleanup iteration is needed — the dicts
    # are GC'd with the ModelLog.

    # Clean tensor labels from model tensors (buffers, etc.)
    _undecorate_model_tensors(model)

    # Clean tensor labels from input tensors
    if input_tensors:
        for t in input_tensors:
            if hasattr(t, "tl_tensor_label_raw"):
                delattr(t, "tl_tensor_label_raw")


_TL_TENSOR_ATTRS = ("tl_tensor_label_raw", "tl_buffer_address", "tl_buffer_parent")


def _strip_tl_attrs(tensor):
    """Remove session-scoped tl_* attributes from a single tensor."""
    for tl_attr in _TL_TENSOR_ATTRS:
        if hasattr(tensor, tl_attr):
            delattr(tensor, tl_attr)


def _undecorate_model_tensors(model: nn.Module) -> None:
    """Remove session-scoped ``tl_*`` attributes from non-parameter tensors in the model.

    Uses ``__dict__`` scan (fast) instead of ``iter_accessible_attributes`` (slow
    dir() + getattr MRO walk). Handles tensors stored directly as attributes,
    inside lists/tuples, and inside dicts.
    """
    for submodule in model.modules():
        for attr_val in submodule.__dict__.values():
            if isinstance(attr_val, torch.Tensor):
                if not isinstance(attr_val, torch.nn.Parameter):
                    _strip_tl_attrs(attr_val)
            elif isinstance(attr_val, (list, tuple, set)):
                for item in attr_val:
                    if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                        _strip_tl_attrs(item)
            elif isinstance(attr_val, dict):
                for val in attr_val.values():
                    if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter):
                        _strip_tl_attrs(val)
    # Also clean any tensors from the registered buffer dict (_buffers)
    for submodule in model.modules():
        for buf_tensor in submodule._buffers.values():
            if buf_tensor is not None:
                _strip_tl_attrs(buf_tensor)


# ---------------------------------------------------------------------------
# Ensure model is prepared (one-time + incremental crawl)
# ---------------------------------------------------------------------------


def _ensure_model_prepared(model: nn.Module) -> None:
    """Orchestrate all one-time preparation steps before a logging session.

    Called at the start of every ``log_forward_pass``. Each step is individually
    idempotent or incremental:

    1. ``wrap_torch()`` — Ensures torch functions are wrapped (no-op if already wrapped,
       re-wraps after ``unwrap_torch()``, first-time decoration on first call).
    2. ``_prepare_model_once(model)`` — Phase 1 model prep (cached per instance).
    3. ``patch_detached_references()`` — Crawl sys.modules for stale refs
       (incremental: only scans newly-imported modules).
    4. ``patch_model_instance(model)`` — Level 4 crawl on model instance attrs.
    """
    from .torch_funcs import wrap_torch, patch_detached_references, patch_model_instance

    wrap_torch()  # idempotent — no-op if already wrapped; auto-rewraps after unwrap
    already_prepared = model in _state._prepared_models
    _prepare_model_once(model)  # idempotent — cached in _state._prepared_models
    patch_detached_references()  # incremental — only new sys.modules entries
    # Phase 1 already patches instance attrs during its DFS, so skip the
    # redundant full crawl for models that were already prepared.
    if not already_prepared:
        patch_model_instance(model)  # level 4 — model instance attrs


# Keep old names as aliases for backward compatibility during transition
prepare_model = _prepare_model_session
cleanup_model = _cleanup_model_session
