"""Model preparation: two-phase setup of nn.Modules for out logging.

This module implements the model preparation pipeline, which has two phases:

**Phase 1 — One-time preparation** (``_prepare_model_once``):
  Runs once per model instance (cached in ``_state._prepared_models`` WeakSet).
  Assigns permanent ``_tl`` module metadata and
  wraps each submodule's ``forward`` with ``module_forward_decorator``. These
  survive across multiple ``trace`` calls.

**Phase 2 — Per-session preparation** (``_prepare_model_session``):
  Runs on every ``trace`` call. Populates session-scoped tracking
  dicts on Trace (pass counters, entered/exited labels), captures module
  metadata, forces ``requires_grad=True`` on all parameters (needed for
  ``grad_fn`` metadata), creates ``Param`` objects, and tags buffer tensors.
  Session data is GC'd with the Trace — no per-module cleanup needed.

The ``module_forward_decorator`` wraps each submodule's ``forward`` with a
toggle-gated wrapper that reads ``trace`` from ``_state._active_trace``
(not closed-over). In exhaustive mode, it calls ``_record_module_entry_metadata`` /
``_record_module_exit_metadata`` to track which tensors enter and exit each module. In
fast mode, it skips entry/exit tracking entirely but still handles
``nn.Identity`` and pass-through detection via ``torch.identity``.
"""

import inspect
import itertools
import copy
import warnings
from collections.abc import Callable
from collections import defaultdict, deque
from functools import wraps
from typing import Any, TYPE_CHECKING, cast

import torch
from torch import nn

from ... import _state
from ...fastlog._halt import HaltSignal
from ._tl import (
    clear_meta,
    get_buffer_address,
    get_label_list,
    get_module_meta,
    get_param_meta,
    get_tensor_label,
    is_forward_call_decorated,
    is_tensor_replacement_wrapped,
    mark_forward_call_decorated,
    mark_tensor_replacement_wrapped,
    promote_label_to_buffer_parent_and_clear_label,
    restore_param_requires_grad,
    set_buffer_address,
    set_module_meta,
    set_param_meta,
    set_tensor_label,
)
from ...data_classes.param_log import ParamAccessor, Param
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.module_log import HookInfo
from ...data_classes._module_role_hints import multi_output_role_from_path, role_hints_for_module
from ...ir import live_record_for_label
from ...utils.tensor_utils import get_memory_amount
from ...utils.introspection import get_vars_of_type_from_obj, iter_accessible_attributes
from ...utils.hashing import make_random_barcode
from .tensor_tracking import _append_module_suffix_to_equivalence_class
from .sources import log_source_tensor
from ...constants import LAYER_PASS_LOG_FIELD_ORDER
from . import module_stack as _mstack

# Cache class-level module metadata (inspect.getsourcelines, inspect.signature, etc.)
# shared across instances of the same class type. Cleared at the start of each
# session in _prepare_model_session to avoid stale data from reloaded modules.
_module_class_metadata_cache: dict[type, dict[str, Any]] = {}

# Pre-computed set of nn.Module attribute names (from MRO). Used to filter out
# inherited custom_methods/attrs when scanning for user-defined extras. Computed once
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
    from ...data_classes.model_log import Trace


# ---------------------------------------------------------------------------
# Shared module traversal
# ---------------------------------------------------------------------------


def _module_address(module: nn.Module) -> str:
    """Return a prepared module's TorchLens address.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    str
        Prepared module address, or ``""`` for an unprepared/root fallback.
    """
    meta = get_module_meta(module)
    return "" if meta is None or meta.address is None else meta.address


def _module_type(module: nn.Module) -> str:
    """Return a prepared module's TorchLens module type.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    str
        Prepared module type, or the Python class name as a fallback.
    """
    meta = get_module_meta(module)
    return type(module).__name__ if meta is None or meta.module_type is None else meta.module_type


def _traverse_model_modules(
    model: nn.Module,
    visitor_fn: Callable[[nn.Module, str, list[tuple[str, nn.Module]], bool], None],
) -> None:
    """DFS over all modules in a model, calling ``visitor_fn`` for each.

    Visits parent before children (pre-order). The visitor receives the module,
    its dotted address, its named children list, and whether it is the root.

    Args:
        model: Root module.
        visitor_fn: Called as ``visitor_fn(module, address, named_children, is_root)``
            for every module. ``named_children`` is ``list(module.named_children())``.
    """
    traversal_queue: deque[tuple[nn.Module, str]] = deque([(model, "")])
    while traversal_queue:
        module, address = traversal_queue.popleft()
        named_children = list(module.named_children())
        child_entries = []
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_entries.append((child_module, child_address))
        # Prepend children to front of deque for DFS pre-order traversal.
        # extendleft reverses, so we reverse child_entries first to maintain order.
        for entry in reversed(child_entries):
            traversal_queue.appendleft(entry)
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

    2. **Assigns permanent metadata** — ``_tl.address`` (dotted path
       like ``"encoder.layer.0.attention"``) and ``_tl.module_type`` (class
       name). These survive across sessions.

    3. **Wraps ``forward``** — Replaces ``module.forward`` with
       ``module_forward_decorator(module.forward, module)``. The wrapper is
       toggle-gated: no-op when logging is off, full entry/exit tracking when on.
       The ``_tl.forward_call_is_decorated`` sentinel prevents double-wrapping.

    The root module is skipped for type annotation and forward wrapping because
    its forward is called directly by ``trace`` with its own
    entry/exit handling.
    """
    if model in _state._prepared_models:
        return

    set_module_meta(model, address="", module_type=str(type(model).__name__))

    def _visit_once(
        module: nn.Module,
        address: str,
        named_children: list[tuple[str, nn.Module]],
        is_root: bool,
    ) -> None:
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
            set_module_meta(
                child_module,
                address=child_address,
                module_type=str(type(child_module).__name__),
            )

        # Root module is handled separately by trace.
        if is_root:
            return

        set_module_meta(module, address=address, module_type=str(type(module).__name__))

        # Wrap forward with toggle-gated decorator (idempotent via sentinel).
        if hasattr(module, "forward") and not is_forward_call_decorated(module.forward):
            module.forward = module_forward_decorator(module.forward, module)
            mark_forward_call_decorated(module.forward)

    _traverse_model_modules(model, _visit_once)
    _state._prepared_models.add(model)


# ---------------------------------------------------------------------------
# Per-session model preparation
# ---------------------------------------------------------------------------


def _prepare_model_session(
    trace: "Trace",
    model: nn.Module,
    optimizer: Any = None,
) -> None:
    """Phase 2: Per-session model preparation, called on every ``trace``.

    Performs setup that must be fresh for each logging session:

    1. Clears metadata caches (class metadata, dir cache).
    2. Captures module metadata (source file, signatures, hooks, etc.) into
       ``trace._module_metadata``.
    3. Sets session-scoped Trace dictionaries for module pass counters and
       tensor entry/exit tracking.
    4. Creates ``Param`` objects and forces ``requires_grad=True`` on all
       parameters (needed so ``grad_fn`` chain is available for metadata).
    5. Tags buffer tensors with ``_tl.address``.

    All session-scoped state is cleaned up by ``_cleanup_model_session``.
    """
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    trace._exhaustive_module_stack = []
    trace.model_class_name = str(type(model).__name__)
    try:
        trace.model_source_file = inspect.getfile(type(model))
        trace.model_source_line = inspect.getsourcelines(type(model))[1]
    except (OSError, TypeError):
        trace.model_source_file = None
        trace.model_source_line = None
    try:
        forward_func = model.forward
        trace.forward_source_file = inspect.getsourcefile(forward_func) or inspect.getfile(
            forward_func
        )
        trace.forward_source_line = inspect.getsourcelines(forward_func)[1]
    except (OSError, TypeError):
        trace.forward_source_file = None
        trace.forward_source_line = None

    # Track seen module ids to detect shared modules (same module at multiple addresses).
    _seen_module_ids: dict[int, str] = {}

    # Use model.modules() + cached module addresses from phase 1, avoiding a
    # second full DFS with string concatenation and list(named_children()) calls.
    for module in model.modules():
        is_root = module is model
        address = _module_address(module)
        named_children = list(module.named_children())
        _capture_module_metadata(
            trace,
            module,
            address,
            named_children,
            _seen_module_ids,
            is_root=is_root,
        )
        if not is_root:
            trace._module_build_data["module_types"][address] = _module_type(module)
            # Session-scoped tracking in Trace dicts (keyed by id(module)).
            mod_id = id(module)
            trace._mod_call_index[mod_id] = 0
            trace._mod_call_labels[mod_id] = []
            trace._mod_entered[mod_id] = []
            trace._mod_exited[mod_id] = []
        _wrap_user_forward_hooks(module)
    if trace.capture_mode != "predicate":
        _create_session_param_logs(trace, model, optimizer)
    prepare_buffer_tensors(trace, model)


def _create_session_param_logs(trace: "Trace", model: nn.Module, optimizer: Any = None) -> None:
    """Create ``Param`` objects and prepare parameter grad tracking.

    Outside ``train_mode``, ``requires_grad`` is forced True so that ``grad_fn``
    metadata is available on all intermediate tensors during the forward pass.
    In ``train_mode``, user-authored ``requires_grad`` values are preserved. The
    original value is always saved to ``_tl.requires_grad_before_capture`` and restored during
    ``_cleanup_model_session``.
    """
    if not hasattr(trace, "_param_log_by_pid"):
        raise AttributeError("Trace._param_log_by_pid must be initialized before param logging.")

    optimized_param_ids: set[int] = set()
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimized_param_ids.add(id(p))

    param_logs: dict[str, Param] = {}
    seen_param_ids: set[int] = set()
    param_id_to_address: dict[int, str] = {}
    for module in model.modules():
        address = _module_address(module)
        module_type = type(module).__name__
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            # Shared parameters: only create one Param per unique tensor.
            pid = id(param)
            if pid in seen_param_ids:
                existing_address = param_id_to_address[pid]
                alias_address = f"{address}.{param_name}" if address else param_name
                if alias_address not in param_logs[existing_address].co_parent_params:
                    param_logs[existing_address].co_parent_params.append(alias_address)
                continue
            seen_param_ids.add(pid)

            module_address = address or "self"
            param_address = f"{address}.{param_name}" if address else param_name
            param_id_to_address[pid] = param_address

            # Save original requires_grad before forcing True.
            requires_grad_before = param.requires_grad
            if not getattr(trace, "train_mode", False):
                param.requires_grad = True

            barcode = make_random_barcode()
            set_param_meta(
                param,
                barcode=barcode,
                address=param_address,
                requires_grad_before=requires_grad_before,
            )

            param_fsize = get_memory_amount(param)
            param_log = Param(
                module_address=module_address,
                name=param_name,
                shape=tuple(param.shape),
                dtype=param.dtype,
                num_params=param.numel(),
                memory=param_fsize,
                trainable=requires_grad_before,
                address=param_address,
                module_type=module_type,
                barcode=barcode,
                has_optimizer=id(param) in optimized_param_ids if optimizer is not None else None,
            )
            param_log._param_ref = param
            param_logs[param_address] = param_log

    trace._param_log_by_pid = param_id_to_address
    trace.param_logs = ParamAccessor(param_logs)


# ---------------------------------------------------------------------------
# Module metadata capture (unchanged from before)
# ---------------------------------------------------------------------------


def _get_class_metadata(module_class: type, save_code_context: bool = False) -> dict[str, Any]:
    """Return class-level metadata for a module class, cached across instances.

    When ``save_code_context`` is False (default), skips expensive
    ``inspect.getsourcelines`` and ``inspect.signature`` calls. Only class
    name and docstrings (already in memory) are captured.

    When True, also fetches source file/line and signatures. Cached per class
    type to avoid redundant filesystem reads.
    """
    cached = _module_class_metadata_cache.get(module_class)
    if cached is not None:
        return cached

    meta: dict[str, Any] = {}
    meta["class_name"] = module_class.__name__
    meta["class_qualname"] = f"{module_class.__module__}.{module_class.__qualname__}"
    meta["cls"] = module_class
    meta["class_docstring"] = module_class.__doc__

    # Cache user-defined custom_methods from class __dict__ (same for all instances of this class).
    user_custom_methods = []
    for attr_name in module_class.__dict__:
        if attr_name.startswith("_") or attr_name.startswith("tl_"):
            continue
        if attr_name in _PYTORCH_INTERNAL or attr_name in _NN_MODULE_ATTRS:
            continue
        val = module_class.__dict__[attr_name]
        if callable(val):
            user_custom_methods.append(attr_name)
    meta["user_custom_methods"] = user_custom_methods

    if save_code_context:
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


def _hook_info_from_registry(registry: Any) -> HookInfo:
    """Build HookInfo for a PyTorch module hook registry.

    Parameters
    ----------
    registry:
        PyTorch hook registry mapping handle ids to callables.

    Returns
    -------
    HookInfo
        Portable hook metadata summary.
    """

    hooks = list(registry.values())
    names: list[str] = []
    qualnames: list[str] = []
    source_locations: list[FuncCallLocation] = []
    for hook in hooks:
        names.append(getattr(hook, "__name__", type(hook).__name__))
        qualname = getattr(hook, "__qualname__", names[-1])
        module_name = getattr(hook, "__module__", "")
        qualnames.append(f"{module_name}.{qualname}" if module_name else qualname)
        try:
            source_file = inspect.getsourcefile(hook) or inspect.getfile(hook)
            source_line = inspect.getsourcelines(hook)[1]
        except (OSError, TypeError):
            continue
        source_locations.append(
            FuncCallLocation(
                file=source_file,
                line_number=source_line,
                func_name=qualnames[-1],
                source_loading_enabled=False,
            )
        )
    return HookInfo(
        count=len(hooks),
        names=names,
        qualnames=qualnames,
        source_locations=source_locations,
    )


def _capture_module_metadata(
    trace: "Trace",
    module: nn.Module,
    parent_address: str,
    module_children: list[tuple[str, nn.Module]],
    seen_module_ids: dict[int, str],
    is_root: bool = False,
) -> None:
    """Capture live module metadata during ``_prepare_model_session``.

    Records source file/line, signatures, docstrings, hooks, training mode,
    child addresses, user-defined attributes/custom_methods, and more. Must be called
    after permanent module metadata has been assigned.

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
        if primary in trace._module_metadata:
            trace._module_metadata[primary]["all_addresses"].append(address)
        return
    seen_module_ids[module_id] = address

    # Start from cached class-level metadata. dict() creates a shallow copy;
    # mutable fields (all_addresses, custom_attributes, custom_methods) are replaced
    # below with fresh instances per module, so no cross-contamination.
    save_source = getattr(trace, "save_code_context", False)
    class_meta = _get_class_metadata(type(module), save_code_context=save_source)
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
    meta["forward_pre_hook_info"] = _hook_info_from_registry(
        getattr(module, "_forward_pre_hooks", {})
    )
    meta["forward_hook_info"] = _hook_info_from_registry(getattr(module, "_forward_hooks", {}))
    meta["backward_pre_hook_info"] = _hook_info_from_registry(
        getattr(module, "_backward_pre_hooks", {})
    )
    meta["backward_hook_info"] = _hook_info_from_registry(getattr(module, "_backward_hooks", {}))
    meta["full_backward_pre_hook_info"] = _hook_info_from_registry(
        getattr(module, "_full_backward_pre_hooks", {})
    )
    meta["full_backward_hook_info"] = _hook_info_from_registry(
        getattr(module, "_full_backward_hooks", {})
    )
    meta["is_train_mode"] = module.training

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
    meta["custom_attributes"] = extra_attrs
    # User-defined custom_methods are cached per class type in _get_class_metadata.
    meta["custom_methods"] = class_meta["user_custom_methods"]

    trace._module_metadata[address] = meta


# ---------------------------------------------------------------------------
# Buffer tensor preparation
# ---------------------------------------------------------------------------


def prepare_buffer_tensors(trace: "Trace", model: nn.Module) -> None:
    """Tag buffer tensors with ``_tl.address`` for later identification.

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
        module_addr = _module_address(submodule)
        # Scan registered buffers
        for buf_name, buf_tensor in submodule.named_buffers(recurse=False):
            if (
                isinstance(buf_tensor, torch.Tensor)
                and not isinstance(buf_tensor, torch.nn.Parameter)
                and get_buffer_address(buf_tensor) is None
            ):
                address = f"{module_addr}.{buf_name}" if module_addr else buf_name
                try:
                    set_buffer_address(buf_tensor, address)
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
                and get_buffer_address(attr_val) is None
            ):
                address = f"{module_addr}.{attr_name}" if module_addr else attr_name
                try:
                    set_buffer_address(attr_val, address)
                    _state._tagged_buffer_ids.add(id(attr_val))
                except Exception:
                    pass
            elif isinstance(attr_val, (list, tuple)):
                for i, item in enumerate(attr_val):
                    if (
                        isinstance(item, torch.Tensor)
                        and not isinstance(item, torch.nn.Parameter)
                        and get_buffer_address(item) is None
                    ):
                        item_addr = (
                            f"{module_addr}.{attr_name}.{i}" if module_addr else f"{attr_name}.{i}"
                        )
                        try:
                            set_buffer_address(item, item_addr)
                            _state._tagged_buffer_ids.add(id(item))
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# Module forward decorator — reads trace from _state
# ---------------------------------------------------------------------------


def _tag_untagged_buffers(module: nn.Module) -> None:
    """Tag any buffers that lack ``_tl.address`` metadata.

    Called during ``_record_module_entry_metadata`` to catch buffers that were created
    dynamically (e.g. in ``forward()``) after the initial ``prepare_buffer_tensors``
    scan. If a buffer already has ``_tl.label_raw`` from being logged as
    an intermediate tensor, that label is moved to ``_tl.buffer_parent`` and cleared
    so the buffer gets a fresh source-tensor entry on next use.
    """
    for buffer_name, buffer_tensor in module.named_buffers():
        if get_buffer_address(buffer_tensor) is not None:
            continue
        module_addr = _module_address(module)
        if module_addr == "":
            address = buffer_name
        else:
            address = f"{module_addr}.{buffer_name}"
        set_buffer_address(buffer_tensor, address)
        # If this buffer was already logged as an intermediate tensor, save the
        # old label as parent and reset so it gets a proper buffer source entry.
        promote_label_to_buffer_parent_and_clear_label(buffer_tensor)


def _record_module_entry_metadata(
    trace: "Trace",
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[set[str], list[str]]:
    """Record pre-forward module metadata for exhaustive mode.

    Called immediately before ``orig_forward(*args, **kwargs)`` in the
    ``module_forward_decorator``. Reads the module's pass counter, records
    input-tensor/module-entry annotations, stores raw forward args for module-log
    construction, and tags dynamically-created buffers.

    Args:
        trace: The active Trace.
        module: The nn.Module about to execute.
        args: Positional arguments to forward.
        kwargs: Keyword arguments to forward.

    Returns:
        Tuple of ``(input_tensor_labels, input_tensor_labels_at_entry)`` —
        needed by ``_record_module_exit_metadata`` for pass-through detection and
        replacement-output recovery.
    """
    address = _module_address(module)
    mod_id = id(module)
    trace._module_build_data["module_training_modes"][address] = module.training
    module_call_index = trace._mod_call_index[mod_id]
    assert module_call_index > 0, "_module_stack.push_frame must increment before entry"
    module_call_label = (address, module_call_index)
    # Push onto stack — popped by _record_module_exit_metadata (or exception handler).
    trace._mod_call_labels[mod_id].append(module_call_label)

    # Stash forward args for later use by _build_module_logs.
    trace._module_forward_args[(address, module_call_index)] = (args, kwargs)

    # Find all tensor arguments (excluding Parameters, which are source tensors).
    input_tensors = get_vars_of_type_from_obj(
        [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
    )
    input_tensor_labels = set()
    input_tensor_labels_at_entry = []
    for t in input_tensors:
        # Lazily register buffer tensors that haven't been logged yet.
        label = get_tensor_label(t)
        address = get_buffer_address(t)
        if label is None and address is not None:
            log_source_tensor(trace, t, "buffer", address)
            label = get_tensor_label(t)
        if label is None:
            # Raw ``register_forward_hook`` replacements run after this
            # module-forward wrapper has already handled the replaced module's
            # exit, so the first recoverable point may be the next downstream
            # module entry.
            capture_events = getattr(trace, "capture_events", None)
            parent_labels = (
                [capture_events.op_events[-1].label_raw]
                if capture_events is not None and capture_events.op_events
                else trace._raw_layer_labels_list[-1:]
            )
            _ensure_module_output_tensor_logged(trace, t, module, parent_labels)
            label = get_tensor_label(t)
        if label is None:
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        layer_entry = live_record_for_label(trace, label).fields
        input_tensor_labels.add(label)
        trace._mod_entered[mod_id].append(label)
        layer_entry["modules_entered"].append(address)
        layer_entry["module_ops_entered"].append(module_call_label)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                layer_entry["module_entry_argnames"][
                    f"{module_call_label[0]}:{module_call_label[1]}"
                ].append(arg_key)
                trace._module_build_data["module_layer_argnames"][
                    (f"{module_call_label[0]}:{module_call_label[1]}")
                ].append((label, arg_key))
        input_tensor_labels_at_entry.append(label)

    # Catch buffers created dynamically (e.g. in forward()) after initial scan.
    _tag_untagged_buffers(module)
    return input_tensor_labels, input_tensor_labels_at_entry


def _next_intervention_replacement_label(trace: "Trace") -> tuple[str, int, int]:
    """Return a fresh raw label for a user-injected module output tensor.

    Parameters
    ----------
    trace:
        Active model log whose raw layer counters should be advanced.

    Returns
    -------
    tuple[str, int, int]
        Raw label, capture index, and per-type index.
    """

    layer_type = "interventionreplacement"
    trace._layer_counter += 1
    trace._raw_layer_type_counter[layer_type] += 1
    capture_index = trace._layer_counter
    type_index = trace._raw_layer_type_counter[layer_type]
    return f"{layer_type}_{type_index}_{capture_index}_raw", capture_index, type_index


def _copy_field_value_for_replacement(value: Any) -> Any:
    """Copy mutable Op field values without cloning tensors.

    Parameters
    ----------
    value:
        Field value from a parent layer entry.

    Returns
    -------
    Any
        A structurally independent copy for container fields, or the original
        immutable/scalar/tensor reference otherwise.
    """

    if isinstance(value, (list, dict, set, defaultdict)):
        return copy.copy(value)
    return value


def _ensure_module_output_tensor_logged(
    trace: "Trace",
    tensor: torch.Tensor,
    module: nn.Module,
    parent_labels: list[str],
) -> None:
    """Log a fresh entry for an unlabeled tensor returned from a module hook.

    Parameters
    ----------
    trace:
        Active model log.
    tensor:
        Replacement output tensor that lacks ``_tl.label_raw``.
    module:
        Module whose forward return value was replaced.
    parent_labels:
        Raw labels for tensors that entered the module.

    Returns
    -------
    None
        The tensor is tagged and a replacement Op is inserted into the raw
        graph so downstream module-exit and op logging can continue.
    """

    from .ops import _make_layer_log_entry

    parent_entries = [live_record_for_label(trace, label).fields for label in parent_labels]
    template_entry = parent_entries[0] if parent_entries else None
    raw_label, capture_index, type_index = _next_intervention_replacement_label(trace)
    fields_dict = {
        field_name: _copy_field_value_for_replacement(
            template_entry.get(field_name) if template_entry is not None else None
        )
        for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
    address = _module_address(module)
    module_call_index = trace._mod_call_index[id(module)]
    modules = [(address, module_call_index)] if address else []
    equivalence_class = _append_module_suffix_to_equivalence_class(raw_label, modules)
    root_ancestors: set[str] = set()
    input_ancestors: set[str] = set()
    internal_source_ancestors: set[str] = set()
    for parent_entry in parent_entries:
        root_ancestors.update(parent_entry["root_ancestors"] or set())
        input_ancestors.update(parent_entry["input_ancestors"] or set())
        internal_source_ancestors.update(parent_entry["internal_source_ancestors"] or set())

    fields_dict.update(
        {
            "_label_raw": raw_label,
            "_layer_label_raw": raw_label,
            "capture_index": capture_index,
            "compute_index": None,
            "source_trace": trace,
            "_tracing_finished": False,
            "_construction_done": False,
            "layer_label": None,
            "layer_label_short": None,
            "layer_label_w_pass": None,
            "layer_label_w_pass_short": None,
            "layer_label_no_pass": None,
            "layer_label_no_pass_short": None,
            "type": "interventionreplacement",
            "type_index": type_index,
            "trace_index": None,
            "call_index": 1,
            "num_calls": 1,
            "lookup_keys": [],
            "out": None,
            "transformed_out": None,
            "has_saved_outs": False,
            "out_postfunc": trace.out_postfunc,
            "annotations": {},
            "interventions": [],
            "intervention_replaced": True,
            "detach_saved_activations": trace.detach_saved_activations,
            "output_device": trace.output_device,
            "has_saved_args": False,
            "saved_args": None,
            "saved_kwargs": None,
            "args_template": None,
            "kwargs_template": None,
            "shape": tuple(tensor.shape),
            "transformed_out_shape": None,
            "dtype": tensor.dtype,
            "transformed_out_dtype": None,
            "memory": get_memory_amount(tensor),
            "transformed_out_memory": None,
            "visualizer_path": None,
            "bytes_delta_at_call": 0,
            "bytes_peak_at_call": 0,
            "autograd_saved_memory": None,
            "num_autograd_saved_tensors": None,
            "has_output_variations": False,
            "output_versions_per_child": {},
            "grad": None,
            "transformed_grad": None,
            "save_grads": trace.save_grads,
            "has_grad": False,
            "grad_shape": None,
            "transformed_grad_shape": None,
            "grad_dtype": None,
            "transformed_grad_dtype": None,
            "grad_memory": 0,
            "transformed_grad_memory": None,
            "func": None,
            "func_call_id": None,
            "func_name": "intervention_replacement",
            "func_qualname": None,
            "code_context": [],
            "func_duration": 0,
            "flops_forward": 0,
            "flops_backward": 0,
            "func_rng_states": {},
            "func_autocast_state": {},
            "arg_names": (),
            "num_args_total": 0,
            "num_pos_args": 0,
            "num_kwargs": 0,
            "non_tensor_pos_args": [],
            "non_tensor_kwargs": {},
            "func_non_tensor_args": [],
            "is_inplace": False,
            "grad_fn_class_name": type(tensor.grad_fn).__name__
            if tensor.grad_fn is not None
            else None,
            "grad_fn_class_qualname": (
                f"{type(tensor.grad_fn).__module__}.{type(tensor.grad_fn).__qualname__}"
                if tensor.grad_fn is not None
                else None
            ),
            "grad_fn_id": id(tensor.grad_fn) if tensor.grad_fn is not None else None,
            "grad_fn": tensor.grad_fn,
            "grad_fn_log": None,
            "is_part_of_iterable_output": False,
            "multi_output_index": None,
            "multi_output_role": None,
            "container_path": (),
            "container_spec": None,
            "parent_params": [],
            "_param_barcodes": [],
            "parent_param_ops": {},
            "_param_logs": [],
            "param_shapes": [],
            "num_params": 0,
            "num_params_trainable": 0,
            "num_params_frozen": 0,
            "param_memory": 0,
            "equivalence_class": equivalence_class,
            "equivalent_ops": trace.equivalent_ops[raw_label],
            "recurrent_ops": [],
            "parents": parent_labels,
            "parent_arg_positions": {"args": {}, "kwargs": {}},
            "edge_uses": [],
            "root_ancestors": root_ancestors or {raw_label},
            "children": [],
            "has_children": False,
            "is_input": False,
            "has_input_ancestor": any(entry["has_input_ancestor"] for entry in parent_entries),
            "input_ancestors": input_ancestors,
            "min_distance_from_input": None,
            "max_distance_from_input": None,
            "is_output": False,
            "is_output_parent": False,
            "is_final_output": False,
            "has_output_descendant": False,
            "output_descendants": set(),
            "min_distance_from_output": None,
            "max_distance_from_output": None,
            "io_role": None,
            "is_buffer": False,
            "address": None,
            "buffer_pass": None,
            "buffer_parent": None,
            "is_internal_source": not parent_entries,
            "has_internal_source_ancestor": any(
                entry["has_internal_source_ancestor"] for entry in parent_entries
            ),
            "internal_source_parents": [],
            "internal_source_ancestors": internal_source_ancestors,
            "is_internal_sink": False,
            "is_terminal_bool": False,
            "is_terminal_conditional_bool": False,
            "conditional_context_kind": None,
            "conditional_wrapper_kind": None,
            "terminal_conditional_id": None,
            "is_scalar_bool": bool(tensor.dtype == torch.bool and tensor.dim() == 0),
            "bool_value": None,
            "in_conditionals": [],
            "terminal_bool_for": None,
            "is_in_conditional_body": False,
            "conditional_branch_stack": [],
            "conditional_branch_depth": 0,
            "conditional_entry_children": [],
            "conditional_then_children": [],
            "conditional_elif_children": {},
            "conditional_else_children": [],
            "conditional_arm_children": {},
            "module": (address, module_call_index),
            "_address_normalized": None,
            "modules": modules,
            "modules_entered": [],
            "module_entry_argnames": defaultdict(list),
            "module_ops_entered": [],
            "output_of_modules": [],
            "output_of_module_calls": [],
            "is_submodule_output": False,
            "is_atomic_module_op": False,
            "atomic_module_call": None,
            "func_config": {},
        }
    )
    trace.equivalent_ops[raw_label].add(raw_label)
    new_entry = _make_layer_log_entry(trace, tensor, fields_dict, (), {}, trace.out_postfunc)
    for parent_entry in parent_entries:
        parent_entry["children"].append(raw_label)
        parent_entry["has_children"] = True
    set_tensor_label(tensor, new_entry._label_raw)
    if trace.save_grads:
        from .ops import _add_tensor_backward_hook

        _add_tensor_backward_hook(trace, tensor, new_entry._label_raw)


def _wrap_user_forward_hooks(module: nn.Module) -> None:
    """Wrap raw PyTorch forward hooks so tensor replacements stay logged.

    Parameters
    ----------
    module:
        Module whose registered forward hooks should be normalized.

    Returns
    -------
    None
        Hook callables in ``module._forward_hooks`` are replaced in place once.
    """

    hooks = getattr(module, "_forward_hooks", None)
    if not hooks:
        return
    for hook_id, hook_fn in list(hooks.items()):
        if is_tensor_replacement_wrapped(hook_fn):
            continue
        hooks[hook_id] = _make_user_forward_hook_wrapper(module, hook_fn)


def _make_user_forward_hook_wrapper(
    module: nn.Module, hook_fn: Callable[..., Any]
) -> Callable[..., Any]:
    """Return a forward-hook wrapper that instruments replacement tensors.

    Parameters
    ----------
    module:
        Module that owns the hook.
    hook_fn:
        User-supplied PyTorch forward hook.

    Returns
    -------
    Callable[..., Any]
        Wrapped hook preserving the original return value.
    """

    @wraps(hook_fn)
    def wrapped_hook(*hook_args: Any, **hook_kwargs: Any) -> Any:
        """Run a raw forward hook and repair TorchLens metadata on replacements."""

        original_output = hook_args[-1] if hook_args else None
        result = hook_fn(*hook_args, **hook_kwargs)
        if result is None or result is original_output:
            return result
        trace = _state._active_trace
        if trace is None or not _state._logging_enabled:
            return result
        parent_labels = [
            label
            for tensor in get_vars_of_type_from_obj(original_output, torch.Tensor, search_depth=4)
            if (label := get_tensor_label(tensor)) is not None
        ]
        for replacement in get_vars_of_type_from_obj(result, torch.Tensor, search_depth=4):
            replacement_label = get_tensor_label(replacement)
            if replacement_label is not None:
                live_record_for_label(trace, replacement_label).fields["intervention_replaced"] = (
                    True
                )
            else:
                _ensure_module_output_tensor_logged(trace, replacement, module, parent_labels)
        return result

    mark_tensor_replacement_wrapped(wrapped_hook)
    return wrapped_hook


def _record_module_exit_metadata(
    trace: "Trace",
    module: nn.Module,
    out: Any,
    input_tensor_labels: set[str],
    input_tensor_labels_at_entry: list[str],
) -> None:
    """Record post-forward module metadata for exhaustive mode.

    Called immediately after ``orig_forward()`` returns in the
    ``module_forward_decorator``. Pops the module call-label stack, creates
    boundary identity ops for pass-through outputs, recovers replacement outputs,
    and annotates output tensors with module-exit metadata.
    """
    address = _module_address(module)
    mod_id = id(module)
    module_call_index = trace._mod_call_index[mod_id]
    trace._mod_call_labels[mod_id].pop()
    from .ops import _walk_output_tensors_with_paths

    output_entries = list(_walk_output_tensors_with_paths(out))
    output_tensors = [entry[0] for entry in output_entries]
    if not output_tensors:
        output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
        output_entries = [(tensor, (), None) for tensor in output_tensors]
    role_hints = role_hints_for_module(module)
    module_call_label = f"{address}:{module_call_index}"
    if output_entries:
        trace._module_build_data.setdefault("module_output_structures", {})[module_call_label] = (
            output_entries[0][2]
        )
    for output_index, (t, container_path, _container_spec) in enumerate(output_entries):
        # nn.Identity modules and pass-through tensors (output is same object
        # as input) need _decorated_identity() to create a distinct log entry
        # so the graph correctly shows the module boundary.
        tensor_label = get_tensor_label(t)
        if (_module_type(module).lower() == "identity") or (
            tensor_label is not None and tensor_label in input_tensor_labels
        ):
            t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            tensor_label = get_tensor_label(t)
        if tensor_label is None:
            parent_labels = list(dict.fromkeys(input_tensor_labels_at_entry))
            _ensure_module_output_tensor_logged(trace, t, module, parent_labels)
            tensor_label = get_tensor_label(t)
        if tensor_label is None:
            continue
        layer_entry = live_record_for_label(trace, tensor_label).fields
        if getattr(module, "_forward_hooks", None):
            layer_entry["intervention_replaced"] = True
        layer_entry["is_submodule_output"] = True
        layer_entry["is_atomic_module_op"] = _is_bottom_level_submodule_exit(trace, t, module)
        layer_entry["output_of_modules"].append(address)
        layer_entry["output_of_module_calls"].append((address, module_call_index))
        if len(output_entries) > 1:
            layer_entry["multi_output_role"] = multi_output_role_from_path(
                container_path,
                output_index,
                hints=role_hints,
            )
        trace._mod_exited[mod_id].append(tensor_label)


def module_forward_decorator(
    orig_forward: Callable[..., Any], module: nn.Module
) -> Callable[..., Any]:
    """Toggle-gated forward wrapper for an nn.Module's ``forward`` method.

    **Closure design**: Closes over ``module`` (a stable instance reference) but
    reads ``trace`` from ``_state._active_trace`` at call time. This is
    necessary because the same wrapper persists across multiple ``trace``
    calls with different Trace instances.

    **Three execution modes**:

    1. **Logging off** (``_state._logging_enabled is False``): Pass through to
       ``orig_forward`` with zero overhead beyond one bool check. This is the
       normal production path.

    2. **Fast mode** (``trace.capture_mode == "fast"``): Runs ``orig_forward``
       first, then handles ``nn.Identity`` and pass-through detection ONLY.
       Skips ``_record_module_entry_metadata``/``_record_module_exit_metadata`` entirely — the fast
       path doesn't track module nesting metadata. The ``torch.identity()`` call
       for nn.Identity/pass-through is still needed to keep tensor counters
       aligned with the exhaustive pass (which already ran and established the
       counter sequence).

    3. **Exhaustive mode**: Full entry/exit bookkeeping via
       ``_record_module_entry_metadata`` and ``_record_module_exit_metadata``.
       Wrapped in try/except for **exception safety**:
       if ``orig_forward`` raises, the module pass label is popped from the stack
       to prevent state corruption in subsequent calls (#122).

    Args:
        orig_forward: The original ``module.forward`` method.
        module: The nn.Module instance (stable across sessions).

    Returns:
        The decorated forward function.
    """

    @wraps(orig_forward)
    def decorated_forward(*args: Any, **kwargs: Any) -> Any:
        # ---- Toggle gate: near-zero overhead when logging is off ----
        if not _state._logging_enabled or _state._active_trace is None:
            return orig_forward(*args, **kwargs)

        trace = _state._active_trace

        # ---- Fast mode: skip module entry/exit tracking ----
        # Only nn.Identity and pass-through detection is needed to keep
        # tensor counters aligned with the exhaustive pass.
        if trace.capture_mode == "fast":
            input_tensors_fast = get_vars_of_type_from_obj(
                [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
            )
            input_tensor_labels = set(get_label_list(input_tensors_fast))
            out = orig_forward(*args, **kwargs)
            output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
            for t in output_tensors:
                # Force _decorated_identity() for nn.Identity modules and pass-throughs
                # to create a new tensor entry, matching what exhaustive mode does.
                tensor_label = get_tensor_label(t)
                if (_module_type(module).lower() == "identity") or (
                    tensor_label is not None and tensor_label in input_tensor_labels
                ):
                    t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            return out

        if trace.capture_mode == "predicate":
            from ...capture.predicates import _evaluate_keep_module
            from ...capture.projections import (
                _build_record_context,
                append_projected_event,
                get_active_recording_state,
            )
            from ...fastlog.types import ActivationRecord, CaptureSpec

            state = get_active_recording_state()
            frame = _mstack.push_frame(trace, state.module_stack, module)
            state.event_index += 1
            enter_ctx = _build_record_context(
                kind="module_enter",
                op_log_or_op_data={
                    "label": f"{frame.address}:enter:{frame.pass_index}",
                    "address": frame.address,
                    "module_type": frame.module_type,
                    "module_pass_index": frame.pass_index,
                },
                module_stack=state.module_stack,
                history=tuple(state.history),
                op_counts=state.op_counts,
                pass_index=state.pass_index,
                event_index=state.event_index,
                compute_index=None,
                time_since_pass_start=0.0,
                include_source_events=state.options.include_source_events,
                sample_id=state.sample_id,
            )
            skipped_spec = CaptureSpec(save_out=False, save_metadata=False)
            enter_spec = skipped_spec
            try:
                enter_spec = _evaluate_keep_module(enter_ctx, state.options)
                if enter_spec.save_out or enter_spec.save_metadata:
                    if state.storage_intent.on_disk:
                        state.add_record(ActivationRecord(ctx=enter_ctx, spec=enter_spec))
                append_projected_event(
                    trace,
                    enter_ctx,
                    enter_spec,
                    predicate_matched=enter_spec.save_out or enter_spec.save_metadata,
                )
            except HaltSignal:
                _mstack.pop_frame(state.module_stack, frame)
                raise
            except Exception as exc:
                state.handle_predicate_exception(enter_ctx, exc)
            finally:
                if not any(
                    event.capture_index == enter_ctx.event_index
                    for event in trace.capture_events.op_events
                ):
                    append_projected_event(
                        trace,
                        enter_ctx,
                        skipped_spec,
                        predicate_matched=False,
                    )
                state.append_context(enter_ctx)
            try:
                return orig_forward(*args, **kwargs)
            finally:
                state.event_index += 1
                exit_ctx = _build_record_context(
                    kind="module_exit",
                    op_log_or_op_data={
                        "label": f"{frame.address}:exit:{frame.pass_index}",
                        "address": frame.address,
                        "module_type": frame.module_type,
                        "module_pass_index": frame.pass_index,
                    },
                    module_stack=state.module_stack,
                    history=tuple(state.history),
                    op_counts=state.op_counts,
                    pass_index=state.pass_index,
                    event_index=state.event_index,
                    compute_index=None,
                    time_since_pass_start=0.0,
                    include_source_events=state.options.include_source_events,
                    sample_id=state.sample_id,
                )
                exit_spec = skipped_spec
                try:
                    exit_spec = _evaluate_keep_module(exit_ctx, state.options)
                    if exit_spec.save_out or exit_spec.save_metadata:
                        if state.storage_intent.on_disk:
                            state.add_record(ActivationRecord(ctx=exit_ctx, spec=exit_spec))
                    append_projected_event(
                        trace,
                        exit_ctx,
                        exit_spec,
                        predicate_matched=exit_spec.save_out or exit_spec.save_metadata,
                    )
                except HaltSignal:
                    raise
                except Exception as exc:
                    state.handle_predicate_exception(exit_ctx, exc)
                finally:
                    if not any(
                        event.capture_index == exit_ctx.event_index
                        for event in trace.capture_events.op_events
                    ):
                        append_projected_event(
                            trace,
                            exit_ctx,
                            skipped_spec,
                            predicate_matched=False,
                        )
                    state.append_context(exit_ctx)
                    _mstack.pop_frame(state.module_stack, frame)

        # ---- Exhaustive mode: full entry -> forward -> exit ----
        frame = _mstack.push_frame(trace, trace._exhaustive_module_stack, module)
        try:
            input_tensor_labels, input_tensor_labels_at_entry = _record_module_entry_metadata(
                trace, module, args, kwargs
            )
            try:
                out = orig_forward(*args, **kwargs)
            except Exception:
                # Exception safety: pop module pass label to keep the stack
                # consistent, preventing corruption in subsequent forward calls (#122).
                mod_id = id(module)
                call_labels = trace._mod_call_labels.get(mod_id)
                if call_labels:
                    call_labels.pop()
                raise
            _record_module_exit_metadata(
                trace, module, out, input_tensor_labels, input_tensor_labels_at_entry
            )
            return out
        finally:
            _mstack.pop_frame(trace._exhaustive_module_stack, frame)

    return decorated_forward


# ---------------------------------------------------------------------------
# Helper: _is_bottom_level_submodule_exit
# ---------------------------------------------------------------------------


def _is_bottom_level_submodule_exit(trace: "Trace", t: torch.Tensor, submodule: nn.Module) -> bool:
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
    tensor_label = get_tensor_label(t)
    if tensor_label is None:
        raise KeyError("Tensor is missing TorchLens metadata")
    layer_entry = live_record_for_label(trace, tensor_label).fields
    subaddress = _module_address(submodule)
    sub_id = id(submodule)

    # Case 1: already determined in a prior call.
    if layer_entry["is_atomic_module_op"]:
        return True

    # Case 2: tensor originated inside the model with no inputs entering
    # this submodule (e.g. a buffer-derived tensor in a leaf module).
    if layer_entry["is_internal_source"] and len(trace._mod_entered[sub_id]) == 0:
        layer_entry["is_atomic_module_op"] = True
        layer_entry["atomic_module_call"] = (
            subaddress,
            trace._mod_call_index[sub_id],
        )
        return True

    # Case 3: check that ALL parent tensors last entered this exact submodule.
    # If any parent was last seen in a different (deeper) submodule, this is
    # NOT a bottom-level exit.
    for parent_label in layer_entry["parents"]:
        parent_tensor = live_record_for_label(trace, parent_label).fields
        parent_modules_entered = parent_tensor["modules_entered"]
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != subaddress):
            layer_entry["is_atomic_module_op"] = False
            return False

    layer_entry["is_atomic_module_op"] = True
    layer_entry["atomic_module_call"] = (
        subaddress,
        trace._mod_call_index[sub_id],
    )
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_all_submodules(model: nn.Module, is_top_level_model: bool = True) -> list[nn.Module]:
    """Return all modules reachable from ``model`` (including itself when top-level).

    Uses ``model.modules()`` which handles shared-module deduplication
    internally via ``id()`` checks.
    """
    return list(model.modules())


def clear_hooks(hook_handles: list[Any]) -> None:
    """Clears a list of hook handles."""
    for hook_handle in hook_handles:
        hook_handle.remove()


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------


def _cleanup_model_session(model: nn.Module, input_tensors: Any = None) -> None:
    """Clean up session-specific state after a ``trace`` call.

    Restores ``requires_grad`` to its original value on all parameters,
    removes all session-scoped parameter metadata, and strips session-scoped
    tensor metadata.

    **Does NOT** remove permanent module metadata or unwrap ``module.forward`` — those persist for
    the lifetime of the model instance.
    """
    # Restore requires_grad and remove session-scoped param attributes
    for param in model.parameters():
        restore_param_requires_grad(param)
        clear_meta(param)

    # Session-scoped module tracking data lives in Trace dicts (not on
    # modules), so no per-module cleanup iteration is needed — the dicts
    # are GC'd with the Trace.

    # Clean tensor labels from model tensors (buffers, etc.)
    _undecorate_model_tensors(model)

    # Clean tensor labels from input tensors
    if input_tensors:
        for t in input_tensors:
            clear_meta(t)


def _undecorate_model_tensors(model: nn.Module) -> None:
    """Remove session-scoped metadata from non-parameter tensors in the model.

    Uses ``__dict__`` scan (fast) instead of ``iter_accessible_attributes`` (slow
    dir() + getattr MRO walk). Handles tensors stored directly as attributes,
    inside lists/tuples, and inside dicts.
    """
    for submodule in model.modules():
        for attr_val in submodule.__dict__.values():
            if isinstance(attr_val, torch.Tensor):
                if not isinstance(attr_val, torch.nn.Parameter):
                    clear_meta(attr_val)
            elif isinstance(attr_val, (list, tuple, set)):
                for item in attr_val:
                    if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                        clear_meta(item)
            elif isinstance(attr_val, dict):
                for val in attr_val.values():
                    if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter):
                        clear_meta(val)
    # Also clean any tensors from the registered buffer dict (_buffers)
    for submodule in model.modules():
        for buf_tensor in submodule._buffers.values():
            if buf_tensor is not None:
                clear_meta(buf_tensor)


# ---------------------------------------------------------------------------
# Ensure model is prepared (one-time + incremental crawl)
# ---------------------------------------------------------------------------


def _ensure_model_prepared(model: nn.Module) -> None:
    """Orchestrate all one-time preparation steps before a logging session.

    Called at the start of every ``trace``. Each step is individually
    idempotent or incremental:

    1. ``wrap_torch()`` — Ensures torch functions are wrapped (no-op if already wrapped,
       re-wraps after ``unwrap_torch()``, first-time decoration on first call).
    2. ``_prepare_model_once(model)`` — Phase 1 model prep (cached per instance).
    3. ``patch_detached_references()`` — Crawl sys.modules for stale refs
       (incremental: only scans newly-imported modules).
    4. ``patch_model_instance(model)`` — Level 4 crawl on model instance attrs.
    """
    from .wrappers import wrap_torch, patch_detached_references, patch_model_instance

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
