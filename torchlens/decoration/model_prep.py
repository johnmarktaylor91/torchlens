"""Model preparation: two-phase setup of nn.Modules for out logging.

This module implements the model preparation pipeline, which has two phases:

**Phase 1 — One-time preparation** (``_prepare_model_once``):
  Runs once per model instance (cached in ``_state._prepared_models`` WeakSet).
  Assigns permanent attributes (``tl_address``, ``tl_module_type``) and
  wraps each submodule's ``forward`` with ``module_forward_decorator``. These
  survive across multiple ``trace`` calls.

**Phase 2 — Per-session preparation** (``_prepare_model_session``):
  Runs on every ``trace`` call. Populates session-scoped tracking
  dicts on Trace (pass counters, entered/exited labels), captures module
  metadata, forces ``requires_grad=True`` on all parameters (needed for
  ``grad_fn`` metadata), creates ``ParamLog`` objects, and tags buffer tensors.
  Session data is GC'd with the Trace — no per-module cleanup needed.

The ``module_forward_decorator`` wraps each submodule's ``forward`` with a
toggle-gated wrapper that reads ``trace`` from ``_state._active_trace``
(not closed-over). In exhaustive mode, it calls ``_handle_module_entry`` /
``_handle_module_exit`` to track which tensors enter and exit each module. In
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

from .. import _state
from ..data_classes.param_log import ParamAccessor, ParamLog
from ..data_classes.func_call_location import FuncCallLocation
from ..data_classes.module_log import HookInfo
from ..utils.tensor_utils import get_memory_amount
from ..utils.introspection import get_vars_of_type_from_obj, iter_accessible_attributes
from ..utils.hashing import make_random_barcode
from ..capture.source_tensors import log_source_tensor
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from . import _module_stack as _mstack

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
    from ..data_classes.model_log import Trace

# Session-scoped attributes set on parameters during _create_session_param_logs.
_SESSION_PARAM_ATTRS = [
    "tl_requires_grad",  # original requires_grad (restored on cleanup)
    "tl_param_barcode",  # unique id for param-sharing detection
    "tl_param_address",  # dotted address (e.g. "encoder.weight")
    "tl_call_index",  # how many times this param has been used
]


# ---------------------------------------------------------------------------
# Shared module traversal
# ---------------------------------------------------------------------------


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
    module_stack: deque[tuple[nn.Module, str]] = deque([(model, "")])
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

    2. **Assigns permanent attributes** — ``tl_address`` (dotted path
       like ``"encoder.layer.0.attention"``) and ``tl_module_type`` (class name).
       These survive across sessions.

    3. **Wraps ``forward``** — Replaces ``module.forward`` with
       ``module_forward_decorator(module.forward, module)``. The wrapper is
       toggle-gated: no-op when logging is off, full entry/exit tracking when on.
       The ``tl_forward_call_is_decorated`` sentinel prevents double-wrapping.

    The root module is skipped for type annotation and forward wrapping because
    its forward is called directly by ``trace`` with its own
    entry/exit handling.
    """
    if model in _state._prepared_models:
        return

    model.tl_address = ""  # type: ignore[assignment]

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
            child_module.tl_address = child_address  # type: ignore[assignment]

        # Root module is handled separately by trace.
        if is_root:
            return

        module.tl_module_type = str(type(module).__name__)  # type: ignore[assignment]

        # Wrap forward with toggle-gated decorator (idempotent via sentinel).
        if hasattr(module, "forward") and not hasattr(
            module.forward, "tl_forward_call_is_decorated"
        ):
            module.forward = module_forward_decorator(module.forward, module)
            module.forward.tl_forward_call_is_decorated = True  # type: ignore[attr-defined]

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
    3. Sets session-scoped ``tl_*`` attributes on each submodule (pass counters,
       tensor entry/exit tracking lists, back-reference to trace).
    4. Creates ``ParamLog`` objects and forces ``requires_grad=True`` on all
       parameters (needed so ``grad_fn`` chain is available for metadata).
    5. Tags buffer tensors with ``tl_buffer_address``.

    All session-scoped state is cleaned up by ``_cleanup_model_session``.
    """
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    trace.model_name = str(type(model).__name__)
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

    # Use model.modules() + cached tl_address from phase 1, avoiding a
    # second full DFS with string concatenation and list(named_children()) calls.
    for module in model.modules():
        is_root = module is model
        address = getattr(module, "tl_address", "")
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
            trace._module_build_data["module_types"][cast(str, module.tl_address)] = cast(
                str, module.tl_module_type
            )
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
    """Create ``ParamLog`` objects and prepare parameter grad tracking.

    Outside ``train_mode``, ``requires_grad`` is forced True so that ``grad_fn``
    metadata is available on all intermediate tensors during the forward pass.
    In ``train_mode``, user-authored ``requires_grad`` values are preserved. The
    original value is always saved to ``tl_requires_grad`` and restored during
    ``_cleanup_model_session``.
    """
    optimized_param_ids: set[int] = set()
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group["params"]:
                optimized_param_ids.add(id(p))

    param_logs: dict[str, ParamLog] = {}
    seen_param_ids: set[int] = set()
    param_id_to_address: dict[int, str] = {}
    for module in model.modules():
        address = getattr(module, "tl_address", "")
        module_type = type(module).__name__
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            # Shared parameters: only create one ParamLog per unique tensor.
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
            param.tl_requires_grad = param.requires_grad  # type: ignore[attr-defined]
            if not getattr(trace, "train_mode", False):
                param.requires_grad = True

            barcode = make_random_barcode()
            param.tl_param_barcode = barcode  # type: ignore[attr-defined]
            param.tl_param_address = param_address  # type: ignore[attr-defined]
            param.tl_call_index = 0  # type: ignore[attr-defined]

            param_fsize = get_memory_amount(param)
            param_log = ParamLog(
                module_address=module_address,
                name=param_name,
                shape=tuple(param.shape),
                dtype=param.dtype,
                num_params=param.numel(),
                memory=param_fsize,
                trainable=param.tl_requires_grad,  # type: ignore[attr-defined]
                address=param_address,
                module_type=module_type,
                barcode=barcode,
                has_optimizer=id(param) in optimized_param_ids if optimizer is not None else None,
            )
            param_log._param_ref = param
            param_logs[param_address] = param_log

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
        module_addr = getattr(submodule, "tl_address", "")
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
# Module forward decorator — reads trace from _state
# ---------------------------------------------------------------------------


def _tag_untagged_buffers(module: nn.Module) -> None:
    """Tag any buffers that lack a ``tl_buffer_address`` attribute.

    Called during ``_handle_module_entry`` to catch buffers that were created
    dynamically (e.g. in ``forward()``) after the initial ``prepare_buffer_tensors``
    scan. If a buffer already has a ``tl__label_raw`` from being logged as
    an intermediate tensor, that label is moved to ``tl_buffer_parent`` and cleared
    so the buffer gets a fresh source-tensor entry on next use.
    """
    for buffer_name, buffer_tensor in module.named_buffers():
        if hasattr(buffer_tensor, "tl_buffer_address"):
            continue
        if module.tl_address == "":
            buffer_address = buffer_name
        else:
            buffer_address = f"{module.tl_address}.{buffer_name}"
        buffer_tensor.tl_buffer_address = buffer_address  # type: ignore[attr-defined]
        # If this buffer was already logged as an intermediate tensor, save the
        # old label as parent and reset so it gets a proper buffer source entry.
        if hasattr(buffer_tensor, "tl__label_raw"):
            buffer_tensor.tl_buffer_parent = buffer_tensor.tl__label_raw  # type: ignore[attr-defined]
            delattr(buffer_tensor, "tl__label_raw")


def _handle_module_entry(
    trace: "Trace",
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[set[str], list[str]]:
    """Pre-forward bookkeeping for exhaustive mode.

    Called immediately before ``orig_forward(*args, **kwargs)`` in the
    ``module_forward_decorator``. Performs:

    1. Increments the module's pass counter (supports multi-pass modules like
       recurrent layers).
    2. Pushes a ``(address, call_index)`` label onto the module's pass stack
       (popped by ``_handle_module_exit``).
    3. Finds all input tensors and annotates their layer entries with module
       entry metadata (``modules_entered``, ``is_submodule_input``, etc.).
    4. Records which argument position each tensor occupies (for arg-name
       reconstruction in ``_build_module_logs``).
    5. Appends a ``("+", address, call_index)`` marker to each input tensor's
       ``_module_boundary_thread_output`` (a chronological log of module
       boundaries that tensors cross).
    6. Tags any dynamically-created buffers that weren't caught during
       ``prepare_buffer_tensors``.

    Args:
        trace: The active Trace.
        module: The nn.Module about to execute.
        args: Positional arguments to forward.
        kwargs: Keyword arguments to forward.

    Returns:
        Tuple of ``(input_tensor_labels, input_tensor_labels_at_entry)`` —
        needed by ``_handle_module_exit`` to trim the entry thread.
    """
    address = cast(str, module.tl_address)
    mod_id = id(module)
    trace._module_build_data["module_training_modes"][address] = module.training
    trace._mod_call_index[mod_id] += 1
    module_call_index = trace._mod_call_index[mod_id]
    module_call_label = (address, module_call_index)
    # Push onto stack — popped by _handle_module_exit (or exception handler).
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
        if (not hasattr(t, "tl__label_raw")) and hasattr(t, "tl_buffer_address"):
            log_source_tensor(trace, t, "buffer", getattr(t, "tl_buffer_address"))
        if not hasattr(t, "tl__label_raw"):
            # Raw ``register_forward_hook`` replacements run after this
            # module-forward wrapper has already handled the replaced module's
            # exit, so the first recoverable point may be the next downstream
            # module entry.
            parent_labels = (
                trace._raw_layer_labels_list[-1:] if trace._raw_layer_labels_list else []
            )
            _ensure_module_output_tensor_logged(trace, t, module, parent_labels)
        if not hasattr(t, "tl__label_raw"):
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        layer_entry = trace._raw_layer_dict[t.tl__label_raw]
        input_tensor_labels.add(t.tl__label_raw)
        trace._mod_entered[mod_id].append(t.tl__label_raw)
        layer_entry.modules_entered.append(address)
        layer_entry.module_ops_entered.append(module_call_label)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                layer_entry.module_entry_argnames[
                    f"{module_call_label[0]}:{module_call_label[1]}"
                ].append(arg_key)
                trace._module_build_data["module_layer_argnames"][
                    (f"{module_call_label[0]}:{module_call_label[1]}")
                ].append((t.tl__label_raw, arg_key))
        # Record module entry in chronological thread (matched by "-" on exit).
        layer_entry._module_boundary_thread_output.append(
            ("+", module_call_label[0], module_call_label[1])
        )
        input_tensor_labels_at_entry.append(t.tl__label_raw)

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
    """Copy mutable OpLog field values without cloning tensors.

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
        Replacement output tensor that lacks ``tl__label_raw``.
    module:
        Module whose forward return value was replaced.
    parent_labels:
        Raw labels for tensors that entered the module.

    Returns
    -------
    None
        The tensor is tagged and a replacement OpLog is inserted into the raw
        graph so downstream module-exit and op logging can continue.
    """

    from ..capture.output_tensors import _make_layer_log_entry

    parent_entries = [trace._raw_layer_dict[label] for label in parent_labels]
    template_entry = parent_entries[0] if parent_entries else None
    raw_label, capture_index, type_index = _next_intervention_replacement_label(trace)
    fields_dict = {
        field_name: _copy_field_value_for_replacement(getattr(template_entry, field_name, None))
        for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
    address = cast(str, module.tl_address)
    module_call_index = trace._mod_call_index[id(module)]
    root_ancestors: set[str] = set()
    input_ancestors: set[str] = set()
    internal_source_ancestors: set[str] = set()
    for parent_entry in parent_entries:
        root_ancestors.update(parent_entry.root_ancestors or set())
        input_ancestors.update(parent_entry.input_ancestors or set())
        internal_source_ancestors.update(parent_entry.internal_source_ancestors or set())

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
            "grad_fn_name": type(tensor.grad_fn).__name__,
            "grad_fn_id": id(tensor.grad_fn) if tensor.grad_fn is not None else None,
            "grad_fn": tensor.grad_fn,
            "grad_fn_log": None,
            "is_part_of_iterable_output": False,
            "multi_output_index": None,
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
            "equivalence_class": raw_label,
            "equivalent_ops": trace.equivalent_ops[raw_label],
            "recurrent_ops": [],
            "parents": parent_labels,
            "parent_arg_positions": {"args": {}, "kwargs": {}},
            "edge_uses": [],
            "root_ancestors": root_ancestors or {raw_label},
            "children": [],
            "has_children": False,
            "is_input": False,
            "has_input_ancestor": any(entry.has_input_ancestor for entry in parent_entries),
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
            "buffer_address": None,
            "buffer_pass": None,
            "buffer_parent": None,
            "is_internal_source": not parent_entries,
            "has_internal_source_ancestor": any(
                entry.has_internal_source_ancestor for entry in parent_entries
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
            "modules": [(address, module_call_index)] if address else [],
            "modules_entered": [],
            "module_entry_argnames": defaultdict(list),
            "module_ops_entered": [],
            "output_of_modules": [],
            "output_of_module_calls": [],
            "is_submodule_output": False,
            "is_atomic_module_op": False,
            "atomic_module_call": None,
            "_module_boundary_threads_inputs": {
                parent._label_raw: parent._module_boundary_thread_output[:]
                for parent in parent_entries
            },
            "_module_boundary_thread_output": [],
            "func_config": {},
        }
    )
    trace.equivalent_ops[raw_label].add(raw_label)
    new_entry = _make_layer_log_entry(trace, tensor, fields_dict, (), {}, trace.out_postfunc)
    for parent_entry in parent_entries:
        parent_entry.children.append(raw_label)
        parent_entry.has_children = True
    tensor.tl__label_raw = new_entry._label_raw  # type: ignore[attr-defined]
    if trace.save_grads:
        from ..capture.output_tensors import _add_backward_hook

        _add_backward_hook(trace, tensor, tensor.tl__label_raw)  # type: ignore[attr-defined]


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
        if getattr(hook_fn, "tl_tensor_replacement_wrapped", False):
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
            getattr(tensor, "tl__label_raw")
            for tensor in get_vars_of_type_from_obj(original_output, torch.Tensor, search_depth=4)
            if hasattr(tensor, "tl__label_raw")
        ]
        for replacement in get_vars_of_type_from_obj(result, torch.Tensor, search_depth=4):
            if hasattr(replacement, "tl__label_raw"):
                trace._raw_layer_dict[replacement.tl__label_raw].intervention_replaced = True
            else:
                _ensure_module_output_tensor_logged(trace, replacement, module, parent_labels)
        return result

    wrapped_hook.tl_tensor_replacement_wrapped = True  # type: ignore[attr-defined]
    return wrapped_hook


def _handle_module_exit(
    trace: "Trace",
    module: nn.Module,
    out: Any,
    input_tensor_labels: set[str],
    input_tensor_labels_at_entry: list[str],
) -> None:
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
       - Appends ``("-", address, call_index)`` to the chronological thread.
    3. Trims the ``_module_boundary_thread_output`` of input tensors — removes
       the ``"+"`` entry marker and everything after it, since the module boundary
       is now closed. This prevents stale entry markers from confusing downstream
       module-nesting analysis.
    """
    address = cast(str, module.tl_address)
    mod_id = id(module)
    module_call_index = trace._mod_call_index[mod_id]
    module_entry_label = trace._mod_call_labels[mod_id].pop()
    output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
    for t in output_tensors:
        # nn.Identity modules and pass-through tensors (output is same object
        # as input) need _decorated_identity() to create a distinct log entry
        # so the graph correctly shows the module boundary.
        if (cast(str, module.tl_module_type).lower() == "identity") or (
            hasattr(t, "tl__label_raw") and t.tl__label_raw in input_tensor_labels
        ):
            t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
        if not hasattr(t, "tl__label_raw"):
            parent_labels = list(dict.fromkeys(input_tensor_labels_at_entry))
            _ensure_module_output_tensor_logged(trace, t, module, parent_labels)
        layer_entry = trace._raw_layer_dict[t.tl__label_raw]
        if getattr(module, "_forward_hooks", None):
            layer_entry.intervention_replaced = True
        layer_entry.is_submodule_output = True
        layer_entry.is_atomic_module_op = _is_bottom_level_submodule_exit(trace, t, module)
        layer_entry.output_of_modules.append(address)
        layer_entry.output_of_module_calls.append((address, module_call_index))
        # Record module exit in chronological thread (matches "+" from entry).
        layer_entry._module_boundary_thread_output.append(
            ("-", module_entry_label[0], module_entry_label[1])
        )
        trace._mod_exited[mod_id].append(t.tl__label_raw)

    # Trim entry threads on input tensors: remove the "+" marker and
    # everything after it. The module boundary is now closed, so input
    # tensors' threads should not carry forward this module's context.
    for entry_label in input_tensor_labels_at_entry:
        layer_entry = trace._raw_layer_dict[entry_label]
        input_module_thread = layer_entry._module_boundary_thread_output[:]
        if (
            "+",
            module_entry_label[0],
            module_entry_label[1],
        ) in input_module_thread:
            module_entry_ix = input_module_thread.index(
                ("+", module_entry_label[0], module_entry_label[1])
            )
            layer_entry._module_boundary_thread_output = layer_entry._module_boundary_thread_output[
                :module_entry_ix
            ]


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
            input_tensor_labels = {
                t.tl__label_raw for t in input_tensors_fast if hasattr(t, "tl__label_raw")
            }
            out = orig_forward(*args, **kwargs)
            output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
            for t in output_tensors:
                # Force _decorated_identity() for nn.Identity modules and pass-throughs
                # to create a new tensor entry, matching what exhaustive mode does.
                if (cast(str, module.tl_module_type).lower() == "identity") or (
                    hasattr(t, "tl__label_raw") and t.tl__label_raw in input_tensor_labels
                ):
                    t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            return out

        if trace.capture_mode == "predicate":
            from ..fastlog._predicate import _evaluate_keep_module
            from ..fastlog._record_context import _build_record_context
            from ..fastlog._state import get_active_recording_state
            from ..fastlog.types import ActivationRecord

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
            try:
                enter_spec = _evaluate_keep_module(enter_ctx, state.options)
                if enter_spec.save_out or enter_spec.save_metadata:
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
                try:
                    exit_spec = _evaluate_keep_module(exit_ctx, state.options)
                    if exit_spec.save_out or exit_spec.save_metadata:
                        state.add_record(ActivationRecord(ctx=exit_ctx, spec=exit_spec))
                except Exception as exc:
                    state.handle_predicate_exception(exit_ctx, exc)
                finally:
                    state.append_context(exit_ctx)
                    _mstack.pop_frame(state.module_stack, frame)

        # ---- Exhaustive mode: full entry → forward → exit ----
        input_tensor_labels, input_tensor_labels_at_entry = _handle_module_entry(
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
        _handle_module_exit(trace, module, out, input_tensor_labels, input_tensor_labels_at_entry)
        return out

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
    layer_entry = trace._raw_layer_dict[getattr(t, "tl__label_raw")]
    subaddress = cast(str, submodule.tl_address)
    sub_id = id(submodule)

    # Case 1: already determined in a prior call.
    if layer_entry.is_atomic_module_op:
        return True

    # Case 2: tensor originated inside the model with no inputs entering
    # this submodule (e.g. a buffer-derived tensor in a leaf module).
    if layer_entry.is_internal_source and len(trace._mod_entered[sub_id]) == 0:
        layer_entry.is_atomic_module_op = True
        layer_entry.atomic_module_call = (
            subaddress,
            trace._mod_call_index[sub_id],
        )
        return True

    # Case 3: check that ALL parent tensors last entered this exact submodule.
    # If any parent was last seen in a different (deeper) submodule, this is
    # NOT a bottom-level exit.
    for parent_label in layer_entry.parents:
        parent_tensor = trace[parent_label]
        parent_modules_entered = parent_tensor.modules_entered
        if (len(parent_modules_entered) == 0) or (parent_modules_entered[-1] != subaddress):
            layer_entry.is_atomic_module_op = False
            return False

    layer_entry.is_atomic_module_op = True
    layer_entry.atomic_module_call = (
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
    removes all session-scoped ``tl_*`` attributes from modules and params,
    and strips ``tl__label_raw`` / ``tl_buffer_address`` from tensors.

    **Does NOT** remove permanent attributes (``tl_address``,
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

    # Session-scoped module tracking data lives in Trace dicts (not on
    # modules), so no per-module cleanup iteration is needed — the dicts
    # are GC'd with the Trace.

    # Clean tensor labels from model tensors (buffers, etc.)
    _undecorate_model_tensors(model)

    # Clean tensor labels from input tensors
    if input_tensors:
        for t in input_tensors:
            if hasattr(t, "tl__label_raw"):
                delattr(t, "tl__label_raw")


_TL_TENSOR_ATTRS = ("tl__label_raw", "tl_buffer_address", "tl_buffer_parent")


def _strip_tl_attrs(tensor: torch.Tensor) -> None:
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

    Called at the start of every ``trace``. Each step is individually
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
