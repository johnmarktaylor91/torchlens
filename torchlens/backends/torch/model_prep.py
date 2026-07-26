"""Prepare torch ``nn.Module`` objects for capture sessions.

One-time preparation installs persistent forward wrappers and module metadata.
Per-session preparation populates Trace state, parameter logs, and buffer labels.
"""

import copy
import inspect
import itertools
import math
import sys
import time
from collections.abc import Callable
from collections import defaultdict, deque
from functools import wraps
from types import ModuleType
from typing import Any, TYPE_CHECKING, cast

import torch
from torch import nn

from ... import _state
from ...fastlog._halt import HaltSignal
from ._tl import (
    begin_label_session,
    clear_meta,
    end_label_session,
    get_buffer_address,
    get_live_tensor_label,
    get_module_meta,
    get_tensor_label,
    is_forward_call_decorated,
    mark_forward_call_decorated,
    mark_tensor_replacement_wrapped,
    promote_label_to_buffer_source_and_clear_label,
    restore_param_requires_grad,
    set_module_meta,
    set_param_meta,
    set_tensor_label,
)
from ...data_classes.param import ParamAccessor, Param
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.module import HookInfo
from ...data_classes._module_role_hints import multi_output_role_from_path, role_hints_for_module
from ...ir import (
    CaptureEvents,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
    replace_op_event,
)
from ...ir.container_registry import ModuleSite, Phase, Role, walk_container
from ...utils.tensor_utils import (
    get_memory_amount,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
)
from ...utils.introspection import (
    _get_code_context,
    get_vars_of_type_from_obj,
)
from ...utils.hashing import make_random_barcode
from .tensor_tracking import _append_module_suffix_to_equivalence_class
from .sources import log_source_tensor
from ...constants import LAYER_PASS_LOG_FIELD_ORDER
from . import module_stack as _mstack
from .escape_detection import (
    expected_original_call,
    mark_expected_original_accounted,
)

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
    from ...data_classes.trace import Trace


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


_QUANTIZED_MODULE_PREFIXES = (
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
)


def _is_quantized_module(module: nn.Module) -> bool:
    """Return whether ``module`` is a PyTorch quantized module.

    Parameters
    ----------
    module:
        Module to inspect.

    Returns
    -------
    bool
        Whether the module class is from a known PyTorch quantized namespace.
    """

    module_name = type(module).__module__
    return module_name.startswith(_QUANTIZED_MODULE_PREFIXES)


def _first_tensor_shape(value: Any) -> tuple[int, ...] | None:
    """Return the shape of the first tensor found in ``value``.

    Parameters
    ----------
    value:
        Object tree to search.

    Returns
    -------
    tuple[int, ...] | None
        First tensor shape, or ``None`` when no tensor is present.
    """

    tensors = get_vars_of_type_from_obj(value, torch.Tensor, search_depth=5)
    if not tensors:
        return None
    return tuple(tensors[0].shape)


def _quantized_module_bias_present(module: nn.Module) -> bool:
    """Return whether a quantized module appears to have a bias term.

    Parameters
    ----------
    module:
        Quantized module to inspect.

    Returns
    -------
    bool
        Whether the module exposes a non-``None`` bias.
    """

    bias = getattr(module, "bias", None)
    if callable(bias):
        try:
            return bias() is not None
        except Exception:
            return False
    return bias is not None


def _estimate_quantized_module_forward_flops(
    module: nn.Module,
    output_shape: tuple[int, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> int | None:
    """Estimate FLOPs for common quantized modules logged as internal sources.

    Parameters
    ----------
    module:
        Module that produced the unwrapped quantized output.
    output_shape:
        Shape of the module output tensor.
    args:
        Positional module-forward arguments.
    kwargs:
        Keyword module-forward arguments.

    Returns
    -------
    int | None
        Estimated forward FLOPs for recognized quantized Linear/Conv modules,
        otherwise ``None``.
    """

    if not _is_quantized_module(module):
        return None
    input_shape = _first_tensor_shape((args, kwargs))
    if input_shape is None:
        return None
    out_numel = int(math.prod(output_shape)) if output_shape else 1
    module_kind = _module_type(module).lower()
    bias_flops = out_numel if _quantized_module_bias_present(module) else 0
    if "linear" in module_kind:
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            return None
        batch = out_numel // out_features if out_features > 0 else 0
        return 2 * batch * in_features * out_features + bias_flops
    if "conv" in module_kind:
        in_channels = getattr(module, "in_channels", None)
        groups = getattr(module, "groups", 1)
        kernel_size = getattr(module, "kernel_size", None)
        if not isinstance(in_channels, int) or not isinstance(groups, int):
            return None
        if isinstance(kernel_size, int):
            kernel_numel = kernel_size
        elif isinstance(kernel_size, tuple) and all(isinstance(v, int) for v in kernel_size):
            kernel_numel = int(math.prod(kernel_size))
        else:
            return None
        channels_per_group = in_channels // groups if groups > 0 else in_channels
        return 2 * out_numel * channels_per_group * kernel_numel + bias_flops
    return None


def _traverse_model_modules(
    model: nn.Module,
    visitor_fn: Callable[[nn.Module, str, list[tuple[str, nn.Module, str]], bool], None],
) -> None:
    """DFS over all modules in a model, calling ``visitor_fn`` for each.

    Visits parent before children (pre-order). The visitor receives the module,
    its dotted address, its child entries, and whether it is the root.

    Args:
        model: Root module.
        visitor_fn: Called as ``visitor_fn(module, address, child_entries, is_root)``
            for every module. Each child entry is ``(name, module, address)``.
    """
    traversal_queue: deque[tuple[nn.Module, str]] = deque([(model, "")])
    while traversal_queue:
        module, address = traversal_queue.popleft()
        named_children = list(module.named_children())
        child_entries: list[tuple[str, nn.Module, str]] = []
        for child_name, child_module in named_children:
            child_address = f"{address}.{child_name}" if address else child_name
            child_entries.append((child_name, child_module, child_address))
        # Prepend children to front of deque for DFS pre-order traversal.
        # extendleft reverses, so we reverse child_entries first to maintain order.
        for _, child_module, child_address in reversed(child_entries):
            traversal_queue.appendleft((child_module, child_address))
        visitor_fn(module, address, child_entries, module is model)


# ---------------------------------------------------------------------------
# One-time model preparation (cached in _state._prepared_models)
# ---------------------------------------------------------------------------


def _restore_undecorated_forward(module: nn.Module) -> None:
    """Undo a stale non-root ``forward`` decoration so ``module`` can be root.

    A module that was prepared as a NON-root submodule in an earlier trace has a
    toggle-gated ``module_forward_decorator`` wrapper installed on its
    ``forward``. If that same module is later traced as its OWN top-level root,
    the wrapper must be removed: ``trace`` invokes and frames the root itself, so
    the root's ``forward`` must be UNDECORATED (the wrapper would otherwise call
    ``push_frame`` for a module that is never registered in the per-session
    module-call dicts, raising ``KeyError``). The original ``forward`` is
    recovered from ``functools.wraps``' ``__wrapped__`` reference; if it is
    absent, the instance-level override is dropped so lookup falls back to the
    (undecorated) class ``forward``.

    Parameters
    ----------
    module:
        Module about to be prepared as a root.

    Returns
    -------
    None
        The module's ``forward`` is restored in place when it was decorated;
        otherwise this is a no-op.
    """
    current_forward = module.__dict__.get("forward", None)
    if current_forward is None or not is_forward_call_decorated(current_forward):
        return
    original_forward = getattr(current_forward, "__wrapped__", None)
    if original_forward is not None:
        module.forward = original_forward
    else:
        module.__dict__.pop("forward", None)


def _prepare_model_once(model: nn.Module) -> None:
    """Phase 1: One-time (per role) model preparation.

    Fast-path cached via ``_state._prepared_models`` (WeakSet) for the common
    case: a model traced repeatedly in a fixed role, or independent models
    traced in any interleaving. Performs three tasks for each submodule:

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
    its forward is called directly by ``trace`` with its own entry/exit handling
    — the root's ``forward`` is deliberately left UNDECORATED.

    **Role swaps.** The address (root-relative) and forward decoration are
    role-DEPENDENT: they differ depending on whether a module is *this* trace's
    root or a non-root submodule. The same module can legitimately be traced in
    both roles across separate traces (e.g. ``trace(outer, ...)`` then
    ``trace(outer.inner, ...)``). When that happens the metadata cached for the
    old role is stale for the new one, so this function re-establishes it:

    * A root that carried a stale non-root ``forward`` decoration is undecorated
      (see :func:`_restore_undecorated_forward`).
    * Re-rooting a descendant under a new model marks the old ancestor root
      stale (via :func:`_state.record_module_root_prep`); the stale root is then
      treated as un-prepared here so its addresses and decorations are refreshed
      for the current root on its next trace.
    """
    if model in _state._prepared_models and not _state.root_prep_is_stale(model):
        return
    _state.clear_root_prep_stale(model)

    set_module_meta(model, address="", module_type=str(type(model).__name__))
    # The root's forward must run undecorated. Restore it if this module carries
    # a stale non-root decoration from an earlier trace where it was a submodule.
    _restore_undecorated_forward(model)

    def _visit_once(
        module: nn.Module,
        address: str,
        child_entries: list[tuple[str, nn.Module, str]],
        is_root: bool,
    ) -> None:
        """Prepare one module and recursively visit its children once."""
        # Stamp this module's current root and flag any prior root it was
        # re-rooted away from as stale (role-swap bookkeeping).
        _state.record_module_root_prep(model, module)

        # Annotate children with their full dotted address path (root-relative).
        for _, child_module, child_address in child_entries:
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
        # A module re-prepared as non-root after having been a root simply gets
        # (re)decorated here, since a root's forward is left undecorated.
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
       parameters (needed so ``grad_fn_handle`` chain is available for metadata).
    5. Tags buffer tensors with ``_tl.address``.

    All session-scoped state is cleaned up by ``_cleanup_model_session``.
    """
    # r83 C1: install this capture's label-anchoring session FIRST, before any
    # path can stamp a label. A previously installed session is dropped here,
    # so a label issued by an earlier capture resolves against nothing. The
    # registry is module-level state in ``_tl`` (not a Trace field): it must be
    # reachable from ``set_tensor_label`` itself, which is the choke point every
    # label stamp flows through and which has no Trace in scope.
    begin_label_session()
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    trace._exhaustive_module_stack = []
    trace.model_class_name = str(type(model).__name__)
    trace.class_docstring = type(model).__doc__
    init_method = getattr(type(model), "__init__", None)
    forward_method = getattr(type(model), "forward", None)
    try:
        trace.init_signature = str(inspect.signature(init_method)) if init_method else None
    except (TypeError, ValueError):
        trace.init_signature = None
    trace.init_docstring = getattr(init_method, "__doc__", None)
    try:
        trace.forward_signature = str(inspect.signature(forward_method)) if forward_method else None
    except (TypeError, ValueError):
        trace.forward_signature = None
    trace.forward_docstring = getattr(forward_method, "__doc__", None)
    try:
        trace.class_source_file = inspect.getfile(type(model))
        trace.class_source_line = inspect.getsourcelines(type(model))[1]
        trace.init_source_file = inspect.getfile(type(model).__init__)
        trace.init_source_line = inspect.getsourcelines(type(model).__init__)[1]
    except (OSError, TypeError):
        trace.class_source_file = None
        trace.class_source_line = None
        trace.init_source_file = None
        trace.init_source_line = None
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
        meta_address = "self" if is_root else address
        meta = trace._module_metadata.get(meta_address)
        if meta is not None:
            capture_events = getattr(trace, "capture_events", None)
            if capture_events is None:
                capture_events = CaptureEvents()
                trace.capture_events = capture_events
            capture_events.module_prep_events.append(
                ModulePrepEvent(
                    address=meta_address,
                    all_addresses=tuple(meta["all_addresses"]),
                    module_type_str=_module_type(module),
                    cls_qualname=meta["class_qualname"],
                    class_name=meta["class_name"],
                    address_children=tuple(meta["address_children"]),
                    class_source_file=meta.get("class_source_file"),
                    class_source_line=meta.get("class_source_line"),
                    init_source_file=meta.get("init_source_file"),
                    init_source_line=meta.get("init_source_line"),
                    forward_source_file=meta.get("forward_source_file"),
                    forward_source_line=meta.get("forward_source_line"),
                    class_docstring=meta.get("class_docstring"),
                    init_signature=meta.get("init_signature"),
                    init_docstring=meta.get("init_docstring"),
                    forward_signature=meta.get("forward_signature"),
                    forward_docstring=meta.get("forward_docstring"),
                    forward_pre_hooks=tuple(meta["forward_pre_hooks"]),
                    forward_hooks=tuple(meta["forward_hooks"]),
                    backward_pre_hooks=tuple(meta["backward_pre_hooks"]),
                    backward_hooks=tuple(meta["backward_hooks"]),
                    full_backward_pre_hooks=tuple(meta["full_backward_pre_hooks"]),
                    full_backward_hooks=tuple(meta["full_backward_hooks"]),
                    training_at_prep=bool(meta["training"]),
                    custom_attributes=tuple(meta["custom_attributes"].items()),
                    custom_methods=tuple(meta["custom_methods"]),
                )
            )
        if not is_root:
            trace._module_build_data["module_types"][address] = _module_type(module)
            # Session-scoped tracking in Trace dicts (keyed by id(module)).
            mod_id = id(module)
            trace._mod_call_index[mod_id] = 0
            trace._mod_call_labels[mod_id] = []
            trace._mod_entered[mod_id] = []
            trace._mod_exited[mod_id] = []
    if trace.capture_mode != "predicate":
        _create_session_param_logs(trace, model, optimizer)
    prepare_buffer_tensors(trace, model)
    if trace.capture_mode == "exhaustive":
        from .buffer_writes import install_buffer_write_tracker

        install_buffer_write_tracker(trace, model)
    from .prehook_provenance import install_prehook_provenance

    install_prehook_provenance(
        trace,
        model,
        forward_hook_wrapper_factory=_make_user_forward_hook_wrapper,
    )


def _create_session_param_logs(trace: "Trace", model: nn.Module, optimizer: Any = None) -> None:
    """Create ``Param`` objects and prepare parameter grad tracking.

    Outside ``backward_ready``, ``requires_grad`` is forced True so that ``grad_fn_handle``
    metadata is available on all intermediate tensors during the forward pass.
    In ``backward_ready``, user-authored ``requires_grad`` values are preserved. The
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
    # r79 session-leak fix: record every parameter this prep STAMPS so cleanup can
    # clear stamps from this inventory instead of re-traversing the live model tree.
    # A param popped from ``_parameters`` mid-forward escapes the re-traversal but
    # never escapes this list (session-scoped strong refs, dropped at cleanup).
    stamped_params: list[nn.Parameter] = []
    for module in model.modules():
        address = _module_address(module)
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            # Shared parameters: only create one Param per unique tensor.
            pid = id(param)
            if pid in seen_param_ids:
                existing_address = param_id_to_address[pid]
                alias_address = f"{address}.{param_name}" if address else param_name
                param_log = param_logs[existing_address]
                if alias_address not in param_log.all_addresses:
                    param_log.all_addresses.append(alias_address)
                alias_module_address = address or "self"
                if alias_module_address not in param_log.all_module_addresses:
                    param_log.all_module_addresses.append(alias_module_address)
                continue
            seen_param_ids.add(pid)

            module_address = address or "self"
            param_address = f"{address}.{param_name}" if address else param_name
            param_id_to_address[pid] = param_address

            # Save original requires_grad before forcing True. Integer/bool-dtype
            # Parameters (e.g. a fixed nn.Parameter(torch.arange(...), requires_grad=False)
            # lookup buffer) are legal PyTorch and never gradient-capable; forcing
            # requires_grad on them raises, so only force floating/complex dtypes.
            requires_grad_before = param.requires_grad
            if not getattr(trace, "backward_ready", False) and (
                torch.is_floating_point(param) or torch.is_complex(param)
            ):
                param.requires_grad = True

            barcode = make_random_barcode()
            set_param_meta(
                param,
                barcode=barcode,
                address=param_address,
                requires_grad_before=requires_grad_before,
            )
            stamped_params.append(param)

            param_fsize = get_memory_amount(param)
            param_log = Param(
                module_address=module_address,
                name=param_name,
                shape=tuple(param.shape),
                dtype=param.dtype,
                num_params=param.numel(),
                param_memory=param_fsize,
                trainable=requires_grad_before,
                address=param_address,
                barcode=barcode,
                has_optimizer=id(param) in optimized_param_ids if optimizer is not None else None,
            )
            param_log._param_ref = param
            param_logs[param_address] = param_log

    trace._param_log_by_pid = param_id_to_address
    trace._session_param_inventory = stamped_params
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
            meta["class_source_file"] = inspect.getfile(module_class)
        except (TypeError, OSError):
            meta["class_source_file"] = None
        try:
            _, line = inspect.getsourcelines(module_class)
            meta["class_source_line"] = line
        except (TypeError, OSError):
            meta["class_source_line"] = None

        init_method = getattr(module_class, "__init__", None)
        try:
            if init_method is not None and init_method is not nn.Module.__init__:
                meta["init_source_file"] = inspect.getsourcefile(init_method) or inspect.getfile(
                    init_method
                )
                meta["init_source_line"] = inspect.getsourcelines(init_method)[1]
            else:
                meta["init_source_file"] = None
                meta["init_source_line"] = None
        except (TypeError, OSError):
            meta["init_source_file"] = None
            meta["init_source_line"] = None
        try:
            meta["init_signature"] = (
                str(inspect.signature(init_method)) if init_method is not None else None
            )
        except (ValueError, TypeError):
            meta["init_signature"] = None
        meta["init_docstring"] = getattr(init_method, "__doc__", None)

        forward_method = getattr(module_class, "forward", None)
        try:
            if forward_method is not None:
                meta["forward_source_file"] = inspect.getsourcefile(
                    forward_method
                ) or inspect.getfile(forward_method)
                meta["forward_source_line"] = inspect.getsourcelines(forward_method)[1]
            else:
                meta["forward_source_file"] = None
                meta["forward_source_line"] = None
        except (TypeError, OSError):
            meta["forward_source_file"] = None
            meta["forward_source_line"] = None
        try:
            meta["forward_signature"] = (
                str(inspect.signature(forward_method)) if forward_method is not None else None
            )
        except (ValueError, TypeError):
            meta["forward_signature"] = None
        meta["forward_docstring"] = getattr(forward_method, "__doc__", None)
    else:
        meta["class_source_file"] = None
        meta["class_source_line"] = None
        meta["init_source_file"] = None
        meta["init_source_line"] = None
        meta["forward_source_file"] = None
        meta["forward_source_line"] = None
        meta["init_signature"] = None
        meta["init_docstring"] = None
        meta["forward_signature"] = None
        meta["forward_docstring"] = None

    _module_class_metadata_cache[module_class] = meta
    return meta


def _hook_info_from_registry(registry: Any) -> list[HookInfo]:
    """Build HookInfo entries for a PyTorch module hook registry.

    Parameters
    ----------
    registry:
        PyTorch hook registry mapping handle ids to callables.

    Returns
    -------
    list[HookInfo]
        Portable hook metadata, one entry per registered hook.
    """

    hooks = list(registry.values())
    hook_infos: list[HookInfo] = []
    for hook in hooks:
        name = getattr(hook, "__name__", type(hook).__name__)
        qualname = getattr(hook, "__qualname__", name)
        module_name = getattr(hook, "__module__", "")
        full_qualname = f"{module_name}.{qualname}" if module_name else qualname
        source_location = None
        try:
            source_file = inspect.getsourcefile(hook) or inspect.getfile(hook)
            source_line = inspect.getsourcelines(hook)[1]
        except (OSError, TypeError):
            pass
        else:
            source_location = FuncCallLocation(
                file=source_file,
                line_number=source_line,
                func_name=full_qualname,
                source_loading_enabled=False,
            )
        hook_infos.append(
            HookInfo(name=name, qualname=full_qualname, source_location=source_location)
        )
    return hook_infos


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
    meta["forward_pre_hooks"] = _hook_info_from_registry(getattr(module, "_forward_pre_hooks", {}))
    meta["forward_hooks"] = _hook_info_from_registry(getattr(module, "_forward_hooks", {}))
    meta["backward_pre_hooks"] = _hook_info_from_registry(
        getattr(module, "_backward_pre_hooks", {})
    )
    meta["backward_hooks"] = _hook_info_from_registry(getattr(module, "_backward_hooks", {}))
    meta["full_backward_pre_hooks"] = _hook_info_from_registry(
        getattr(module, "_full_backward_pre_hooks", {})
    )
    meta["full_backward_hooks"] = _hook_info_from_registry(
        getattr(module, "_full_backward_hooks", {})
    )
    meta["training"] = module.training

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

    r79 session-leak fix: every tensor stamped here is ALSO recorded in
    ``trace._session_buffer_inventory`` (session-scoped strong refs) so cleanup
    clears the stamps from the recorded inventory instead of relying on a model
    re-traversal that a mid-forward ``_buffers.pop(...)`` can escape.

    r81 buffer-rung parity: every stamp routes through
    ``register_session_buffer_stamp`` so it also joins the session identity
    registry (``trace._session_buffer_identity``) consulted by the buffer-rung
    storage-identity belt; the registry is reset here at session start.
    """
    from .buffer_writes import register_session_buffer_stamp

    _state._tagged_buffer_ids.clear()
    trace._session_buffer_inventory = []
    trace._session_buffer_identity = {}
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
                    register_session_buffer_stamp(trace, buf_tensor, address)
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
                    register_session_buffer_stamp(trace, attr_val, address)
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
                            register_session_buffer_stamp(trace, item, item_addr)
                            _state._tagged_buffer_ids.add(id(item))
                        except Exception:
                            pass


# ---------------------------------------------------------------------------
# Module forward decorator — reads trace from _state
# ---------------------------------------------------------------------------


def _tag_untagged_buffers(trace: "Trace", module: nn.Module) -> None:
    """Tag any buffers that lack ``_tl.address`` metadata.

    Called during ``_record_module_entry_metadata`` to catch buffers that were created
    dynamically (e.g. in ``forward()``) after the initial ``prepare_buffer_tensors``
    scan. If a buffer already has ``_tl.label_raw`` from being logged as
    an intermediate tensor, that label is moved to ``_tl.buffer_source`` and cleared
    so the buffer gets a fresh source-tensor entry on next use.

    Dynamically stamped buffers join ``trace._session_buffer_inventory`` so the
    r79 inventory-driven cleanup clears them even if they are popped from
    ``_buffers`` later in the same forward. r81: the stamp routes through
    ``register_session_buffer_stamp`` so it also joins the session identity
    registry consulted by the buffer-rung storage-identity belt.
    """
    from .buffer_writes import register_session_buffer_stamp

    for buffer_name, buffer_tensor in module.named_buffers():
        if get_buffer_address(buffer_tensor) is not None:
            continue
        module_addr = _module_address(module)
        if module_addr == "":
            address = buffer_name
        else:
            address = f"{module_addr}.{buffer_name}"
        register_session_buffer_stamp(trace, buffer_tensor, address)
        # If this buffer was already logged as an intermediate tensor, save the
        # previous label as parent and reset so it gets a proper buffer source entry.
        promote_label_to_buffer_source_and_clear_label(buffer_tensor)


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
    from .buffer_writes import session_validated_buffer_address

    module_address = _module_address(module)
    mod_id = id(module)
    trace._module_build_data["module_training_modes"][module_address] = module.training
    module_call_index = trace._mod_call_index[mod_id]
    assert module_call_index > 0, "_module_stack.push_frame must increment before entry"
    module_call_label = (module_address, module_call_index)
    # Push onto stack — popped by _record_module_exit_metadata (or exception handler).
    trace._mod_call_labels[mod_id].append(module_call_label)

    # Stash forward args for later use by _build_module_logs.
    trace._module_forward_args[(module_address, module_call_index)] = (args, kwargs)
    module_call_label_str = f"{module_address}:{module_call_index}"
    _register_module_input_container_snapshots(
        trace,
        args,
        kwargs,
        module_call_label=module_call_label_str,
    )
    should_capture_template = bool(
        getattr(trace, "intervention_ready", False) or getattr(trace, "save_arg_templates", False)
    )
    forward_args_template = None
    forward_kwargs_template = None
    if should_capture_template:
        from .ops import _build_args_template

        captured_template = _build_args_template(module.forward, args, kwargs)
        forward_args_template = captured_template
        forward_kwargs_template = captured_template if kwargs else None
        trace._module_build_data.setdefault("module_forward_templates", {})[
            module_call_label_str
        ] = (
            forward_args_template,
            forward_kwargs_template,
        )
    forward_start_time = time.time()
    trace._module_build_data.setdefault("module_forward_start_times", {})[module_call_label_str] = (
        forward_start_time
    )
    code_context_cache = getattr(trace, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        trace._code_context_cache = code_context_cache
    code_context = _get_code_context(
        num_context_lines=trace.num_context_lines,
        source_loading_enabled=trace.save_code_context,
        context_cache=code_context_cache,
    )
    trace._module_build_data.setdefault("module_code_contexts", {})[module_call_label_str] = (
        code_context
    )
    call_stack = [
        f"{frame.address}:{frame.pass_index}" for frame in trace._exhaustive_module_stack[:-1]
    ]
    trace._module_build_data.setdefault("module_call_stacks", {})[module_call_label_str] = (
        call_stack
    )

    # Find all tensor arguments (excluding Parameters, which are source tensors).
    input_tensors = get_vars_of_type_from_obj(
        [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=5
    )
    input_tensor_labels = set()
    input_tensor_labels_at_entry = []
    for t in input_tensors:
        if is_functorch_wrapped_tensor(t):
            continue
        # Lazily register buffer tensors that haven't been logged yet. r81: the
        # module-entry gate validates the static stamp through the session belt
        # (current-session object + storage identity), never raw.
        label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        buffer_address = session_validated_buffer_address(trace, t)
        if label is None and buffer_address is not None:
            log_source_tensor(trace, t, "buffer", buffer_address)
            label = get_tensor_label(t)
        if label is None:
            # An untagged tensor enters a module. A genuine raw
            # ``register_forward_hook`` output replacement is already tagged at
            # module exit by ``_make_user_forward_hook_wrapper`` (with
            # intervention_replaced=True), so anything still untagged here is an
            # internally generated tensor whose construction TorchLens could not
            # trace (e.g. an attention mask built inside ``torch.vmap``). Log it
            # as a clean graph source -- NOT a user intervention -- so it
            # validates legitimately rather than getting a functionless
            # intervention-replacement placeholder.
            _ensure_module_output_tensor_logged(
                trace, t, module, parent_labels=[], kind="internal_source"
            )
            label = get_tensor_label(t)
        if label is None:
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        input_tensor_labels.add(label)
        trace._mod_entered[mod_id].append(label)
        trace.capture_events.live_index.note_module_entry(mod_id, label, module_address)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                trace._module_build_data["module_layer_argnames"][
                    (f"{module_call_label[0]}:{module_call_label[1]}")
                ].append((label, arg_key))
        input_tensor_labels_at_entry.append(label)

    # Catch buffers created dynamically (e.g. in forward()) after initial scan.
    _tag_untagged_buffers(trace, module)
    trace.capture_events.module_enter_events.append(
        ModuleEnterEvent(
            address=module_address,
            call_index=module_call_index,
            call_label=module_call_label_str,
            training=module.training,
            code_context=tuple(code_context),
            call_stack=tuple(call_stack),
            forward_start_time=forward_start_time,
            forward_args=args,
            forward_kwargs=kwargs,
            forward_args_template=forward_args_template,
            forward_kwargs_template=forward_kwargs_template,
            layer_argnames=tuple(
                trace._module_build_data["module_layer_argnames"][module_call_label_str]
            ),
            input_labels=tuple(input_tensor_labels_at_entry),
        )
    )
    return input_tensor_labels, input_tensor_labels_at_entry


def _register_module_input_container_snapshots(
    trace: "Trace",
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    module_call_label: str,
) -> None:
    """Register tensor-bearing module input containers at call entry.

    Parameters
    ----------
    trace:
        Active trace.
    args:
        Positional module inputs.
    kwargs:
        Keyword module inputs.
    module_call_label:
        Stable module-call label for this invocation.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    registry = trace._ensure_build_state().container_registry
    event_index = int(getattr(trace, "_layer_counter", 0))
    for index, arg in enumerate(args):
        result = walk_container(arg, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            arg,
            site=ModuleSite(module_call_label=module_call_label, position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in kwargs.items():
        result = walk_container(value, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            value,
            site=ModuleSite(module_call_label=module_call_label, position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )


def _register_module_output_container_snapshot(
    trace: "Trace",
    output: Any,
    *,
    module_call_label: str,
) -> None:
    """Register a tensor-bearing module output container at call exit.

    Parameters
    ----------
    trace:
        Active trace.
    output:
        Raw module output object.
    module_call_label:
        Stable module-call label for this invocation.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    result = walk_container(output, role=Role.CALL_OUTPUT, capability="full_spec")
    if result is None:
        return
    trace._ensure_build_state().container_registry.register_snapshot(
        output,
        site=ModuleSite(module_call_label=module_call_label, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=int(getattr(trace, "_layer_counter", 0)),
        spec=result.spec,
        leaf_occurrences=result.leaf_occurrences,
        reconstructable=result.reconstructable,
    )


def _next_untagged_tensor_label(trace: "Trace", layer_type: str) -> tuple[str, int, int]:
    """Return a fresh raw label for an untagged tensor surfaced mid-forward.

    Two distinct kinds of untagged tensors reach this helper:

    * ``"interventionreplacement"`` -- a genuine raw ``register_forward_hook``
      output replacement injected by the user; the tensor lacks traceable
      provenance because the user substituted it for a module's real output.
    * ``"internalsource"`` -- an internally generated tensor (e.g. an attention
      mask built inside ``torch.vmap``, whose construction TorchLens cannot
      trace) that enters a module untagged during plain capture. It is a
      legitimate graph source, not a user intervention.

    Parameters
    ----------
    trace:
        Active model log whose raw layer counters should be advanced.
    layer_type:
        ``"interventionreplacement"`` or ``"internalsource"``.

    Returns
    -------
    tuple[str, int, int]
        Raw label, capture index, and per-type index.
    """

    trace._layer_counter += 1
    trace._raw_layer_type_counter[layer_type] += 1
    raw_index = trace._layer_counter
    type_index = trace._raw_layer_type_counter[layer_type]
    return f"{layer_type}_{type_index}_{raw_index}_raw", raw_index, type_index


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
    kind: str = "intervention_replacement",
) -> str:
    """Log a fresh entry for an unlabeled tensor surfaced mid-forward.

    This handles two distinct, legitimately untraceable cases and must keep
    them distinguishable so validation stays armed (see project CLAUDE.md
    "Validation Integrity"):

    * ``kind="intervention_replacement"`` -- a genuine raw
      ``register_forward_hook`` that replaced a module's output with a fresh
      tensor the user injected. The op is marked ``intervention_replaced`` and
      is legitimately functionless (the user-supplied callable is opaque).
    * ``kind="internal_source"`` -- an internally generated tensor (e.g. an
      attention mask built inside ``torch.vmap``, whose construction TorchLens
      cannot trace) that enters a module untagged during plain capture. It is a
      real graph source (like a buffer/constant), NOT a user intervention, so
      it is logged as an internal source and validates legitimately.

    Parameters
    ----------
    trace:
        Active model log.
    tensor:
        Untagged tensor that lacks ``_tl.label_raw``.
    module:
        Module the tensor is associated with (the replaced module for a hook
        replacement, or the module the tensor first entered for an internal
        source).
    parent_labels:
        Raw labels for tensors that entered the module. Only meaningful for the
        intervention-replacement kind; an internal source has no parents.
    kind:
        ``"intervention_replacement"`` or ``"internal_source"``.

    Returns
    -------
    str
        Raw label of the inserted boundary Op. The tensor is tagged so downstream
        module-exit and op logging can continue.
    """

    from .ops import _make_layer_log_entry, _pop_tensor_live_fire_results

    is_internal_source = kind == "internal_source"
    if is_internal_source:
        # An internal source has no real dataflow parents -- the construction
        # ops are untraceable, so attaching the previous op as a "parent" would
        # be a fabricated edge. Register it as a clean graph source.
        parent_labels = []
    from ...capture.projections import LiveOpView

    parent_entries = [
        LiveOpView(trace, trace.capture_events.live_index.require_event(label))
        for label in parent_labels
    ]
    template_entry = parent_entries[0] if parent_entries else None
    layer_type = "internalsource" if is_internal_source else "interventionreplacement"
    raw_label, raw_index, type_index = _next_untagged_tensor_label(trace, layer_type)
    fields_dict = {
        field_name: _copy_field_value_for_replacement(
            getattr(template_entry, field_name, None) if template_entry is not None else None
        )
        for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }
    address = _module_address(module)
    # The TOP-LEVEL model is never registered in `_mod_call_index` -- its
    # `forward` is deliberately left undecorated ("Root module is handled
    # separately by trace", `_prepare_model_once`'s `_visit_once`), so
    # `push_frame` (the sole incrementer) never runs for it. A raw
    # `register_forward_hook` on the root module itself (depth 0) therefore
    # hits this lookup with a module never present in the dict. Default to 1,
    # matching the codebase's fixed "self:1" convention for the root's single
    # canonical call (see `torchlens/postprocess/finalization.py`). This
    # default is provably inert for the root: `address` is `""` for the root
    # (see `_module_address`), so every use of `module_call_index` below that
    # feeds the module-stack/equivalence-class machinery is gated on
    # `if address` and skips it entirely for the root case; only the
    # `"module"` field write further down carries the value, and it is
    # explicitly `None`-gated there too so a bogus `":1"` label never reaches
    # postprocessing.
    module_call_index = trace._mod_call_index.get(id(module), 1)
    # Both kinds must carry the FULL exhaustive module stack -- exactly like every
    # real op (see sources.py / ops.py) -- not just the innermost frame. Truncating
    # to [(address, idx)] mis-parents any synthesized op whose module is nested 2+
    # address levels deep, because downstream call-tree construction
    # (_finalize.py / finalization.py) treats stack index 0 as "top-level" and wires
    # the op as a direct child of the root, while the module's real ops (which do
    # carry the correct full stack) simultaneously wire the same call label under
    # its true parent -- a bidirectionality conflict that trips the
    # [module_hierarchy] MetadataInvariantError.
    #
    # * internal_source: the untagged tensor enters the CURRENTLY-EXECUTING module
    #   (e.g. esmfold's trunk.structure_module.ipa, synthesized when a vmap/state-
    #   leaked tensor enters a module untagged 2+ levels deep), whose frame is still
    #   on `trace._exhaustive_module_stack` -- the plain snapshot already includes it.
    # * intervention_replacement: a raw `register_forward_hook` fires AFTER the
    #   hooked module's own `decorated_forward` has returned and popped its frame.
    #   The replacement is therefore a module-exit boundary op owned by the live
    #   PARENT scope that consumes the hooked module's output. Re-entering the
    #   hooked module here would make its ModuleCall its own parent and child.
    from .sources import _snapshot_exhaustive_module_stack

    modules = _snapshot_exhaustive_module_stack(trace)
    equivalence_class = _append_module_suffix_to_equivalence_class(raw_label, modules)
    module_args, module_kwargs = trace._module_forward_args.get(
        (address, module_call_index), ((), {})
    )
    quantized_flops_forward = _estimate_quantized_module_forward_flops(
        module,
        tuple(tensor.shape),
        module_args,
        module_kwargs,
    )
    root_ancestors: set[str] = set()
    input_ancestors: set[str] = set()
    internal_source_ancestors: set[str] = set()
    for parent_entry in parent_entries:
        root_ancestors.update(parent_entry.root_ancestors or set())
        input_ancestors.update(parent_entry.input_ancestors or set())
        internal_source_ancestors.update(parent_entry.internal_source_ancestors or set())
    if is_internal_source:
        # A self-rooted internal source: it is its own root and internal-source
        # ancestor, with no input ancestry (mirrors buffer source tagging).
        root_ancestors = {raw_label}
        internal_source_ancestors = {raw_label}

    fields_dict.update(
        {
            "_label_raw": raw_label,
            "_layer_label_raw": raw_label,
            "raw_index": raw_index,
            "step_index": None,
            "source_trace": trace,
            "_tracing_finished": False,
            "_construction_done": False,
            "label": None,
            "label_short": None,
            "layer_label": None,
            "layer_label_short": None,
            "type": layer_type,
            "type_index": type_index,
            "pass_index": 1,
            "num_passes": 1,
            "lookup_keys": [],
            "out": None,
            "transformed_out": None,
            "has_saved_activation": False,
            "activation_transform": trace.activation_transform,
            "annotations": {},
            "interventions": [],
            "intervention_replaced": not is_internal_source,
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
            "activation_memory": get_memory_amount_from_metadata(
                tensor,
                tuple(tensor.shape),
                tensor.dtype,
            ),
            "transformed_activation_memory": None,
            "visualizer_path": None,
            "bytes_delta_at_call": 0,
            "bytes_peak_at_call": 0,
            "autograd_memory": None,
            "num_autograd_tensors": None,
            "has_out_variations": False,
            "out_versions_by_child": {},
            "grad": None,
            "transformed_grad": None,
            "save_grads": getattr(trace, "save_grads", None) not in (None, False),
            "has_grad": False,
            "grad_shape": None,
            "transformed_grad_shape": None,
            "grad_dtype": None,
            "transformed_grad_dtype": None,
            "gradient_memory": 0,
            "transformed_gradient_memory": None,
            "func": None,
            "func_call_id": None,
            "func_name": (
                f"quantized_{_module_type(module).lower()}"
                if is_internal_source and quantized_flops_forward is not None
                else "none"
                if is_internal_source
                else "intervention_replacement"
            ),
            "func_qualname": None,
            "code_context": [],
            "func_duration": 0,
            "flops_forward": quantized_flops_forward or 0,
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
            "grad_fn_object_id": id(tensor.grad_fn) if tensor.grad_fn is not None else None,
            "grad_fn_handle": tensor.grad_fn,
            "grad_fn": None,
            "in_multi_output": False,
            "multi_output_index": None,
            "multi_output_name": None,
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
            "equivalent_ops": trace.op_equivalence_classes[raw_label],
            "recurrent_ops": [],
            "parents": parent_labels,
            "parent_arg_positions": {"args": {}, "kwargs": {}},
            "_edge_uses": [],
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
            "min_distance_to_output": None,
            "max_distance_to_output": None,
            "io_role": None,
            "is_buffer": False,
            "address": None,
            "buffer_pass": None,
            "buffer_source": None,
            "buffer_write_kind": None,
            "buffer_value_changed": None,
            "buffer_replay_validated": None,
            "buffer_source_func_name": None,
            "is_internal_source": is_internal_source,
            "has_internal_source_ancestor": is_internal_source
            or any(entry.has_internal_source_ancestor for entry in parent_entries),
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
            # Match ordinary op ownership: the innermost live frame owns the op.
            # For intervention replacements this is the parent scope consuming
            # the exited module's output, never the exited module itself.
            "module": modules[-1] if modules else None,
            "_address_normalized": None,
            "modules": modules,
            "module_call_stack": [],
            "module_entry_arg_keys": defaultdict(list),
            "input_to_module_calls": [],
            "output_of_modules": [],
            "output_of_module_calls": [],
            "is_module_output": False,
            "is_atomic_module": False,
            "atomic_module_call": None,
            "func_config": {},
        }
    )
    fire_results = _pop_tensor_live_fire_results(tensor)
    if fire_results:
        fields_dict["fire_results"] = fire_results
        fields_dict["interventions"] = [
            result.fire_record for result in fire_results if result.fire_record is not None
        ]
        fields_dict["intervention_replaced"] = any(result.replaced for result in fire_results)
    trace.op_equivalence_classes[raw_label].add(raw_label)
    new_entry = _make_layer_log_entry(
        trace, tensor, fields_dict, (), {}, trace.activation_transform
    )
    if is_internal_source:
        # Keep the special-list <-> flag invariant satisfied: every op with
        # is_internal_source set must appear in trace.internal_source_ops.
        trace.internal_source_ops.append(new_entry._label_raw)
    set_tensor_label(tensor, new_entry._label_raw)
    from .ops import _add_tensor_backward_hook

    _add_tensor_backward_hook(trace, tensor, new_entry._label_raw)
    return new_entry._label_raw


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
        expected_token = None
        if (
            _state._escape_detector_mode == "shadow"
            or _state._completeness_witness_mode == "shadow"
        ):
            with expected_original_call(hook_fn, "module_forward_hook:user") as expected_token:
                result = hook_fn(*hook_args, **hook_kwargs)
        else:
            result = hook_fn(*hook_args, **hook_kwargs)
        if result is None or result is original_output:
            mark_expected_original_accounted(expected_token, captured=False)
            return result
        trace = _state._active_trace
        if trace is None or not _state._logging_enabled:
            mark_expected_original_accounted(expected_token, captured=False)
            return result
        parent_labels = [
            label
            for tensor in get_vars_of_type_from_obj(original_output, torch.Tensor, search_depth=4)
            if (
                label := get_live_tensor_label(tensor, trace.capture_events.live_index.by_raw_label)
            )
            is not None
        ]
        captured_replacement_boundary = False
        for replacement in get_vars_of_type_from_obj(result, torch.Tensor, search_depth=4):
            replacement_label = get_live_tensor_label(
                replacement, trace.capture_events.live_index.by_raw_label
            )
            if replacement_label is not None:
                replace_op_event(trace, replacement_label, intervention_replaced=True)
            else:
                _ensure_module_output_tensor_logged(trace, replacement, module, parent_labels)
                captured_replacement_boundary = True
        mark_expected_original_accounted(
            expected_token,
            captured=captured_replacement_boundary,
        )
        return result

    mark_tensor_replacement_wrapped(wrapped_hook)
    return wrapped_hook


def _record_module_exit_metadata(
    trace: "Trace",
    module: nn.Module,
    out: Any,
    input_tensor_labels: set[str],
    input_tensor_labels_at_entry: list[str],
) -> bool:
    """Record post-forward module metadata for exhaustive mode.

    Called immediately after ``orig_forward()`` returns in the
    ``module_forward_decorator``. Pops the module call-label stack, creates
    boundary identity ops for pass-through outputs, recovers replacement outputs,
    and annotates output tensors with module-exit metadata.

    Returns
    -------
    bool
        Whether module-exit reconciliation inserted an explicit boundary Op for
        an otherwise untraceable output tensor.
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
    start_times = trace._module_build_data.setdefault("module_forward_start_times", {})
    forward_duration = 0.0
    if module_call_label in start_times:
        forward_duration = time.time() - start_times[module_call_label]
        trace._module_build_data.setdefault("module_forward_durations", {})[module_call_label] = (
            forward_duration
        )
    output_structure = None
    if output_entries:
        output_structure = output_entries[0][2]
        trace._module_build_data.setdefault("module_output_structures", {})[module_call_label] = (
            output_structure
        )
    _register_module_output_container_snapshot(
        trace,
        out,
        module_call_label=module_call_label,
    )
    output_tensor_labels_raw: list[str] = []
    output_paths: list[tuple[object, ...]] = []
    per_output_atomic: list[tuple[str, tuple[ModuleFrame, ...], bool, tuple[str, int] | None]] = []
    output_names: list[str | None] = []
    captured_untraceable_output = False
    for output_index, (t, container_path, _container_spec) in enumerate(output_entries):
        # nn.Identity modules and pass-through tensors (output is same object
        # as input) need _decorated_identity() to create a distinct log entry
        # so the graph correctly shows the module boundary.
        tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        fire_results = tuple(getattr(t, "_tl_live_fire_results", ()))
        if (_module_type(module).lower() == "identity") or (
            tensor_label is not None and tensor_label in input_tensor_labels
        ):
            intervention_parent_labels: list[str] = list(
                getattr(t, "_tl_module_intervention_parent_labels", ())
            )
            t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            if fire_results:
                try:
                    setattr(t, "_tl_live_fire_results", fire_results)
                    setattr(
                        t,
                        "_tl_module_intervention_parent_labels",
                        tuple(intervention_parent_labels),
                    )
                except Exception:
                    pass
            tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        if tensor_label is None:
            # A live module-boundary intervention deliberately clears copied op
            # labels and leaves typed fire metadata on its fresh output. Preserve
            # that value as an explicit replacement op. Without fire metadata, an
            # untagged module return remains an internal source whose construction
            # TorchLens could not trace (for example, inside ``torch.vmap``).
            intervention_parent_labels = list(
                getattr(t, "_tl_module_intervention_parent_labels", ())
            )
            _ensure_module_output_tensor_logged(
                trace,
                t,
                module,
                parent_labels=intervention_parent_labels,
                kind="intervention_replacement" if fire_results else "internal_source",
            )
            captured_untraceable_output = True
            tensor_label = get_tensor_label(t)
        if tensor_label is None:
            continue
        if fire_results:
            from .ops import _pop_tensor_live_fire_results

            remaining_fire_results = _pop_tensor_live_fire_results(t)
            if remaining_fire_results:
                replace_op_event(
                    trace,
                    tensor_label,
                    intervention_fired=True,
                    intervention_replaced=any(result.replaced for result in remaining_fire_results),
                    fire_results=remaining_fire_results,
                )
        is_atomic_module = _is_bottom_level_submodule_exit(trace, t, module)
        atomic_module_call = (address, module_call_index) if is_atomic_module else None
        output_tensor_labels_raw.append(tensor_label)
        output_paths.append(tuple(container_path))
        event = trace.capture_events.live_index.require_event(tensor_label)
        per_output_atomic.append(
            (
                tensor_label,
                event.module_stack,
                bool(is_atomic_module),
                atomic_module_call,
            )
        )
        output_name = None
        if len(output_entries) > 1:
            output_name = multi_output_role_from_path(
                container_path,
                output_index,
                hints=role_hints,
            )
        output_names.append(output_name)
        trace._mod_exited[mod_id].append(tensor_label)
    trace.capture_events.module_exit_events.append(
        ModuleExitEvent(
            address=address,
            call_index=module_call_index,
            call_label=module_call_label,
            forward_duration=forward_duration,
            output_structure=output_structure,
            output_tensor_labels_raw=tuple(output_tensor_labels_raw),
            output_paths=tuple(output_paths),
            per_output_atomic=tuple(per_output_atomic),
            output_names=tuple(output_names),
        )
    )
    return captured_untraceable_output


def module_forward_decorator(
    orig_forward: Callable[..., Any], module: nn.Module
) -> Callable[..., Any]:
    """Toggle-gated forward wrapper for an nn.Module's ``forward`` method.

    **Closure design**: Closes over ``module`` (a stable instance reference) but
    reads ``trace`` from ``_state._active_trace`` at call time. This is
    necessary because the same wrapper persists across multiple ``trace``
    calls with different Trace instances.

    **Execution modes**:

    1. **Logging off** (``_state._logging_enabled is False``): Pass through to
       ``orig_forward`` with zero overhead beyond one bool check. This is the
       normal production path.

    2. **Exhaustive mode**: Full entry/exit bookkeeping via
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
        """Route one module forward call through TorchLens capture bookkeeping."""
        # ---- Toggle gate: near-zero overhead when logging is off ----
        if not _state._logging_enabled or _state._active_trace is None:
            return orig_forward(*args, **kwargs)

        trace = _state._active_trace

        if trace.capture_mode == "predicate":
            from ...capture.predicates import (
                _evaluate_halt,
                _evaluate_keep_module,
                _is_halt_only_capture,
            )
            from ...capture.projections import (
                _build_record_context,
                append_projected_event,
                get_active_recording_state,
            )
            from ...fastlog.types import ActivationRecord, CaptureSpec

            state = get_active_recording_state()
            frame = _mstack.push_frame(trace, state.module_stack, module)
            from .prehook_provenance import bind_invocation

            bind_invocation(trace, module, frame.address, frame.pass_index, args, kwargs)
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
                step_index=None,
                time_since_pass_start=0.0,
                include_source_events=state.options.include_source_events,
                sample_id=state.sample_id,
            )
            skipped_spec = CaptureSpec(save_out=False, save_metadata=False)
            enter_spec = skipped_spec
            halt_only = _is_halt_only_capture(state.options)
            try:
                if halt_only:
                    _evaluate_halt(enter_ctx, state.options)
                else:
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
                    _evaluate_halt(enter_ctx, state.options)
            except HaltSignal:
                _mstack.pop_frame(state.module_stack, frame)
                raise
            except Exception as exc:
                state.handle_predicate_exception(enter_ctx, exc)
            finally:
                if not halt_only:
                    if not any(
                        event.raw_index == enter_ctx.event_index
                        for event in trace.capture_events.op_events
                    ):
                        append_projected_event(
                            trace,
                            enter_ctx,
                            skipped_spec,
                            predicate_matched=False,
                        )
                    state.append_context(enter_ctx)
            out = None
            try:
                if (
                    _state._escape_detector_mode == "shadow"
                    or _state._completeness_witness_mode == "shadow"
                ):
                    with expected_original_call(orig_forward, "module_forward:predicate"):
                        out = orig_forward(*args, **kwargs)
                else:
                    out = orig_forward(*args, **kwargs)
                return out
            finally:
                active_model_exc = sys.exc_info()[1]
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
                    step_index=None,
                    time_since_pass_start=0.0,
                    include_source_events=state.options.include_source_events,
                    sample_id=state.sample_id,
                )
                exit_spec = skipped_spec
                try:
                    if halt_only:
                        _evaluate_halt(exit_ctx, state.options, frontier_output=out)
                    else:
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
                        _evaluate_halt(exit_ctx, state.options, frontier_output=out)
                except HaltSignal:
                    if active_model_exc is None:
                        raise
                except Exception as exc:
                    if active_model_exc is None:
                        state.handle_predicate_exception(exit_ctx, exc)
                    else:
                        state.add_predicate_failure(exit_ctx, exc)
                finally:
                    if not halt_only:
                        if not any(
                            event.raw_index == exit_ctx.event_index
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
        from .prehook_provenance import bind_invocation

        bind_invocation(trace, module, frame.address, frame.pass_index, args, kwargs)
        try:
            input_tensor_labels, input_tensor_labels_at_entry = _record_module_entry_metadata(
                trace, module, args, kwargs
            )
            expected_token = None
            try:
                if (
                    _state._escape_detector_mode == "shadow"
                    or _state._completeness_witness_mode == "shadow"
                ):
                    with expected_original_call(
                        orig_forward, "module_forward:exhaustive"
                    ) as expected_token:
                        out = orig_forward(*args, **kwargs)
                else:
                    out = orig_forward(*args, **kwargs)
                from ...intervention.runtime import _apply_module_boundary_live_hooks

                out = _apply_module_boundary_live_hooks(
                    out,
                    module_address=frame.address,
                    module_call_index=frame.pass_index,
                    module_type=_module_type(module),
                    call_args=args,
                    call_kwargs=dict(kwargs),
                )
            except Exception:
                # Exception safety: pop module pass label to keep the stack
                # consistent, preventing corruption in subsequent forward calls (#122).
                mod_id = id(module)
                call_labels = trace._mod_call_labels.get(mod_id)
                if call_labels:
                    call_labels.pop()
                raise
            captured_untraceable_output = _record_module_exit_metadata(
                trace, module, out, input_tensor_labels, input_tensor_labels_at_entry
            )
            mark_expected_original_accounted(
                expected_token,
                captured=captured_untraceable_output,
            )
            options = getattr(trace, "_predicate_save_options", None)
            if options is not None and options.halt is not None:
                from ...capture.predicates import _evaluate_halt
                from ...capture.projections import _build_record_context

                exit_ctx = _build_record_context(
                    kind="module_exit",
                    op_log_or_op_data={
                        "label": f"{frame.address}:exit:{frame.pass_index}",
                        "address": frame.address,
                        "module_type": _module_type(module),
                        "module_pass_index": frame.pass_index,
                    },
                    module_stack=[],
                    history=(),
                    op_counts={},
                    pass_index=1,
                    event_index=trace._layer_counter,
                    step_index=None,
                    time_since_pass_start=0.0,
                    include_source_events=False,
                    sample_id=None,
                )
                _evaluate_halt(exit_ctx, options, frontier_output=out)
            return out
        finally:
            _mstack.pop_frame(trace._exhaustive_module_stack, frame)

    return decorated_forward


# ---------------------------------------------------------------------------
# Helper: _is_bottom_level_submodule_exit
# ---------------------------------------------------------------------------


def _is_bottom_level_submodule_exit(trace: "Trace", t: torch.Tensor, submodule: nn.Module) -> bool:
    """Reserved capture-time hook for bottom-level submodule exits.

    Atomic (single-op leaf) module detection is computed in postprocess from the
    finalized op-to-module map (see ``_materialize._module_output_fields``), where
    the full set of ops contained by each module call is available. Computing it
    here at capture time cannot see sibling/side ops (e.g. a BatchNorm's
    ``num_batches_tracked`` increment) and so would mis-flag multi-op leaves as
    atomic. This stub keeps the call site stable and always defers.
    """
    tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
    if tensor_label is None:
        raise KeyError("Tensor is missing TorchLens metadata")
    trace.capture_events.live_index.require_event(tensor_label)
    trace.capture_events.live_index.module_entry_count(id(submodule))
    return False


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


def _restore_session_param_state(trace: "Trace", model: nn.Module) -> None:
    """Restore parameter grad flags and remove session-scoped parameter metadata.

    This cleanup is deliberately independent of the exception that ended the
    capture. Callers invoke it from teardown paths that may be handling any
    ``BaseException`` raised by user code.

    r79 session-leak fix: the AUTHORITATIVE clear iterates the RECORDED prep
    inventory (``trace._session_param_inventory``), never only the live model
    tree -- a parameter popped from ``_parameters`` mid-forward escapes a
    ``model.parameters()`` re-traversal, and its surviving prep stamp would let
    a LATER capture accept stale provenance (false VERIFIED / wrong-bind). The
    live-tree walk is kept as belt-and-suspenders; both passes are idempotent
    (``clear_meta`` pops with a default, ``restore_param_requires_grad`` no-ops
    once the stamp is gone).

    Parameters
    ----------
    trace
        Trace whose session prep recorded the stamped-parameter inventory.
    model
        Model whose parameters were prepared for the capture session.
    """

    inventory = getattr(trace, "_session_param_inventory", None)
    for param in inventory or ():
        restore_param_requires_grad(param)
        clear_meta(param)
    if inventory:
        trace._session_param_inventory = []
    for param in model.parameters():
        restore_param_requires_grad(param)
        clear_meta(param)


def _cleanup_model_session(
    trace: "Trace",
    model: nn.Module,
    input_tensors: Any = None,
    input_objects: Any = None,
) -> None:
    """Clean up session-specific state after a ``trace`` call.

    Restores ``requires_grad`` to its original value on all parameters,
    removes all session-scoped parameter metadata, and strips session-scoped
    tensor metadata.

    **Does NOT** remove permanent module metadata or unwrap ``module.forward`` — those persist for
    the lifetime of the model instance.
    """
    from .prehook_provenance import rollback_prehook_provenance

    try:
        rollback_prehook_provenance(trace)
    finally:
        # Restore requires_grad and remove session-scoped param attributes
        _restore_session_param_state(trace, model)

    # Session-scoped module tracking data lives in Trace dicts (not on
    # modules), so no per-module cleanup iteration is needed — the dicts
    # are GC'd with the Trace.

    # Clean tensor labels from model tensors (buffers, etc.)
    _undecorate_model_tensors(trace, model)

    # Clean tensor labels from input tensors
    seen: set[int] = set()
    if input_tensors:
        for t in input_tensors:
            clear_meta(t)
            seen.add(id(t))
    if input_objects is not None:
        _clear_session_tensor_metadata(input_objects, seen)

    # r83 C1: retire this capture's label-anchoring session LAST, after every
    # cleanup pass that consults it. Any label still carried by an object that
    # outlives the capture is now anchored to a retired session and can never
    # be accepted as provenance by a later capture, whatever route it escaped by.
    end_label_session()


def _clear_session_tensor_metadata(value: Any, seen: set[int], depth: int = 0) -> None:
    """Clear TorchLens tensor metadata from a model-owned object graph.

    Parameters
    ----------
    value
        Candidate object reachable from a prepared model.
    seen
        Object ids already visited during this cleanup scan.
    depth
        Current recursion depth, used to bound traversal through arbitrary
        third-party helper objects.

    Returns
    -------
    None
        Mutates reachable tensors in place by removing TorchLens metadata.
    """

    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    if isinstance(value, ModuleType):
        # r81 (r80 F1 root B): a stamped tensor stashed as a DIRECT attribute of
        # a ``types.ModuleType`` escaped the belt entirely (this walk returned
        # immediately for modules). Sweep the module namespace SHALLOWLY for
        # plain tensors only -- deep recursion into arbitrary imported modules
        # (``torch``, ``numpy``) would be unbounded; deeper stashes are covered
        # by the session identity belt, which never trusts an unregistered
        # stamp anyway.
        obj_id = id(value)
        if obj_id in seen or depth >= 12:
            return
        seen.add(obj_id)
        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, dict):
            for item in list(namespace.values()):
                if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                    clear_meta(item)
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            clear_meta(value)
        return
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    if isinstance(value, dict):
        for key, item in value.items():
            _clear_session_tensor_metadata(key, seen, depth + 1)
            _clear_session_tensor_metadata(item, seen, depth + 1)
        return
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            _clear_session_tensor_metadata(item, seen, depth + 1)
        return
    if isinstance(value, nn.Module):
        return
    namespace = getattr(value, "__dict__", None)
    if namespace is None:
        return
    for item in namespace.values():
        _clear_session_tensor_metadata(item, seen, depth + 1)


def _clear_callable_session_tensor_metadata(callable_obj: Any, seen: set[int]) -> None:
    """Clear TorchLens tensor metadata captured by a callable object.

    Parameters
    ----------
    callable_obj
        Candidate callable, such as a model's bound ``forward`` method.
    seen
        Object ids already visited during this cleanup scan.

    Returns
    -------
    None
        Mutates reachable tensor metadata in place.
    """

    raw_callable = getattr(callable_obj, "__func__", callable_obj)
    defaults = getattr(raw_callable, "__defaults__", None) or ()
    _clear_session_tensor_metadata(defaults, seen)
    kwdefaults = getattr(raw_callable, "__kwdefaults__", None) or {}
    _clear_session_tensor_metadata(kwdefaults, seen)
    closure = getattr(raw_callable, "__closure__", None) or ()
    for cell in closure:
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        _clear_session_tensor_metadata(cell_value, seen)
    globals_dict = getattr(raw_callable, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    code = getattr(raw_callable, "__code__", None)
    if code is None:
        return
    for name in code.co_names:
        if name in globals_dict:
            _clear_session_tensor_metadata(globals_dict[name], seen)


def _undecorate_model_tensors(trace: "Trace", model: nn.Module) -> None:
    """Remove session-scoped metadata from non-parameter tensors in the model.

    Uses a bounded ``__dict__`` scan instead of ``iter_accessible_attributes``
    (slow dir() + getattr MRO walk). Handles tensors stored directly as
    attributes, inside Python containers, and inside model-owned helper objects.

    r79 session-leak fix: the AUTHORITATIVE clear iterates the RECORDED buffer
    inventory (``trace._session_buffer_inventory``) first -- a buffer popped
    from ``_buffers`` mid-forward escapes the ``model.modules()`` re-traversal
    below, and its surviving ``TensorMeta`` address would let a later capture
    accept stale buffer provenance. The traversal is kept as belt-and-suspenders
    for unstamped model-owned tensors; ``clear_meta`` is idempotent.

    r81: the session identity registry (``trace._session_buffer_identity``) is
    cleared alongside -- its entries pin the stamped objects and their stamp-time
    storages, so releasing it both clears any stamp the inventory might ever
    miss and drops the storage keepers.
    """
    buffer_inventory = getattr(trace, "_session_buffer_inventory", None)
    for stamped_tensor in buffer_inventory or ():
        clear_meta(stamped_tensor)
    if buffer_inventory:
        trace._session_buffer_inventory = []
    identity_registry = getattr(trace, "_session_buffer_identity", None)
    if identity_registry:
        for stamp_entry in identity_registry.values():
            clear_meta(stamp_entry.tensor)
        trace._session_buffer_identity = {}
    seen: set[int] = set()
    for submodule in model.modules():
        for attr_val in submodule.__dict__.values():
            _clear_session_tensor_metadata(attr_val, seen)
        _clear_callable_session_tensor_metadata(getattr(submodule, "forward", None), seen)
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
    3. ``patch_detached_references(model=model)`` — Incremental identity crawl
       plus model-provenance candidates under scoped policy.
    4. ``patch_model_instance(model)`` — Per-capture Level 4 scan, including
       callable attributes reassigned since a prior capture.
    """
    from .wrappers import wrap_torch, patch_detached_references, patch_model_instance

    wrap_torch()  # idempotent — no-op if already wrapped; auto-rewraps after unwrap
    _prepare_model_once(model)  # idempotent — cached in _state._prepared_models
    patch_detached_references(model=model)
    patch_model_instance(model)
