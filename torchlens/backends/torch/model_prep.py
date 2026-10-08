"""Prepare torch ``nn.Module`` objects for capture sessions.

One-time preparation installs persistent forward wrappers and module metadata.
Per-session preparation populates Trace state, parameter logs, and buffer labels.
"""

import copy
import inspect
import itertools
import math
import sys
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from ... import _state
from ..._capture_state_helpers import _is_uninitialized_param
from ..._errors import CaptureContextError
from ...constants import LAYER_PASS_LOG_FIELD_ORDER
from ...data_classes._module_role_hints import multi_output_role_from_path, role_hints_for_module
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.module import HookInfo
from ...data_classes.param import Param, ParamAccessor
from ...fastlog._halt import HaltSignal
from ...ir import (
    CaptureEvents,
    ModuleEnterEvent,
    ModuleExitEvent,
    ModuleFrame,
    ModulePrepEvent,
)
from ...ir.container_registry import ModuleSite, Phase, Role, walk_container
from ...ir.op_record import (
    amend_module_boundary_retention,
    amend_module_exit_intervention,
    amend_raw_hook_intervention,
)
from ...quantities import Bytes
from ...utils.hashing import make_random_barcode
from ...utils.introspection import (
    _get_code_context,
    get_arg_tensors_for_resolution,
    get_vars_of_type_from_obj,
)
from ...utils.tensor_utils import (
    get_memory_amount,
    get_memory_amount_from_metadata,
    is_functorch_wrapped_tensor,
)
from . import module_stack as _mstack
from ._held_refs import normalize_held_torch_function_refs, register_released_model
from ._module_arg_stubs import first_stub_shape, stub_module_arg_payloads
from ._module_boundary_adoption import (
    collect_pre_forward_tensor_ids,
    record_module_boundary_adoption,
)
from ._tl import (
    begin_label_session,
    clear_meta,
    clear_param_meta,
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
from .escape_detection import (
    expected_original_call,
    mark_expected_original_accounted,
)
from .sources import log_source_tensor
from .tensor_tracking import _append_module_suffix_to_equivalence_class

# Cache class-level module metadata (inspect.getsourcelines, inspect.signature, etc.)
# shared across instances during one capture. Cleanup releases it at the end of
# the session; the next capture rebuilds from the current class definitions.
_module_class_metadata_cache: dict[type, dict[str, Any]] = {}

# Process-stable memo for source start lines. ``inspect.findsource`` re-tokenizes
# (functions) or fully AST-parses (classes, CPython < 3.13) the defining file on
# every call, a fixed multi-ms tax repaid each capture because the per-session
# cache above is cleared. The start line is a fact of the definition site, so it
# is keyed on the class object itself or on a function's code object: a
# reloaded/redefined class or a monkey-patched method is a *new* object and can
# never be served a stale entry. Weak keys keep dynamically created classes and
# code collectable; failures are never cached, so exception behavior is unchanged.
_source_line_cache: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()


def _source_start_line(obj: Any) -> int:
    """Memoized ``inspect.getsourcelines(obj)[1]`` for classes and functions."""
    key = obj if isinstance(obj, type) else getattr(obj, "__code__", None)
    if key is not None:
        try:
            line = _source_line_cache.get(key)
        except TypeError:
            key = None
        else:
            if line is not None:
                return line
    line = inspect.getsourcelines(obj)[1]
    if key is not None:
        _source_line_cache[key] = line
    return line


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
    if tensors:
        return tuple(tensors[0].shape)
    # F20 W1a: module-arg stashes carry payload-free stubs; their recorded
    # shape serves the same estimation read.
    return first_stub_shape(value)


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
    if original_forward is None:
        module.__dict__.pop("forward", None)
        return
    # When the recovered forward is just the module's own class method, drop
    # the instance override instead of pinning the bound method as an instance
    # attribute: an instance-level forward churns the implementation
    # fingerprint (`_fingerprint_model_implementation` folds it), so the
    # documented trace -> release_model -> trace(cache=True) workflow missed
    # the cache on every released model.
    original_func = getattr(original_forward, "__func__", None)
    if original_func is not None and original_func is inspect.getattr_static(
        type(module), "forward", None
    ):
        module.__dict__.pop("forward", None)
    else:
        module.forward = original_forward


def _refuse_release_during_active_capture() -> None:
    """Refuse model release while a capture owns the logging globals.

    Sibling of the ``unwrap_torch`` mid-capture guard: releasing a model
    strips the ``._tl`` module metadata and forward decorations the live
    capture's module-attribution reads, so a mid-forward release (reachable
    single-threaded from a forward hook or ``activation_transform``) let the
    capture finish ``capture_verified`` with silently emptied module
    attribution instead of failing loudly.

    Raises
    ------
    CaptureContextError
        If ``_active_trace`` is set or logging is enabled.
    """

    if _state._active_trace is None and not _state._logging_enabled:
        return
    trace = _state._active_trace
    model_label = getattr(trace, "model_label", None) or getattr(trace, "model_class_name", None)
    raise CaptureContextError(
        "tl.release_model() was called while a TorchLens capture is still active"
        + (f" for model {model_label!r}" if model_label else ""),
        code="release_during_active_capture",
        remedy=(
            "let the capture finish before releasing the model — releasing "
            "mid-forward strips the module metadata the capture is reading and "
            "silently empties module attribution"
        ),
        owner_thread_id=_state._active_owner_thread_id,
        calling_thread_id=threading.get_ident(),
    )


def release_model(model: nn.Module) -> None:
    """Remove persistent TorchLens preparation from a PyTorch module tree.

    Parameters
    ----------
    model:
        Root module whose full current module tree should be released.

    Returns
    -------
    None
        The model is modified in place. Releasing an unprepared model is a no-op.

    Raises
    ------
    CaptureContextError
        If a capture is currently active (code
        ``release_during_active_capture``). Releasing mid-capture strips the
        ``tl_*`` / ``._tl`` metadata the capture's module attribution reads,
        so the call is refused instead of finishing a silently degraded trace.

    Notes
    -----
    Persistent non-root ``forward`` wrappers make whole-model pickling fail.
    This operation restores those forwards, clears TorchLens-owned module
    metadata and legacy ``tl_*`` instance attributes, and evicts all related
    preparation bookkeeping so a later trace prepares the tree from scratch.
    It also normalizes plain module attributes holding epoch-mismatched torch
    function references (``self.act = F.relu`` captured in the other wrap
    state) to the values currently live at their public names, so
    ``pickle``/``torch.save`` succeed at release time, and registers the
    model so a later wrap-state flip (``unwrap_torch()``/re-wrap)
    re-normalizes it: a released model stays serializable in every epoch.

    The refusal check and the release run under ``_capture_admission_lock``
    (the same seam the ``unwrap_torch`` guard uses): a capture racing this
    call is either seen by the refusal or blocks at admission until the
    release is complete and then prepares the tree from scratch — it can
    never interleave with a half-stripped module tree. No other lock is
    acquired inside the window.
    """
    with _state._capture_admission_lock:
        _refuse_release_during_active_capture()
        modules = tuple(model.modules())
        try:
            for module in modules:
                _restore_undecorated_forward(module)
                for attr_name in tuple(module.__dict__):
                    if attr_name.startswith("tl_"):
                        module.__dict__.pop(attr_name, None)
                clear_meta(module)
                normalize_held_torch_function_refs(module)
        finally:
            # Evict preparation bookkeeping on EVERY exit: a fault (or ^C)
            # mid-loop otherwise left a half-stripped tree the registry still
            # certified as prepared, and the next capture took the
            # already-prepared fast path and silently emitted incomplete
            # module containment. Evicted, the next trace re-prepares the
            # tree from scratch as the docstring promises.
            _state.release_model_prep(model, modules)
        register_released_model(model)


def _prepare_model_once(model: nn.Module) -> None:
    """Phase 1: One-time (per role) model preparation.

    Fast-path cached via ``_state._prepared_models`` (WeakSet) for the common
    case: a model traced repeatedly in a fixed role, or independent models
    traced in any interleaving. Performs three tasks for each submodule:

    1. **Assigns permanent metadata** — ``_tl.address`` (dotted path
       like ``"encoder.layer.0.attention"``) and ``_tl.module_type`` (class
       name). These survive across sessions.

    2. **Wraps ``forward``** — Replaces ``module.forward`` with
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
       ``trace._module_capture_ws.module_metadata``.
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
    _state._dir_cache.clear()
    trace._module_capture_ws.exhaustive_module_stack = []
    # F41 bound-method roots: the synthetic TL-authored root's identity reads
    # the OWNER (type(owner).__name__, the owner's class/source metadata) and
    # the bound METHOD as the entry callable -- never the wrapper class name
    # or the string "method".
    from .bound_root import TLBoundMethodRoot

    if isinstance(model, TLBoundMethodRoot):
        identity_cls = type(model.tl_owner)
        entry_callable = getattr(identity_cls, model.tl_method_name, None)
    else:
        identity_cls = type(model)
        entry_callable = getattr(identity_cls, "forward", None)
    trace.model_class_name = str(identity_cls.__name__)
    trace.class_docstring = identity_cls.__doc__
    init_method = getattr(identity_cls, "__init__", None)
    forward_method = entry_callable
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
        trace.class_source_file = inspect.getfile(identity_cls)
        trace.class_source_line = _source_start_line(identity_cls)
        trace.init_source_file = inspect.getfile(identity_cls.__init__)
        trace.init_source_line = _source_start_line(identity_cls.__init__)
    except (OSError, TypeError):
        trace.class_source_file = None
        trace.class_source_line = None
        trace.init_source_file = None
        trace.init_source_line = None
    try:
        forward_func = model.forward if entry_callable is None else entry_callable
        trace.forward_source_file = inspect.getsourcefile(forward_func) or inspect.getfile(
            forward_func
        )
        trace.forward_source_line = _source_start_line(forward_func)
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
        meta = trace._module_capture_ws.module_metadata.get(meta_address)
        if meta is not None:
            capture_events = getattr(trace, "capture_events", None)
            if capture_events is None:
                capture_events = CaptureEvents()
                trace.capture_events = capture_events
            capture_events.append_module_prep(
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
            trace._module_capture_ws.module_build_data["module_types"][address] = _module_type(
                module
            )
            # Session-scoped tracking in Trace dicts (keyed by id(module)).
            mod_id = id(module)
            trace._module_capture_ws.mod_call_index[mod_id] = 0
            trace._module_capture_ws.mod_call_labels[mod_id] = []
            trace._module_capture_ws.mod_entered[mod_id] = []
            trace._module_capture_ws.mod_exited[mod_id] = []
    if trace.capture_mode != "predicate":
        _create_session_param_logs(trace, model, optimizer)
    prepare_buffer_tensors(trace, model)
    # Pre-forward ownership snapshot for the R16 module-entry adoption
    # disclosure: tensors reachable NOW (nested caches, forward globals) are
    # model-owned known sources; anything first seen mid-forward is not.
    owned_at_entry, forward_outside_ids = collect_pre_forward_tensor_ids(model)
    trace._module_capture_ws.module_build_data["model_owned_tensor_ids_at_entry"] = owned_at_entry
    trace._module_capture_ws.module_build_data["forward_outside_tensor_ids_at_entry"] = (
        forward_outside_ids
    )
    if trace.capture_mode == "exhaustive":
        from .buffer_writes import install_buffer_write_tracker

        install_buffer_write_tracker(trace, model)
    from .prehook_provenance import install_prehook_provenance

    install_prehook_provenance(
        trace,
        model,
        forward_hook_wrapper_factory=_make_user_forward_hook_wrapper,
    )
    # Accelerate offload/dispatch hooks (lane F37): run hook internals under
    # pause_logging and re-stamp hook-materialized params so offloaded models
    # capture with clean graphs and attributed weights.
    from .offload_hooks import install_offload_hook_shims

    install_offload_hook_shims(trace, model)


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

            # Lazy modules (nn.LazyLinear etc.) hold UninitializedParameter
            # until the first forward; shape/numel/dtype-kind access raises.
            # The pre-forward scan TOLERATES them: register the identity now
            # with deferred geometry, and finalize the inventory after the one
            # captured forward materializes them in place (postprocess step 15).
            param_is_lazy = _is_uninitialized_param(param)

            # Save original requires_grad before forcing True. Integer/bool-dtype
            # Parameters (e.g. a fixed nn.Parameter(torch.arange(...), requires_grad=False)
            # lookup buffer) are legal PyTorch and never gradient-capable; forcing
            # requires_grad on them raises, so only force floating/complex dtypes.
            requires_grad_before = param.requires_grad
            if (
                not param_is_lazy
                and param.is_leaf  # a non-leaf already requires grad; its flag is read-only
                and not getattr(trace, "backward_ready", False)
                and (torch.is_floating_point(param) or torch.is_complex(param))
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

            param_fsize = Bytes(0) if param_is_lazy else get_memory_amount(param)
            param_log = Param(
                module_address=module_address,
                name=param_name,
                shape=() if param_is_lazy else tuple(param.shape),
                dtype=param.dtype,
                num_params=0 if param_is_lazy else param.numel(),
                param_memory=param_fsize,
                trainable=requires_grad_before,
                address=param_address,
                barcode=barcode,
                has_optimizer=id(param) in optimized_param_ids if optimizer is not None else None,
            )
            param_log._param_ref = param
            if param_is_lazy:
                param_log._lazy_at_prep = True  # type: ignore[attr-defined]
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
            meta["class_source_line"] = _source_start_line(module_class)
        except (TypeError, OSError):
            meta["class_source_line"] = None

        init_method = getattr(module_class, "__init__", None)
        try:
            if init_method is not None and init_method is not nn.Module.__init__:
                meta["init_source_file"] = inspect.getsourcefile(init_method) or inspect.getfile(
                    init_method
                )
                meta["init_source_line"] = _source_start_line(init_method)
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
                meta["forward_source_line"] = _source_start_line(forward_method)
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
            source_line = _source_start_line(hook)
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
        if primary in trace._module_capture_ws.module_metadata:
            trace._module_capture_ws.module_metadata[primary]["all_addresses"].append(address)
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

    trace._module_capture_ws.module_metadata[address] = meta


# ---------------------------------------------------------------------------
# Buffer tensor preparation
# ---------------------------------------------------------------------------


def prepare_buffer_tensors(trace: "Trace", model: nn.Module) -> None:
    """Tag buffer tensors with ``_tl.address`` for later identification.

    Buffers are non-parameter tensors registered via ``register_buffer()`` or
    held as plain tensors in module state (attribute, list/tuple item, dict value,
    bounded nesting; see ``iter_module_held_plain_tensors``). They are tagged here so
    that when a buffer first appears as an argument to a wrapped torch function, the
    interceptor can call ``log_source_tensor`` with the correct address.

    Uses ``named_buffers()`` for registered buffers and a ``__dict__`` scan for
    module-held plain tensors (faster than ``iter_accessible_attributes`` which
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
    from .buffer_writes import (
        iter_module_held_plain_tensors,
        register_session_buffer_stamp,
        warn_held_scan_truncated,
    )

    _state._tagged_buffer_ids.clear()
    trace._session_buffer_inventory = []
    trace._session_buffer_identity = {}
    unstampable: list[str] = []
    held_truncations: list[str] = []

    def _stamp(tensor: torch.Tensor, address: str) -> None:
        """Stamp one buffer into the session registries, collecting failures."""
        # A failed stamp is EVIDENCE LOSS -- the buffer's reads may log as
        # internal sources instead of buffer versions -- so it is collected
        # and disclosed once below instead of silently swallowed (B1-13a).
        try:
            register_session_buffer_stamp(trace, tensor, address)
            _state._tagged_buffer_ids.add(id(tensor))
        except Exception as exc:
            unstampable.append(f"{address} ({type(exc).__name__}: {exc})")

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
                _stamp(buf_tensor, address)
        # Module-held plain tensors (attribute, list/tuple item, dict value such as a
        # warm-filled cache) are buffer sources too, never dangling reads.
        held = iter_module_held_plain_tensors(submodule, held_truncations, module_addr)
        for held_name, held_tensor in held:
            if get_buffer_address(held_tensor) is None:
                _stamp(held_tensor, f"{module_addr}.{held_name}" if module_addr else held_name)
    warn_held_scan_truncated(held_truncations, trace)
    if unstampable:
        import warnings

        shown = "; ".join(unstampable[:5])
        suffix = "" if len(unstampable) <= 5 else f" (+{len(unstampable) - 5} more)"
        warnings.warn(
            f"TorchLens could not stamp buffer provenance on "
            f"{len(unstampable)} model tensor(s): {shown}{suffix}. Reads of "
            "these tensors may log as internal sources instead of buffers.",
            stacklevel=2,
        )


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
    trace._module_capture_ws.module_build_data["module_training_modes"][module_address] = (
        module.training
    )
    module_call_index = trace._module_capture_ws.mod_call_index[mod_id]
    if module_call_index <= 0:
        # A hard raise, not an assert: ``python -O`` strips asserts, and this
        # guard catches a broken push_frame/entry ordering that would silently
        # mislabel every module call in the capture (R24-5).
        raise RuntimeError(
            "TorchLens internal error: _module_stack.push_frame must increment "
            f"before module entry (module {module_address!r} has call index "
            f"{module_call_index})"
        )
    module_call_label = (module_address, module_call_index)
    # Push onto stack — popped by _record_module_exit_metadata (or exception handler).
    trace._module_capture_ws.mod_call_labels[mod_id].append(module_call_label)

    # Stash forward args for later use by _build_module_logs.
    trace._module_capture_ws.module_forward_args[(module_address, module_call_index)] = (
        args,
        kwargs,
    )
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
        trace._module_capture_ws.module_build_data.setdefault("module_forward_templates", {})[
            module_call_label_str
        ] = (
            forward_args_template,
            forward_kwargs_template,
        )
    forward_start_time = time.time()
    trace._module_capture_ws.module_build_data.setdefault("module_forward_start_times", {})[
        module_call_label_str
    ] = forward_start_time
    code_context_cache = getattr(trace, "_code_context_cache", None)
    if code_context_cache is None:
        code_context_cache = {}
        trace._code_context_cache = code_context_cache
    code_context = _get_code_context(
        num_context_lines=trace.num_context_lines,
        source_loading_enabled=trace.save_code_context,
        context_cache=code_context_cache,
    )
    trace._module_capture_ws.module_build_data.setdefault("module_code_contexts", {})[
        module_call_label_str
    ] = code_context
    call_stack = [
        f"{frame.address}:{frame.pass_index}"
        for frame in trace._module_capture_ws.exhaustive_module_stack[:-1]
    ]
    trace._module_capture_ws.module_build_data.setdefault("module_call_stacks", {})[
        module_call_label_str
    ] = call_stack

    # Find all tensor arguments (excluding Parameters, which are source tensors).
    input_tensors = get_arg_tensors_for_resolution(args, kwargs)
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
            # R16: adoption must not LAUNDER an escape (see the helper).
            record_module_boundary_adoption(trace, t, label, "entry", module_address)
        if label is None:
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        input_tensor_labels.add(label)
        trace._module_capture_ws.mod_entered[mod_id].append(label)
        trace.capture_events.live_index.note_module_entry(mod_id, label, module_address)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                trace._module_capture_ws.module_build_data["module_layer_argnames"][
                    (f"{module_call_label[0]}:{module_call_label[1]}")
                ].append((label, arg_key))
        input_tensor_labels_at_entry.append(label)

    # Catch buffers created dynamically (e.g. in forward()) after initial scan.
    _tag_untagged_buffers(trace, module)
    # F20 W1a (release-at-emission): neither the workspace stash nor the
    # enter event may pin live payloads for the rest of the forward. Every
    # consumer on the torch trace path reads shape/dtype facts only, and
    # GC-11 nulls ``ModuleCall.forward_args`` before the trace is returned.
    # Stubbing waits until here so the input-adoption loop above has stamped
    # labels the stubs can carry.
    stub_args = stub_module_arg_payloads(args)
    stub_kwargs = stub_module_arg_payloads(kwargs)
    trace._module_capture_ws.module_forward_args[(module_address, module_call_index)] = (
        stub_args,
        stub_kwargs,
    )
    trace.capture_events.append_module_enter(
        ModuleEnterEvent(
            address=module_address,
            call_index=module_call_index,
            call_label=module_call_label_str,
            training=module.training,
            code_context=tuple(code_context),
            call_stack=tuple(call_stack),
            forward_start_time=forward_start_time,
            forward_args=stub_args,
            forward_kwargs=stub_kwargs,
            forward_args_template=forward_args_template,
            forward_kwargs_template=forward_kwargs_template,
            layer_argnames=tuple(
                trace._module_capture_ws.module_build_data["module_layer_argnames"][
                    module_call_label_str
                ]
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
    registry = trace._wrapper_runtime_ws.container_registry
    event_index = trace._raw_graph_ws.layer_counter
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
    trace._wrapper_runtime_ws.container_registry.register_snapshot(
        output,
        site=ModuleSite(module_call_label=module_call_label, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
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

    trace._raw_graph_ws.layer_counter += 1
    trace._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
    raw_index = trace._raw_graph_ws.layer_counter
    type_index = trace._raw_graph_ws.raw_layer_type_counter[layer_type]
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


def _note_replacement_event(
    trace: "Trace",
    raw_label: str | None,
    *,
    origin: str = "raw_forward_hook",
) -> None:
    """Append the journal edit record for one genuinely observed replacement.

    Interventions are EDITS in the capture journal
    (``InterventionAppliedEvent`` referencing the edited label), never op
    kinds and never a side ledger. The record is appended ONLY at the sites
    that directly observe the replacement event itself (a raw
    ``register_forward_hook`` returning a new object, or a live-fire
    intervention hook reporting ``replaced=True`` while an intervention spec
    or hook plan is actually armed for this capture), so it is the
    trace-level ground truth the functionless-op validation carve-out
    requires: a placeholder minted during PLAIN capture (a capture gap, or
    forged per-op attributes) can never mint one and must STILL fail
    validation (2026-06-02 lesson).

    Parameters
    ----------
    trace:
        Active model log.
    raw_label:
        Raw label of the op whose value was genuinely replaced.
    origin:
        Which observation site directly witnessed the edit.
    """

    if trace is None or not isinstance(raw_label, str):
        return
    events = getattr(trace, "capture_events", None)
    if events is None:
        return
    from ...ir.events import InterventionAppliedEvent

    # Causal binding stamped at the observation site: the edited op's event
    # already exists in this journal (the boundary/replacement op was logged
    # before the edit is noted), so the edit records the run nonce and the
    # exact target event instance. Validation refuses an edit without a live
    # binding, so a record appended anywhere else stays inert.
    target_event = events.op_event_by_label_raw.get(raw_label)
    events.append_intervention(
        InterventionAppliedEvent(
            label_raw=raw_label,
            kind="replaced",
            origin=origin,  # type: ignore[arg-type]
            timestamp=time.time(),
            run_token=events.run_nonce,
            target_seq=int(getattr(target_event, "seq", 0) or 0),
            target_func_call_id=getattr(target_event, "func_call_id", None),
        )
    )


def _live_intervention_machinery_armed() -> bool:
    """Return whether live-fire intervention dispatch is armed for this capture.

    ``_tl_live_fire_results`` is a plain Python attribute on tensor OBJECTS and
    can survive across traces on user-retained tensors (a cross-trace leak). A
    fire result observed while NO intervention spec or hook plan is armed is
    therefore definitionally stale and must not mint replacement-event
    evidence for the current (plain) capture.

    Returns
    -------
    bool
        True when the current capture has intervention machinery armed.
    """

    return _state._active_intervention_spec is not None or _state._active_hook_plan is not None


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
    module_call_index = trace._module_capture_ws.mod_call_index.get(id(module), 1)
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
    #   on `trace._module_capture_ws.exhaustive_module_stack` -- the plain snapshot already includes it.
    # * intervention_replacement: a raw `register_forward_hook` fires AFTER the
    #   hooked module's own `decorated_forward` has returned and popped its frame.
    #   The replacement is therefore a module-exit boundary op owned by the live
    #   PARENT scope that consumes the hooked module's output. Re-entering the
    #   hooked module here would make its ModuleCall its own parent and child.
    from .sources import _snapshot_exhaustive_module_stack

    modules = _snapshot_exhaustive_module_stack(trace)
    equivalence_class = _append_module_suffix_to_equivalence_class(raw_label, modules)
    module_args, module_kwargs = trace._module_capture_ws.module_forward_args.get(
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
            "bytes_delta_at_call": None,
            "bytes_peak_at_call": None,
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
        # Fire results only mint replacement-event evidence when the live-fire
        # machinery is actually armed for THIS capture; a stale
        # ``_tl_live_fire_results`` attribute leaked from an earlier intervened
        # trace stays unledgered, so validation refuses the placeholder it
        # would otherwise launder into a plain capture.
        if fields_dict["intervention_replaced"] and _live_intervention_machinery_armed():
            _note_replacement_event(trace, raw_label, origin="live_fire")
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
        replacement_boundaries: list[tuple[torch.Tensor, str]] = []
        for replacement in get_vars_of_type_from_obj(result, torch.Tensor, search_depth=4):
            replacement_label = get_live_tensor_label(
                replacement, trace.capture_events.live_index.by_raw_label
            )
            if replacement_label is not None:
                if replacement_label in parent_labels:
                    # Recontainer: the hook returned the module's own original
                    # output tensor (possibly rewrapped in a new container).
                    # That rewires the module boundary but does not replace the
                    # producing op's value, so it must not mint replacement
                    # evidence that would exempt the native op from replay
                    # validation.
                    continue
                replaced_event = trace.capture_events.op_event_by_label_raw.get(replacement_label)
                if replaced_event is not None:
                    trace.capture_events.append_amendment(
                        amend_raw_hook_intervention(
                            replaced_event.seq,
                            replacement_label,
                            intervention_replaced=True,
                        )
                    )
                # The hook returned an ALREADY-TRACED tensor: that REWIRES the
                # module boundary to reuse an existing op's value, it does not
                # replace that op's own computation. The durable
                # ``intervention_replaced`` stamp above stays as the
                # intervened-capture disclosure (runnable save keys its
                # user_intervention_not_replayable refusal on it), but NO
                # trace-level replacement-event ledger entry is minted here:
                # that ledger corroboration is what exempts FUNCTIONLESS
                # ``intervention_replacement`` ops from validation, and a
                # traced tensor's producing op is an input/buffer/real op with
                # its own honest exemption or replayable function. Minting it
                # blessed ANY functionless op a hook happened to return --
                # masking exactly the lost-func plain-capture gap the
                # 2026-06-02 tripwire rule requires to STILL fail. Genuine
                # opaque replacements (fresh untraced tensors) mint their
                # ledger evidence on the synthesized boundary op below.
            else:
                boundary_label = _ensure_module_output_tensor_logged(
                    trace, replacement, module, parent_labels
                )
                _note_replacement_event(trace, boundary_label)
                replacement_boundaries.append((replacement, boundary_label))
        mark_expected_original_accounted(
            expected_token,
            captured=bool(replacement_boundaries),
            boundary_outputs=tuple(replacement_boundaries),
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
) -> tuple[tuple[torch.Tensor, str], ...]:
    """Record post-forward module metadata for exhaustive mode.

    Called immediately after ``orig_forward()`` returns in the
    ``module_forward_decorator``. Pops the module call-label stack, creates
    boundary identity ops for pass-through outputs, recovers replacement outputs,
    and annotates output tensors with module-exit metadata.

    Returns
    -------
    tuple[tuple[torch.Tensor, str], ...]
        Exact output tensors and raw labels of boundary Ops inserted for otherwise
        untraceable outputs.
    """
    address = _module_address(module)
    mod_id = id(module)
    module_call_index = trace._module_capture_ws.mod_call_index[mod_id]
    trace._module_capture_ws.mod_call_labels[mod_id].pop()
    from ...intervention.runtime import (
        _peek_module_intervention_parent_labels,
        _peek_tensor_live_fire_results,
        _record_module_intervention_parent_labels,
        _record_tensor_live_fire_results,
    )
    from .ops import _walk_output_tensors_with_paths

    output_entries = list(_walk_output_tensors_with_paths(out))
    output_tensors = [entry[0] for entry in output_entries]
    if not output_tensors:
        output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
        output_entries = [(tensor, (), None) for tensor in output_tensors]
    role_hints = role_hints_for_module(module)
    module_call_label = f"{address}:{module_call_index}"
    start_times = trace._module_capture_ws.module_build_data.setdefault(
        "module_forward_start_times", {}
    )
    forward_duration = 0.0
    if module_call_label in start_times:
        forward_duration = time.time() - start_times[module_call_label]
        trace._module_capture_ws.module_build_data.setdefault("module_forward_durations", {})[
            module_call_label
        ] = forward_duration
    output_structure = None
    if output_entries:
        output_structure = output_entries[0][2]
        trace._module_capture_ws.module_build_data.setdefault("module_output_structures", {})[
            module_call_label
        ] = output_structure
    _register_module_output_container_snapshot(
        trace,
        out,
        module_call_label=module_call_label,
    )
    output_tensor_labels_raw: list[str] = []
    output_paths: list[tuple[object, ...]] = []
    per_output_atomic: list[tuple[str, tuple[ModuleFrame, ...], bool, tuple[str, int] | None]] = []
    output_names: list[str | None] = []
    untraceable_output_boundaries: list[tuple[torch.Tensor, str]] = []
    exit_outputs: list[tuple[int, torch.Tensor]] = []
    for output_index, (t, container_path, _container_spec) in enumerate(output_entries):
        # nn.Identity modules and pass-through tensors (output is same object
        # as input) need _decorated_identity() to create a distinct log entry
        # so the graph correctly shows the module boundary.
        tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        # Peek through BOTH evidence channels (plain attribute + the
        # storage-owned side table): a replacement tensor that rejects dynamic
        # attributes must not read as evidence-free here, or the fresh value
        # below is silently misclassified as an internal source.
        fire_results = _peek_tensor_live_fire_results(t)
        if (_module_type(module).lower() == "identity") or (
            tensor_label is not None and tensor_label in input_tensor_labels
        ):
            intervention_parent_labels: list[str] = list(
                _peek_module_intervention_parent_labels(t, trace)
            )
            t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            if fire_results:
                # Re-attach through the robust recorders: the side channels
                # absorb attr-rejecting tensors, and a tensor writable through
                # NEITHER channel refuses typed instead of silently losing the
                # intervention provenance (B1-13a, strengthened).
                _record_tensor_live_fire_results(t, fire_results)
                if intervention_parent_labels:
                    _record_module_intervention_parent_labels(
                        t, tuple(intervention_parent_labels), trace
                    )
            tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        if tensor_label is None:
            # A live module-boundary intervention deliberately clears copied op
            # labels and leaves typed fire metadata on its fresh output. Preserve
            # that value as an explicit replacement op. Without fire metadata, an
            # untagged module return remains an internal source whose construction
            # TorchLens could not trace (for example, inside ``torch.vmap``).
            intervention_parent_labels = list(_peek_module_intervention_parent_labels(t, trace))
            boundary_label = _ensure_module_output_tensor_logged(
                trace,
                t,
                module,
                parent_labels=intervention_parent_labels,
                kind="intervention_replacement" if fire_results else "internal_source",
            )
            # NOTE: the replacement-event ledger entry for a genuine live-fire
            # replacement is minted inside ``_ensure_module_output_tensor_logged``
            # (gated on the intervention machinery being armed); a stale
            # ``_tl_live_fire_results`` leak in a plain capture stays unledgered.
            untraceable_output_boundaries.append((t, boundary_label))
            tensor_label = get_tensor_label(t)
            # A module returning its own Parameter (a learned query, prompt or
            # scale) is a known model-owned source, not an escape; the entry
            # twin never sees Parameters (``get_arg_tensors_for_resolution``
            # drops them) and buffers sit in the pre-forward ownership snapshot.
            if not fire_results and not isinstance(t, nn.Parameter):
                record_module_boundary_adoption(trace, t, tensor_label, "exit", address)
        if tensor_label is None:
            continue
        if fire_results:
            from .ops import _pop_tensor_live_fire_results

            remaining_fire_results = _pop_tensor_live_fire_results(t)
            if remaining_fire_results:
                any_replaced = any(result.replaced for result in remaining_fire_results)
                exit_event = trace.capture_events.op_event_by_label_raw.get(tensor_label)
                if exit_event is not None:
                    trace.capture_events.append_amendment(
                        amend_module_exit_intervention(
                            exit_event.seq,
                            tensor_label,
                            intervention_fired=True,
                            intervention_replaced=any_replaced,
                            fire_results=remaining_fire_results,
                        )
                    )
                if any_replaced and _live_intervention_machinery_armed():
                    _note_replacement_event(trace, tensor_label, origin="live_fire")
        is_atomic_module = _is_bottom_level_submodule_exit(trace, t, module)
        atomic_module_call = (address, module_call_index) if is_atomic_module else None
        output_tensor_labels_raw.append(tensor_label)
        output_paths.append(tuple(container_path))
        event = trace.capture_events.live_index.require_event(tensor_label)
        exit_outputs.append((event.raw_index, t))
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
        trace._module_capture_ws.mod_exited[mod_id].append(tensor_label)
    trace.capture_events.append_module_exit(
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
            output_tensor_leaf_count=len(output_entries),
        )
    )
    from ...capture.session import capture_session_for

    capture_session = capture_session_for(trace)
    if capture_session is not None:
        capture_session.escrow_module_exit_outputs(module_call_label, tuple(exit_outputs))
    return tuple(untraceable_output_boundaries)


def _record_predicate_module_boundary_outputs(
    trace: "Trace",
    state: Any,
    out: Any,
    *,
    module_address: str,
    module_call_index: int,
) -> None:
    """Retain sparse output-op records selected by ``tl.module`` at module exit.

    Parameters
    ----------
    trace:
        Active predicate-mode trace.
    state:
        Active fastlog recording state.
    out:
        Module output after live boundary interventions.
    module_address:
        Address of the exiting module.
    module_call_index:
        One-based call index of the exiting module.

    Returns
    -------
    None
        Matching output ops are added to the sparse recording once.
    """

    from ...capture.predicates import _evaluate_keep_op
    from ...capture.projections import _record_from_record_context
    from ...fastlog.types import ActivationRecord
    from ...intervention.runtime import _peek_module_intervention_parent_labels
    from ...intervention.selectors import BaseSelector
    from ...ir.predicate import RetroactiveCaptureDecision
    from ...ir.selector_eval import selector_contains_kind
    from .ops import _walk_output_tensors_with_paths

    predicate = state.options.keep_op
    if not isinstance(predicate, BaseSelector) or not selector_contains_kind(predicate, "module"):
        return
    contexts_by_label = {
        ctx.raw_label: ctx
        for ctx in state.all_contexts
        if ctx.kind == "op" and ctx.raw_label is not None
    }
    existing = {
        (record.ctx.raw_label, record.ctx.pass_index)
        for record in state.recording.records
        if record.ctx.raw_label is not None
    }
    module_call_label = f"{module_address}:{module_call_index}"
    labeled_outputs: list[tuple[torch.Tensor, tuple[Any, ...], str]] = []
    walked_leaf_count = 0
    for tensor, container_path, _container_spec in _walk_output_tensors_with_paths(out):
        walked_leaf_count += 1
        raw_label = get_tensor_label(tensor)
        if raw_label is None:
            parent_labels = _peek_module_intervention_parent_labels(tensor, trace)
            raw_label = parent_labels[0] if parent_labels else None
        if raw_label is not None:
            labeled_outputs.append((tensor, tuple(container_path), raw_label))
    trace.capture_events.append_module_exit(
        ModuleExitEvent(
            address=module_address,
            call_index=module_call_index,
            call_label=module_call_label,
            forward_duration=0.0,
            output_structure=None,
            output_tensor_labels_raw=tuple(label for _tensor, _path, label in labeled_outputs),
            output_paths=tuple(path for _tensor, path, _label in labeled_outputs),
            per_output_atomic=(),
            output_names=tuple(None for _tensor, _path, _label in labeled_outputs),
            output_tensor_leaf_count=walked_leaf_count,
        )
    )
    for tensor, container_path, raw_label in labeled_outputs:
        ctx = contexts_by_label.get(raw_label)
        if ctx is None:
            continue
        boundary_ctx = replace(
            ctx,
            output_of_module_calls=tuple(
                dict.fromkeys((*ctx.output_of_module_calls, module_call_label))
            ),
        )
        decision = _evaluate_keep_op(boundary_ctx, state.options)
        if isinstance(decision, RetroactiveCaptureDecision):
            continue
        if not decision.save_out and not decision.save_metadata:
            continue
        trace._tl_save_selector_fire_count = (
            int(getattr(trace, "_tl_save_selector_fire_count", 0)) + 1
        )
        key = (boundary_ctx.raw_label, boundary_ctx.pass_index)
        if key in existing:
            continue
        existing.add(key)
        ram_payload, disk_payload, transformed_ram, transformed_disk = state.resolve_storage(
            tensor,
            decision,
            ctx=boundary_ctx,
        )
        state.add_record(
            ActivationRecord(
                ctx=boundary_ctx,
                spec=decision,
                ram_payload=ram_payload,
                disk_payload=disk_payload,
                transformed_ram_payload=transformed_ram,
                transformed_disk_payload=transformed_disk,
            )
        )
        selected_event = _record_from_record_context(
            boundary_ctx,
            decision,
            tensor=tensor,
            ram_payload=ram_payload,
            transformed_ram_payload=transformed_ram,
            predicate_matched=True,
            container_path=tuple(container_path),
        )
        selected_policy = selected_event.policy
        if selected_policy is None:  # pragma: no cover - sparse freeze always stamps one
            raise RuntimeError("sparse freeze produced a record without a policy facet")
        boundary_label = boundary_ctx.raw_label or boundary_ctx.label
        boundary_event = trace.capture_events.op_event_by_label_raw.get(boundary_label)
        if boundary_event is not None:
            trace.capture_events.append_amendment(
                amend_module_boundary_retention(
                    boundary_event.seq,
                    boundary_label,
                    output=selected_event.output,
                    policy=selected_policy,
                    predicate_matched=True,
                    capture_spec=decision,
                    record_context=boundary_ctx,
                )
            )


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
                _is_halt_only_capture,
                _module_capture_spec,
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

            bind_invocation(trace, module, frame.address, frame.pass_index, (args, kwargs))
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
                    enter_spec = _module_capture_spec(state.options)
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
                    if not _predicate_event_was_appended(
                        trace, enter_ctx.raw_label or enter_ctx.label
                    ):
                        append_projected_event(
                            trace,
                            enter_ctx,
                            skipped_spec,
                            predicate_matched=False,
                        )
                    state.append_context(enter_ctx)
                    # Echo narrator slot (snoop D1, module seam): structure
                    # lines ride the events already in the stream. Duck-typed
                    # session read: the hot path imports nothing.
                    echo_session = trace.__dict__.get("_echo_session")
                    if echo_session is not None:
                        echo_session.emit_module_enter(frame.address, frame.module_type)
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
                from ...intervention.runtime import _apply_module_boundary_live_hooks

                out = _apply_module_boundary_live_hooks(
                    out,
                    module_address=frame.address,
                    module_call_index=frame.pass_index,
                    module_type=frame.module_type,
                    call_args=args,
                    call_kwargs=dict(kwargs),
                )
                _record_predicate_module_boundary_outputs(
                    trace,
                    state,
                    out,
                    module_address=frame.address,
                    module_call_index=frame.pass_index,
                )
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
                        exit_spec = _module_capture_spec(state.options)
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
                        if not _predicate_event_was_appended(
                            trace, exit_ctx.raw_label or exit_ctx.label
                        ):
                            append_projected_event(
                                trace,
                                exit_ctx,
                                skipped_spec,
                                predicate_matched=False,
                            )
                        state.append_context(exit_ctx)
                        # Echo module-exit line: only a COMPLETED module call
                        # claims completion; a raising module abandons its
                        # frame silently (the enter line without an exit is
                        # the honest crash shape).
                        echo_session = trace.__dict__.get("_echo_session")
                        if echo_session is not None:
                            if active_model_exc is None:
                                echo_session.emit_module_exit(frame.address)
                            else:
                                echo_session.abandon_module_frame()
                    _mstack.pop_frame(state.module_stack, frame)

        # ---- Exhaustive mode: full entry -> forward -> exit ----
        frame = _mstack.push_frame(trace, trace._module_capture_ws.exhaustive_module_stack, module)
        from .prehook_provenance import bind_invocation

        bind_invocation(trace, module, frame.address, frame.pass_index, (args, kwargs))
        try:
            input_tensor_labels, input_tensor_labels_at_entry = _record_module_entry_metadata(
                trace, module, args, kwargs
            )
            # Echo narrator slot (snoop D1, module seam, exhaustive tier).
            # Duck-typed session read: the hot path imports nothing.
            echo_session = trace.__dict__.get("_echo_session")
            if echo_session is not None:
                echo_session.emit_module_enter(frame.address, _module_type(module))
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
                call_labels = trace._module_capture_ws.mod_call_labels.get(mod_id)
                if call_labels:
                    call_labels.pop()
                # A raising module never completed: abandon the echo frame
                # without an exit line (the enter without exit is the honest
                # crash shape; depth stays consistent if user code catches).
                if echo_session is not None:
                    echo_session.abandon_module_frame()
                raise
            untraceable_output_boundaries = _record_module_exit_metadata(
                trace, module, out, input_tensor_labels, input_tensor_labels_at_entry
            )
            if echo_session is not None:
                echo_session.emit_module_exit(frame.address)
            mark_expected_original_accounted(
                expected_token,
                captured=bool(untraceable_output_boundaries),
                boundary_outputs=untraceable_output_boundaries,
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
                        # tl.module evaluates capture-time subjects through
                        # output_of_module_calls, so a module-boundary halt
                        # (incl. compiled stop_after= module addresses) can only
                        # fire if this exit ctx names its own module call; the
                        # exhaustive path previously left it empty and every
                        # tl.module halt silently ran the full forward.
                        "output_of_module_calls": (f"{frame.address}:{frame.pass_index}",),
                    },
                    module_stack=[],
                    history=(),
                    op_counts={},
                    pass_index=1,
                    event_index=trace._raw_graph_ws.layer_counter,
                    step_index=None,
                    time_since_pass_start=0.0,
                    include_source_events=False,
                    sample_id=None,
                )
                _evaluate_halt(exit_ctx, options, frontier_output=out)
            return out
        finally:
            _mstack.pop_frame(trace._module_capture_ws.exhaustive_module_stack, frame)

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


def _predicate_event_was_appended(trace: "Trace", label_raw: str) -> bool:
    """Return whether predicate capture already appended ``label_raw``.

    Parameters
    ----------
    trace:
        Active trace whose predicate event buffer may already contain the
        projected event.
    label_raw:
        Raw label used when appending the projected event.

    Returns
    -------
    bool
        True when ``trace.capture_events.op_event_by_label_raw`` already owns
        ``label_raw``.
    """

    capture_events = getattr(trace, "capture_events", None)
    if capture_events is None:
        return False
    return label_raw in capture_events.op_event_by_label_raw


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
        clear_param_meta(param)
    if inventory:
        trace._session_param_inventory = []
    for param in model.parameters():
        restore_param_requires_grad(param)
        clear_param_meta(param)


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
    from .offload_hooks import uninstall_offload_hook_shims
    from .prehook_provenance import rollback_prehook_provenance

    try:
        uninstall_offload_hook_shims(trace)
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

    # The class-metadata cache is SESSION-scoped by design, but it was only ever
    # cleared at the START of the next capture -- so a process whose last capture
    # used generated / function-local module classes pinned those classes (and,
    # through ``meta["cls"]``, their code objects and closures) alive
    # indefinitely. Release it here too: the entries have no consumer after
    # module capture, and the start-of-session clear stays as the staleness
    # guard for captures that die before this epilogue runs.
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()

    # r83 C1: retire this capture's label-anchoring session LAST, after every
    # cleanup pass that consults it. Any label still carried by an object that
    # outlives the capture is now anchored to a retired session and can never
    # be accepted as provenance by a later capture, whatever route it escaped by.
    end_label_session()


def _is_isinstance_hostile_deprecation_shim(value: Any) -> bool:
    """Return whether ``value`` is torch's ``reduce_op`` deprecation singleton.

    ``torch.distributed.reduce_op`` is a ``_reduce_op`` instance whose
    ``__getattribute__`` emits a ``FutureWarning`` on ANY attribute access --
    including the ``__class__`` read every ``isinstance()`` performs -- so an
    initialized distributed process running under ``-W error`` had its capture
    CLEANUP aborted by the namespace walks below (SF-45). ``type()`` bypasses
    the instance's ``__getattribute__``, so this exact-name check is silent;
    structural matching (not an import of the private class) follows the
    ``_distributed.py`` fallback convention.
    """

    value_type = type(value)
    return (
        value_type.__name__ == "_reduce_op"
        and value_type.__module__ == "torch.distributed.distributed_c10d"
    )


def _clear_session_tensor_metadata(
    value: Any,
    seen: set[int],
    depth: int = 0,
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
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
    visit
        Optional action applied to each reachable non-Parameter tensor instead
        of the default ``clear_meta`` -- the pre-forward ownership snapshot
        (:func:`_collect_model_owned_tensor_ids`) reuses this exact traversal
        so the "model-owned" surfaces of the snapshot and the session-end
        clear can never drift apart.

    Returns
    -------
    None
        Mutates reachable tensors in place by removing TorchLens metadata
        (or applies ``visit`` when given).
    """

    action = clear_meta if visit is None else visit
    if value is None or _is_isinstance_hostile_deprecation_shim(value):
        return
    if isinstance(value, (str, bytes, int, float, bool)):
        return
    if isinstance(value, ModuleType):
        # r81 (r80 F1 root B): a stamped tensor stashed as a DIRECT attribute of
        # a ``types.ModuleType`` escaped the belt entirely (this walk returned
        # immediately for modules). r33/r83: a stamp nested inside a plain
        # CONTAINER in the module namespace escaped the shallow sweep too and
        # laundered provenance across sessions, so namespace containers now get
        # a dedicated PURE container-tree descent (tensors + plain containers
        # only). The descent deliberately never enters objects or nested
        # modules: this walk reaches broadly-imported modules (``torch``,
        # ``math``) through model-owned helper objects, and generic recursion
        # from a module namespace would explode through ``sys.modules`` into
        # every imported module in the process. Object stashes inside module
        # namespaces stay covered by the session identity belt, which never
        # trusts an unregistered stamp anyway.
        obj_id = id(value)
        if obj_id in seen or depth >= 12:
            return
        seen.add(obj_id)
        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, dict):
            # No slot memo (r8 b4 R39, sol probe): the memo invalidated on
            # ``len(namespace)`` only, so REPLACING a value at a constant
            # length (``mod.stash = 5`` becoming ``mod.stash = tensor``) kept
            # the stale slot list and the stamped tensor escaped the session
            # clear -- provenance laundering across sessions. No cheaper
            # invalidation is sound: detecting a new tensor/container slot
            # requires type-inspecting every item, which IS the rebuild, so
            # the filter runs fresh each walk (one isinstance pass per
            # reachable module namespace per session end).
            slot_names = tuple(
                name
                for name, item in namespace.items()
                if not _is_isinstance_hostile_deprecation_shim(item)
                and isinstance(
                    item,
                    (torch.Tensor, dict, list, tuple, set, frozenset, deque),
                )
            )
            for name in slot_names:
                item = namespace.get(name)
                if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                    action(item)
                elif isinstance(item, (dict, list, tuple, set, frozenset, deque)):
                    _clear_container_tree_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            action(value)
        return
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    # Scalar leaves cannot carry tl_* stamps; skipping them INLINE (instead of
    # paying a full call that immediately returns) halves the cost of walking
    # broadly-imported module namespaces like ``torch`` (round-3 b4 F1: the
    # pre-forward ownership snapshot made every capture pay this walk twice).
    # EXACT type() membership on purpose: isinstance() falls back to reading
    # ``__class__`` on a non-matching type, which the reduce_op deprecation
    # shim answers with a FutureWarning (SF-45) -- the recursion's entry guard
    # never runs for inline-skipped items, so this check must stay hostile-safe.
    # Non-exact scalar subclasses fall through to the guarded recursive call.
    _scalar_leaves = (str, bytes, int, float, bool)
    if isinstance(value, dict):
        for key, item in value.items():
            if key is not None and type(key) not in _scalar_leaves:
                _clear_session_tensor_metadata(key, seen, depth + 1, visit)
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, nn.Module):
        return
    namespace = getattr(value, "__dict__", None)
    if namespace is None:
        return
    for item in namespace.values():
        if item is not None and type(item) not in _scalar_leaves:
            _clear_session_tensor_metadata(item, seen, depth + 1, visit)


def _clear_container_tree_tensor_metadata(
    value: Any,
    seen: set[int],
    depth: int,
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Clear tensor metadata from a pure container tree (no object descent).

    Restricted companion to ``_clear_session_tensor_metadata`` for
    ``types.ModuleType`` namespaces: walks plain containers and clears
    non-Parameter tensors, but never descends into objects, ``nn.Module``
    instances, or nested modules, so sweeping a broadly-imported module's
    namespace cannot fan out through the whole process object graph.

    Parameters
    ----------
    value
        Container member reached from a module-namespace container.
    seen
        Object ids already visited during this cleanup scan.
    depth
        Current recursion depth, shared with the main walk's bound.

    Returns
    -------
    None
        Mutates reachable tensors in place by removing TorchLens metadata.
    """

    if _is_isinstance_hostile_deprecation_shim(value):
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            (clear_meta if visit is None else visit)(value)
        return
    if not isinstance(value, (dict, list, tuple, set, frozenset, deque)):
        return
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    items: Iterable[Any]
    if isinstance(value, dict):
        items = [item for pair in value.items() for item in pair]
    else:
        items = list(value)
    # Inline scalar-leaf skip: module-namespace container trees are dominated
    # by strings (``torch.__all__`` alone is ~1400), and each full call here
    # costs more than the check (round-3 b4 F1 walk-cost finding). EXACT
    # type() membership on purpose: isinstance() reads ``__class__`` on a
    # non-matching type, which the reduce_op deprecation shim answers with a
    # FutureWarning (SF-45); subclasses fall through to the guarded recursion.
    for item in items:
        if item is None or type(item) in (str, bytes, int, float, bool):
            continue
        _clear_container_tree_tensor_metadata(item, seen, depth + 1, visit)


def _clear_callable_session_tensor_metadata(
    callable_obj: Any,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Clear TorchLens tensor metadata captured by a callable object.

    The globals scan targets the callable that actually runs USER code, so
    TorchLens's own ``module_forward_decorator`` wrapper is unwrapped first (via
    ``functools.wraps``' ``__wrapped__``, the same link
    :func:`_restore_undecorated_forward` uses). The wrapper is defined in THIS
    module, so scanning ITS ``__code__.co_names`` against ITS ``__globals__``
    describes TorchLens internals -- ``_state``'s decoration registries and
    ``sys.modules`` -- and never the user's forward. That was both a coverage gap
    (every decorated submodule's real globals went unscanned) and the whole cost
    of the per-capture namespace sweep: those two roots alone accounted for
    ~33.7 k of the 33.7 k container visits on a 3-op model and ~35.1 k of 35.1 k
    on resnet50, reaching zero tensors. Only the globals scan moves inward;
    defaults, keyword defaults and closures are still swept at EVERY link of the
    chain, so nothing a wrapper legitimately captures is skipped. A callable that
    is not a TorchLens forward wrapper is inspected exactly as before.

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
    chain: list[Any] = []
    chain_ids: set[int] = set()
    current = raw_callable
    while current is not None and id(current) not in chain_ids:
        chain_ids.add(id(current))
        chain.append(current)
        if not is_forward_call_decorated(current):
            break
        wrapped = getattr(current, "__wrapped__", None)
        current = None if wrapped is None else getattr(wrapped, "__func__", wrapped)

    for link in chain:
        defaults = getattr(link, "__defaults__", None) or ()
        _clear_session_tensor_metadata(defaults, seen, visit=visit)
        kwdefaults = getattr(link, "__kwdefaults__", None) or {}
        _clear_session_tensor_metadata(kwdefaults, seen, visit=visit)
        closure = getattr(link, "__closure__", None) or ()
        for cell in closure:
            try:
                cell_value = cell.cell_contents
            except ValueError:
                continue
            _clear_session_tensor_metadata(cell_value, seen, visit=visit)

    user_callable = chain[-1]
    globals_dict = getattr(user_callable, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    code = getattr(user_callable, "__code__", None)
    if code is None:
        return
    for name in code.co_names:
        if name in globals_dict:
            _clear_session_tensor_metadata(globals_dict[name], seen, visit=visit)


def _collect_model_owned_tensor_ids(model: nn.Module) -> dict[int, torch.Tensor]:
    """Snapshot tensors reachable from the model's PRE-FORWARD state, PINNED.

    Walks exactly the surfaces the session-end clear walks -- submodule
    ``__dict__`` object graphs plus each forward callable's defaults, keyword
    defaults, closure cells, and referenced globals -- via the shared ``visit``
    traversal, so a tensor the previous session's cleanup could reach (a
    nested model-owned cache, a forward-global mask) is recognized as
    model-owned by the next capture. The R16 module-entry adoption disclosure
    consults this snapshot: a pre-forward model-owned tensor is a KNOWN
    internal source (its stale labels were legitimately cleared between
    sessions), while a tensor first appearing MID-forward is absent from the
    snapshot and keeps the escape disclosure.

    The mapping VALUES are strong references, deliberately (round-3 b1/b3/b4
    merged finding): a bare ``set[int]`` of recyclable ids had no liveness
    pinning, so a model that dropped a snapshotted cache tensor mid-forward
    freed the object and a later stale-pre-wrap escape product could reuse the
    exact id -- classified "model-owned known source", silently suppressing
    the adoption disclosure (the exact laundering 3c721316 closed). Pinning
    every snapshot member for the session makes id reuse impossible; the
    workspace drop at the transient-state cleanup seam releases the pins.

    Parameters
    ----------
    model
        The prepared root model.

    Returns
    -------
    dict[int, torch.Tensor]
        ``id() -> tensor`` for every reachable non-Parameter tensor. Consumers
        test membership (``id(t) in snapshot``), identical to the historical
        set semantics; the values exist only to pin the ids.
    """

    owned, _ = collect_pre_forward_tensor_ids(model)
    return owned


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
    ``wrap_torch()`` performs the incremental stale-reference belt sweep as part
    of wrapper installation/revalidation, so this chokepoint does not repeat it.
    """
    from .wrappers import wrap_torch

    wrap_torch()  # idempotent — no-op if already wrapped; auto-rewraps after unwrap
    _prepare_model_once(model)  # idempotent — cached in _state._prepared_models
