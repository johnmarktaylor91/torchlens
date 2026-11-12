"""Prepare torch ``nn.Module`` objects for capture sessions.

One-time preparation installs persistent forward wrappers and module metadata.
Per-session preparation populates Trace state, parameter logs, and buffer labels.
"""

import inspect
import itertools
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from torchlens._ir.capture_events import CaptureEvents

from ... import _state
from ..._capture.projections import LiveOpView, active_recording_state_or_none
from ..._capture.session import capture_session_for
from ..._capture_honesty import warn_held_scan_truncated
from ..._capture_state_helpers import _is_uninitialized_param
from ..._data_classes._module_role_hints import multi_output_role_from_path, role_hints_for_module
from ..._data_classes.func_call_location import FuncCallLocation
from ..._data_classes.module import HookInfo
from ..._data_classes.param import Param, ParamAccessor
from ..._errors import CaptureNotActiveError
from ..._ir import ModuleEnterEvent, ModuleExitEvent, ModuleFrame, ModulePrepEvent
from ..._ir.events import InterventionAppliedEvent
from ..._ir.op_record import (
    amend_module_exit_intervention,
    amend_raw_hook_intervention,
)
from ..._ir.refs import ParamRef
from ..._quantities import Bytes
from ..._utils._model_walk import model_modules, own_named_buffers
from ..._utils._tensor_label import (
    begin_label_session,
    clear_meta,
    end_label_session,
    get_buffer_address,
    get_live_tensor_label,
    get_module_meta,
    get_tensor_label,
    is_forward_call_decorated,  # noqa: F401  (re-exported for tests)
    mark_tensor_replacement_wrapped,
    set_module_meta,
    set_tensor_label,
)
from ..._utils.hashing import make_random_barcode
from ..._utils.introspection import get_arg_tensors_for_resolution, get_vars_of_type_from_obj
from ..._utils.tensor_utils import (
    get_memory_amount,
    get_memory_amount_from_metadata,
)
from . import buffer_writes as _buffer_writes, module_stack as _mstack
from ._functorch_check import functorch_wrapped_check
from ._held_refs import normalize_held_torch_function_refs, register_released_model
from ._held_tensor_scan import iter_module_held_plain_tensors
from ._model_prep_containers import (
    _register_module_input_container_snapshots,
    _register_module_output_container_snapshot,
)
from ._module_aliases import shared_module_addresses
from ._module_arg_stubs import entry_live_label, input_labels_by_id, stub_module_call_args
from ._module_boundary_adoption import (
    _tag_untagged_buffers,
    pre_forward_snapshot_build_data,
    record_module_boundary_adoption,
)
from ._module_class_metadata import (
    _NN_MODULE_ATTRS,
    _PYTORCH_INTERNAL,
    _get_class_metadata,
    _module_class_metadata_cache,
    _source_start_line,
)
from ._module_entry_site import module_entry_code_context
from ._ops_containers import _walk_output_tensors_with_paths
from ._ops_finalize import _make_layer_log_entry
from ._ops_interventions import _pop_tensor_live_fire_results
from ._predicate_forward import _predicate_module_forward
from ._prepared_cache import _KEPT, install_session_forwards as _install_forwards
from ._quantized_flops import estimate_quantized_module_forward_flops
from ._replacement_op_fields import _next_untagged_tensor_label
from ._replacement_template import _replacement_template_fields
from ._root_prep import (
    prepare_bound_owner_once,
    record_model_identity,
    register_bound_owner_calls,
    register_session_module_calls,
)
from ._session_cleanup import (  # noqa: F401  (re-exported for tests and sibling modules)
    _clear_container_tree_tensor_metadata,
    _clear_model_tree_tensor_metadata,
    _clear_module_attribute_tensor_metadata,
    _clear_module_callables_tensor_metadata,
    _clear_session_tensor_metadata,
    method_index,
)
from ._session_forwards import (
    _NO_INSTANCE_FORWARD,
    install_session_forward_wrappers as _install_session_forward_wrappers,
    restore_session_forward_wrappers as _restore_session_forward_wrappers,
    restore_undecorated_forward as _restore_undecorated_forward,
)
from ._session_param_logs import create_session_param_logs
from ._session_param_release import restore_session_param_state
from .bound_root import capture_root_module
from .buffer_writes import session_validated_buffer_address
from .escape_detection import (
    expected_original_call,
    mark_expected_original_accounted,
)
from .offload_hooks import uninstall_offload_hook_shims
from .ops import _VIEW_MUTATION_RELABELS, _add_tensor_backward_hook
from .prehook_provenance import bind_invocation
from .sources import log_closure_constant_if_known, log_source_tensor
from .tensor_tracking import _append_module_suffix_to_equivalence_class
from .wrappers import wrap_torch

# Cache class-level module metadata (inspect.getsourcelines, inspect.signature, etc.)
# shared across instances during one capture. Cleanup releases it at the end of
# the session; the next capture rebuilds from the current class definitions.

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from ...recording._types import ModuleStackFrame


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
    CaptureNotActiveError
        If ``_active_trace`` is set or logging is enabled.
    """

    if _state._active_trace is None and not _state._logging_enabled:
        return
    trace = _state._active_trace
    model_name = getattr(trace, "model_name", None) or (
        trace._source_class_name if trace is not None else None
    )
    raise CaptureNotActiveError(
        "tl.release_model() was called while a TorchLens capture is still active"
        + (f" for model {model_name!r}" if model_name else ""),
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
    """Remove TorchLens preparation bookkeeping from a PyTorch module tree.

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
    CaptureNotActiveError
        If a capture is currently active (code
        ``release_during_active_capture``). Releasing mid-capture strips the
        ``tl_*`` / ``._tl`` metadata the capture's module attribution reads,
        so the call is refused instead of finishing a silently degraded trace.

    Notes
    -----
    A capture leaves no ``forward`` wrapper on the model (they are
    session-scoped); this operation still undoes any leftover decoration,
    clears TorchLens-owned module metadata and legacy ``tl_*`` instance
    attributes, and evicts all related
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


def _make_once_visitor(
    model: nn.Module, addressed_module_ids: set[int]
) -> Callable[[nn.Module, str, list[tuple[str, nn.Module, str]], bool], None]:
    """Return the per-module visitor ``_prepare_model_once`` traverses ``model`` with.

    ``addressed_module_ids`` collects the modules already given a canonical
    address, so a shared module keeps its first registration path.
    """

    def _visit_once(
        module: nn.Module,
        address: str,
        _child_entries: list[tuple[str, nn.Module, str]],
        is_root: bool,
    ) -> None:
        """Prepare one module and recursively visit its children once."""
        # Stamp this module's current root and flag any prior root it was
        # re-rooted away from as stale (role-swap bookkeeping).
        _state.record_module_root_prep(model, module)

        # Root module is handled separately by trace.
        if is_root:
            return

        # The traversal is a pre-order DFS, the order ``named_modules()`` reports, and
        # visits a shared module once per registration path. Its canonical address is
        # the path that reaches it FIRST in that order (the address stock-torch module
        # hooks and the glossary use), so it is stamped here, when the module itself is
        # reached, and a later path never overwrites it; the other paths stay alias
        # addresses. Stamping children while visiting their parent would let a shallow
        # later alias (``self.B = self.A[0]``) win over the earlier nested path.
        if id(module) not in addressed_module_ids:
            addressed_module_ids.add(id(module))
            set_module_meta(module, address=address, module_type=str(type(module).__name__))

    return _visit_once


def _prepare_model_once(model: nn.Module) -> None:
    """Phase 1: One-time (per role) model preparation.

    Fast-path cached via ``_state._prepared_models`` (WeakSet) for the common
    case: a model traced repeatedly in a fixed role, or independent models
    traced in any interleaving. **Assigns permanent metadata** to each
    submodule: ``_tl.address`` (dotted path like
    ``"encoder.layer.0.attention"``) and ``_tl.module_type`` (class name), held
    in TorchLens's identity-keyed registry (never on the module object), so they
    survive across sessions without changing the module.

    The ``forward`` wrappers are NOT installed here: they are session-scoped
    (:func:`_install_session_forward_wrappers` at session start,
    :func:`_restore_session_forward_wrappers` at session cleanup), so a model
    between captures carries no TorchLens callable and deep-copies, pickles and
    saves exactly like a model TorchLens never touched. The root module is
    skipped for type annotation because its forward is called directly by
    ``trace`` with its own entry/exit handling.

    **Role swaps.** The address (root-relative) is role-DEPENDENT: they differ depending on whether a module is *this* trace's
    root or a non-root submodule. The same module can legitimately be traced in
    both roles across separate traces (e.g. ``trace(outer, ...)`` then
    ``trace(outer.inner, ...)``). When that happens the metadata cached for the
    old role is stale for the new one, so this function re-establishes it:

    * A root that still carries a TorchLens ``forward`` decoration (only
      possible when a cleanup was cut short) is undecorated (see
      :func:`_restore_undecorated_forward`).
    * Re-rooting a descendant under a new model marks the old ancestor root
      stale (via :func:`_state.record_module_root_prep`); the stale root is then
      treated as un-prepared here so its addresses are refreshed for the
      current root on its next trace.
    """
    if model in _state._prepared_models and not _state.root_prep_is_stale(model):
        return
    _state.clear_root_prep_stale(model)

    set_module_meta(model, address="", module_type=str(type(model).__name__))
    # The root's forward must run undecorated. Restore it if this module carries
    # a stale non-root decoration from an earlier trace where it was a submodule.
    _restore_undecorated_forward(model)

    addressed_module_ids: set[int] = set()
    _traverse_model_modules(model, _make_once_visitor(model, addressed_module_ids))
    prepare_bound_owner_once(model, _restore_undecorated_forward)
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
    4. Creates ``Param`` objects and stamps each parameter's barcode, address
       and user ``requires_grad`` flag (the flag itself is never changed).
    5. Tags buffer tensors with ``_tl.address``.

    All session-scoped state is cleaned up by ``_cleanup_model_session``.
    """
    # Install this capture's label-anchoring session first, before any path can stamp a label;
    # an earlier capture's labels then resolve against nothing. It is module state in ``_tl``
    # because ``set_tensor_label``, which every label stamp flows through, has no Trace.
    begin_label_session()
    _state._dir_cache.clear()
    trace._module_capture_ws.exhaustive_module_stack = []
    if _KEPT:  # a kept_prepared block is open; the default path makes no extra call
        _install_forwards(trace, model, _install_session_forward_wrappers, _NO_INSTANCE_FORWARD)
    else:
        _install_session_forward_wrappers(trace, model)
    record_model_identity(trace, model)

    _seen_module_ids: dict[int, str] = {}  # finds a module shared at several addresses
    # A module registered under several names is visited once; its aliases, from one walk.
    _alias_addresses = shared_module_addresses(model)

    # One recorded walk per module tree plus the addresses phase 1 cached.
    root_module = capture_root_module(model)  # the owner, on a bound-method root
    for module in model_modules(model):
        is_root = module is root_module
        address = _module_address(module)
        module_type_str = _module_type(module)
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
        if meta is not None and id(module) in _alias_addresses:
            for alias_address in _alias_addresses[id(module)]:
                if alias_address != meta_address and alias_address not in meta["all_addresses"]:
                    meta["all_addresses"].append(alias_address)
        if meta is not None:
            capture_events = getattr(trace, "capture_events", None)
            if capture_events is None:
                capture_events = CaptureEvents()
                trace.capture_events = capture_events
            capture_events.append_module_prep(
                ModulePrepEvent(
                    address=meta_address,
                    all_addresses=tuple(meta["all_addresses"]),
                    module_type_str=module_type_str,
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
            trace._module_capture_ws.module_build_data["module_types"][address] = module_type_str
            register_session_module_calls(trace, module)
    register_bound_owner_calls(trace, model)
    if trace._options_mode != "predicate":
        create_session_param_logs((Param, ParamAccessor), trace, model, optimizer)
    else:
        _register_predicate_session_params(model)
    prepare_buffer_tensors(trace, model)
    # Pre-forward ownership snapshot for the R16 module-entry adoption
    # disclosure: tensors reachable NOW (nested caches, forward globals) are
    # model-owned known sources; anything first seen mid-forward is not. An
    # unstamped one is late-indexed as a buffer at first use (_held_late_index).
    trace._module_capture_ws.module_build_data.update(pre_forward_snapshot_build_data(model))
    if trace._options_mode == "exhaustive":
        from .buffer_writes import install_buffer_write_tracker

        install_buffer_write_tracker(trace, model)
    from .prehook_provenance import install_prehook_provenance

    install_prehook_provenance(
        trace,
        model,
        forward_hook_wrapper_factory=_make_user_forward_hook_wrapper,
    )
    # Accelerate offload/dispatch hooks: run hook internals under
    # pause_logging and re-stamp hook-materialized params so offloaded models
    # capture with clean graphs and attributed weights.
    from .offload_hooks import install_offload_hook_shims

    install_offload_hook_shims(trace, model)


def _register_predicate_session_params(model: nn.Module) -> None:
    """Register the model's own Parameters for this predicate pass.

    Predicate capture skips ``create_session_param_logs`` (no ``Param`` logs, no
    stamps), which left a Parameter write unable to tell prepared model state
    from a Parameter the forward created. Registering identities (weakly, with
    the lazy-at-prep flag) lets ``parameter_mutation_output_for_logging`` keep a
    prepared ``param.data = rhs`` rebind and a lazy module's in-forward
    initialization out of the op graph, exactly as exhaustive capture does.
    """

    state = active_recording_state_or_none()
    if state is None:
        return
    registry: dict[int, tuple[weakref.ref[torch.nn.Parameter], bool]] = {}
    param_refs: dict[int, ParamRef] = {}
    param_logs: dict[str, Param] = {}
    for module in model.modules():
        address = _module_address(module)
        for param_name, param in module._parameters.items():
            if param is None:
                continue
            pid = id(param)
            if pid in registry:
                _alias_prepared_param(param_logs, param_refs[pid], address, param_name)
                continue
            param_is_lazy = _is_uninitialized_param(param)
            registry[pid] = (weakref.ref(param), param_is_lazy)
            ref, param_log = _prepared_param_identity(
                param, address, param_name, lazy=param_is_lazy
            )
            param_refs[pid] = ref
            param_logs[ref.address] = param_log
    state.prepared_params = registry
    state.prepared_param_refs = param_refs
    state.prepared_param_logs = param_logs


def _prepared_param_identity(
    param: nn.Parameter,
    module_address: str,
    param_name: str,
    *,
    lazy: bool,
) -> tuple[ParamRef, Param]:
    """Return the capture-time identity of one prepared parameter.

    Predicate capture stamps nothing on the parameter: the barcode lives only
    in the recording state's registry, keyed by object identity, and the
    ``Param`` log carries no tensor reference (a Recording never extends a
    parameter's lifetime).

    Parameters
    ----------
    param
        The model's own Parameter.
    module_address
        Address of the owning module (``""`` for the root).
    param_name
        The parameter's name inside its module.
    lazy
        Whether the parameter is still uninitialized at prep.

    Returns
    -------
    tuple[ParamRef, Param]
        The op-record reference and the Trace-side log it resolves to.
    """

    address = f"{module_address}.{param_name}" if module_address else param_name
    barcode = make_random_barcode()
    shape: tuple[int, ...] = () if lazy else tuple(param.shape)
    ref = ParamRef(
        barcode=barcode,
        address=address,
        shape=shape,
        dtype=str(param.dtype),
        trainable=bool(param.requires_grad),
        module_address=None,
    )
    param_log = Param(
        module_address=module_address or "self",
        name=param_name,
        shape=shape,
        dtype=param.dtype,
        num_params=0 if lazy else param.numel(),
        param_memory=Bytes(0) if lazy else get_memory_amount(param),
        trainable=bool(param.requires_grad),
        address=address,
        barcode=barcode,
        has_optimizer=None,
    )
    if lazy:
        param_log._lazy_at_prep = True  # type: ignore[attr-defined]
    return ref, param_log


def _alias_prepared_param(
    param_logs: dict[str, Param],
    ref: ParamRef,
    module_address: str,
    param_name: str,
) -> None:
    """Record a shared parameter's second address on its one ``Param`` log."""

    param_log = param_logs[ref.address]
    alias_address = f"{module_address}.{param_name}" if module_address else param_name
    if alias_address not in param_log.all_addresses:
        param_log.all_addresses.append(alias_address)
    alias_module_address = module_address or "self"
    if alias_module_address not in param_log.all_module_addresses:
        param_log.all_module_addresses.append(alias_module_address)


# ---------------------------------------------------------------------------
# Module metadata capture (unchanged from before)
# ---------------------------------------------------------------------------


#: ``(metadata field, module attribute)`` for each torch hook registry.
_HOOK_REGISTRIES = tuple(
    (name, f"_{name}")
    for name in (
        "forward_pre_hooks",
        "forward_hooks",
        "backward_pre_hooks",
        "backward_hooks",
        "full_backward_pre_hooks",
        "full_backward_hooks",
    )
)


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
    save_source = trace._options_save_source_context
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

    # Instance-specific fields. An empty registry (the usual case) yields no
    # entries, so it skips the per-registry call.
    state = module.__dict__
    for field, attr in _HOOK_REGISTRIES:
        registry = state[attr] if attr in state else getattr(module, attr, {})
        meta[field] = _hook_info_from_registry(registry) if registry else []
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
        if attr_name[:1] == "_" or attr_name[:3] == "tl_":  # startswith, without a call
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

    Every tensor stamped here is ALSO recorded in
    ``trace._session_buffer_inventory`` (session-scoped strong refs) so cleanup
    clears the stamps from the recorded inventory instead of relying on a model
    re-traversal that a mid-forward ``_buffers.pop(...)`` can escape.

    Every stamp routes through
    ``register_session_buffer_stamp`` so it also joins the session identity
    registry (``trace._session_buffer_identity``) consulted by the buffer-rung
    storage-identity belt; the registry is reset here at session start.
    """

    _state._tagged_buffer_ids.clear()
    trace._session_buffer_inventory = []
    trace._session_buffer_identity = {}
    unstampable: list[str] = []
    held_truncations: list[str] = []

    def _stamp(tensor: torch.Tensor, address: str) -> None:
        """Stamp one buffer into the session registries, collecting failures."""
        # A failed stamp is EVIDENCE LOSS -- the buffer's reads may log as
        # internal sources instead of buffer versions -- so it is collected
        # and disclosed once below instead of silently swallowed.
        try:
            _buffer_writes.register_session_buffer_stamp(trace, tensor, address)
            _state._tagged_buffer_ids.add(id(tensor))
        except Exception as exc:
            unstampable.append(f"{address} ({type(exc).__name__}: {exc})")

    for submodule in model_modules(model):
        module_addr = _module_address(submodule)
        # Scan registered buffers
        for buf_name, buf_tensor in own_named_buffers(submodule):
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


def _record_module_entry_build_data(trace: "Trace") -> tuple[float, Any, list[str]]:
    """Return the forward start time, code context and call stack for one entry.

    The enter event carries them; ingest rebuilds the module side channel from the journal.
    """
    forward_start_time = time.time()
    code_context = module_entry_code_context(trace)
    # Ancestor module calls, root excluded: a plain root is never pushed, and a
    # bound-method root's owner (address ``self``) is dropped here so the
    # containment facts read exactly as under a plain trace of the owner.
    call_stack = [
        f"{frame.address}:{frame.pass_number}"
        for frame in trace._module_capture_ws.exhaustive_module_stack[:-1]
        if frame.address != "self"
    ]
    return forward_start_time, code_context, call_stack


def _record_module_entry_metadata(
    trace: "Trace",
    module: nn.Module,
    frame: "ModuleStackFrame",
    call: tuple[tuple[Any, ...], dict[str, Any]],
) -> tuple[set[str], list[str], float, Any]:
    """Record pre-forward module metadata for exhaustive mode.

    Called right before ``orig_forward`` in ``module_forward_decorator``. Reads the call's
    frame (pushed on entry, popped on exit), records input-tensor/module-entry annotations,
    stores raw forward args for module-log construction, tags dynamically-created buffers.

    Args:
        trace: The active Trace.
        module: The nn.Module about to execute.
        frame: The frame ``push_frame`` created for this call.
        call: ``(args, kwargs)`` of the forward call.

    Returns:
        ``(input_tensor_labels, input_tensor_labels_at_entry, forward_start_time, session)``
        for ``_record_module_exit_metadata`` (pass-through, replacement, duration, escrow).
    """

    args, kwargs = call
    workspace = trace._module_capture_ws
    module_address = "" if frame.address is None else frame.address
    mod_id = frame.module_id
    workspace.module_build_data["module_training_modes"][module_address] = module.training
    module_call_index = frame.pass_number
    if module_call_index <= 0:
        # A hard raise, not an assert: ``python -O`` strips asserts, and this
        # guard catches a broken push_frame/entry ordering that would silently
        # mislabel every module call in the capture.
        raise RuntimeError(
            "TorchLens internal error: _module_stack.push_frame must increment "
            f"before module entry (module {module_address!r} has call index "
            f"{module_call_index})"
        )
    # Stash forward args for the input-adoption loop below (stubbed at the end).
    workspace.module_forward_args[(module_address, module_call_index)] = (args, kwargs)
    module_call_label_str = f"{module_address}:{module_call_index}"

    capture_session = capture_session_for(trace)
    if capture_session is not None:
        capture_session.enter_module_pass(module_call_label_str)
    _register_module_input_container_snapshots(
        trace, args, kwargs, module_call_label=module_call_label_str
    )
    # Tensor arguments (not Parameters, which are source tensors), each label read once: the
    # template, the adoption loop and the payload stubs reuse it until a stamp changes.
    input_tensors = get_arg_tensors_for_resolution(args, kwargs)
    labels_at_entry = input_labels_by_id(input_tensors)
    forward_args_template = forward_kwargs_template = None
    if getattr(trace, "is_replayable", False) or trace._options_save_arg_templates:
        from torchlens.backends.torch._ops_arguments import module_args_template

        forward_args_template = module_args_template(trace, module, call, labels_at_entry)
        forward_kwargs_template = forward_args_template if kwargs else None
    forward_start_time, code_context, call_stack = _record_module_entry_build_data(trace)

    input_tensor_labels = set()
    input_tensor_labels_at_entry = []
    layer_argnames: list[tuple[str, Any]] = []
    stamps_changed = False
    is_wrapped = functorch_wrapped_check(trace.capture_events.run_nonce) if input_tensors else None
    for t in input_tensors:
        if is_wrapped is not None and is_wrapped(t):
            continue
        known = None if stamps_changed else labels_at_entry
        label, cleared = entry_live_label(t, known, trace.capture_events.live_index.by_raw_label)
        stamps_changed = stamps_changed or cleared
        # Lazily register buffer tensors that haven't been logged yet. The
        # module-entry gate validates the static stamp through the session belt
        # (current-session object + storage identity), never raw.
        buffer_address = None if label is not None else session_validated_buffer_address(trace, t)
        if buffer_address is not None:
            log_source_tensor(trace, t, "buffer", buffer_address)
            label = get_tensor_label(t)
            stamps_changed = True
        if label is None and log_closure_constant_if_known(trace, t):
            label = get_tensor_label(t)
            stamps_changed = True
        if label is None:
            # An untagged tensor enters a module. A genuine raw
            # ``register_forward_hook`` output replacement is already tagged at
            # module exit by ``_make_user_forward_hook_wrapper`` (with
            # is_edited=True), so anything still untagged here is an
            # internally generated tensor whose construction TorchLens could not
            # trace (e.g. an attention mask built inside ``torch.vmap``). Log it
            # as a clean graph source -- NOT a user intervention -- so it
            # validates legitimately rather than getting a functionless
            # intervention-replacement placeholder.
            _ensure_module_output_tensor_logged(
                trace, t, module, parent_labels=[], kind="internal_source"
            )
            label = get_tensor_label(t)
            stamps_changed = True
            # R16: adoption must not LAUNDER an escape (see the helper).
            record_module_boundary_adoption(trace, t, label, "entry", module_address)
        if label is None:
            continue  # Skip untracked tensors (e.g. external constants) (#117)
        input_tensor_labels.add(label)
        trace.capture_events.live_index.note_module_entry(mod_id, label, module_address)
        # Record which arg position this tensor occupies for this module pass.
        for arg_key, arg_val in itertools.chain(enumerate(args), kwargs.items()):
            if arg_val is t:
                layer_argnames.append((label, arg_key))
        input_tensor_labels_at_entry.append(label)

    # Catch buffers created dynamically (e.g. in forward()) after initial scan.
    stamps_changed = _tag_untagged_buffers(trace, module) or stamps_changed
    # Release at emission: neither the workspace stash nor the
    # enter event may pin live payloads for the rest of the forward. Every
    # consumer on the torch trace path reads shape/dtype facts only, and
    # ``ModuleCall.forward_args`` is nulled before the trace is returned.
    # Stubbing waits until here so the input-adoption loop above has stamped
    # labels the stubs can carry.
    stub_labels = None if stamps_changed else labels_at_entry
    stub_args, stub_kwargs = stub_module_call_args(args, kwargs, trace, stub_labels)
    workspace.module_forward_args[(module_address, module_call_index)] = (stub_args, stub_kwargs)
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
            layer_argnames=tuple(layer_argnames),
            input_labels=tuple(input_tensor_labels_at_entry),
        )
    )
    return input_tensor_labels, input_tensor_labels_at_entry, forward_start_time, capture_session


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


def _attach_live_fire_results(
    trace: "Trace", tensor: torch.Tensor, fields_dict: dict[str, Any], raw_label: str
) -> None:
    """Fold an untagged tensor's pending live-fire results into its boundary op fields."""

    fire_results = _pop_tensor_live_fire_results(tensor)
    if fire_results:
        # The record's ``edits`` derive from ``fire_results`` at scatter.
        fields_dict["fire_results"] = fire_results
        fields_dict["is_edited"] = any(result.replaced for result in fire_results)
        # Fire results only mint replacement-event evidence when the live-fire
        # machinery is actually armed for THIS capture; a stale
        # ``_tl_live_fire_results`` attribute leaked from an earlier intervened
        # trace stays unledgered, so validation refuses the placeholder it
        # would otherwise launder into a plain capture.
        if fields_dict["is_edited"] and _live_intervention_machinery_armed():
            _note_replacement_event(trace, raw_label, origin="live_fire")


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
      tensor the user injected. The op is marked ``is_edited`` and
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

    is_internal_source = kind == "internal_source"
    if is_internal_source:
        # An internal source has no real dataflow parents -- the construction
        # ops are untraceable, so attaching the previous op as a "parent" would
        # be a fabricated edge. Register it as a clean graph source.
        parent_labels = []

    parent_entries = [
        LiveOpView(trace, trace.capture_events.live_index.require_event(label))
        for label in parent_labels
    ]
    template_entry = parent_entries[0] if parent_entries else None
    layer_type = "internalsource" if is_internal_source else "interventionreplacement"
    raw_label, raw_index, type_index = _next_untagged_tensor_label(trace, layer_type)
    fields_dict = _replacement_template_fields(template_entry)
    address = _module_address(module)
    # The TOP-LEVEL model is never registered in `_mod_call_index` -- its
    # `forward` is deliberately left undecorated ("Root module is handled
    # separately by trace", `_prepare_model_once`'s `_visit_once`), so
    # `push_frame` (the sole incrementer) never runs for it. A raw
    # `register_forward_hook` on the root module itself (depth 0) therefore
    # hits this lookup with a module never present in the dict. Default to 1,
    # matching the codebase's fixed "self:1" convention for the root's single
    # canonical call (see `torchlens/_postprocess/finalization.py`). This
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
    quantized_flops_forward = estimate_quantized_module_forward_flops(
        module,
        tuple(tensor.shape),
        module_args,
        module_kwargs,
    )
    root_ancestors: set[str] = set()
    input_ancestors: set[str] = set()
    internal_source_ancestors: set[str] = set()
    for parent_entry in parent_entries:
        root_ancestors.update(parent_entry._graph_root_ancestors or set())
        input_ancestors.update(parent_entry.input_ancestors or set())
        internal_source_ancestors.update(parent_entry._graph_internal_source_ancestors or set())
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
            "short_label": None,
            "layer_label": None,
            "short_layer_label": None,
            "type": layer_type,
            "type_index": type_index,
            "pass_number": 1,
            "num_passes": 1,
            "lookup_keys": [],
            "out": None,
            "transformed_out": None,
            "has_saved_activation": False,
            "activation_transform": trace._options_activation_transform,
            "annotations": {},
            "edits": [],
            "is_edited": not is_internal_source,
            "detach_saved_activations": trace._options_detach_saved_activations,
            "output_device": trace._options_output_device,
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
            "save_grads": trace._options_save_grads not in (None, False),
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
            "equivalent_ops": trace._graph_op_equivalence_classes[raw_label],
            "recurrent_ops": [],
            "parents": parent_labels,
            "parent_arg_positions": {"args": {}, "kwargs": {}},
            "_edge_uses": [],
            "root_ancestors": root_ancestors or {raw_label},
            "children": [],
            "has_children": False,
            "is_input": False,
            "has_input_ancestor": any(entry._graph_has_input_ancestor for entry in parent_entries),
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
            or any(entry._graph_has_internal_source_ancestor for entry in parent_entries),
            "internal_source_parents": [],
            "internal_source_ancestors": internal_source_ancestors,
            "is_internal_sink": False,
            "is_terminal_bool": False,
            "is_terminal_conditional_bool": False,
            "conditional_context_kind": None,
            "conditional_wrapper_kind": None,
            "terminal_conditional_id": None,
            "is_scalar_bool": bool(tensor.dtype == torch.bool and tensor.ndim == 0),
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
    _attach_live_fire_results(trace, tensor, fields_dict, raw_label)
    trace._graph_op_equivalence_classes[raw_label].add(raw_label)
    new_entry = _make_layer_log_entry(
        trace, tensor, fields_dict, (), {}, trace._options_activation_transform
    )
    if is_internal_source:
        # Keep the special-list <-> flag invariant satisfied: every op with
        # is_internal_source set must appear in trace.internal_source_ops.
        trace.internal_source_ops.append(new_entry._label_raw)
    set_tensor_label(tensor, new_entry._label_raw)

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
                            is_edited=True,
                        )
                    )
                # The hook returned an ALREADY-TRACED tensor: that REWIRES the
                # module boundary to reuse an existing op's value, it does not
                # replace that op's own computation. The durable
                # ``is_edited`` stamp above stays as the
                # intervened-capture disclosure (runnable save keys its
                # user_intervention_unsupported refusal on it), but NO
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
    frame: "ModuleStackFrame",
    out: Any,
    entry: tuple[set[str], list[str], float, Any],
) -> tuple[tuple[torch.Tensor, str], ...]:
    """Record post-forward module metadata for exhaustive mode.

    Called immediately after ``orig_forward()`` returns in the
    ``module_forward_decorator`` with the call's frame and its entry facts. Creates
    boundary identity ops for pass-through outputs, recovers replacement outputs,
    and annotates output tensors with module-exit metadata.

    Returns
    -------
    tuple[tuple[torch.Tensor, str], ...]
        Exact output tensors and raw labels of boundary Ops inserted for otherwise
        untraceable outputs.
    """
    input_tensor_labels, _labels_at_entry, forward_start_time, capture_session = entry
    address = "" if frame.address is None else frame.address
    module_call_index = frame.pass_number
    module_type = type(module).__name__ if frame.module_type is None else frame.module_type
    is_identity_module = module_type.lower() == "identity"
    from torchlens.intervention._runtime import (
        _peek_module_intervention_parent_labels,
        _peek_tensor_live_fire_results,
        _record_module_intervention_parent_labels,
        _record_tensor_live_fire_results,
    )

    output_entries = list(_walk_output_tensors_with_paths(out))
    output_tensors = [entry[0] for entry in output_entries]
    if not output_tensors:
        output_tensors = get_vars_of_type_from_obj(out, torch.Tensor, search_depth=4)
        output_entries = [(tensor, (), None) for tensor in output_tensors]
    role_hints = role_hints_for_module(module) if len(output_entries) > 1 else None
    module_call_label = f"{address}:{module_call_index}"
    forward_duration = time.time() - forward_start_time
    output_structure = output_entries[0][2] if output_entries else None
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
        label_is_live = tensor_label is not None  # the live lookup just accepted it
        # Peek through BOTH evidence channels (plain attribute + the
        # storage-owned side table): a replacement tensor that rejects dynamic
        # attributes must not read as evidence-free here, or the fresh value
        # below is silently misclassified as an internal source.
        fire_results = _peek_tensor_live_fire_results(t)
        if is_identity_module or (tensor_label is not None and tensor_label in input_tensor_labels):
            label_is_live = False
            intervention_parent_labels: list[str] = list(
                _peek_module_intervention_parent_labels(t, trace)
            )
            t = cast(Callable[[torch.Tensor], torch.Tensor], _state._decorated_identity)(t)
            if fire_results:
                # Re-attach through the robust recorders: the side channels
                # absorb attr-rejecting tensors, and a tensor writable through
                # NEITHER channel refuses typed instead of silently losing the
                # intervention provenance.
                _record_tensor_live_fire_results(t, fire_results)
                if intervention_parent_labels:
                    _record_module_intervention_parent_labels(
                        t, tuple(intervention_parent_labels), trace
                    )
            tensor_label = get_live_tensor_label(t, trace.capture_events.live_index.by_raw_label)
        if tensor_label is None and not fire_results and log_closure_constant_if_known(trace, t):
            # A module returning a tensor its forward closes over (SPEC 2 rule 1)
            # returns a closure constant, recorded as a ``constant`` source op.
            tensor_label = get_tensor_label(t)
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
                            is_edited=any_replaced,
                            fire_results=remaining_fire_results,
                        )
                    )
                if any_replaced and _live_intervention_machinery_armed():
                    _note_replacement_event(trace, tensor_label, origin="live_fire")
        # The stub only re-validates the label; a label the live lookup returned
        # for this very tensor needs no second lookup.
        is_atomic_module = not label_is_live and _is_bottom_level_submodule_exit(trace, t, module)
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
                container_path, output_index, hints=role_hints
            )
        output_names.append(output_name)
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

    if capture_session is not None:
        capture_session.escrow_module_exit_outputs(module_call_label, tuple(exit_outputs))
    return tuple(untraceable_output_boundaries)


def _evaluate_module_exit_halt(
    trace: "Trace", frame: "ModuleStackFrame", module_type: str, options: Any, out: Any
) -> None:
    """Evaluate the session's halt predicate at one exhaustive module exit."""
    from ..._capture.predicates import _evaluate_halt
    from ..._capture.projections import _build_record_context

    exit_ctx = _build_record_context(
        kind="module_exit",
        op_log_or_op_data={
            "label": f"{frame.address}:exit:{frame.pass_number}",
            "address": frame.address,
            "module_type": module_type,
            "module_pass_number": frame.pass_number,
            # tl.module evaluates capture-time subjects through
            # output_of_module_calls, so a module-boundary halt
            # (incl. compiled stop_after= module addresses) can only
            # fire if this exit ctx names its own module call; the
            # exhaustive path previously left it empty and every
            # tl.module halt silently ran the full forward.
            "output_of_module_calls": (f"{frame.address}:{frame.pass_number}",),
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


def module_forward_decorator(
    orig_forward: Callable[..., Any], module: nn.Module
) -> Callable[..., Any]:
    """Toggle-gated forward wrapper for an nn.Module's ``forward`` method.

    **Closure design**: Closes over ``module`` but reads ``trace`` from
    ``_state._active_trace`` at call time. The session installs it inside a
    ``_SessionForward`` (``_session_forwards.py``), which keeps deepcopies and
    pickles of the module TorchLens-free, and removes it at session cleanup;
    a reference the user kept can still be called later, in another session.

    **Execution modes**:

    1. **Logging off** (``_state._logging_enabled is False``), or ``module``
       not registered in the active session (a wrapper that outlived its own
       session): pass through to ``orig_forward`` unrecorded.

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

    module_id = id(module)
    # ``[run nonce, ModuleMeta]``: the registry lookup is paid once per capture
    # run (an entry is only ever removed by ``release_model``, refused mid-capture).
    meta_of_run: list[Any] = [None, None]

    @wraps(orig_forward)
    def decorated_forward(*args: Any, **kwargs: Any) -> Any:
        """Route one module forward call through TorchLens capture bookkeeping."""
        # ---- Toggle gate: near-zero overhead when logging is off ----
        if not _state._logging_enabled or _state._active_trace is None:
            return orig_forward(*args, **kwargs)

        trace = _state._active_trace
        workspace = trace._module_capture_ws
        # A wrapper that outlived its own session (a stashed ``module.forward``,
        # a shallow module copy) reaching an UNRELATED capture: the module is not
        # registered there, so it runs unrecorded, like any helper function.
        if module_id not in workspace.mod_call_index:
            return orig_forward(*args, **kwargs)
        if trace._options_mode == "predicate":
            _state.note_fingerprint_token(_state.module_token(_module_address(module)))
            return _predicate_module_forward(trace, module, orig_forward, args, kwargs)
        run_nonce = trace.capture_events.run_nonce
        if meta_of_run[0] != run_nonce:
            meta_of_run[:] = [run_nonce, get_module_meta(module)]
        meta = meta_of_run[1]
        # Module-entry fingerprint token, before any of this module's ops; the
        # native twin is ``_call_fingerprint.install_module_token_hooks``. The
        # address is read per call because role swaps re-address modules.
        address = "" if meta is None or meta.address is None else meta.address
        _state.note_fingerprint_token(_state.module_token(address))

        # ---- Exhaustive mode: full entry -> forward -> exit ----
        frame = _mstack.push_frame(trace, workspace.exhaustive_module_stack, module, meta)
        module_type = type(module).__name__ if frame.module_type is None else frame.module_type
        bind_invocation(trace, module, frame.address, frame.pass_number, (args, kwargs))
        try:
            entry = _record_module_entry_metadata(trace, module, frame, (args, kwargs))
            # Echo narrator slot (module seam, exhaustive tier).
            # Duck-typed session read: the hot path imports nothing.
            echo_session = trace.__dict__.get("_echo_session")
            if echo_session is not None:
                echo_session.emit_module_enter(frame.address, module_type)
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
                from torchlens.intervention._runtime import _apply_module_boundary_live_hooks

                out = _apply_module_boundary_live_hooks(
                    out,
                    module_address=frame.address,
                    module_call_index=frame.pass_number,
                    module_type=module_type,
                    call_args=args,
                    call_kwargs=dict(kwargs),
                )
            except Exception:
                # Exception safety (#122): the frame is the call's only record and
                # the ``finally`` below pops it.
                # A raising module never completed: abandon the echo frame
                # without an exit line (the enter without exit is the honest
                # crash shape; depth stays consistent if user code catches).
                if echo_session is not None:
                    echo_session.abandon_module_frame()
                raise
            untraceable_output_boundaries = _record_module_exit_metadata(
                trace, module, frame, out, entry
            )
            if echo_session is not None:
                echo_session.emit_module_exit(frame.address)
            mark_expected_original_accounted(
                expected_token,
                captured=bool(untraceable_output_boundaries),
                boundary_outputs=untraceable_output_boundaries,
            )
            options = trace.__dict__.get("_predicate_save_options")
            if options is not None and options.halt is not None:
                _evaluate_module_exit_halt(trace, frame, module_type, options, out)
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

    Puts every submodule ``forward`` back FIRST (plain dict writes that cannot
    fail), so no later cleanup fault can leave a session wrapper on the model.
    **Does NOT** remove the permanent module metadata, which lives in
    TorchLens's identity-keyed registry, never on the module object.
    """

    _restore_session_forward_wrappers(trace)
    try:
        uninstall_offload_hook_shims(trace)
        from .prehook_provenance import rollback_prehook_provenance

        rollback_prehook_provenance(trace)
    finally:
        # Restore requires_grad and remove session-scoped param attributes
        restore_session_param_state(trace, model)

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

    # Session-scoped state released here, not only at the START of the next capture: the
    # class-metadata cache pinned generated / function-local module classes (and, through
    # ``meta["cls"]``, their code objects and closures) alive indefinitely, and the
    # view-write relabels kept every later capture off the per-call emptiness check. A
    # capture that dies before this epilogue leaves stale entries that the next start clears
    # (class metadata) or that its retired session token keeps inert (relabels).
    _module_class_metadata_cache.clear()
    _state._dir_cache.clear()
    _VIEW_MUTATION_RELABELS.clear()

    # retire this capture's label-anchoring session LAST, after every
    # cleanup pass that consults it. Any label still carried by an object that
    # outlives the capture is now anchored to a retired session and can never
    # be accepted as provenance by a later capture, whatever route it escaped by.
    end_label_session()


def _undecorate_model_tensors(trace: "Trace", model: nn.Module) -> None:
    """Remove session-scoped metadata from non-parameter tensors in the model.

    Uses a bounded ``__dict__`` scan instead of ``iter_accessible_attributes``
    (slow dir() + getattr MRO walk). Handles tensors stored directly as
    attributes, inside Python containers, and inside model-owned helper objects.

    The AUTHORITATIVE clear iterates the RECORDED buffer
    inventory (``trace._session_buffer_inventory``) first -- a buffer popped
    from ``_buffers`` mid-forward escapes the ``model.modules()`` re-traversal
    below, and its surviving ``TensorMeta`` address would let a later capture
    accept stale buffer provenance. The traversal is kept as belt-and-suspenders
    for unstamped model-owned tensors; ``clear_meta`` is idempotent.

    the session identity registry (``trace._session_buffer_identity``) is
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
    _clear_model_tree_tensor_metadata(model, set())
    # Also clean any tensors from the registered buffer dict (_buffers)
    for submodule in model_modules(model):
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

    wrap_torch()  # idempotent — no-op if already wrapped; auto-rewraps after unwrap
    _prepare_model_once(model)  # idempotent — cached in _state._prepared_models
