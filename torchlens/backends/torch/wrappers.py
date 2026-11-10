"""Lazy torch function wrapping for capture-time operation interception.

Wrappers persist after first installation and branch on ``_state._logging_enabled``.
This module also patches detached torch references and torch transform boundaries.
"""

import copy
import inspect
import os
import sys
import threading
import time
import types
import warnings
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import torch

# Imported into THIS module's globals so torch.jit.script can resolve the
# torch.overrides boilerplate when it compiles a wrapped torch.nn.functional
# Python op (e.g. softsign): jit pulls the original op's source but resolves
# names against the wrapper's globals (this module), not torch.nn.functional's.
# Every functional op shares the
# ``if has_torch_function_unary(x): return handle_torch_function(...)`` preamble;
# jit treats has_torch_function_unary as always-False, so the branch is elided.
from torch.overrides import handle_torch_function, has_torch_function_unary  # noqa: F401

from ... import _state
from ..._capture.arg_positions import _ensure_schema_tensor_position_corrections
from ..._capture_state_helpers import _is_uninitialized_param
from ..._constants import _get_torchvision_funcs, get_orig_torch_funcs
from ..._data_classes.internal_types import FuncExecutionContext
from ..._errors import CaptureNotActiveError
from ..._utils import _compiled_write_guard
from ..._utils._compiled_write_guard import materialize_deferred_for_call
from ..._utils._dispatch_modes import TensorSubclassUnsupportedError, pause_own_dispatch_modes
from ..._utils._hop_entry_compile import compile_disabled
from ..._utils._tensor_label import (
    _DETACHED_ACTIVATION_PROPAGATION_FUNCS,
    DescriptorCompatProperty,
    get_tensor_label,
    has_detached_saved_activations,
    is_decorated_function,
    is_tensor_data_alias,
    mark_decorated_function,
    mark_tensor_data_alias,
    propagate_detached_saved_activation,
    set_tensor_label,
)
from ..._utils._torch_compat import (
    HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE,
    dynamo_is_compiling,
    fix_tensor_sequence_slot,
    get_current_function_mode_stack,
    get_device_constructors,
    get_device_context_type,
    get_functorch_transform_probe,
    get_jit_boolean_dispatch_table,
    get_jit_builtin_table,
    get_optional_torch_namespace,
    get_torch_function_mode_stack_length,
    mark_torch_capability_missing,
)
from ..._utils._torch_symbols import tensor_read
from ..._utils.display import identity
from ..._utils.hashing import make_random_barcode
from ..._utils.introspection import get_vars_of_type_from_obj
from ..._utils.rng import log_current_autocast_state, log_op_rng_states
from ..._utils.tensor_utils import (
    _DEFER_ENABLED as _COW_ENABLED,
    _DEFER_GATE as _COW_GATE,
    arm_deferred_payload_window,
    disarm_deferred_payload_window,
    print_override,
)
from . import _storage_exports, _tracing_pass
from ._completeness_audited import LAZY_MODULE_MATERIALIZE_WRAPPER
from ._completeness_boundaries import completeness_scope_for_wrapper
from ._completeness_cross_thread import (
    observe_nonowner_operands,
    record_host_string_escape_source,
    string_escape_is_owner_thread,
)
from ._completeness_dispatch import record_uncaptured_owner_callsite
from ._completeness_escape_state import record_data_alias_mutation
from ._completeness_origins import internal_scalar_read
from ._foreign_data import FOREIGN_DATA_FUNC_NAME, HOST_CONSTRUCTOR_FUNC_NAMES, resolve_host_entry
from ._foreign_wrappers import rebase_buried_wrapper, warn_buried_sites
from ._function_boundary import (
    _BARCODE_TRANSPARENT_FUNCS,
    BOUNDARY_NOTE_FUNC_NAMES,
    funcs_not_to_log,
    inside_function_apply,
    note_function_boundary,
    print_funcs,
    unlogged_wrapper_name,
)
from ._fused_norm_writes import fused_call_writes
from ._logged_arguments import logged_args, logged_receiver_output
from ._op_markers import _pop_op_markers, _push_op_markers
from ._ops_autograd import register_call_input_container_snapshots
from ._ops_capture_records import (
    _is_inplace_augmented_assignment_dunder,
    _record_label_version_snapshot,
    _stamp_same_object,
)
from ._ops_containers import _walk_output_tensors_with_paths
from ._ops_emission import apply_live_hooks_to_outputs, log_function_output_tensors
from ._output_tensors import _collect_output_tensors
from ._protocol_redispatch import STILL_DECLINED, redispatch_declined
from ._torch_ops_calls import install_torch_ops_call_recorders, uninstall_torch_ops_call_recorders
from ._tracing_pass import call_inside_tracing_pass, inside_tracing_tensor_pass
from ._transform_wrappers import (  # noqa: F401  (re-exported by their pre-split spelling)
    DIRECT_TRANSFORM_SITES,
    TRANSFORM_BUILDER_SITES,
    _callable_code_location,
    _decorate_direct_transforms,
    _decorate_transform_builders,
    _set_transform_metadata,
    _setattr_ignoring_advisories,
    _transform_builder_config,
    _transform_tags,
    _warn_transform_boundary_collapse,
    direct_transform_decorator,
    transform_builder_decorator,
)
from ._user_grad_fn import grad_fn_read_refused
from ._view_writeback import overlapping_labeled_aliases, view_mutation_propagation_enabled
from ._warm_originals import originals_exposed
from ._weightsfree_ctx import active_factory_device as _active_factory_device
from ._wrappers_refusals import (
    _constructs_strict_subclass,
    _pauses_for_subclass_conversion,
    _raise_if_subclass_ctor_under_dispatch_mode,
)
from .aliasing import _tensors_alias
from .arg_names import get_arg_names
from .belt import restore_belt_references, sweep_stale_belt_references
from .buffer_writes import (
    record_op_buffer_writes,
    resolve_or_late_index_buffer_address,
    snapshot_buffer_args,
)
from .completeness_witness import CompletenessWitnessMode
from .escape_detection import (
    EscapeDetectorMode,
    expected_original_call,
    mark_expected_original_accounted,
    reset_detector_tables,
)
from .param_mutation import (
    adopt_minted_parameter_label as _adopt_minted_parameter_label,
    frozen_parameter_receiver_context as _frozen_parameter_receiver_context,
    is_mutated_parameter_receiver as _is_mutated_parameter_receiver,
    late_register_forward_created_parameter,
    materialized_lazy_slots,
    nested_write_relabeled_receiver as _nested_write_relabeled_receiver,
    note_lazy_slot_materializing,
    parameter_mutation_output_for_logging as _parameter_mutation_output_for_logging,
    receiver_kwarg_as_positional as _receiver_kwarg_as_positional,
    receiver_mutation_label as _receiver_mutation_label,
    same_object_logging_copy as _same_object_logging_copy,
)
from .sources import log_closure_constant_if_known, log_source_tensor
from .tensor_tracking import _add_tensor_backward_hook

if TYPE_CHECKING:
    pass


def _diagnostic_edge_armed() -> bool:
    """Return whether either exact wrapper-edge diagnostic is enabled.

    Returns
    -------
    bool
        ``True`` when a shared one-shot token is required.
    """

    return _state.diagnostic_observer_armed()


# ---------------------------------------------------------------------------
# CPython slot fixup for Tensor sequence protocol
# ---------------------------------------------------------------------------


def _fix_tensor_sequence_slot() -> None:
    """Clear the stale sq_item C slot on torch.Tensor after dunder changes."""

    fix_tensor_sequence_slot()


def _is_inside_functorch_transform() -> bool:
    """Return True if inside a vmap/grad/etc. functorch transform."""
    transform_probe = get_functorch_transform_probe()
    if transform_probe is None:
        return False
    return transform_probe() is not None


def _is_inside_dynamo_compilation() -> bool:
    """Return True while Dynamo is tracing the current frame.

    Returns
    -------
    bool
        Whether a ``torch.compile`` region is being traced right now. False when
        the capability probe is unavailable, so an absent probe degrades to
        "not compiling" and leaves capture behavior exactly as it was.
    """

    return dynamo_is_compiling()


def _warn_dynamo_region_not_logged() -> None:
    """Warn once that a compiled region's interior was not logged.

    Returns
    -------
    None
        Emits a ``UserWarning`` describing the honest gap in the Trace.
    """

    import warnings

    warnings.warn(
        "TorchLens detected a torch.compile (Dynamo) region during this forward pass. "
        "Operations that run inside the compiled region are not logged: while Dynamo "
        "traces the region (cold compile), the tensors it passes through the wrappers "
        "are data-free FakeTensors, and a warm-cache execution bypasses the Python "
        "wrappers entirely. The returned Trace contains only operations that ran "
        "OUTSIDE the compiled region. Compiled child nn.Modules are unwrapped to their "
        "eager source automatically; a compiled plain-attribute callable or free "
        "function cannot be, so call the eager function during capture if you need its "
        "interior logged (on torch >= 2.6, TorchLens instead runs compiled callables "
        "eagerly via torch.compiler.set_stance and this gap does not arise).",
        UserWarning,
        stacklevel=2,
    )


def _warn_functorch_region_not_logged() -> None:
    """Emit the once-per-forward functorch transform-boundary warning."""

    warnings.warn(
        "TorchLens detected a functorch/vmap/grad/jacfwd transform "
        "during this forward pass. Operations that run inside the "
        "transform are not logged. The returned Trace will only "
        "contain operations that ran OUTSIDE the transform.",
        UserWarning,
        stacklevel=3,
    )


# Names of torch factory functions that accept a ``device`` kwarg.
# When a ``torch.device`` context manager is active (TorchFunctionMode),
# normal C dispatch injects the device automatically — but our Python
# wrappers bypass that dispatch, so we must inject it ourselves.
_DEVICE_CONSTRUCTOR_NAMES: set[str] = set()

_FULL_DECORATION_COMPLETED = False
"""True once ``decorate_all_once()`` has run to COMPLETION at least once.

Distinct from ``_state._is_decorated`` (which tracks whether wrappers are currently
INSTALLED, and flips back to False on ``unwrap_torch()``). ``_wrap_torch_locked`` keyed
its "full decoration vs re-install from the existing maps" choice on
``_state._orig_to_decorated`` being non-empty -- but a decoration that failed partway
through pass 2 leaves that map PARTIALLY populated, so the retry took the re-install
branch, reinstalled only the partial map, and stamped ``_is_decorated = True``. That
permanently disarmed the #138 retry guard ``decorate_all_once`` is built around and left
never-decorated functions silently unlogged in every future capture. Completion, not
map non-emptiness, is the correct predicate.
"""

# Argument leaf types that never require the recursive object crawler. Keeping
# these tuples module-local avoids rebuilding the isinstance chains for every op.
_SIMPLE_ARG_TYPES = (int, float, bool, str, type(None))
_FLAT_ARG_TYPES = (*_SIMPLE_ARG_TYPES, torch.dtype, torch.device)
# Exact callable types that the BFS provably yields no tensors for when the
# instance ``__dict__`` is empty: plain/builtin function attribute crawls see
# only dunder (filtered) class attributes, so expanding one finds nothing.
# ``tensor.register_hook(fn)`` — TorchLens's own per-output gradient hook —
# is the hot case: without this, every such call took the full BFS fall-back.
# Bound methods are deliberately NOT listed: ``dir()`` on a method surfaces the
# underlying function's attributes, so they keep the BFS.
_LEAF_CALLABLE_TYPES = (types.FunctionType, types.BuiltinFunctionType)

# Lazy imports cached at first use.
_torch_function_mode_len = None
_DeviceContext = None


def _recorded_func_name(namespace_name: str, attr_name: str) -> str:
    """Return the TorchLens function name for a decorated torch callable.

    Parameters
    ----------
    namespace_name:
        Dotted torch namespace path containing the callable attribute.
    attr_name:
        Attribute name used for installation on the namespace.

    Returns
    -------
    str
        User-facing function name recorded on captured Ops.
    """

    if namespace_name.startswith("torch.ops.torchvision.") and attr_name == "_op":
        return namespace_name.rsplit(".", 1)[-1]
    return attr_name


def _get_active_device() -> str | None:
    """Return the device from the innermost active ``DeviceContext``, or ``None``.

    Walks the ``TorchFunctionMode`` stack in reverse (innermost first) to find
    the most recently pushed ``DeviceContext``. This is the same device that
    PyTorch's C dispatch would inject — but since our Python wrappers bypass
    C dispatch, we must query it manually.
    """
    global _DeviceContext
    if _DeviceContext is None:
        _DeviceContext = get_device_context_type()
    if _DeviceContext is None:
        return None
    mode_stack = get_current_function_mode_stack()
    if mode_stack is None:
        return None
    for mode in reversed(list(mode_stack)):
        if isinstance(mode, _DeviceContext):
            return str(mode.device)
    return None


def _maybe_inject_device_kwarg(func_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject ``device`` kwarg for factory functions when a ``DeviceContext`` is active.

    Python wrappers bypass PyTorch's C-level ``TorchFunctionMode`` dispatch, so
    ``torch.device('meta')`` context (used by e.g. HuggingFace ``from_pretrained``)
    won't inject the device kwarg automatically. We replicate that injection here.

    Only applies to known factory functions (``torch.zeros``, ``torch.ones``, etc.)
    whose names were collected into ``_DEVICE_CONSTRUCTOR_NAMES`` at decoration time.
    """
    # Early exit: not a factory function, or caller already pinned a real device.
    if not (_DEVICE_CONSTRUCTOR_NAMES and func_name in _DEVICE_CONSTRUCTOR_NAMES):
        return kwargs
    # An EXPLICIT ``device=None`` (e.g. ``nn.Linear`` forwards ``factory_kwargs``
    # with ``device=None``) means "defer to the active device context", exactly
    # like an absent kwarg -- native C dispatch would still inject the context
    # device. Only a non-None device pins the result, so bail solely in that case;
    # otherwise fall through and inject the active DeviceContext device. Using
    # ``"device" in kwargs`` here would treat ``device=None`` as pinned and skip
    # injection, silently placing meta-context tensors on CPU.
    if kwargs.get("device") is not None:
        return kwargs
    # During an admitted weights-free capture TorchLens owns the factory
    # device through its thread-scoped slot -- no torch DeviceContext mode is
    # on the stack (its catch-all re-entry respells dunder ops), so the slot
    # supplies the placement a native device context would have injected. A caller-pinned non-None
    # device above always wins.
    slot_device = _active_factory_device()
    if slot_device is not None:
        return {**kwargs, "device": slot_device}
    stack_length = get_torch_function_mode_stack_length()
    if stack_length is not None and stack_length > 0:
        device = _get_active_device()
        if device is not None:
            return {**kwargs, "device": device}
    return kwargs


def _collect_tensor_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[torch.Tensor]:
    """Fast inline tensor extraction from function arguments.

    Most torch function calls have flat args (tensors, ints, bools, etc.).
    This avoids the full BFS crawl of get_vars_of_type_from_obj for the
    common case. Falls back to BFS only when nested containers are found.
    """
    if not kwargs and len(args) == 1:
        arg = args[0]
        if isinstance(arg, torch.Tensor):
            return [arg]
        if isinstance(arg, _FLAT_ARG_TYPES):
            return []

    tensors = []
    needs_bfs = False
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                elif not isinstance(item, _SIMPLE_ARG_TYPES):
                    needs_bfs = True
        elif isinstance(arg, dict):
            for val in arg.values():
                if isinstance(val, torch.Tensor):
                    tensors.append(val)
        elif type(arg) in _LEAF_CALLABLE_TYPES and not getattr(arg, "__dict__", None):
            # A plain function with no instance attributes cannot hold tensors
            # (the BFS crawl of it provably finds nothing) — not a BFS trigger.
            pass
        elif not isinstance(arg, _FLAT_ARG_TYPES):
            needs_bfs = True
    for val in kwargs.values():
        if isinstance(val, torch.Tensor):
            tensors.append(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
    if needs_bfs:
        all_args = args if not kwargs else (*args, *kwargs.values())
        return get_vars_of_type_from_obj(all_args, torch.Tensor)
    return tensors


def _label_mutated_prepared_parameter(
    trace: Any, param: torch.nn.Parameter, out_label: str | None
) -> None:
    """Advance a prepared Parameter's provenance to its in-place mutation op.

    A prepared Parameter normally carries no tensor label: every read of it is a
    parameter edge to its ``Param`` source. Once an in-place op mutates it inside
    the pass, later reads see the mutated value, so the live Parameter takes the
    op's label and ``_extract_arg_tensors_and_params`` then binds those reads as
    graph parents of the mutation op (reads before the mutation keep their
    parameter edge). Plain captures never reach this, so their graphs are
    unchanged.

    A gradient hook is registered only when the mutation left the Parameter with
    autograd history (a frozen Parameter written with a grad-requiring operand ends
    a non-leaf, as in eager): the gradient then flows through the live Parameter,
    so it takes the op's gradient the way any live in-place result does. Such a
    hook lives on this pass's ``grad_fn`` node, which stays in the model's
    autograd history (later forwards chain onto it), so it records only the
    owning trace's own backward and ``cleanup()`` removes it. A Parameter still a
    leaf (mutated under ``no_grad``) gets none: no gradient flows through the
    write.

    Parameters
    ----------
    trace:
        Active capture trace.
    param:
        Live prepared Parameter that the op mutated in place.
    out_label:
        Raw label of the logged mutation op, or ``None`` when nothing was logged.
    """

    if out_label is None:
        return
    set_tensor_label(param, out_label)
    _propagate_mutation_label_to_storage_aliases(trace, param, out_label)
    with internal_scalar_read():
        carries_history = param.grad_fn is not None
    if carries_history:
        _register_inplace_live_grad_hook(trace, param, out_label, owning_backward_only=True)


def _canonical_capture_callable(
    func: Callable[..., Any],
    func_name: str,
    property_accessor: str | None = None,
) -> tuple[Callable[..., Any], str]:
    """Return the replay-safe callable identity for one wrapped operation.

    ``Tensor.data`` is a C descriptor whose GETTER dispatches ``aten.detach`` and
    returns the same storage-sharing, autograd-detached value as ``Tensor.detach``.
    Record that operation under the canonical detach callable so live validation
    and portable replay agree without admitting the unsafe ``data`` descriptor
    through the callable resolver. The SETTER (``t.data = rhs``) rebinds the
    receiver onto RHS's storage: the receiver's old
    value has zero dataflow into the result, so the op is RECORDED as the
    canonical single-argument ``detach(rhs)`` call (the wrapper logs only the
    RHS argument; see ``wrapped_func``) while keeping the user-facing ``"data"``
    op name. That makes the emitted op value- and alias-exact for validation
    replay and gives the runnable producer/resolver a trusted, already-supported
    callable identity -- never a bogus two-argument ``detach`` and never the
    unresolvable raw descriptor ``__set__``. Non-rebinding mutating setters
    (``real`` / ``imag``) write through the receiver's own storage -- genuine
    receiver dataflow -- and keep their descriptor-``__set__`` identity.

    Parameters
    ----------
    func:
        Original wrapped callable.
    func_name:
        TorchLens name associated with the wrapped namespace entry.
    property_accessor:
        Accessor kind when ``func`` came from a wrapped getset property.

    Returns
    -------
    tuple[Callable[..., Any], str]
        Callable and operation name to persist for capture/replay.
    """

    if func_name != "data" or property_accessor == "del":
        return func, func_name
    decorated_detach = torch.Tensor.detach
    original_detach = _state._decorated_to_orig.get(id(decorated_detach), decorated_detach)
    if property_accessor == "set":
        # Keep the user-facing "data" op name; the recorded/replayed callable is
        # the canonical single-argument detach over the RHS-only logged args.
        return cast(Callable[..., Any], original_detach), "data"
    return cast(Callable[..., Any], original_detach), "detach"


def _untyped_storage_key(t: torch.Tensor) -> tuple[int, int, str] | None:
    """Return a tensor's storage identity key ``(data_ptr, nbytes, device)``.

    Reads run under ``pause_logging`` + ``internal_scalar_read`` so TorchLens's
    own pointer read is never recorded as a user raw-pointer escape (which
    would fail-close every runnable capture). ``None`` means the storage is
    unreadable; callers must fail closed.
    """

    from ._completeness_origins import internal_scalar_read

    try:
        with _state.pause_logging(), internal_scalar_read():
            storage = t.untyped_storage()
            return (storage.data_ptr(), storage.nbytes(), str(storage.device))
    except Exception:
        return None


# Per-trace cache for the clone-on-write eligibility decision. Keyed weakly:
# a Trace attribute would trip the portable-state completeness scrub
# (_PORTABLE_STATE_SPEC), and the decision is wrapper-internal state anyway.
_COW_STATE_PTRS_CACHE: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()
_COW_UNSET = object()


def _cow_payload_state_ptrs(trace: Any) -> Any:
    """Return the clone-on-write payload window state for one capture trace.

    Computed once per trace and cached weakly: ``None`` when deferred payload
    clones must stay disabled for this capture, else the frozenset of model
    param/buffer storage pointers that must never be deferred (their bytes can
    move through C++ side effects no wrapped-call signature announces, e.g.
    train-mode ``batch_norm`` running-stat updates).

    Eligibility is deliberately narrow — the plain default capture only. Every
    mode where saved payloads interact with other machinery (runnable
    witnesses, backward capture, gradient saving, user activation transforms
    that may mutate the payload in place, predicate recording, streaming
    writers, non-copy save modes) keeps the historical eager clone.
    ``inference_only`` (implied by ``no_grad``) only adds ``no_grad``: eligible.
    """
    cached: Any = _COW_UNSET
    with suppress(TypeError):
        cached = _COW_STATE_PTRS_CACHE.get(trace, _COW_UNSET)
    if cached is not _COW_UNSET:
        return None if _compiled_write_guard.compiled_writes_seen() else cached
    ptrs: Any = None
    if (
        _COW_ENABLED
        and not getattr(trace, "is_replayable", False)
        and not trace._options_backward_ready
        and trace._options_save_grads in (None, False)
        and trace._options_activation_transform is None
        and trace._options_mode != "predicate"
        and trace._options_save_mode == "copy"
        and getattr(trace, "_out_writer", None) is None
        and not _compiled_write_guard.compiled_writes_possible()
    ):
        model_ref = getattr(trace, "_source_model_ref", None)
        model = model_ref() if callable(model_ref) else None
        if model is not None and hasattr(model, "parameters"):
            try:
                with _state.pause_logging(), internal_scalar_read():
                    collected: set[int] = set()
                    for p in model.parameters():
                        collected.add(tensor_read(p, "untyped_storage")(p).data_ptr())
                    for b in model.buffers():
                        collected.add(tensor_read(b, "untyped_storage")(b).data_ptr())
                ptrs = frozenset(collected)
            except Exception:
                ptrs = None
    with suppress(TypeError):
        _COW_STATE_PTRS_CACHE[trace] = ptrs
    return ptrs


def _func_mutates_receiver(func_name: str) -> bool:
    """Return whether a wrapped function mutates its first tensor argument.

    Parameters
    ----------
    func_name : str
        TorchLens wrapper name for the callable.

    Returns
    -------
    bool
        ``True`` for in-place methods, augmented-assignment dunders, and
        setter-style tensor operations.
    """

    return (
        _is_inplace_augmented_assignment_dunder(func_name)
        or func_name in {"__setitem__", "__delitem__"}
        or (func_name.endswith("_") and not func_name.startswith("__"))
    )


def _positional_inplace_index(func: Any) -> int | None:
    """Return the positional index of ``func``'s ``inplace`` parameter, if any.

    Python-level functionals (``F.hardswish``, ``F.hardsigmoid``, ...) accept
    ``inplace`` as an ordinary positional-or-keyword parameter, and their
    ``nn.Module`` conveniences pass it POSITIONALLY (torch's
    ``Hardswish.forward`` runs ``F.hardswish(input, self.inplace)``), so a
    kwargs-only ``inplace`` probe misses the mutation request entirely — and
    some of those functionals mutate through UNWRAPPED builtins
    (``torch._C._nn.hardswish_``) that no inner wrapper intercepts. Only plain
    Python functions are probed: C builtins expose no signature and cannot
    carry a Python-level ``inplace`` parameter.
    """
    if not isinstance(func, types.FunctionType):
        return None
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return None
    for index, (name, param) in enumerate(parameters.items()):
        if name == "inplace":
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                return index
            return None
    return None


def _call_requests_inplace(
    inplace_param_index: int | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Return whether this call's ``inplace`` argument requests mutation.

    Checks the keyword spelling first, then the positional slot located at
    decoration time. Truthy non-``True`` values (``inplace=1``) still mutate,
    and an unreadable flag counts as a request — a spurious materialization
    is one wasted clone, never a missed pre-mutation copy-out.
    """
    if "inplace" in kwargs:
        value = kwargs["inplace"]
    elif inplace_param_index is not None and len(args) > inplace_param_index:
        value = args[inplace_param_index]
    else:
        return False
    try:
        return bool(value)
    except Exception:  # noqa: BLE001 - unreadable flag must count as a mutation request (fail-safe)
        return True


def _propagate_data_alias_provenance(
    func_name: str,
    data_alias_inputs: tuple[torch.Tensor, ...],
    output_tensors: list[torch.Tensor],
) -> None:
    """Propagate ``Tensor.data`` provenance through storage-sharing outputs.

    Parameters
    ----------
    func_name : str
        Original wrapped callable name before replay canonicalization.
    data_alias_inputs : tuple[torch.Tensor, ...]
        Input tensors already known to descend from ``Tensor.data``.
    output_tensors : list[torch.Tensor]
        Live output tensors after capture labels have been assigned.
    """

    if func_name == "data":
        for output in output_tensors:
            mark_tensor_data_alias(output)
        return
    if not data_alias_inputs:
        return
    with internal_scalar_read():
        for output in output_tensors:
            if any(_tensors_alias(output, source) for source in data_alias_inputs):
                mark_tensor_data_alias(output)


def _register_inplace_live_grad_hook(
    trace: Any, tensor: Any, raw_label: str, *, owning_backward_only: bool = False
) -> None:
    """Hook the live in-place result so its gradient is captured under ``raw_label``.

    In-place ops log their output against a ``safe_copy`` whose grad_fn is a
    dead-end ``CloneBackward`` node; the live, in-place-modified tensor is what
    downstream ops consume, so the real gradient flows through it and this hooks
    it (``_add_tensor_backward_hook`` dedups by ``(label, id(tensor))`` and only
    hooks autograd-participating tensors). A tensor whose ``grad_fn`` read raises
    or warns (``grad_fn_read_refused``: a stale multi-output or custom-Function
    view of the written base) is skipped: eager raises on any use of it, so it
    carries no gradient. ``owning_backward_only``: see ``_add_tensor_backward_hook``.
    """

    if not isinstance(tensor, torch.Tensor) or grad_fn_read_refused(tensor):
        return

    # The live tensor is what downstream ops consume, so it takes gradient
    # ownership of the label; the logged copy's hook (if any) stops emitting.
    _add_tensor_backward_hook(
        trace,
        tensor,
        raw_label,
        take_ownership=True,
        owning_backward_only=owning_backward_only,
    )


def _propagate_mutation_label_to_storage_aliases(
    trace: Any, mutated: torch.Tensor, out_label: str
) -> None:
    """Advance live labels of storage aliases overlapping an in-place op's target.

    An in-place op that mutates a VIEW (``v = y[0]; v.add_(100.)``) changes the
    BASE tensor's content, but the base is a different Python object: only the
    view's label used to advance, so the mutation op was a dead end and later
    consumers of the base bound to the stale pre-mutation parent while storing
    the post-mutation value (a silent error). Resolve every OTHER live
    labeled tensor whose storage byte range overlaps the mutated target and
    advance its label to the mutating op, exactly like the direct same-object
    propagation. Value replay is untouched: the consumer's version-snapshot
    machinery (``_get_parent_output_version_snapshot``) records the alias's
    actual content per child, so replay and perturbation both see the real
    full-tensor values.

    Parameters
    ----------
    trace
        Active capture trace.
    mutated
        The live tensor object the op actually wrote through (``args[0]`` for
        in-place/setter ops, each destination for ``out=`` ops).
    out_label
        The mutating op's freshly issued raw label.
    """

    from ._ops_capture_records import _record_label_version_snapshot

    # SCOPE: descriptor/grad-bound captures keep the unadvanced (historical)
    # topology. The runnable recipe + numeric attestation key payloads PER
    # LABEL, and advancing a base tensor's label to a view-mutation op makes
    # one label denote two different values (the op's view-shaped output AND
    # the full post-mutation base as the consumer's parent) -- byte-exact
    # attestation then fails on a genuinely-verifiable run.
    # Those modes stay honest without the edge: the view-lineage gate
    # fail-closes view-mediated input mutation, and validation's
    # version-snapshot replay is green with EITHER topology. Default captures
    # (receptive fields, influence geometry, collapse -- the impact
    # surface) get the mutation edge. Follow-up: teach the runnable reader
    # per-(label, consumer) payload keying, then lift this scope.
    if not view_mutation_propagation_enabled(trace):
        return

    # ``untyped_storage()`` / ``data_ptr()`` / ``stride()`` / ``storage_offset()``
    # are WITNESSED host-escape / metadata surfaces, and a genuine user
    # ``data_ptr()`` read fail-closes runnable captures to UNVERIFIABLE.
    # These are TorchLens's OWN bookkeeping reads, so they run under
    # ``pause_logging`` (no spurious op capture) plus ``internal_scalar_read``
    # (kept off the escape census / metadata patches) -- the same sanctioned
    # pattern as ``completeness_witness``'s storage-site indexers.
    with _state.pause_logging(), internal_scalar_read():
        for alias, alias_label in overlapping_labeled_aliases(mutated, out_label):
            set_tensor_label(alias, out_label)
            _register_inplace_live_grad_hook(trace, alias, out_label)
            _record_label_version_snapshot(alias, mutated_via=(mutated, alias_label))


# Positional index of the target Tensor subclass ``cls`` argument for each
# wrapped callable whose ORIGINAL can crash under an active python
# TorchDispatchMode on torch 2.1/2.2 (``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``).
# ``__new__``/``_make_subclass`` are called as ``cls(...)``/
# ``cls._make_subclass(cls, ...)`` (class first); ``as_subclass`` is an
# instance method, ``tensor.as_subclass(cls)`` (class second). Shared by the
# LOGGED path's pause (see ``constructs_tensor_subclass`` below -- ``__new__``
# unconditionally, ``_make_subclass``/``as_subclass`` only where the
# capability is absent, so a modern torch build's LOGGED-path behavior for
# those two is byte-for-byte unchanged) and the FAST-PATH gate just below,
# which REFUSES typed instead of pausing: pausing a direct top-level
# (non-reentrant) construction call was tried and reverted after it
# corrupted interpreter state when exercised across several capture paths in
# one process (segfault on torch 2.1.2) -- see
# ``TensorSubclassUnsupportedError``.
_SUBCLASS_CLS_ARG_INDEX: dict[str, int] = {"__new__": 0, "_make_subclass": 0, "as_subclass": 1}


_MUTATING_TENSOR_PROPERTY_SETTERS = frozenset({"real", "imag", "data"})

# Setters that rebind the receiver to the RHS's storage instead of writing in
# place. The mutation label must NOT propagate to other tensors on that (RHS)
# storage: their bytes were never written.
_STORAGE_REBINDING_PROPERTY_SETTERS = frozenset({"data"})

_SUBCLASS_CTOR_REFUSAL_MARKER = "already associated to a python object"


# ---------------------------------------------------------------------------
# Cold arms of ``wrapped_func``. Each runs only on a path a plain forward never
# takes (a strict Tensor-subclass constructor, a non-owner thread, a functorch
# or Dynamo region, a print function, a ``.data=`` rebind, torch's own
# subclass-construction crash), so the hot path pays no extra call per op.
# ---------------------------------------------------------------------------


def _call_subclass_ctor_unlogged(
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cls_index: int,
) -> Any:
    """Run a strict-subclass constructor with logging off, translating torch's refusal.

    Translate-on-FAILURE, never a preemptive refusal: whether an active
    TorchLens mode actually makes THIS construction crash depends on details
    (e.g. ``torch.nn.Parameter`` construction via ``_make_subclass`` routinely
    succeeds here even under an active mode) this gate cannot cheaply predict,
    so it costs nothing extra on the common (succeeding) case and only
    replaces torch's own cryptic error with a typed one when it actually raises.
    """

    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        if _SUBCLASS_CTOR_REFUSAL_MARKER in str(exc):
            raise TensorSubclassUnsupportedError(
                f"torchlens cannot construct the Tensor subclass "
                f"{args[cls_index]!r} via {func_name!r} "
                "while a TorchLens dispatch mode is active on this torch "
                "build. Remedy: upgrade torch "
                "(HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE reports False here) or "
                "avoid constructing/converting into a custom Tensor subclass "
                "inside a model forward, an intervention hook, or a "
                "validate_forward_pass replay on this build.",
                code="subclass_ctor_under_dispatch_mode_unsupported",
                func_name=func_name,
            ) from exc
        raise


def _call_unlogged_off_owner(
    func: Callable[..., Any],
    func_name: str,
    wrapped_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run a wrapped call that no trace owns on this thread, unrecorded.

    While a runnable capture is armed, a NON-owner thread's op that
    consumes a captured tensor as an operand ceilings replay proof to
    ``unverifiable`` (the worker-DERIVED cross-thread escape sibling: fresh
    worker-side storage the owner-only census never registered).
    ``_nonowner_belt_armed`` is False for every plain trace and the whole
    steady state, so the disarmed path pays one bool read.
    """

    if _state._nonowner_belt_armed and _state.active_capture()[0] is not None:
        observe_nonowner_operands(args, kwargs)
    if func_name in _DEVICE_CONSTRUCTOR_NAMES:
        kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
    try:
        out = func(*args, **kwargs)
    except TypeError as exc:
        out = redispatch_declined(exc, wrapped_func, func_name, args, kwargs)
        if out is STILL_DECLINED:
            raise
    if func_name in _DETACHED_ACTIVATION_PROPAGATION_FUNCS and has_detached_saved_activations():
        propagate_detached_saved_activation(
            func_name,
            _collect_tensor_args(args, kwargs),
            _collect_output_tensors(out),
        )
    return out


def _call_inside_functorch_transform(
    func: Callable[..., Any],
    func_name: str,
    wrapped_func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Pass a call inside a functorch transform through unlogged, warning once per forward.

    Internal TorchLens operations (``safe_copy``, ``torch.equal``, ``.item()``)
    have no vmap batching rules and would crash; the original function is
    already vmap-compatible. A raw transform interior is outside the witness
    claim; the witness-off route retains its original logging state (declined
    dispatches still redispatch).
    """

    if not _state._functorch_warning_emitted:
        _state._functorch_warning_emitted = True
        trace, _enabled = _state.active_capture()
        if trace is not None:
            trace._raw_transform_escape_detected = True
        _warn_functorch_region_not_logged()
    shadow_witness = _state._completeness_witness_mode == "shadow"
    edge_ctx: Any = nullcontext()
    if _state._escape_detector_mode == "shadow":
        edge_ctx = expected_original_call(func, f"torch_func:{func_name}:functorch")
    with _state.pause_logging() if shadow_witness else nullcontext(), edge_ctx:
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            out = redispatch_declined(exc, wrapped_func, func_name, args, kwargs)
            if out is STILL_DECLINED:
                raise
            return out


def _call_inside_dynamo_region(
    trace: Any, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Pass a call inside a Dynamo-traced region through unlogged.

    TorchLens's internal operations read tensor VALUES and the tensors flowing
    through a compiled region are data-free FakeTensors, so logging there used
    to die with a raw ``InternalTorchDynamoError``. Compiled *submodules* are
    unwrapped before capture (``_capture_state_helpers.unwrap_compiled_submodules``);
    a compiled *callable* held as a plain attribute cannot be swapped out, so
    this is the boundary for those. ``_raw_dynamo_region_detected`` is the
    specific cause and outranks every other verification reason;
    ``_raw_transform_escape_detected`` additionally licenses the
    unattributable-model-output tolerance the functorch boundary uses.
    """

    trace._raw_dynamo_region_detected = True
    trace._raw_transform_escape_detected = True
    if not _state._dynamo_warning_emitted:
        _state._dynamo_warning_emitted = True
        _warn_dynamo_region_not_logged()
    if _state._completeness_witness_mode == "shadow":
        with _state.pause_logging():
            return func(*args, **kwargs)
    return func(*args, **kwargs)


def _call_print_func(
    trace: Any,
    func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Format a captured tensor, recording the host string escape first.

    Stringifying a captured tensor extracts its VALUES into the
    returned string (a tensor-to-host value escape the user can fold back into
    control flow). ``print_override`` runs that extraction under
    ``pause_logging()``, blinding the ordinary escape observers, so the source
    is recorded BEFORE the paused format. A NON-OWNER thread must
    never flip the global pause toggle mid-forward, so a worker calls the
    original torch string function unchanged.
    """

    for stringified in _collect_tensor_args(args, kwargs):
        record_host_string_escape_source(trace, stringified)
    if string_escape_is_owner_thread(trace):
        return print_override(args[0], func_name)
    return func(*args, **kwargs)


def _rebind_setter_swaps_storage(args: tuple[Any, ...]) -> bool:
    """Whether a ``t.data = rhs`` rebind moves the receiver onto a foreign storage.

    Whether the rebind SWAPS the storage object (rhs on foreign storage) or
    PRESERVES it (rhs a view of the receiver's own storage) decides the
    ancestry barrier after the call, and is only observable BEFORE the setter
    runs. Fail closed: an unreadable storage on either side counts as a swap.
    """

    if (
        len(args) < 2
        or not isinstance(args[0], torch.Tensor)
        or not isinstance(args[1], torch.Tensor)
    ):
        return False
    receiver_storage_key = _untyped_storage_key(args[0])
    rhs_storage_key = _untyped_storage_key(args[1])
    return (
        receiver_storage_key is None
        or rhs_storage_key is None
        or receiver_storage_key != rhs_storage_key
    )


def _call_lazy_slot_plumbing(
    trace: Any, func: Callable[..., Any], func_name: str, args: tuple[Any, ...], kwargs: Any
) -> Any:
    """Pass one lazy-module materialization call on a pending slot through unlogged.

    torch's lazy pre-hook materializes pending ``UninitializedBuffer`` /
    ``UninitializedParameter`` slots through wrapped calls whose RECEIVER is the
    pending tensor (``.data`` device read or setter, ``empty_like`` on the data
    alias). A pending tensor has no shape or storage to log and the plumbing is
    pre-forward materialization, not model dataflow: the materialized buffer
    registers at the module-entry gate, the materialized parameter of a module
    built inside forward at the wrapped call that first consumes it
    (``late_register_forward_created_parameter``). The watched slot is recorded
    by identity, and under the dispatch census the call runs as the audited
    ``lazy_module:materialize`` expected-opaque owner, so its ``.data`` reads are
    accounted constructor plumbing rather than unowned drops.
    """

    note_lazy_slot_materializing(trace, args[0])
    if _diagnostic_edge_armed():
        with expected_original_call(
            func,
            LAZY_MODULE_MATERIALIZE_WRAPPER,
            func_name=func_name,
            census_scope=completeness_scope_for_wrapper(LAZY_MODULE_MATERIALIZE_WRAPPER),
        ):
            return func(*args, **kwargs)
    return func(*args, **kwargs)


def _register_first_use_state(trace: Any, arg_tensorlike: Iterable[torch.Tensor]) -> None:
    """Register buffers and late lazy Parameters among a call's tensor arguments.

    Buffers are tagged with ``_tl.address`` during model prep but get no
    ``_tl.label_raw`` until the first function actually uses them. A Parameter
    torch materialized from a pending lazy slot earlier in THIS forward (a lazy
    module built inside forward) registers late here, at its first consuming
    call; the watched-slot set is ``None`` until such a slot was seen, so the
    prepared-parameter path pays one attribute read.
    """

    watched_lazy_slots = materialized_lazy_slots(trace)
    for t in arg_tensorlike:
        if isinstance(t, torch.nn.Parameter):
            if watched_lazy_slots is not None and t in watched_lazy_slots:
                late_register_forward_created_parameter(trace, t)
            continue
        # The identity ladder + lazy late-index live in
        # buffer_writes.resolve_or_late_index_buffer_address.
        address = resolve_or_late_index_buffer_address(trace, t)
        if address is not None and get_tensor_label(t) is None:
            log_source_tensor(trace, t, "buffer", address)
        elif address is None:
            # A tensor the forward closes over (SPEC 2 rule 1) becomes a
            # ``constant`` source op on first use; anything not proven
            # pre-forward stays unattributed.
            log_closure_constant_if_known(trace, t)


def torch_func_decorator(
    func: Callable[..., Any],
    func_name: str,
    property_accessor: str | None = None,
    mutates_first_arg: bool = False,
) -> Callable[..., Any]:
    """Wrap a single torch function with toggle-gated logging.

    When ``_state._logging_enabled`` is ``False``, the wrapper logs nothing but still counts the
    call fingerprint and runs the armed checks (compile depth, copy-on-write gate, device
    injection, fast-run collector, subclass constructors) around the original.  When ``True``, it:

    1. Registers any buffer tensors seen for the first time.
    2. Snapshots args (if ``save_arg_values``), timing, RNG, and autocast state.
    3. Calls the original function.
    4. Detects **nested calls** via a barcode mechanism (see below).
    5. Handles **in-place ops** by copying the output and propagating the label back.
    6. Logs all output tensors into the active ``Trace``.

    **Barcode nesting detection**: Before calling the original function, a random
    barcode is written to ``trace._wrapper_runtime_ws.current_func_barcode``.  If the
    original function internally calls *other* wrapped torch functions, those
    inner calls will overwrite the barcode.  After the call returns, if the
    barcode still matches, this is a "bottom-level" function (leaf in the call
    tree).  Bottom-level functions get richer metadata capture.

    **In-place op handling**: When a function returns the same object as its
    first argument (``id(out) == id(args[0])``), the output is ``safe_copy``-ed
    to create a distinct tensor for logging.  For true in-place ops (trailing
    ``_`` or ``__i*`` dunder), the new label is propagated back to the original
    tensor so subsequent operations see it.  Non-mutating self-returns (e.g. ``contiguous()``
    on an already-contiguous tensor) are copied but NOT propagated back; barcode identity
    detection drops them downstream.

    Args:
        func: The original (unwrapped) torch function.
        func_name: The attribute name of the function (e.g. ``"cos"``, ``"__add__"``).
        property_accessor: ``"set"`` / ``"del"`` / ``"get"`` when ``func`` is one
            accessor of a wrapped getset property (``Tensor.real`` and friends),
            else ``None``. Mutating property SETTERS return ``None`` from a call
            that rewrites the receiver, so the wrapper reconstructs the receiver
            as the logged output for names in
            ``_MUTATING_TENSOR_PROPERTY_SETTERS``.
        mutates_first_arg: The callable writes its first tensor argument in place and
            returns it, whatever its name (a ``torch.ops`` operator whose schema
            mutates its first argument and returns nothing; the recorder hands it a
            callable that returns the mutated argument). Treated exactly like a
            trailing-underscore in-place method.

    Returns:
        The wrapped function.
    """
    is_mutating_property_setter = (
        property_accessor == "set" and func_name in _MUTATING_TENSOR_PROPERTY_SETTERS
    )
    is_storage_rebinding_setter = (
        is_mutating_property_setter and func_name in _STORAGE_REBINDING_PROPERTY_SETTERS
    )
    needs_device_injection = func_name in _DEVICE_CONSTRUCTOR_NAMES
    is_unlogged_func = func_name in funcs_not_to_log
    is_print_func = func_name in print_funcs
    is_boundary_func = func_name in BOUNDARY_NOTE_FUNC_NAMES  # an export or a file load
    is_host_constructor = func_name in HOST_CONSTRUCTOR_FUNC_NAMES
    mutates_receiver = mutates_first_arg or _func_mutates_receiver(func_name)
    inplace_param_index = _positional_inplace_index(func)
    reconstructs_receiver_output = func_name in {"__setitem__", "zero_", "__delitem__"}
    writes_args = mutates_receiver or reconstructs_receiver_output or is_mutating_property_setter
    exports_storage = func_name in _storage_exports.STORAGE_EXPORT_FUNC_NAMES
    has_inplace_signature = (
        mutates_first_arg
        or func_name.endswith("_")
        or func_name.startswith("__i")
        or func_name in {"__setitem__", "__delitem__"}
        or is_mutating_property_setter
    )
    force_distinct_return = func_name == "identity"
    # ``TensorBase.__new__`` is the one wrapped callable whose ORIGINAL refuses
    # to run under any python TorchDispatchMode when handed a strict Tensor
    # subclass cls, on EVERY torch version (see pause_own_dispatch_modes);
    # ``_make_subclass``/``as_subclass`` share the same original limitation but
    # ONLY on a torch build lacking ``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE`` --
    # torch's own nested ``__torch_function__`` return-conversion reaches one
    # of these TWO through the LOGGED path just as often as ``__new__`` (e.g.
    # ``scratch.sum()`` on a subclass reconstructing via ``as_subclass``
    # inside the SAME logged call), so they need the identical unconditional
    # treatment there, narrowed to the torch versions that actually need it.
    # Every other op pays nothing (dict lookup on a decoration-time constant).
    _logged_subclass_cls_index = (
        0
        if func_name == "__new__"
        else (
            None if HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE else _SUBCLASS_CLS_ARG_INDEX.get(func_name)
        )
    )
    constructs_tensor_subclass = _logged_subclass_cls_index is not None
    # FAST-PATH-only gate (direct top-level ``__new__``/``_make_subclass``/
    # ``as_subclass`` calls made with logging disabled -- e.g. a user model or
    # intervention hook constructing a subclass during a replay/validation
    # pass, or TorchLens's own paused internal calls): ``None`` whenever the
    # active torch build tolerates subclass construction under a dispatch
    # mode (``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``, the common case -- zero
    # cost there) or ``func_name`` is not one of the three. A direct top-level
    # call here REFUSES typed instead of pausing (see the table's own note).
    _fast_path_subclass_cls_index = (
        None if HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE else _SUBCLASS_CLS_ARG_INDEX.get(func_name)
    )
    # Decoration-time constant: ``propagate_detached_saved_activation`` is a
    # guaranteed no-op for any name outside the propagation allowlist, but its
    # ARGUMENTS (two tensor collections, each with a BFS fall-back for nested
    # args such as ``register_hook``'s callable) were evaluated eagerly on
    # every paused internal call. Gating on the closure constant skips exactly
    # the calls the allowlist check inside the helper would discard.
    is_detached_propagation_func = func_name in _DETACHED_ACTIVATION_PROPAGATION_FUNCS
    canonical_capture_callable = None
    if func_name == "__deepcopy__":
        # Recorded as ``copy.deepcopy(self)``: replay needs a fresh memo per call.
        canonical_capture_callable = (copy.deepcopy, func_name)
    elif func_name != "data" or property_accessor == "del":
        canonical_capture_callable = (func, func_name)
    # See the barcode-transparency note inside ``wrapped_func``.
    is_barcode_transparent = func_name in _BARCODE_TRANSPARENT_FUNCS
    # TorchLens's synthetic ``identity`` op (Identity modules, pass-through tensors)
    # never occurs in a native forward, so it stays out of the call fingerprint.
    fingerprint_token = None if func_name == "identity" else _state.call_token(func_name)

    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        """Dispatch a decorated torch callable through the logging gate."""
        # Dynamo tracing a lifted operator entry compile sees data-free values (_tracing_pass).
        if _tracing_pass.hop_compile_depth and _tracing_pass.inside_hop_internal_compile():
            return func(*args, **kwargs)
        # Ordered call fingerprint, counted identically with and without capture
        # (unpaused owner-thread calls only; see ``_state.CallFingerprint``).
        fingerprint = _state._call_fingerprint
        if fingerprint is not None and fingerprint_token is not None:
            fingerprint.note(fingerprint_token)
        # ---- Fast path ---- (logging off: pass through; DeviceContext injection still runs for
        # torch.device('meta') contexts). Op-logging is OWNER-thread-scoped, so a
        # NON-owner thread's ops are never logged; its host escapes still meet the method belt.
        if not _state._logging_enabled:
            # Deferred payload clones: pending aliases stay zero-copy for the process lifetime, so
            # the fast path carries the SAME pre-execution interception as the logging path: any
            # call that can write through a tensor argument first copies out pending aliases on
            # those storages. Disarmed cost is one flag check (Dynamo-guardable: _DeferGate).
            if _COW_GATE.armed and (
                writes_args
                or "out" in kwargs
                or _call_requests_inplace(inplace_param_index, args, kwargs)
                or fused_call_writes(func_name, args, kwargs)
            ):
                _COW_GATE.materialize(_collect_tensor_args(args, kwargs))
            if needs_device_injection:
                kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            # Opened before the call: the fast collector mirrors capture's nesting barcode.
            fast_collector: Any = _state._active_fast_run_collector
            fast_event = fast_collector and fast_collector.enter_event(is_barcode_transparent)
            if _fast_path_subclass_cls_index is not None and _constructs_strict_subclass(
                args, _fast_path_subclass_cls_index
            ):
                out = _call_subclass_ctor_unlogged(
                    func, func_name, args, kwargs, _fast_path_subclass_cls_index
                )
            else:
                try:
                    out = func(*args, **kwargs)
                except TypeError as exc:
                    out = redispatch_declined(exc, wrapped_func, func_name, args, kwargs)
                    if out is STILL_DECLINED:
                        raise
            if fast_event is not None:
                fast_collector.exit_event(fast_event, func_name, out, args, property_accessor)
            if is_detached_propagation_func and has_detached_saved_activations():
                propagate_detached_saved_activation(
                    func_name,
                    _collect_tensor_args(args, kwargs),
                    _collect_output_tensors(out),
                )
            return out

        active_trace = _state._active_trace
        owner_thread_id = _state._active_owner_thread_id
        current_thread_id = threading.get_ident()
        if active_trace is None or owner_thread_id != current_thread_id:
            return _call_unlogged_off_owner(func, func_name, wrapped_func, args, kwargs)

        trace = cast(Any, active_trace)
        if exports_storage and not inside_function_apply():
            _storage_exports.note_storage_export(trace, args)
        if has_inplace_signature and not args:
            # ``torch.clamp_(input=p)``: an in-place receiver passed by keyword.
            args, kwargs = _receiver_kwarg_as_positional(args, kwargs)
        if needs_device_injection:
            kwargs = _maybe_inject_device_kwarg(func_name, kwargs)

        # A HOP Autograd kernel's fake-tensor pass over its branches is not the model's.
        if inside_tracing_tensor_pass(args, kwargs):
            return call_inside_tracing_pass(func, args, kwargs)
        # Skip logging inside vmap/functorch transforms — internal TorchLens
        # operations (safe_copy, torch.equal, .item()) don't have vmap batching
        # rules and will crash. The original function is already vmap-compatible.
        # Warn once per forward pass so the user knows their Trace is
        # missing whatever runs inside the transform.
        if _is_inside_functorch_transform():
            return _call_inside_functorch_transform(func, func_name, wrapped_func, args, kwargs)

        # Skip logging inside a Dynamo-traced region, as the functorch guard above:
        # TorchLens' internal operations read tensor VALUES and a compiled region's
        # tensors are data-free FakeTensors (a raw ``'FakeTensor' object has no
        # attribute 'fake_mode'``). Compiled *submodules* are unwrapped before
        # capture; a compiled plain-attribute or free *callable* cannot be, so this
        # is its boundary: log the eager source; torch.compile internals untraced.
        if _is_inside_dynamo_compilation():
            return _call_inside_dynamo_region(trace, func, args, kwargs)

        # Lazy-module completion: torch's materialization plumbing on a
        # pending slot receiver passes through unlogged (``_call_lazy_slot_plumbing``;
        # the materialized state registers later), as the region guards above.
        if args and isinstance(args[0], torch.Tensor) and _is_uninitialized_param(args[0]):
            return _call_lazy_slot_plumbing(trace, func, func_name, args, kwargs)

        # Usage stats: count every decorated function call during logging.
        if _state._collect_usage_stats:
            _state._function_call_counts[func_name] = (
                _state._function_call_counts.get(func_name, 0) + 1
            )
            _state._function_call_models.setdefault(func_name, set()).add(
                _state._current_model_name
            )

        # Reset barcode; skip metadata-only functions that would cause recursion.
        # ``as_subclass`` (and ``__index__``) is barcode-TRANSPARENT. Torch's default
        # ``__torch_function__`` return conversion calls ``ret.as_subclass(cls)``
        # INSIDE the enclosing wrapped call (``torch.tanh(subclass_tensor)``),
        # which used to steal the enclosing call's bottom-level barcode: the
        # real op (tanh) never logged, and the trace showed a parentless
        # bookkeeping ``as_subclass`` node flagged only by the provenance
        # heuristic. The conversion still logs its own value flow, then
        # restores the enclosing barcode so the outer call keeps its identity.
        enclosing_barcode = (
            trace._wrapper_runtime_ws.current_func_barcode if is_barcode_transparent else 0
        )
        trace._wrapper_runtime_ws.current_func_barcode = 0
        if is_boundary_func:
            note_function_boundary(func_name, args)
        if is_unlogged_func:
            if _diagnostic_edge_armed():
                wrapper_name = unlogged_wrapper_name(func_name)
                with expected_original_call(
                    func,
                    wrapper_name,
                    func_name=func_name,
                    census_scope=completeness_scope_for_wrapper(wrapper_name),
                ):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        # Inline tensor extraction — avoids BFS crawl for the common case
        # where args are flat tensors. Falls back to BFS only for nested containers.
        arg_tensorlike = _collect_tensor_args(args, kwargs)
        data_alias_inputs = tuple(
            tensor for tensor in arg_tensorlike if is_tensor_data_alias(tensor)
        )
        mutates_data_alias = bool(
            args
            and isinstance(args[0], torch.Tensor)
            and is_tensor_data_alias(args[0])
            and mutates_receiver
        )

        # Register buffers (and a Parameter materialized from a pending lazy slot
        # earlier in this forward) on first encounter.
        _register_first_use_state(trace, arg_tensorlike)

        # Intercept print functions to show TorchLens label info in repr.
        if is_print_func and arg_tensorlike:
            return _call_print_func(trace, func, func_name, args, kwargs)

        # Snapshot args before the call in case in-place ops mutate them.
        receiver_label_before = _receiver_mutation_label(args)
        if trace._options_save_args:
            from ._ops_saved_arg_copies import _copy_saved_call_args

            arg_copies, kwarg_copies = _copy_saved_call_args(func_name, args, kwargs)
        else:
            arg_copies = args
            kwarg_copies = kwargs

        buffer_snapshots = snapshot_buffer_args(trace, func_name, arg_tensorlike, kwargs, args)

        # ---- Storage-rebinding setter pre-call snapshot ----
        # ``t.data = rhs`` rebinds the receiver onto RHS's storage. Whether the
        # rebind SWAPS the storage object (rhs on foreign storage) or PRESERVES
        # it (rhs a view of the receiver's own storage) decides the ancestry
        # barrier below, and is only observable BEFORE the setter runs.
        rebind_setter_swaps_storage = is_storage_rebinding_setter and _rebind_setter_swaps_storage(
            args
        )

        # ---- Deferred-clone interception (clone-on-write) ----
        # Any wrapped call that can WRITE through a tensor argument must first copy out pending
        # deferred payload aliases sharing those storages — this runs BEFORE the mutation, so saved
        # bytes stay capture-time exact. Triggers cover every wrapped mutation surface: in-place
        # methods and augmented-assignment dunders (``mutates_receiver``),
        # ``__setitem__``/``zero_``/``__delitem__``, mutating property setters, ``out=`` destinations,
        # and ``inplace=True`` conveniences (``F.relu`` / ``F.dropout``) whose actual underscore
        # mutation may run below this wrapper — including POSITIONALLY-passed ``inplace`` (torch's
        # ``Hardswish.forward`` runs ``F.hardswish(input, self.inplace)``, and
        # ``torch._C._nn.hardswish_`` below it is not a wrapped surface, so this wrapper is the ONLY
        # interception point; missing it left stale pending aliases that tripped the belt at the next
        # mutating call — the mobilenet_v3_small failure). NOT ``has_inplace_signature`` — that flag
        # is true for EVERY dunder (``"__add__".endswith("_")``) and is only ever meaningful gated
        # behind a same-object return. Storage-rebinding ``.data=`` writes no bytes but rides along
        # via its property-setter signature — a spurious materialization is merely a wasted clone,
        # never a correctness risk.
        if _COW_GATE.armed and (
            writes_args
            or "out" in kwargs
            or _call_requests_inplace(inplace_param_index, args, kwargs)
            or fused_call_writes(func_name, args, kwargs)
        ):
            _COW_GATE.materialize(arg_tensorlike)

        # ---- Execute the original function ----
        # Write a unique barcode BEFORE the call. If any inner wrapped functions
        # execute during this call, they will overwrite it. After the call,
        # matching barcode => this is the bottom-level (leaf) function.
        func_call_barcode = make_random_barcode()
        trace._wrapper_runtime_ws.current_func_barcode = func_call_barcode
        rng_states = log_op_rng_states(args, kwargs) if trace._options_save_rng else {}
        autocast_state = log_current_autocast_state()
        func_call_id = _state.next_func_call_id()
        register_call_input_container_snapshots(
            trace,
            args,
            kwargs,
            func_call_id=func_call_id,
            event_index=func_call_id,
        )
        from ...intervention._runtime import snapshot_call_inputs_for_inplace_intervention_site

        call_input_snapshots = snapshot_call_inputs_for_inplace_intervention_site(
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            trace=trace,
            func_call_id=func_call_id,
        )
        expected_token = None
        pauses_owned_modes = constructs_tensor_subclass and _pauses_for_subclass_conversion(
            func_name, args, _logged_subclass_cls_index
        )
        # per-op duration must measure the USER op, not TorchLens
        # bookkeeping. The clock starts here -- after RNG/autocast snapshots
        # and container/intervention-site registration -- and stops right
        # after the call returns, so ``func_duration`` no longer
        # systematically overstates cheap ops in instrumented captures.
        # Attempted-op marker: pushed BEFORE execution (one dict
        # read when echo is off); semantics on EchoSession.attempted_push.
        _echo = trace.__dict__.get("_echo_session")
        _echo_depth = _echo.attempted_push(func_name, args, kwargs) if _echo is not None else 0
        # The markers open exactly where the op clock opens, so a range never encloses
        # TorchLens's own pre-call bookkeeping (RNG/autocast snapshots,
        # container registration).
        op_marker_tokens = _push_op_markers(trace, func_name, func_call_id)
        func_exec_start = time.time()
        device_memory_before = None
        if getattr(trace, "track_device_memory", False):
            # Observe item 15; deferred: cycle _observe.
            from ..._observe._device_memory import read_before as _dm_read_before

            device_memory_before = _dm_read_before(trace)
        mode_pause = pause_own_dispatch_modes() if pauses_owned_modes else nullcontext(())
        paused_modes: tuple[Any, ...] = ()
        frozen_receiver_ctx = (
            _frozen_parameter_receiver_context(
                trace,
                args,
                kwargs,
                has_inplace_signature or _call_requests_inplace(inplace_param_index, args, kwargs),
            )
            if args and isinstance(args[0], torch.nn.Parameter)
            else nullcontext()
        )
        try:
            with mode_pause as paused_modes, frozen_receiver_ctx:
                logged_edge_ctx: Any = nullcontext()
                if _diagnostic_edge_armed():
                    logged_edge_ctx = expected_original_call(
                        func,
                        f"torch_func:{func_name}:logged",
                        func_name=func_name,
                        func_call_id=func_call_id,
                        call_barcode=func_call_barcode,
                    )
                with logged_edge_ctx as expected_token:
                    try:
                        out_orig = func(*args, **kwargs)
                    except TypeError as exc:
                        out_orig = redispatch_declined(exc, wrapped_func, func_name, args, kwargs)
                        if out_orig is STILL_DECLINED:
                            raise
        except torch.cuda.OutOfMemoryError:
            # Observe item 15: the OOM raised BEFORE this op committed -- name
            # the ATTEMPTED call, never the last committed op.
            if getattr(trace, "track_device_memory", False):  # deferred import: cycle _observe
                from ..._observe._device_memory import settle_op_bracket

                settle_op_bracket(trace, func_name, func_call_id, device_memory_before, oom=True)
            raise
        except RuntimeError as exc:
            # The LOGGED path's __new__ pause above is unconditional (every torch
            # version) and normally suffices; it is known NOT to on torch 2.1/2.2
            # when this forward is also observed by the private aten-profile
            # recorder (a reentrant torch limitation reproduced on stock torch
            # with a no-op mode -- popping TorchLens's OWN mode first did not
            # help there, confirmed by direct instrumentation). Translate torch's
            # own crash into the same typed, disclosed refusal the fast path
            # raises, rather than let a cryptic RuntimeError escape.
            _raise_if_subclass_ctor_under_dispatch_mode(exc, func_name, pauses_owned_modes)
            raise
        finally:
            # Markers close FIRST so the range covers the user op, not the
            # mode-paused interior bookkeeping recorded below.
            _pop_op_markers(op_marker_tokens)
            if paused_modes:
                from ._aten_capture import _record_mode_paused_interior

                _record_mode_paused_interior(trace, owner_func_call_id=func_call_id)
        func_exec_duration = time.time() - func_exec_start
        if device_memory_before is not None or getattr(trace, "track_device_memory", False):
            from ..._observe._device_memory import settle_op_bracket  # deferred: cycle _observe

            settle_op_bracket(trace, func_name, func_call_id, device_memory_before)
        if _echo is not None:
            _echo.attempted_pop(_echo_depth)
        if mutates_data_alias:
            record_data_alias_mutation(trace)
        return_value = out_orig
        exec_ctx = FuncExecutionContext(
            time_elapsed=func_exec_duration,
            rng_states=rng_states,
            autocast_state=autocast_state,
        )
        is_bottom_level_func = trace._wrapper_runtime_ws.current_func_barcode == func_call_barcode
        if isinstance(out_orig, torch.nn.Parameter):
            # ``Parameter(data)`` minted in the pass is unlogged; it adopts data's label.
            _adopt_minted_parameter_label(trace, out_orig, args)

        # A receiver-mutating call that returns ``None`` (or always, for __setitem__,
        # zero_, __delitem__) logs its mutated receiver(s) as the output.
        out_orig = logged_receiver_output(
            out_orig, args, func_name, reconstructs_receiver_output, is_mutating_property_setter
        )

        # ---- In-place detection and safe copy ----
        same_object_returned = len(args) > 0 and id(out_orig) == id(args[0])
        # True in-place ops (add_, mul_, etc.) modify the tensor and return self.
        # No-op functions (to(same_dtype), contiguous() on contiguous tensor)
        # also return self but don't modify anything.
        # Both cases need safe_copy so logging doesn't overwrite the original's
        # label, but only true in-place ops should propagate the new label back.
        # A functional called with ``inplace=True`` (``F.relu6(x, inplace=True)``,
        # ``nn.Hardtanh(inplace=True)``) mutates its receiver through an unwrapped
        # private builtin under a plain name, so the name signature alone misses it.
        mutation_signature = has_inplace_signature or (
            same_object_returned and _call_requests_inplace(inplace_param_index, args, kwargs)
        )
        was_inplace = same_object_returned and mutation_signature
        # The live intervention site's in-place verdict. A replacement at a
        # MUTATING site (``x.add_(1)``, ``torch.add(a, b, out=c)``) only takes
        # effect when copied into the mutated receiver, so the hook runner
        # needs the verdict; a NON-mutating same-object return (``x.to(device)``
        # already there, ``x.contiguous()`` on a contiguous tensor) hands its
        # replacement to the caller like any functional op. Label presence is
        # not mutation: keying on it refused every replacement at an identity
        # return and recorded ``replaced=False`` on a rule that had fired.
        record_is_inplace = was_inplace or isinstance(kwargs.get("out"), torch.Tensor)
        # An in-place write to a prepared Parameter is a mutated-receiver site
        # too: a live intervention replacing its output must be copied into the
        # Parameter (``_apply_inplace_replacement_to_mutated_storage``) so later
        # reads see it, as for any labeled in-place receiver.
        mutates_parameter_receiver = (
            was_inplace
            and not is_storage_rebinding_setter
            and _is_mutated_parameter_receiver(trace, args[0])
        )
        # The internal identity-forcing decorator (_state._decorated_identity)
        # exists precisely to MINT a distinct logged tensor at module boundaries
        # (nn.Identity / pass-through outputs). Unlike user-visible no-ops such as
        # x.contiguous(), it must NOT preserve the input's Python object identity,
        # otherwise the module exit re-reads the input's label and the boundary
        # node (e.g. identity_1_2) never attaches to the module's output_ops.
        if same_object_returned:
            out_orig = _same_object_logging_copy(out_orig, trace, was_inplace)
            # A nested bottom-level write already recorded this Parameter
            # mutation; the enclosing initializer keeps its Parameter return
            # and logs nothing (``nested_write_relabeled_receiver``).
            if is_bottom_level_func or not (
                was_inplace and _nested_write_relabeled_receiver(args[0], receiver_label_before)
            ):
                out_orig = _parameter_mutation_output_for_logging(
                    trace,
                    out_orig,
                    source=args[0],
                    was_inplace=was_inplace,
                    is_storage_rebind=is_storage_rebinding_setter,
                )
            # A storage rebind is recorded as the non-mutating ``detach(rhs)`` call.
            if not is_storage_rebinding_setter:
                _stamp_same_object(out_orig, args[0], mutation_signature, force_distinct_return)

        if canonical_capture_callable is None:
            capture_func, capture_func_name = _canonical_capture_callable(
                func, func_name, property_accessor
            )
        else:
            capture_func, capture_func_name = canonical_capture_callable
        host_entry = (
            resolve_host_entry(trace, func_name, (args, kwargs), out_orig, _set_transform_metadata)
            if is_host_constructor and not arg_tensorlike
            else None
        )
        if host_entry is not None:
            capture_func, capture_func_name = host_entry.replay, FOREIGN_DATA_FUNC_NAME
        logged = logged_args(func_name, args, arg_copies, is_storage_rebinding_setter, host_entry)
        if logged is None:
            log_args, log_kwargs = args, kwargs
            log_arg_copies, log_kwarg_copies = arg_copies, kwarg_copies
        else:
            log_args, log_arg_copies = logged
            log_kwargs, log_kwarg_copies = {}, {}
        out_before_hooks = out_orig
        out_orig = apply_live_hooks_to_outputs(
            trace,
            capture_func,
            capture_func_name,
            log_args,
            log_kwargs,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
            func_call_id,
            call_input_snapshots,
            record_is_inplace,
        )

        # Log all output tensors (excluding Parameters, which are source tensors).
        # Fast inline check for the common single-tensor output case.
        if getattr(trace, "is_replayable", False):
            output_tensors = [entry[0] for entry in _walk_output_tensors_with_paths(out_orig)]
        else:
            output_tensors = _collect_output_tensors(out_orig)

        call_emitted_op = False
        if len(output_tensors) > 0:
            # Deferred payload clones (clone-on-write): for eligible plain
            # captures, payload ``safe_copy`` calls issued while this window is
            # armed return registered aliases instead of eager clones. The
            # window is scoped strictly to the payload-saving call so buffer
            # snapshots, arg copies, and every other capture-time copy keep
            # their historical eager semantics.
            cow_state_ptrs = _cow_payload_state_ptrs(trace)
            if cow_state_ptrs is not None:
                arm_deferred_payload_window(cow_state_ptrs)
            try:
                # Hide TorchLens bookkeeping dispatches only from the opt-in user-op census.
                if _state._completeness_witness_mode == "shadow":
                    with _state.pause_logging():
                        call_emitted_op = log_function_output_tensors(
                            trace,
                            capture_func,
                            capture_func_name,
                            log_args,
                            log_kwargs,
                            log_arg_copies,
                            log_kwarg_copies,
                            out_orig,
                            exec_ctx,
                            is_bottom_level_func,
                            func_call_id,
                        )
                else:
                    call_emitted_op = log_function_output_tensors(
                        trace,
                        capture_func,
                        capture_func_name,
                        log_args,
                        log_kwargs,
                        log_arg_copies,
                        log_kwarg_copies,
                        out_orig,
                        exec_ctx,
                        is_bottom_level_func,
                        func_call_id,
                    )
            finally:
                if cow_state_ptrs is not None:
                    disarm_deferred_payload_window()

            _propagate_data_alias_provenance(
                func_name,
                data_alias_inputs,
                output_tensors,
            )

            # Same-object returns are logged against out_orig (a safe_copy with
            # the op's new label). When Python object identity is preserved we
            # actually return the LIVE tensor (return_value / args[0]), which
            # still carries its OLD label and OLD autograd grad_fn -- so without
            # repair the live graph bypasses this op entirely:
            #  * label: downstream ops would see the input's label, so the op
            #    (e.g. an eval-mode Dropout no-op, or in-place add_) drops out of
            #    the graph and the model output traces straight back to the input
            #    -- which then spuriously trips the module-boundary identity-node
            #    synthesis, inflating the node count.
            #  * grad: the op's grad hook sits on the dead safe_copy, so any
            #    module whose output descends from it loses grad attribution.
            # Propagate the op's label onto the live tensor(s) and hook them so
            # both the forward graph and backward grads stay attached. Only do
            # this when we will actually hand back the live tensor (the default
            # for same-object returns that no hook replaced); when out_orig is
            # returned instead the safe_copy already carries everything.
            propagate_to_live = (
                same_object_returned and out_orig is out_before_hooks and not force_distinct_return
            )
            if propagate_to_live and not isinstance(args[0], torch.nn.Parameter):
                out_label = get_tensor_label(out_orig)
                if out_label is not None:
                    if was_inplace:
                        set_tensor_label(args[0], out_label)
                        _register_inplace_live_grad_hook(trace, args[0], out_label)
                        # A storage-SWAPPING ``.data=`` rebind threads its
                        # consumers to the RHS producer (correct dataflow),
                        # but verdict-steering attribution must never root
                        # THROUGH the swap. Register the op label as an
                        # ancestry barrier so layout/witness rooting fails
                        # closed, as for an unattributed break.
                        if is_storage_rebinding_setter and rebind_setter_swaps_storage:
                            from ._completeness_metadata import record_storage_rebind_barrier

                            record_storage_rebind_barrier(trace, out_label)
                        # the write may have gone through a VIEW; every
                        # other live labeled alias whose storage bytes overlap
                        # the target (its base, an overlapping sibling view)
                        # saw its content change too, so consumers of THOSE
                        # objects must also bind to this mutation op. A
                        # storage-REBINDING setter (``t.data = rhs``) wrote no
                        # bytes: after the rebind the receiver shares RHS's
                        # storage, and advancing RHS-side aliases would invent
                        # mutation edges for values that never changed.
                        if not is_storage_rebinding_setter:
                            _propagate_mutation_label_to_storage_aliases(trace, args[0], out_label)
                    if isinstance(return_value, torch.Tensor):
                        set_tensor_label(return_value, out_label)
                        _register_inplace_live_grad_hook(trace, return_value, out_label)
            elif mutates_parameter_receiver and not force_distinct_return:
                # Unhooked, or a live intervention whose replacement was copied
                # into the Parameter: either way the Parameter now holds the
                # logged op's output, so later reads bind to that op.
                _label_mutated_prepared_parameter(trace, args[0], get_tensor_label(out_orig))

            # the module-boundary identity mint (force_distinct_return)
            # logs against a distinct safe copy so the boundary op attaches to
            # the module's output_ops -- but the CALLER keeps the original
            # object. Without advancing the live original's label, every
            # downstream consumer bound to the pre-module label and the
            # boundary node dangled (nn.Identity / pass-through modules).
            # Advance the live object exactly like same-object propagation;
            # the minted copy still carries the label for module bookkeeping.
            if (
                force_distinct_return
                and out_orig is out_before_hooks
                and len(args) > 0
                and isinstance(args[0], torch.Tensor)
                and not isinstance(args[0], torch.nn.Parameter)
            ):
                boundary_label = get_tensor_label(out_orig)
                if boundary_label is not None:
                    live_label_before_mint = get_tensor_label(args[0])
                    set_tensor_label(args[0], boundary_label)
                    _register_inplace_live_grad_hook(trace, args[0], boundary_label)
                    _record_label_version_snapshot(args[0])
                    # The mint is value-preserving by construction (an internal
                    # no-op identity): tell the container registry so snapshot
                    # dedup of an unchanged threaded container survives the
                    # label advance (a mutation's advance is never reported).
                    if live_label_before_mint is not None and getattr(
                        trace, "_capture_container_structure", False
                    ):
                        trace._wrapper_runtime_ws.container_registry.note_value_preserving_relabel(
                            live_label_before_mint, boundary_label
                        )

            # ``out=`` family: an ``out=`` destination may itself be a view
            # of a larger live tensor (``torch.add(x, 1, out=y[0])``); the
            # destination object's label advances at logging time, but its
            # overlapping aliases need the same mutation-provenance advance.
            out_kwarg_destinations = kwargs.get("out")
            if isinstance(out_kwarg_destinations, torch.Tensor):
                out_destination_tensors: tuple[torch.Tensor, ...] = (out_kwarg_destinations,)
            elif isinstance(out_kwarg_destinations, (list, tuple)):
                out_destination_tensors = tuple(
                    item for item in out_kwarg_destinations if isinstance(item, torch.Tensor)
                )
            else:
                out_destination_tensors = ()
            for out_destination in out_destination_tensors:
                destination_label = get_tensor_label(out_destination)
                if destination_label is not None:
                    _propagate_mutation_label_to_storage_aliases(
                        trace, out_destination, destination_label
                    )

        mark_expected_original_accounted(expected_token, captured=call_emitted_op)
        if (
            expected_token is not None
            and not call_emitted_op
            and _state._completeness_witness_mode == "shadow"
        ):
            record_uncaptured_owner_callsite(expected_token)

        producer_label = None
        if isinstance(out_orig, torch.Tensor):
            producer_label = get_tensor_label(out_orig)
        elif output_tensors:
            producer_label = get_tensor_label(output_tensors[0])
        record_buffer_writes_here = is_bottom_level_func
        if not record_buffer_writes_here and buffer_snapshots:
            # On an ADMITTED meta capture the
            # fused mutator's Python decomposition strips the USER call of its
            # bottom-level status (the inner allocator calls claim it), so the
            # declared buffer writes would silently vanish. The snapshot-holding
            # call is the ONE that resolved the buffer args, so it records its
            # own writes; the decomposition's inner calls never carry
            # snapshots (their names are not mutator-classified).
            from ..._capture._weightsfree_admission import weightsfree_meta_active

            record_buffer_writes_here = weightsfree_meta_active(trace)
        if record_buffer_writes_here:
            record_op_buffer_writes(
                trace,
                capture_func_name,
                buffer_snapshots,
                producer_label,
            )

        if is_barcode_transparent and enclosing_barcode:
            trace._wrapper_runtime_ws.current_func_barcode = enclosing_barcode

        if out_orig is not out_before_hooks and not mutates_parameter_receiver:
            return out_orig
        if force_distinct_return:
            return out_orig
        return return_value

    # ---- __wrapped__ removal for JIT compatibility ----
    # @wraps sets __wrapped__ on the wrapper. For C builtins (no __code__),
    # inspect.unwrap() follows __wrapped__ and fails because builtins have
    # no inspectable source. torch.jit.script (e.g. via timm) calls
    # inspect.unwrap internally, so we must remove __wrapped__ to prevent
    # the failure chain: jit.script -> inspect.unwrap -> inspect.getsource -> crash.
    if not hasattr(func, "__code__"):
        try:
            del wrapped_func.__wrapped__
        except AttributeError:
            pass

    setattr(wrapped_func, "__tl_original_id__", id(func))
    setattr(wrapped_func, "__tl_wrapper_name__", f"torch_func:{func_name}")
    setattr(wrapped_func, "__tl_detector_excluded__", is_unlogged_func)

    # ---- __prepare_scriptable__ for JIT compatibility ----
    # torch.jit.script and torch.jit._recursive.try_compile_fn both honor this
    # hook BEFORE building the resolution callback. Without it, jit pulled the
    # ORIGINAL functional's source (inspect.unwrap follows __wrapped__) but
    # resolved its globals against THIS module -- so any wrapped pure-Python
    # functional whose source needs names beyond the torch.overrides
    # boilerplate imported above (``F.interpolate`` -> ``undefined value
    # math``) failed to script, process-wide, once wrappers installed.
    # Returning the original hands jit a self-consistent (source, globals)
    # pair; scripted artifacts run raw torch by contract (never logged).
    setattr(wrapped_func, "__prepare_scriptable__", lambda: func)

    return wrapped_func


def _is_jit_incompatible_dtype_annotation(annotation: Any) -> bool:
    """Return whether an annotation is the JIT-incompatible ``DType`` marker.

    Parameters
    ----------
    annotation:
        Annotation object or string copied onto a decorated wrapper.

    Returns
    -------
    bool
        Whether the annotation names a ``DType`` type that TorchScript cannot parse.
    """

    if annotation == "DType":
        return True
    if getattr(annotation, "__name__", None) == "DType":
        return True
    return "DType" in repr(annotation)


def _sanitize_jit_wrapper_annotations(func: Callable[..., Any]) -> None:
    """Replace wrapper annotations that TorchScript cannot parse.

    Parameters
    ----------
    func:
        Decorated function being registered as a JIT builtin.
    """

    annotations = getattr(func, "__annotations__", None)
    if not isinstance(annotations, dict):
        return

    sanitized_annotations = dict(annotations)
    changed = False
    for key, annotation in sanitized_annotations.items():
        if _is_jit_incompatible_dtype_annotation(annotation):
            sanitized_annotations[key] = int
            changed = True
    if changed:
        func.__annotations__ = sanitized_annotations


def _register_jit_builtin_wrappers() -> None:
    """Register decorated torch wrappers in TorchScript's builtin table."""

    builtin_table = get_jit_builtin_table()
    if builtin_table is None:
        return
    for orig_id, decorated_func in _state._orig_to_decorated.items():
        builtin_name = builtin_table.get(orig_id)
        if builtin_name is not None:
            if callable(decorated_func):
                _sanitize_jit_wrapper_annotations(decorated_func)
            builtin_table[id(decorated_func)] = builtin_name
            # For properties, also register getter/setter/deleter individually
            # since JIT may call them directly.
            if isinstance(decorated_func, property):
                for accessor in (decorated_func.fget, decorated_func.fset, decorated_func.fdel):
                    if accessor is not None:
                        _sanitize_jit_wrapper_annotations(accessor)
                        builtin_table[id(accessor)] = builtin_name


def _register_jit_boolean_dispatch_wrappers() -> None:
    """Register wrappers of boolean-dispatched functionals in torch's table.

    ``torch._jit_internal.boolean_dispatched`` is a WeakKeyDictionary keyed by
    the ORIGINAL function objects (the whole ``F.max_pool*`` family plus
    ``fractional_max_pool*`` / ``adaptive_max_pool*``). TorchScript's
    sugared-value layer consults it BY OBJECT before source compilation, so
    once decoration replaced the namespace slot with a wrapper,
    ``torch.jit.script`` on any max-pool-using module hard-failed
    (``NotSupportedError`` on the wrapper's varargs). Registering each wrapper
    as an additional key sharing the original's dispatch record keeps jit
    compiling the SAME if_true/if_false originals under either alias. Entries
    persist like the builtin-table wrapper ids (wrappers live in the
    append-only ledger), so stale post-unwrap wrapper references still script.
    """

    table = get_jit_boolean_dispatch_table()
    if table is None:
        return
    for orig, record in list(table.items()):
        wrapper = _state._orig_to_decorated.get(id(orig))
        if wrapper is None or not callable(wrapper) or isinstance(wrapper, property):
            continue
        try:
            if table.get(wrapper) is None:
                table[wrapper] = record
        except TypeError:
            # Non-weakref-able wrapper object: leave the original-only entry.
            continue


# ---------------------------------------------------------------------------
# One-time decoration at import time
# ---------------------------------------------------------------------------


def decorate_all_once() -> None:
    """Decorate all torch functions (internal, called once by ``wrap_torch``).

    Iterates over every ``(namespace, func_name)`` pair in ``ORIG_TORCH_FUNCS``
    and replaces each function with a ``torch_func_decorator`` wrapper. Also:

    - Pre-computes ``_state._arg_names`` for metadata capture.
    - Populates ``_state._orig_to_decorated`` / ``_state._decorated_to_orig``
      bidirectional mappings (keyed by ``id()``).
    - Registers wrappers in ``torch.jit._builtins._builtin_table`` so JIT
      compilation recognizes wrapped functions as known ATen ops.
    - Collects ``_DEVICE_CONSTRUCTOR_NAMES`` for DeviceContext bypass.
    - Creates ``_state._decorated_identity`` (a no-op that forces new log
      entries at module boundaries).

    **Shared-original deduplication**: Multiple torch namespaces can alias the
    same C builtin (e.g. ``torch.cos`` and ``torch._VF.cos``). When the same
    ``id(orig_func)`` is encountered again, we reuse the existing wrapper
    rather than creating a second one. This ensures the JIT builtin table and
    ``_orig_to_decorated`` stay consistent (one original -> one wrapper).

    Idempotent: returns immediately if already decorated.
    """
    if _state._is_decorated:
        return  # already fully decorated
    # NOTE: Do NOT guard on `_orig_to_decorated` being non-empty: a prior partial
    # failure may have populated it without completing decoration; guarding on
    # _is_decorated (set at the end) lets that retry run (#138).

    _warm_derived_identity_caches()

    _decorate_torch_func_pairs(get_orig_torch_funcs())

    # ---- JIT builtin table registration ----
    # torch.jit._builtins._builtin_table maps id(func) -> ATen op name.
    # We must register our wrappers so JIT recognizes them as the same ops.
    # Without this, torch.jit.script fails on any code using wrapped functions.
    _register_jit_builtin_wrappers()
    _register_jit_boolean_dispatch_wrappers()

    # ---- DeviceContext bypass ----
    # ``_DEVICE_CONSTRUCTOR_NAMES`` was collected from the PRE-wrap warm above,
    # and torch's ``_device_constructors()`` lru-cache deliberately stays keyed
    # by ORIGINALS. ``DeviceContext.__torch_function__`` always receives
    # the original C function, so re-materializing the cache with wrappers here
    # (the historical behavior) made the C-level membership test miss and
    # silently skipped device injection for stale pre-wrap factory references
    # (``from torch import zeros`` before the first capture, then
    # ``with torch.device('meta')``: tensors landed on CPU). Wrapped calls
    # never need the C-level path -- ``_maybe_inject_device_kwarg`` replicates
    # the injection inside the wrapper.

    # Create the decorated identity — a no-op that forces a new log entry at
    # module boundaries (nn.Identity, pass-through outputs).  Stored on _state
    # instead of monkey-patching torch.identity (which doesn't exist in PyTorch
    # type stubs and causes mypy errors).
    _state._decorated_identity = torch_func_decorator(identity, "identity")
    _decorate_transform_builders()
    _decorate_direct_transforms()
    global _FULL_DECORATION_COMPLETED
    _FULL_DECORATION_COMPLETED = True
    _state._is_decorated = True

    # Wrapping __getitem__ on torch.Tensor pollutes the C-level sq_item slot,
    # making PySequence_Check(tensor) return True.  Clear it so torch.tensor()
    # doesn't try to iterate 0-d tensor elements as sequences.
    _fix_tensor_sequence_slot()


_AT_FORK_HYGIENE_INSTALLED = False
"""True once the fork-child capture-state clear is registered (per process)."""


def _install_fork_capture_hygiene() -> None:
    """Clear inherited capture state in ``os.fork()`` children, once per process.

    a fork DURING a traced forward (e.g. a fork-start
    ``DataLoader`` constructed inside ``forward``) inherits
    ``_logging_enabled=True`` and ``_active_trace``, and -- because the
    forking thread's ident is preserved as the child's main thread -- passes
    the owner-thread gate, so child-side torch ops silently log into the
    child's inherited trace copy (child-local corruption plus per-op
    overhead; the parent is unaffected through COW). The child's inherited
    mid-capture state can never be a capture the CHILD owns, so it is cleared
    unconditionally. NEW captures in the child stay governed by the existing
    guards (``warn_parallel``'s import-PID stamp; the sanctioned
    distributed-rank path starts its own captures and is untouched -- ranks
    fork/spawn BEFORE capturing, so they inherit no mid-capture state).
    """

    global _AT_FORK_HYGIENE_INSTALLED
    if _AT_FORK_HYGIENE_INSTALLED or not hasattr(os, "register_at_fork"):
        return

    def _clear_inherited_capture_state() -> None:
        """Reset capture globals inherited across ``fork`` in the child process."""
        _state._logging_enabled = False
        _state._active_trace = None

    os.register_at_fork(after_in_child=_clear_inherited_capture_state)
    _AT_FORK_HYGIENE_INSTALLED = True


#: ``torch._dynamo.trace_rules`` lru-caches whose keys are LIVE torch callables.
#: The string-keyed siblings (``dynamo_dir``, ``get_mod_inlinelist``, ...) are
#: identity-safe and deliberately excluded.
_DYNAMO_IDENTITY_RULE_CACHE_NAMES = ("get_torch_obj_rule_map", "get_tensor_method")


@originals_exposed(_setattr_ignoring_advisories, TRANSFORM_BUILDER_SITES + DIRECT_TRANSFORM_SITES)
def _warm_derived_identity_caches() -> None:
    """Warm torch's identity-keyed DERIVED caches while the namespace holds originals.

    Invariant: no torchlens path may FIRST-call a cached torch introspection table while
    wrappers are installed. Both install paths call this before their setattr loop;
    ``originals_exposed`` also shows originals on slots a failed decoration pass left
    wrapped (the #138 retry). Covered:

    - the two ``torch.overrides`` tables;
    - torch's ``_device_constructors()`` set -- ``DeviceContext.__torch_function__`` always
      receives the ORIGINAL C function, so a wrapper-keyed set silently skips device injection
      for stale pre-wrap factory references; the warm also collects
      ``_DEVICE_CONSTRUCTOR_NAMES`` for the wrapper-side injection path;
    - dynamo's identity-keyed rule tables, when dynamo is already imported (never
      force-imported -- tables a user materializes mid-epoch are dropped at ``unwrap_torch()``
      instead); Dynamo then inlines the wrappers, so their payload copy-out runs compile-disabled.
    """

    from torch.overrides import get_overridable_functions, get_testing_overrides

    get_overridable_functions()
    get_testing_overrides()

    device_constructors = get_device_constructors()
    if device_constructors is not None:
        try:
            device_constructors.cache_clear()
            for ctor in device_constructors():
                name = getattr(ctor, "__name__", None)
                if name:
                    _DEVICE_CONSTRUCTOR_NAMES.add(name)
        except (AttributeError, TypeError):
            mark_torch_capability_missing(
                "HAS_DEVICE_CONSTRUCTORS",
                "factory-function device injection inventory could not be evaluated",
            )

    for rule_cache in _dynamo_identity_rule_caches():
        try:
            rule_cache()
        except Exception:  # pragma: no cover - dynamo-internal failure
            from ..._errors import TorchLensWarning

            warnings.warn(
                "torchlens could not pre-warm a torch._dynamo.trace_rules "
                "cache before wrapping; torch.compile identity rules may "
                "re-derive against torchlens wrappers until unwrap_torch().",
                TorchLensWarning,
                stacklevel=2,
            )
    if _dynamo_identity_rule_caches():
        _COW_GATE.materialize = compile_disabled(materialize_deferred_for_call)


def _dynamo_identity_rule_caches() -> list[Any]:
    """Return dynamo's identity-keyed rule caches, without importing dynamo.

    ``torch._dynamo.trace_rules`` memoizes rule tables keyed by the live callable objects
    resolved from the torch namespace at materialization time. Like the ``torch.overrides``
    tables these are DERIVED caches: the attribute-identity restore census cannot see them, so
    a table built while torchlens wrappers are installed silently poisons ``torch.compile`` for
    the rest of the process (``lookup(torch.cos)`` degrades to ``SkipFunctionVariable`` after
    ``unwrap_torch()``). Feature-detected via ``sys.modules`` + ``getattr`` -- never
    force-imports dynamo, and degrades to an empty list on any future torch that renames the
    getters.
    """

    trace_rules = sys.modules.get("torch._dynamo.trace_rules")
    if trace_rules is None:
        return []
    caches: list[Any] = []
    for cache_name in _DYNAMO_IDENTITY_RULE_CACHE_NAMES:
        cache = getattr(trace_rules, cache_name, None)
        if cache is not None and hasattr(cache, "cache_clear"):
            caches.append(cache)
    return caches


def _stamp_wrapper_provenance(
    wrapper: Callable[..., Any], namespace_name: str, func_name: str
) -> None:
    """Stamp install-site ``__module__``/``__qualname__`` onto a wrapper.

    ``@wraps`` copies the ORIGINAL's metadata, which for torch's C descriptors
    names classes that are not importable attributes (``pickle.dumps(torch.cos)``
    died on ``_VariableFunctionsClass.cos``). The install-site stamp
    makes a bare wrapper pickle by reference to its public torch name while
    wrappers are installed (loading as the ORIGINAL in a fresh process), and
    introspection reports the namespace the user actually reached the callable
    through. Shared originals keep their FIRST (public-namespace-first) stamp
    via the dedup branch below. The ``inspect.signature`` fabrication for C
    builtins stays a documented residual: ``__wrapped__`` must remain deleted
    for JIT compatibility, and functions cannot raise from attribute access.

    MODULE namespaces only — CLASS-namespace wrappers (tensor methods) are
    deliberately NOT stamped. C-level tensor methods carry no ``__module__``,
    so their wrappers keep the honest ``torchlens.backends.torch.wrappers``
    module. Stamping them ``"torch"`` would make a wrapped storage-unsafe
    method (``Tensor.resize_``/``set_``/``apply_``/``map_``) CLAIM torch
    purity to every string-based safety gate — the exact spoof surface the
    tensor-method smuggling defense pins as denied on REAL identity with the
    wrappers module visible. The
    security disclosure wins over introspection fidelity there; a bare wrapped
    tensor method staying unpicklable-by-reference is the accepted residual.
    """

    namespace_obj = get_optional_torch_namespace(namespace_name)
    if isinstance(namespace_obj, type):
        return
    try:
        wrapper.__module__ = namespace_name
        wrapper.__qualname__ = func_name
    except (AttributeError, TypeError):
        pass


# Callable types ``_install_function_wrapper`` wraps; the others (getset
# descriptors) take the property path in ``_decorate_torch_func_pairs``. These
# MUST come from ``types`` constants, not live torch attributes (e.g.
# ``type(torch.mean)``): after decoration the torch attributes are wrappers, and
# probing them would misclassify every pristine C callable.
_WRAPPABLE_FUNCTION_TYPES = (
    types.FunctionType,  # <class 'function'>
    types.BuiltinFunctionType,  # <class 'builtin_function_or_method'>
    types.MethodDescriptorType,  # <class 'method_descriptor'>
    types.WrapperDescriptorType,  # <class 'wrapper_descriptor'>
)


def _install_function_wrapper(
    local_func_namespace: Any, namespace_name: str, func_name: str, orig_func: Any
) -> None:
    """Wrap ``orig_func`` into ``namespace.func_name`` and record it in the ledgers.

    Shared by first install, late torchvision ops and re-install adoption of a
    callable that drifted while torch was unwrapped (a peer that patched the
    slot in between): first install and re-install then agree on what gets
    wrapped. A shared original (``torch.cos`` and ``torch._VF.cos`` are one C
    builtin) reuses its existing wrapper; a second wrapper would break the 1:1
    mapping in ``_orig_to_decorated`` and the JIT builtin table.
    """

    if id(orig_func) in _state._orig_to_decorated:
        existing = _state._orig_to_decorated[id(orig_func)]
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, existing)
        except (AttributeError, TypeError):
            pass
        return

    recorded_name = _recorded_func_name(namespace_name, func_name)
    new_func = torch_func_decorator(orig_func, recorded_name)
    _stamp_wrapper_provenance(new_func, namespace_name, func_name)
    try:
        _setattr_ignoring_advisories(local_func_namespace, func_name, new_func)
        mark_decorated_function(new_func)
        # Bidirectional id-keyed mappings for fast lookup.
        _state._orig_to_decorated[id(orig_func)] = new_func
        _state._decorated_to_orig[id(new_func)] = orig_func
        # Object-keyed mappings for cases where we have the object, not its id.
        _state._decorated_func_mapper[new_func] = orig_func
        _state._decorated_func_mapper[orig_func] = new_func
    except (AttributeError, TypeError):
        pass


def _decorate_torch_func_pairs(func_pairs: list[tuple[str, str]]) -> None:
    """Collect argument names, then decorate one batch of torch func targets.

    Shared by ``decorate_all_once`` (the full inventory) and
    ``_ensure_torchvision_ops_decorated`` (torchvision custom ops imported
    after the first wrap). Idempotent per pair: already-registered arg names
    and already-decorated functions are skipped.

    Parameters
    ----------
    func_pairs:
        ``(namespace, func_name)`` targets to decorate.
    """
    # From ``types``, never a live torch attribute (see _WRAPPABLE_FUNCTION_TYPES).
    getset_class = types.GetSetDescriptorType  # <class 'getset_descriptor'> (properties)

    # --- Pass 1: Collect argument names before any decoration ---
    # inspect.signature() must run against the pristine torch namespace.
    # Python 3.14+ (PEP 649) evaluates annotations lazily; if we decorate
    # Tensor.bool first, then inspect Tensor.dim_order, the annotation
    # bool | list[...] resolves bool to our wrapper -> TypeError (#138).
    for namespace_name, func_name in func_pairs:
        # Exact-name dedup: ``add``, ``add_``, and ``__add__`` have
        # meaningfully different signatures, so each registers its own entry.
        # The historical stripped-key dedup made whichever spelling appeared
        # first swallow all the others' registrations.
        if func_name in _state._arg_names:
            continue
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        get_arg_names(orig_func, func_name)

    # --- Pass 2: Decorate all functions ---
    for namespace_name, func_name in func_pairs:
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)

        # Guard against double-decoration (ORIG_TORCH_FUNCS may list duplicates).
        if is_decorated_function(orig_func):
            continue

        if type(orig_func) in _WRAPPABLE_FUNCTION_TYPES:
            _install_function_wrapper(local_func_namespace, namespace_name, func_name, orig_func)

        elif type(orig_func) is getset_class:
            # getset_descriptors (e.g. Tensor.real, Tensor.imag) are C-level
            # properties. We wrap getter/setter/deleter individually and
            # reassemble as a Python property.
            orig_descriptor = cast(Any, orig_func)
            getter_orig, setter_orig, deleter_orig = (
                orig_descriptor.__get__,
                orig_descriptor.__set__,
                orig_descriptor.__delete__,
            )
            getter_dec = torch_func_decorator(getter_orig, func_name, property_accessor="get")
            setter_dec = torch_func_decorator(setter_orig, func_name, property_accessor="set")
            deleter_dec = torch_func_decorator(deleter_orig, func_name, property_accessor="del")
            mark_decorated_function(getter_dec)
            mark_decorated_function(setter_dec)
            mark_decorated_function(deleter_dec)
            new_property = DescriptorCompatProperty(
                getter_dec, setter_dec, deleter_dec, doc=func_name
            )
            new_property.__objclass__ = getattr(
                orig_descriptor, "__objclass__", local_func_namespace
            )
            new_property.__name__ = func_name
            try:
                _setattr_ignoring_advisories(local_func_namespace, func_name, new_property)
                # #31: Only add mapper entries if setattr succeeded — otherwise
                # we'd have dangling entries pointing to an uninstalled property.
                cast(dict[int, Any], _state._orig_to_decorated)[id(orig_func)] = new_property
                cast(dict[int, Any], _state._decorated_to_orig)[id(new_property)] = orig_func
                cast(dict[Any, Any], _state._decorated_func_mapper)[new_property] = orig_func
                cast(dict[Any, Any], _state._decorated_func_mapper)[orig_func] = new_property
            except (AttributeError, TypeError):
                pass


_torchvision_ops_ensured = False
"""Whether torchvision custom ops are confirmed decorated (or confirmed no-op)."""


_wrapper_install_lock = threading.RLock()
"""Serializes wrapper INSTALL / UNINSTALL so the torch namespaces never interleave.

``wrap_torch`` and ``unwrap_torch`` are check-then-mutate sequences over
``_state._is_decorated``, the ``_orig_to_decorated`` / ``_decorated_to_orig`` id
maps, and hundreds of torch namespace attributes. Unsynchronized, two threads
reaching their FIRST capture together can double-populate the id maps, double
register the JIT builtins, or -- worst -- store thread A's WRAPPER as
``_ORIGINAL_AUTOGRAD_BACKWARD``, which permanently leaks a wrapper into torch on
the next uninstall. Both entry points run wholly under this lock.

Reentrant (``RLock``) because the install path legitimately re-enters itself:
``unwrap_torch`` -> ``uninstall_autograd_wrappers`` and
``wrap_torch`` -> ``_ensure_torchvision_ops_decorated`` -> ``_register_jit_builtin_wrappers``
sit under the same top-level call, and ``TorchBackend.wrap``/``unwrap`` may be
reached from a caller that already holds it. Install is once-per-process (and
per explicit re-wrap), so the lock is never on the capture hot path.
"""


def _ensure_torchvision_ops_decorated() -> None:
    """Decorate torchvision custom ops when torchvision appears after first wrap.

    TorchLens never imports torchvision itself (the eager probe cost ~1.9 s
    and ~150 MB RSS on the first capture of ANY model). ``decorate_all_once``
    therefore only covers ``torch.ops.torchvision.*`` targets when torchvision
    was already imported by the user at first-wrap time. A user may import
    torchvision *after* the first capture; this per-wrap re-check decorates
    those ops before the next capture begins, so torchvision models wrap
    exactly as they did under the eager probe. A model cannot call a
    torchvision op without torchvision imported (the ops only register with
    the dispatcher during ``import torchvision``), so checking at wrap time is
    exact, not heuristic. Once confirmed, this reduces to one flag check.
    """
    global _torchvision_ops_ensured
    if _torchvision_ops_ensured or not _state._is_decorated:
        return
    torchvision_pairs = _get_torchvision_funcs()
    if not torchvision_pairs:
        return  # torchvision absent or mid-import; re-check on the next wrap
    _decorate_torch_func_pairs(torchvision_pairs)
    # Idempotent re-registration keeps parity with the imported-before-wrap
    # path, where these pairs were present during the one-time registration.
    _register_jit_builtin_wrappers()
    _torchvision_ops_ensured = True


def unwrap_torch() -> None:
    """Remove torchlens wrappers and restore original torch callables.

    After calling this, ``torch.cos``, ``torch.Tensor.__add__``, etc. are the
    originals shipped by PyTorch.  TorchLens logging will not work until
    ``wrap_torch()`` is called (or ``trace`` auto-wraps).

    Safe to call multiple times — no-op if already unwrapped.

    Raises
    ------
    CaptureNotActiveError
        If a capture is currently active. Removing the wrappers mid-forward
        leaves the rest of that forward unlogged and returns a silently
        truncated Trace, so the call is refused instead (code
        ``unwrap_during_active_capture``). This is reachable single-threaded —
        from a forward hook, an ``activation_transform``, or any user callback
        that runs inside the traced forward.
    """
    with _wrapper_install_lock:
        # R54: the refusal reads _active_trace/_logging_enabled, which are
        # PUBLISHED under _capture_admission_lock (a different lock domain), so
        # a capture admitted between the refusal check and the uninstall was
        # silently truncated (reproduced with a deterministic barrier). Holding
        # the admission lock across refusal AND teardown makes the two domains
        # atomic: a racing capture is either seen by the refusal or blocks
        # until torch is fully restored (and then fails the wrapped-epoch check
        # at admission instead of running an unlogged forward). Lock order is
        # install -> admission only; admission never acquires the install lock.
        with _state._capture_admission_lock:
            _refuse_unwrap_during_active_capture()
            from .identity_shims import remove_identity_shims

            remove_identity_shims()
            _unwrap_torch_locked()


def _refuse_unwrap_during_active_capture() -> None:
    """Refuse wrapper removal while a capture owns the logging globals.

    Raises
    ------
    CaptureNotActiveError
        If ``_active_trace`` is set or logging is enabled.
    """

    if _state._active_trace is None and not _state._logging_enabled:
        return
    trace = _state._active_trace
    model_name = (trace.model_name or trace._source_class_name) if trace is not None else None
    raise CaptureNotActiveError(
        "unwrap_torch() was called while a TorchLens capture is still active"
        + (f" for model {model_name!r}" if model_name else ""),
        code="unwrap_during_active_capture",
        remedy=(
            "let the capture finish before removing the wrappers (pass "
            "unwrap_when_done=True to tl.trace, or call unwrap_torch() after "
            "trace() returns) — unwrapping mid-forward silently truncates the Trace"
        ),
        owner_thread_id=_state._active_owner_thread_id,
        calling_thread_id=threading.get_ident(),
    )


def _unwrap_torch_locked() -> None:
    """Remove torchlens wrappers; caller holds ``_wrapper_install_lock``."""
    _compiled_write_guard.materialize_all_deferred_payloads()  # unwrapped writes go unseen
    _state._logging_enabled = False
    _state._active_trace = None
    reset_detector_tables()
    _state._escape_detector_mode = "off"
    _state._completeness_witness_mode = "off"

    restore_belt_references()
    from .backward import uninstall_autograd_wrappers

    buried_sites: list[str] = []
    uninstall_autograd_wrappers(buried_sites)
    from .legacy_ctors import uninstall_legacy_constructor_wrappers

    uninstall_legacy_constructor_wrappers()

    uninstall_torch_ops_call_recorders()

    if not _state._decorated_to_orig:
        _state._is_decorated = False
        warn_buried_sites(buried_sites)
        return

    for namespace_name, func_name in get_orig_torch_funcs():
        # install tolerates namespace drift; teardown/re-install must
        # too, or unwrap_torch() dies mid-loop on the exact drift install absorbs,
        # leaving torch partially wrapped (a process-global leak).
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            # a drifted slot whose foreign wrapper chains to a live
            # torchlens wrapper is rebased onto the restored original; only a
            # burial that survives the rebase is reported (at the very end).
            if (
                id(current) not in _state._orig_to_decorated
                and callable(current)
                and rebase_buried_wrapper(current)
            ):
                buried_sites.append(f"{namespace_name}.{func_name}")
            continue
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    for namespace_name, func_name, _transform_kind in TRANSFORM_BUILDER_SITES:
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            continue
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    for namespace_name, func_name, _transform_kind, _label_name in DIRECT_TRANSFORM_SITES:
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            continue
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    _state._is_decorated = False

    # Restoring Tensor.__getitem__ doesn't clear the stale sq_item slot.
    _fix_tensor_sequence_slot()

    # Torch's ``_device_constructors()`` is an lru_cache keyed on nothing; it
    # memoizes the SET of factory callables that ``DeviceContext.__torch_function__``
    # injects a device into. ``wrap_torch`` warms it PRE-wrap so it stays keyed
    # by originals for the whole wrapped epoch (R56: repopulating it post-wrap
    # broke C-level injection for stale pre-wrap factory refs). The clear here
    # is defense-in-depth: if any third-party path cleared and re-materialized
    # the cache mid-epoch it would hold the now-replaced wrappers, so drop it
    # and let torch re-derive against the restored originals.
    device_constructors = get_device_constructors()
    if device_constructors is not None:
        try:
            device_constructors.cache_clear()
        except (AttributeError, TypeError):
            pass

    # R56 derived-cache class: dynamo rule tables materialized during the
    # wrapped epoch (a user importing/compiling after the first capture) are
    # keyed by torchlens wrappers and would SURVIVE this unwrap -- the poisoned
    # ``get_torch_obj_rule_map`` made post-unwrap ``torch.compile(fullgraph=True)``
    # fail on skipped-function lookups. Drop them so the next materialization
    # re-derives from the restored originals.
    for rule_cache in _dynamo_identity_rule_caches():
        try:
            rule_cache.cache_clear()
        except (AttributeError, TypeError):
            pass

    # Released models were normalized to the WRAPPED epoch's live values; with
    # the originals now restored, re-normalize them so the documented
    # release_model serializability remedy survives the unwrap instead of
    # inverting into the unrecoverable pickle shape.
    _renormalize_released_models_after_flip()

    # Teardown is complete here; the report may raise under a warning filter,
    # so it comes after the flag flip and the cache clears.
    warn_buried_sites(buried_sites)


def _configure_escape_detector(mode: EscapeDetectorMode | None) -> EscapeDetectorMode:
    """Validate and apply the process-level diagnostic detector mode.

    Parameters
    ----------
    mode:
        ``None`` preserves the current mode; ``"shadow"`` reports without
        enforcement and ``"off"`` disables profiling.

    Returns
    -------
    EscapeDetectorMode
        Effective mode for subsequent captures.
    """

    if mode is not None:
        if mode not in {"off", "shadow"}:
            raise ValueError("escape_detector must be 'off' or 'shadow'.")
        _state._escape_detector_mode = mode
    return cast(EscapeDetectorMode, _state._escape_detector_mode)


def _configure_completeness_witness(
    mode: bool | CompletenessWitnessMode | None,
) -> CompletenessWitnessMode:
    """Validate and apply the process-level dispatcher witness mode.

    Parameters
    ----------
    mode:
        ``True`` enables diagnostic shadow mode, ``False`` disables it, and
        ``None`` preserves the current wrapper-epoch setting.

    Returns
    -------
    CompletenessWitnessMode
        Effective mode for subsequent captures.
    """

    if mode is not None:
        normalized: str = "shadow" if mode is True else "off" if mode is False else mode
        if normalized not in {"off", "shadow"}:
            raise ValueError("completeness_witness must be a bool, 'off', or 'shadow'.")
        _state._completeness_witness_mode = normalized
    return cast(CompletenessWitnessMode, _state._completeness_witness_mode)


def wrap_torch(
    *,
    escape_detector: EscapeDetectorMode | None = None,
    completeness_witness: bool | CompletenessWitnessMode | None = None,
) -> None:
    """Install (or re-install) torchlens wrappers on all torch functions.

    If this is the first call, performs full decoration (``decorate_all_once``
    plus the mechanical belt sweep).  If wrappers were
    previously removed via ``unwrap_torch()``, re-installs them from the
    cached maps without re-creating wrapper objects.

    Safe to call multiple times. Patching is process-global, so policy and
    allowlist configuration also apply process-wide for the current epoch.

    Parameters
    ----------
    escape_detector:
        Opt-in callable diagnostic mode. ``"shadow"`` reports exact raw-call
        escapes and marks traces unverified; the release default is ``"off"``.
    completeness_witness:
        Opt-in aten dispatcher census. ``True`` or ``"shadow"`` reports
        unaccounted dispatches and marks traces unverified; default is off.
    """
    # Whole install under one lock: every mutation below is a check-then-mutate
    # over process-global wrapper state (see ``_wrapper_install_lock``).
    with _wrapper_install_lock:
        _wrap_torch_locked(
            escape_detector=escape_detector,
            completeness_witness=completeness_witness,
        )
        # Identity shims keep torch-internal `x is F.y` checks truthful while
        # wrappers are installed (transformer fastpath flag, CausalBias sdpa
        # dispatch, expanded-weights per-sample-grads). Installed here so every
        # wrap path (first decoration, already-decorated, re-install) has them.
        from .identity_shims import install_identity_shims

        install_identity_shims()


def _reinstall_torch_func_wrappers() -> None:
    """Put the recorded wrappers back after a prior ``unwrap_torch``.

    A slot holding a recorded original gets its wrapper back; a slot that
    DRIFTED while torch was unwrapped (a peer patched it in between, or
    ``unwrap_torch`` rebased a foreign wrapper onto the original) is adopted as
    a new original, exactly as first install wraps whatever is current, so a
    capture after ``tl.validate`` still logs the call. Non-callable drift
    (a property, a deleted attribute) is left alone.
    """

    for namespace_name, func_name in get_orig_torch_funcs():
        # install tolerates namespace drift; teardown/re-install must
        # too, or unwrap_torch() dies mid-loop on the exact drift install absorbs,
        # leaving torch partially wrapped (a process-global leak).
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        if id(current) in _state._decorated_to_orig:
            continue
        if id(current) in _state._orig_to_decorated:
            decorated = _state._orig_to_decorated[id(current)]
        elif type(current) in _WRAPPABLE_FUNCTION_TYPES:
            _install_function_wrapper(local_func_namespace, namespace_name, func_name, current)
            continue
        else:
            continue
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, decorated)
        except (AttributeError, TypeError):
            pass


def _wrap_torch_locked(
    *,
    escape_detector: EscapeDetectorMode | None,
    completeness_witness: bool | CompletenessWitnessMode | None,
) -> None:
    """Install torchlens wrappers; caller holds ``_wrapper_install_lock``.

    Parameters
    ----------
    escape_detector:
        Optional diagnostic detector mode, as passed to ``wrap_torch``.
    completeness_witness:
        Optional dispatcher-census mode, as passed to ``wrap_torch``.
    """

    from .backward import install_autograd_wrappers

    # Torch-only setup deferred out of arg_positions import time: the corrected
    # spec table must exist before any wrapper can build an op record.
    _ensure_schema_tensor_position_corrections()

    # Fork children must never keep logging into an inherited mid-capture
    # trace; registered once per process, on every install path.
    _install_fork_capture_hygiene()

    _configure_escape_detector(escape_detector)
    _configure_completeness_witness(completeness_witness)

    # Torchvision is probed lazily (never imported by TorchLens); a user import
    # that landed after the first wrap gets its custom ops decorated here.
    _ensure_torchvision_ops_decorated()

    from .legacy_ctors import install_legacy_constructor_wrappers

    # Legacy ``torch.<dtype>Tensor`` / ``Variable`` classes are patched in place
    # (idempotent; restored by ``_unwrap_torch_locked``).
    install_legacy_constructor_wrappers()
    install_torch_ops_call_recorders()

    if _state._is_decorated:
        install_autograd_wrappers()
        sweep_stale_belt_references()
        return

    _state._wrap_epoch += 1

    if not _FULL_DECORATION_COMPLETED:
        # No decoration pass has ever COMPLETED (first wrap, or a retry after one
        # failed partway). Run the full pass: it is per-pair idempotent, so it
        # finishes exactly the targets a partial failure left undecorated.
        decorate_all_once()
        install_autograd_wrappers()
        sweep_stale_belt_references()
        _renormalize_released_models_after_flip()
        return

    # Re-install from existing maps (after a prior unwrap_torch).
    # The warm invariant holds per EPOCH: unwrap_torch() cleared the derived
    # identity caches (device constructors, any dynamo rule tables), so they
    # must be re-warmed against the restored originals BEFORE the setattr loop
    # repoints the namespace; otherwise epoch 2+ is correct only if something
    # happened to materialize them between unwrap and re-wrap.
    _warm_derived_identity_caches()

    _reinstall_torch_func_wrappers()

    _decorate_transform_builders()
    _decorate_direct_transforms()

    # Recreate decorated identity in case wrapper references shifted
    _state._decorated_identity = torch_func_decorator(identity, "identity")
    _state._is_decorated = True
    # A torchvision import that landed while torch was unwrapped is invisible
    # to the reinstall loop above (its ops were never in the wrapper maps).
    _ensure_torchvision_ops_decorated()
    install_autograd_wrappers()
    sweep_stale_belt_references()

    # Re-wrapping __getitem__ pollutes sq_item again; clear it.
    _fix_tensor_sequence_slot()

    _renormalize_released_models_after_flip()


def _renormalize_released_models_after_flip() -> None:
    """Re-point released models' held refs at the values live in this epoch.

    ``release_model`` normalizes held torch-function attrs to the values live
    at release time; without this hook a later wrap-state flip inverted the
    documented serializability remedy into the UNRECOVERABLE pickle shape (a
    released-while-wrapped model permanently held the transient epoch's
    wrapper). Best-effort with a routed warning: a teardown/install seam must
    never die on one model's exotic state, but it must not go silent either.
    """

    from ._held_refs import renormalize_released_models

    try:
        renormalize_released_models()
    except Exception as error:  # pragma: no cover - defensive seam belt
        from ..._errors import TorchLensWarning

        warnings.warn(
            "TorchLens could not re-normalize held torch-function references "
            f"on a released model after a wrap-state change ({type(error).__name__}: "
            f"{error}). That model may fail whole-model pickle/torch.save until "
            "tl.release_model(model) is called again.",
            TorchLensWarning,
            stacklevel=3,
        )


@contextmanager
def wrapped(
    *,
    escape_detector: EscapeDetectorMode | None = None,
    completeness_witness: bool | CompletenessWitnessMode | None = None,
) -> Iterator[None]:
    """Context manager: wrap torch on entry, unwrap on exit.

    Usage::

        from torchlens.backends.torch.wrappers import wrapped

        with wrapped():
            log = torchlens.trace(model, x)
        # torch is clean again here

    Parameters
    ----------
    escape_detector:
        Optional ``"off"`` or diagnostic ``"shadow"`` mode.
    completeness_witness:
        Optional bool or ``"off"``/``"shadow"`` dispatcher witness mode.
    """
    wrap_torch(
        escape_detector=escape_detector,
        completeness_witness=completeness_witness,
    )
    try:
        yield
    finally:
        unwrap_torch()
