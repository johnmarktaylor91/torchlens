"""Lazy torch function wrapping for capture-time operation interception.

Wrappers persist after first installation and branch on ``_state._logging_enabled``.
This module also patches detached torch references and torch transform boundaries.
"""

import inspect
import os
import sys
import threading
import time
import types
import warnings
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import partial, wraps
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
from ..._capture_state_helpers import _is_uninitialized_param
from ..._errors import CaptureContextError
from ...capture.arg_positions import _ensure_schema_tensor_position_corrections
from ...constants import _get_torchvision_funcs, get_orig_torch_funcs
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.internal_types import FuncExecutionContext
from ...utils._torch_compat import (
    HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE,
    HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE,
    dynamo_is_compiling,
    fix_tensor_sequence_slot,
    get_current_function_mode_stack,
    get_device_constructors,
    get_device_context_type,
    get_functorch_maybe_current_level,
    get_jit_boolean_dispatch_table,
    get_jit_builtin_table,
    get_optional_torch_namespace,
    get_torch_function_mode_stack_length,
    mark_torch_capability_missing,
)
from ...utils.arg_handling import copy_arg_tree
from ...utils.display import identity
from ...utils.hashing import make_random_barcode
from ...utils.introspection import get_vars_of_type_from_obj
from ...utils.rng import log_current_autocast_state, log_current_rng_states
from ...utils.tensor_utils import (
    _DEFER_ENABLED as _COW_ENABLED,
    _DEFER_PENDING as _COW_PENDING,
    arm_deferred_payload_window,
    disarm_deferred_payload_window,
    materialize_deferred_for_call,
    print_override,
    safe_copy,
)
from ._modes import (
    SubclassConstructionUnderDispatchModeError,
    _any_owned_dispatch_mode_active,
    pause_own_dispatch_modes,
)
from ._op_markers import _pop_op_markers, _push_op_markers
from ._tl import (
    _DETACHED_ACTIVATION_PROPAGATION_FUNCS,
    get_param_meta,
    get_tensor_label,
    has_detached_saved_activations,
    is_decorated_function,
    is_tensor_data_alias,
    mark_decorated_function,
    mark_tensor_data_alias,
    propagate_detached_saved_activation,
    set_tensor_label,
)
from ._weightsfree_ctx import active_factory_device as _active_factory_device
from .aliasing import _tensors_alias
from .buffer_writes import (
    record_op_buffer_writes,
    resolve_or_late_index_buffer_address,
    snapshot_buffer_args,
)
from .completeness_witness import (
    CompletenessWitnessMode,
    completeness_scope_for_wrapper,
    internal_scalar_read,
    observe_nonowner_operands,
    record_data_alias_mutation,
    record_host_string_escape_source,
    record_uncaptured_owner_callsite,
    string_escape_is_owner_thread,
)
from .escape_detection import (
    EscapeDetectorMode,
    expected_original_call,
    mark_expected_original_accounted,
    reset_detector_tables,
)
from .ops import (
    _is_inplace_augmented_assignment_dunder,
    _record_label_version_snapshot,
    _walk_output_tensors_with_paths,
    apply_live_hooks_to_outputs,
    log_function_output_tensors,
    register_call_input_container_snapshots,
)
from .sources import log_source_tensor

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
    maybe_current_level = get_functorch_maybe_current_level()
    if maybe_current_level is None:
        return False
    return maybe_current_level() is not None


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


def _warn_transform_boundary_collapse(transform_kind: str) -> None:
    """Warn that a transform boundary was collapsed.

    Parameters
    ----------
    transform_kind:
        Transform kind whose inner operations are intentionally not logged.
    """

    import warnings

    warnings.warn(
        "TorchLens captured a "
        f"{transform_kind} transform as a boundary op. Operations that run inside "
        "the functorch/vmap/grad/jacfwd transform are not logged. The returned "
        "Trace will only contain the transform boundary and operations that ran "
        "outside the transform.",
        UserWarning,
        stacklevel=3,
    )


TRANSFORM_BUILDER_SITES: tuple[tuple[str, str, str], ...] = (
    ("torch", "vmap", "vmap"),
    ("torch.func", "vmap", "vmap"),
    ("torch.func", "grad", "grad"),
    ("torch._functorch.apis", "vmap", "vmap"),
    ("torch._functorch.apis", "grad", "grad"),
)
"""Torch transform builders instrumented as returned-callable boundaries."""

DIRECT_TRANSFORM_SITES: tuple[tuple[str, str, str, str], ...] = (
    ("torch.autograd.functional", "jacobian", "autograd.jacobian", "autogradjacobian"),
    ("torch.autograd.functional", "hessian", "autograd.hessian", "autogradhessian"),
    ("torch.autograd.functional", "vjp", "autograd.vjp", "autogradvjp"),
    ("torch.autograd.functional", "jvp", "autograd.jvp", "autogradjvp"),
    ("torch.autograd.functional", "hvp", "autograd.hvp", "autogradhvp"),
    ("torch.autograd.functional", "vhp", "autograd.vhp", "autogradvhp"),
)
"""Torch autograd.functional direct-call transform entry points."""


def _transform_tags(callable_obj: Callable[..., Any]) -> tuple[str, ...]:
    """Return TorchLens transform tags carried by a callable.

    Parameters
    ----------
    callable_obj:
        Callable or partial to inspect.

    Returns
    -------
    tuple[str, ...]
        Transform tags, outermost first, or an empty tuple when unavailable.
    """

    tags = getattr(callable_obj, "__tl_transform_tags__", ())
    if tags:
        return tuple(tags)
    if isinstance(callable_obj, partial):
        return _transform_tags(callable_obj.func)
    return ()


def _callable_code_location(callable_obj: Callable[..., Any]) -> str | None:
    """Return a best-effort code-location fingerprint for a callable.

    Parameters
    ----------
    callable_obj:
        Callable, partial, or bound method to inspect.

    Returns
    -------
    str | None
        ``filename:firstlineno`` when available.
    """

    target = callable_obj
    if isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    code = getattr(target, "__code__", None)
    if code is None:
        return None
    return f"{code.co_filename}:{code.co_firstlineno}"


def _callable_source_location(callable_obj: Callable[..., Any]) -> FuncCallLocation | None:
    """Return a lazy source location for a callable when code metadata is available.

    Parameters
    ----------
    callable_obj:
        Callable, partial, or bound method to inspect.

    Returns
    -------
    FuncCallLocation | None
        Lazy source location, or ``None`` for native/builtin callables.
    """

    target = callable_obj
    if isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    code = getattr(target, "__code__", None)
    if code is None:
        return None
    return FuncCallLocation(
        file=code.co_filename,
        line_number=code.co_firstlineno,
        func_name=getattr(target, "__name__", type(target).__name__),
        num_context_lines_requested=1,
        _frame_func_obj=target,
        code_firstlineno=code.co_firstlineno,
        func_qualname=getattr(target, "__qualname__", None),
        source_loading_enabled=True,
    )


def _transform_builder_config(
    transform_kind: str,
    builder_args: tuple[Any, ...],
    builder_kwargs: dict[str, Any],
    inner_fn: Callable[..., Any],
) -> dict[str, Any]:
    """Build serializable metadata for a transform builder invocation.

    Parameters
    ----------
    transform_kind:
        Transform kind being built.
    builder_args:
        Positional builder arguments after the inner function.
    builder_kwargs:
        Keyword builder arguments.
    inner_fn:
        User callable being transformed.

    Returns
    -------
    dict[str, Any]
        Best-effort transform configuration.
    """

    config = dict(builder_kwargs)
    if transform_kind == "vmap":
        if builder_args:
            config.setdefault("in_dims", builder_args[0])
        if len(builder_args) > 1:
            config.setdefault("out_dims", builder_args[1])
        config.setdefault("in_dims", 0)
        config.setdefault("out_dims", 0)
    elif transform_kind == "grad":
        if builder_args:
            config.setdefault("argnums", builder_args[0])
        config.setdefault("argnums", 0)
    code_location = _callable_code_location(inner_fn)
    if code_location is not None:
        config["fn_code_location"] = code_location
    return config


def _set_transform_metadata(
    callable_obj: Callable[..., Any],
    *,
    transform_kind: str,
    tags: tuple[str, ...],
    transform_config: dict[str, Any],
    inner_fn: Callable[..., Any],
) -> None:
    """Attach TorchLens transform metadata to a callable when possible.

    Parameters
    ----------
    callable_obj:
        Callable receiving metadata.
    transform_kind:
        Unsanitized transform kind.
    tags:
        Transform chain tags.
    transform_config:
        Captured transform configuration.
    inner_fn:
        User function being transformed.

    Returns
    -------
    None
        Metadata is attached best-effort.
    """

    metadata = {
        "__tl_is_transform_boundary__": True,
        "__tl_transform_tags__": tags,
        "__tl_transform_kind__": transform_kind,
        "__tl_transform_config__": transform_config,
        "__tl_transform_fn_name__": getattr(inner_fn, "__name__", None),
        "__tl_transform_fn_qualname__": getattr(inner_fn, "__qualname__", None),
        "__tl_transform_fn_source__": _callable_source_location(inner_fn),
    }
    for name, value in metadata.items():
        try:
            setattr(callable_obj, name, value)
        except (AttributeError, TypeError):
            pass


def transform_builder_decorator(
    builder: Callable[..., Any],
    transform_kind: str,
) -> Callable[..., Any]:
    """Wrap a torch.func-style transform builder.

    Parameters
    ----------
    builder:
        Original transform builder such as ``torch.func.vmap``.
    transform_kind:
        Unsanitized transform kind recorded on returned callables.

    Returns
    -------
    Callable[..., Any]
        Builder wrapper that instruments returned callables unconditionally.
    """

    @wraps(builder)
    def wrapped_builder(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Build a transform callable and attach TorchLens boundary metadata."""
        built = builder(func, *args, **kwargs)
        if not callable(built):
            return built

        inner_tags = _transform_tags(func)
        tags = (transform_kind, *inner_tags)
        raw_built = built
        transform_config = _transform_builder_config(transform_kind, args, kwargs, func)
        _set_transform_metadata(
            raw_built,
            transform_kind=transform_kind,
            tags=tags,
            transform_config=transform_config,
            inner_fn=func,
        )

        @wraps(raw_built)
        def wrapped_transform(*call_args: Any, **call_kwargs: Any) -> Any:
            """Record one call to a torch.func-style transform boundary."""
            if not _state._logging_enabled or _state._active_trace is None:
                return raw_built(*call_args, **call_kwargs)

            trace = cast(Any, _state._active_trace)
            capture_start_time = time.time()
            save_rng = getattr(trace, "save_rng_states", False)
            rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
            autocast_state = log_current_autocast_state()
            func_call_id = _state.next_func_call_id()
            _warn_transform_boundary_collapse(transform_kind)
            with _state.pause_logging():
                out_orig = raw_built(*call_args, **call_kwargs)
            exec_ctx = FuncExecutionContext(
                time_elapsed=time.time() - capture_start_time,
                rng_states=rng_states,
                autocast_state=autocast_state,
            )
            out_orig = apply_live_hooks_to_outputs(
                trace,
                raw_built,
                transform_kind,
                call_args,
                call_kwargs,
                out_orig,
                exec_ctx,
                True,
                func_call_id,
            )
            if _collect_output_tensors(out_orig):
                # Hide TorchLens bookkeeping dispatches only from the opt-in user-op census.
                if _state._completeness_witness_mode == "shadow":
                    with _state.pause_logging():
                        log_function_output_tensors(
                            trace,
                            raw_built,
                            transform_kind,
                            call_args,
                            call_kwargs,
                            call_args,
                            call_kwargs,
                            out_orig,
                            exec_ctx,
                            True,
                            func_call_id,
                        )
                else:
                    log_function_output_tensors(
                        trace,
                        raw_built,
                        transform_kind,
                        call_args,
                        call_kwargs,
                        call_args,
                        call_kwargs,
                        out_orig,
                        exec_ctx,
                        True,
                        func_call_id,
                    )
            return out_orig

        _set_transform_metadata(
            wrapped_transform,
            transform_kind=transform_kind,
            tags=tags,
            transform_config=transform_config,
            inner_fn=func,
        )
        return wrapped_transform

    return wrapped_builder


def direct_transform_decorator(
    direct_func: Callable[..., Any],
    transform_kind: str,
    func_name: str,
) -> Callable[..., Any]:
    """Wrap an autograd.functional direct-call transform.

    Parameters
    ----------
    direct_func:
        Original direct-call transform.
    transform_kind:
        Unsanitized transform kind.
    func_name:
        Sanitized TorchLens label type.

    Returns
    -------
    Callable[..., Any]
        Toggle-gated direct transform wrapper.
    """

    @wraps(direct_func)
    def wrapped_direct(user_fn: Callable[..., Any], inputs: Any, *args: Any, **kwargs: Any) -> Any:
        """Record one direct-call transform as a boundary operation."""
        if not _state._logging_enabled or _state._active_trace is None:
            return direct_func(user_fn, inputs, *args, **kwargs)

        trace = cast(Any, _state._active_trace)
        call_args = (inputs,)
        raw_replay = partial(direct_func, user_fn, *args, **kwargs)
        transform_config = dict(kwargs)
        transform_config["fn_code_location"] = _callable_code_location(user_fn)
        _set_transform_metadata(
            raw_replay,
            transform_kind=transform_kind,
            tags=(transform_kind,),
            transform_config=transform_config,
            inner_fn=user_fn,
        )
        capture_start_time = time.time()
        save_rng = getattr(trace, "save_rng_states", False)
        rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
        autocast_state = log_current_autocast_state()
        func_call_id = _state.next_func_call_id()
        _warn_transform_boundary_collapse(transform_kind)
        with _state.pause_logging():
            out_orig = direct_func(user_fn, inputs, *args, **kwargs)
        exec_ctx = FuncExecutionContext(
            time_elapsed=time.time() - capture_start_time,
            rng_states=rng_states,
            autocast_state=autocast_state,
        )
        out_orig = apply_live_hooks_to_outputs(
            trace,
            raw_replay,
            func_name,
            call_args,
            {},
            out_orig,
            exec_ctx,
            True,
            func_call_id,
        )
        if _collect_output_tensors(out_orig):
            # Hide TorchLens bookkeeping dispatches only from the opt-in user-op census.
            if _state._completeness_witness_mode == "shadow":
                with _state.pause_logging():
                    log_function_output_tensors(
                        trace,
                        raw_replay,
                        func_name,
                        call_args,
                        {},
                        call_args,
                        {},
                        out_orig,
                        exec_ctx,
                        True,
                        func_call_id,
                    )
            else:
                log_function_output_tensors(
                    trace,
                    raw_replay,
                    func_name,
                    call_args,
                    {},
                    call_args,
                    {},
                    out_orig,
                    exec_ctx,
                    True,
                    func_call_id,
                )
        return out_orig

    return wrapped_direct


def _decorate_transform_builders() -> None:
    """Install transform-builder decorators listed in ``TRANSFORM_BUILDER_SITES``.

    Returns
    -------
    None
        Torch namespaces and decoration maps are updated in place.
    """

    for namespace_name, attr_name, transform_kind in TRANSFORM_BUILDER_SITES:
        namespace = get_optional_torch_namespace(namespace_name)
        if namespace is None:
            continue
        if not hasattr(namespace, attr_name):
            continue
        current = getattr(namespace, attr_name)
        if id(current) in _state._decorated_to_orig:
            continue
        if id(current) in _state._orig_to_decorated:
            decorated = _state._orig_to_decorated[id(current)]
        else:
            decorated = transform_builder_decorator(current, transform_kind)
            mark_decorated_function(decorated)
            _state._orig_to_decorated[id(current)] = decorated
            _state._decorated_to_orig[id(decorated)] = current
            _state._decorated_func_mapper[decorated] = current
            _state._decorated_func_mapper[current] = decorated
        try:
            _setattr_ignoring_advisories(namespace, attr_name, decorated)
        except (AttributeError, TypeError):
            pass


def _decorate_direct_transforms() -> None:
    """Install direct-call transform decorators.

    Returns
    -------
    None
        Torch namespaces and decoration maps are updated in place.
    """

    for namespace_name, attr_name, transform_kind, func_name in DIRECT_TRANSFORM_SITES:
        # r-b4 R26-5b: tolerant resolution, matching _decorate_transform_builders.
        namespace = get_optional_torch_namespace(namespace_name)
        if namespace is None or not hasattr(namespace, attr_name):
            continue
        current = getattr(namespace, attr_name)
        if id(current) in _state._decorated_to_orig:
            continue
        if id(current) in _state._orig_to_decorated:
            decorated = _state._orig_to_decorated[id(current)]
        else:
            decorated = direct_transform_decorator(current, transform_kind, func_name)
            mark_decorated_function(decorated)
            _state._orig_to_decorated[id(current)] = decorated
            _state._decorated_to_orig[id(decorated)] = current
            _state._decorated_func_mapper[decorated] = current
            _state._decorated_func_mapper[current] = decorated
        try:
            _setattr_ignoring_advisories(namespace, attr_name, decorated)
        except (AttributeError, TypeError):
            pass


# Functions that should never be logged — they are metadata queries, not
# computational operations, and logging them would cause infinite recursion
# (e.g. size() is called internally by logging code).
funcs_not_to_log = ["numpy", "__array__", "size", "dim"]

# Print functions get special handling: intercepted to add TorchLens label
# info to the repr without creating a new logged operation.
print_funcs = ["__repr__", "__str__", "_str"]

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
    # W1-CTX (weightsfree memo D4): during an ADMITTED weights-free capture
    # TorchLens owns the factory device through its thread-scoped slot — no
    # torch DeviceContext mode is on the stack (its catch-all re-entry
    # respells dunder ops, defect L3), so the slot supplies the placement a
    # native device context would have injected. A caller-pinned non-None
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


def _collect_output_tensors(out: Any) -> list[torch.Tensor]:
    """Fast inline output tensor extraction.

    Most torch functions return a single tensor. This handles that case
    with a simple isinstance check, falling back to BFS for compound outputs.
    """
    if isinstance(out, torch.Tensor):
        if isinstance(out, torch.nn.Parameter):
            return []
        return [out]
    if isinstance(out, (list, tuple)):
        tensors = []
        for item in out:
            if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                tensors.append(item)
        return tensors
    if out is None:
        return []
    if type(out) is torch.utils.hooks.RemovableHandle:
        # ``register_hook``-family output: holds only ints and weakrefs
        # (weakrefs are callable, so the BFS skips them) — provably tensor-free.
        return []
    # Rare: dict, custom object, etc. — fall back to BFS.
    return get_vars_of_type_from_obj(
        out, which_type=torch.Tensor, subclass_exceptions=[torch.nn.Parameter]
    )


def _is_unregistered_parameter(trace: Any, value: Any) -> bool:
    """Return whether ``value`` is a Parameter outside the prepared model state.

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    value:
        Candidate operation output.

    Returns
    -------
    bool
        ``True`` for a Parameter that is not the exact prepared parameter object
        recorded at its stamped address in this capture.
    """

    if not isinstance(value, torch.nn.Parameter):
        return False
    meta = get_param_meta(value)
    address = None if meta is None else meta.param_address
    if not address:
        return True
    param_logs = getattr(trace, "param_logs", None)
    if param_logs is None or address not in param_logs:
        return True
    return getattr(param_logs[address], "_param_ref", None) is not value


def _parameter_mutation_output_for_logging(
    trace: Any,
    value: Any,
    *,
    source: Any,
    was_inplace: bool,
) -> Any:
    """Convert an unregistered Parameter mutation result into a loggable Tensor.

    PyTorch constructs module Parameters from ordinary factory tensors, and the
    conversion intentionally drops TorchLens tensor labels. Initializers such as
    ``uniform_`` then return the new Parameter itself. Parameter outputs are normally
    excluded because prepared model state remains a source rather than an op output,
    but that rule also dropped real initialization ops for modules created inside
    ``forward``.

    Only unregistered Parameters are converted. Mutations of prepared model state
    remain excluded and therefore remain visible to the completeness tripwire instead
    of being laundered into a disconnected op.

    Parameters
    ----------
    trace:
        Active capture trace.
    value:
        Safe-copied operation output.
    source:
        Live same-object return whose current-session registration is authoritative.
    was_inplace:
        Whether the wrapped callable has an in-place mutation signature.

    Returns
    -------
    Any
        A plain Tensor snapshot for capture-local Parameter mutations; otherwise
        ``value`` unchanged.
    """

    if not was_inplace or not _is_unregistered_parameter(trace, source):
        return value
    with _state.pause_logging():
        if HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE:
            plain_value = value.as_subclass(torch.Tensor)
        else:
            plain_value = torch.ops.aten.detach.default(value)
        tensor = safe_copy(plain_value, detach_tensor=True)
        tensor.requires_grad_(value.requires_grad)
    return tensor


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
    through the callable resolver. The SETTER (``t.data = rhs``, round-31 M6 +
    r28 reconcile) rebinds the receiver onto RHS's storage: the receiver's old
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

    from .completeness_witness import internal_scalar_read

    try:
        with _state.pause_logging(), internal_scalar_read():
            storage = t.untyped_storage()
            return (storage.data_ptr(), storage.nbytes(), str(storage.device))
    except Exception:
        return None


# Per-trace cache for the clone-on-write eligibility decision. Keyed weakly:
# a Trace attribute would trip the portable-state completeness scrub
# (PORTABLE_STATE_SPEC), and the decision is wrapper-internal state anyway.
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
    witnesses, backward capture, gradient saving, inference-mode tensors,
    user activation transforms that may mutate the payload in place, predicate
    recording, streaming writers, non-copy save modes) keeps the historical
    eager clone.
    """
    try:
        cached = _COW_STATE_PTRS_CACHE.get(trace, _COW_UNSET)
    except TypeError:
        cached = _COW_UNSET
    if cached is not _COW_UNSET:
        return cached
    ptrs: Any = None
    if (
        _COW_ENABLED
        and not getattr(trace, "intervention_ready", False)
        and not getattr(trace, "backward_ready", False)
        and not getattr(trace, "inference_only", False)
        and getattr(trace, "save_grads", None) in (None, False)
        and getattr(trace, "activation_transform", None) is None
        and getattr(trace, "capture_mode", None) != "predicate"
        and getattr(trace, "save_mode", "copy") == "copy"
        and getattr(trace, "_out_writer", None) is None
    ):
        model_ref = getattr(trace, "_source_model_ref", None)
        model = model_ref() if callable(model_ref) else None
        if model is not None and hasattr(model, "parameters"):
            try:
                with _state.pause_logging(), internal_scalar_read():
                    collected: set[int] = set()
                    for p in model.parameters():
                        collected.add(p.untyped_storage().data_ptr())
                    for b in model.buffers():
                        collected.add(b.untyped_storage().data_ptr())
                ptrs = frozenset(collected)
            except Exception:
                ptrs = None
    try:
        _COW_STATE_PTRS_CACHE[trace] = ptrs
    except TypeError:
        pass
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


def _register_inplace_live_grad_hook(trace: Any, tensor: Any, raw_label: str) -> None:
    """Hook the live in-place result so its gradient is captured under ``raw_label``.

    In-place ops log their output against a ``safe_copy`` whose grad_fn is a
    dead-end ``CloneBackward`` node. When same-object identity is preserved the
    live tensor (the original, in-place-modified one) is what downstream ops
    consume, so the real gradient flows through it -- not the logged copy. This
    registers the standard backward grad hook on the live tensor so the grad is
    captured. ``_add_tensor_backward_hook`` dedups by ``(label, id(tensor))`` and
    only hooks autograd-participating tensors, so the call is safe and cheap.
    """

    if not isinstance(tensor, torch.Tensor):
        return
    from .tensor_tracking import _add_tensor_backward_hook

    # The live tensor is what downstream ops consume, so it takes gradient
    # ownership of the label; the logged copy's hook (if any) stops emitting.
    _add_tensor_backward_hook(trace, tensor, raw_label, take_ownership=True)


def _storage_overlap_byte_interval(t: torch.Tensor) -> tuple[int, int]:
    """Return the ``[start, end)`` byte interval ``t`` can address in its storage.

    Conservative interval form (stride gaps ignored): a strided view's
    addressable extent is treated as one contiguous byte range, so two views
    with interleaved-but-disjoint elements inside the same range are treated
    as overlapping. That errs toward LINKING a mutation to a possibly-affected
    alias, never toward missing one. Negative strides do not exist in torch;
    zero strides (``expand``) contribute nothing to the span.
    """

    element_size = t.element_size()
    start = int(t.storage_offset()) * element_size
    if t.numel() == 0:
        return (start, start)
    span = 1 + sum((int(size) - 1) * int(stride) for size, stride in zip(t.shape, t.stride()))
    return (start, start + span * element_size)


# Exact element-overlap scans are vectorized O(min(n1, n2)); above this bound
# fall back to the conservative byte-interval answer instead of a large scan.
_EXACT_OVERLAP_SCAN_LIMIT = 65536


def _effective_1d_element_layout(t: torch.Tensor) -> tuple[int, int, int] | None:
    """Return ``(start_byte, stride_bytes, count)`` when element starts form one
    arithmetic progression.

    Size-1 and stride-0 (``expand``) dims contribute no distinct addresses and
    are dropped. A single remaining strided dim maps directly; a dense
    (memory-contiguous) multi-dim block telescopes to stride ``element_size``.
    Anything else (genuinely multi-strided ``as_strided`` lattices) returns
    ``None`` so the caller keeps the conservative interval answer.

    Parameters
    ----------
    t:
        Live view whose element addresses are being described.

    Returns
    -------
    tuple[int, int, int] | None
        Progression of element start addresses in storage bytes, or ``None``
        when the layout is not a single progression.
    """

    element_size = t.element_size()
    start_byte = int(t.storage_offset()) * element_size
    if t.numel() == 0:
        return (start_byte, element_size, 0)
    dims = [
        (int(size), int(stride))
        for size, stride in zip(t.shape, t.stride())
        if size > 1 and stride != 0
    ]
    if not dims:
        return (start_byte, element_size, 1)
    if len(dims) == 1:
        size, stride = dims[0]
        return (start_byte, stride * element_size, size)
    expected_stride = 1
    total = 1
    for size, stride in sorted(dims, key=lambda dim: dim[1]):
        if stride != expected_stride:
            return None
        expected_stride = stride * size
        total *= size
    return (start_byte, element_size, total)


def _strided_views_share_storage_elements(mutated: torch.Tensor, alias: torch.Tensor) -> bool:
    """Return whether two same-storage views share at least one element's bytes.

    The byte-interval intersection is kept as the exact NEGATIVE test (disjoint
    intervals can never share bytes) and as the conservative fallback for
    layouts the exact test cannot express. For the common case -- both views
    reducible to one arithmetic progression of equal-width element starts --
    the answer is computed exactly, so element-DISJOINT interleaved views
    (``base[::2]`` vs ``base[1::2]``) no longer receive an invented mutation
    edge (round-31 M4), while genuinely overlapping views keep theirs.

    Parameters
    ----------
    mutated:
        The live tensor the op wrote through.
    alias:
        Another live labeled tensor on the same storage.

    Returns
    -------
    bool
        True when the views provably or possibly share storage bytes; False
        only on proof of disjointness.
    """

    mutated_lo, mutated_hi = _storage_overlap_byte_interval(mutated)
    alias_lo, alias_hi = _storage_overlap_byte_interval(alias)
    if not (alias_lo < mutated_hi and mutated_lo < alias_hi):
        return False
    if mutated.element_size() != alias.element_size():
        # Different element widths break the shared start-address grid; the
        # windows can partially overlap without equal starts. Stay conservative.
        return True
    mutated_layout = _effective_1d_element_layout(mutated)
    alias_layout = _effective_1d_element_layout(alias)
    if mutated_layout is None or alias_layout is None:
        return True
    start_a, stride_a, count_a = mutated_layout
    start_b, stride_b, count_b = alias_layout
    if count_a == 0 or count_b == 0:
        return False
    if count_a > count_b:
        start_a, stride_a, count_a, start_b, stride_b, count_b = (
            start_b,
            stride_b,
            count_b,
            start_a,
            stride_a,
            count_a,
        )
    if count_a > _EXACT_OVERLAP_SCAN_LIMIT:
        return True
    scan_starts = start_a + torch.arange(count_a, dtype=torch.long) * stride_a
    relative = scan_starts - start_b
    if stride_b <= 0:
        return True
    hits = (relative >= 0) & (relative < count_b * stride_b) & (relative % stride_b == 0)
    return bool(hits.any())


def _propagate_mutation_label_to_storage_aliases(
    trace: Any, mutated: torch.Tensor, out_label: str
) -> None:
    """Advance live labels of storage aliases overlapping an in-place op's target.

    An in-place op that mutates a VIEW (``v = y[0]; v.add_(100.)``) changes the
    BASE tensor's content, but the base is a different Python object: only the
    view's label used to advance, so the mutation op was a dead end and later
    consumers of the base bound to the stale pre-mutation parent while storing
    the post-mutation value (W3 audit F1, silent). Resolve every OTHER live
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

    from ._tl import session_storage_alias_candidates, storage_alias_index_key
    from .ops import _record_label_version_snapshot

    # SCOPE (r26 reconcile): descriptor/grad-bound captures keep the HISTORICAL
    # topology. The runnable recipe + numeric attestation key payloads PER
    # LABEL, and advancing a base tensor's label to a view-mutation op makes
    # one label denote two different values (the op's view-shaped output AND
    # the full post-mutation base as the consumer's parent) -- byte-exact
    # attestation then fails on a genuinely-verifiable run (r29 suite).
    # Those modes stay honest without the edge: the r29 view-lineage gate
    # fail-closes view-mediated input mutation, and validation's
    # version-snapshot replay is green with EITHER topology. Default captures
    # (receptive fields, influence geometry, collapse -- the W3-F1 impact
    # surface) get the mutation edge. Follow-up: teach the runnable reader
    # per-(label, consumer) payload keying, then lift this scope.
    if (
        getattr(trace, "intervention_ready", False)
        or getattr(trace, "backward_ready", False)
        or getattr(trace, "save_grads", None) not in (None, False)
    ):
        return

    # ``untyped_storage()`` / ``data_ptr()`` / ``stride()`` / ``storage_offset()``
    # are WITNESSED host-escape / metadata surfaces, and a genuine user
    # ``data_ptr()`` read fail-closes runnable captures to UNVERIFIABLE
    # (r15-H1). These are TorchLens's OWN bookkeeping reads, so they run under
    # ``pause_logging`` (no spurious op capture) plus ``internal_scalar_read``
    # (kept off the escape census / metadata patches) -- the same sanctioned
    # pattern as ``completeness_witness``'s storage-site indexers.
    with _state.pause_logging(), internal_scalar_read():
        try:
            # Meta-safe key (D20): data_ptr() reads 0 for EVERY meta storage.
            storage_ptr = storage_alias_index_key(mutated.untyped_storage())
        except Exception:
            return
        if storage_ptr is None:
            return
        candidates = session_storage_alias_candidates(storage_ptr)
        if not candidates or (len(candidates) == 1 and candidates[0] is mutated):
            return
        mutated_lo, mutated_hi = _storage_overlap_byte_interval(mutated)
        if mutated_hi <= mutated_lo:
            return
        for alias in candidates:
            if (
                alias is mutated
                or not isinstance(alias, torch.Tensor)
                or isinstance(alias, torch.nn.Parameter)
            ):
                continue
            # Gated read: rejects foreign-session stamps and storage-rebound
            # objects, so a stale index entry can never act as a live alias.
            alias_label = get_tensor_label(alias)
            if alias_label is None or alias_label == out_label:
                continue
            try:
                if (
                    storage_alias_index_key(alias.untyped_storage()) != storage_ptr
                    or alias.device != mutated.device
                ):
                    continue
                # Element-exact where provable (round-31 M4): interleaved
                # element-disjoint views must not inherit the mutation edge.
                shares_elements = _strided_views_share_storage_elements(mutated, alias)
            except Exception:
                continue
            if shares_elements:
                set_tensor_label(alias, out_label)
                _register_inplace_live_grad_hook(trace, alias, out_label)
                _record_label_version_snapshot(alias)


# Tensor getset properties whose SETTER mutates the receiver's forward data:
# ``t.real = rhs`` / ``t.imag = rhs`` write through the receiver's storage,
# ``t.data = rhs`` REBINDS the receiver onto ``rhs``'s storage. These execute
# real dataflow yet return ``None``, so without receiver reconstruction no op
# is ever emitted and consumers keep stale/absent parents (round-31 M6).
# ``requires_grad`` / ``grad`` and similar setters change autograd bookkeeping,
# not forward values, and are deliberately NOT listed.
def _setattr_ignoring_advisories(namespace: Any, name: str, value: Any) -> None:
    """Set a torch namespace attribute, suppressing ADVISORY warnings only.

    Wrap/unwrap setattr over deprecated torch aliases legitimately fires
    deprecation-family advisories, but the historical bare
    ``simplefilter("ignore")`` also hid every OTHER warning category raised in
    scope and invalidated the process ``__warningregistry__`` per entry
    (B8-39). Only the advisory categories are ignored; a genuine torch
    ``RuntimeWarning`` (or anything else) still reaches the user.
    """

    with warnings.catch_warnings():
        for category in (
            DeprecationWarning,
            PendingDeprecationWarning,
            FutureWarning,
            UserWarning,
        ):
            warnings.simplefilter("ignore", category)
        setattr(namespace, name, value)


# Positional index of the target Tensor subclass ``cls`` argument for each
# wrapped callable whose ORIGINAL can crash under an active python
# TorchDispatchMode on torch 2.1/2.2 (``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``).
# ``__new__``/``_make_subclass`` are called as ``cls(...)``/
# ``cls._make_subclass(cls, ...)`` (class first); ``as_subclass`` is an
# instance method, ``tensor.as_subclass(cls)`` (class second). This is the
# FAST-PATH-only table: the LOGGED path's own ``__new__``-only gap-tracked
# pause (see ``constructs_tensor_subclass`` below) is unrelated and
# untouched. A direct top-level call here REFUSES typed instead of pausing:
# pausing a top-level (non-reentrant) construction call was tried and
# reverted after it corrupted interpreter state when exercised across
# several capture paths in one process (segfault on torch 2.1.2) -- see
# ``SubclassConstructionUnderDispatchModeError``.
_FAST_PATH_SUBCLASS_CLS_ARG_INDEX: dict[str, int] = {
    "__new__": 0,
    "_make_subclass": 0,
    "as_subclass": 1,
}


def _fast_path_constructs_strict_subclass(args: tuple[Any, ...], idx: int) -> bool:
    """Return whether ``args[idx]`` is a strict (non-``torch.Tensor``) subclass cls.

    Parameters
    ----------
    args:
        Positional call arguments.
    idx:
        Index of the candidate subclass ``cls`` argument.

    Returns
    -------
    bool
        ``True`` when ``args[idx]`` is a type strictly below ``torch.Tensor``.
    """

    if len(args) <= idx:
        return False
    cls = args[idx]
    return isinstance(cls, type) and cls is not torch.Tensor and issubclass(cls, torch.Tensor)


_MUTATING_TENSOR_PROPERTY_SETTERS = frozenset({"real", "imag", "data"})

# Setters that rebind the receiver to the RHS's storage instead of writing in
# place. The mutation label must NOT propagate to other tensors on that (RHS)
# storage: their bytes were never written.
_STORAGE_REBINDING_PROPERTY_SETTERS = frozenset({"data"})


def torch_func_decorator(
    func: Callable[..., Any],
    func_name: str,
    property_accessor: str | None = None,
) -> Callable[..., Any]:
    """Wrap a single torch function with toggle-gated logging.

    When ``_state._logging_enabled`` is ``False``, the wrapper is a near-noop
    (one bool check, then call original).  When ``True``, it:

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
    tensor so subsequent operations see it.  Non-mutating self-returns (e.g.
    ``contiguous()`` on an already-contiguous tensor) are copied but NOT
    propagated back — they are silently dropped by barcode identity detection
    downstream.

    Args:
        func: The original (unwrapped) torch function.
        func_name: The attribute name of the function (e.g. ``"cos"``, ``"__add__"``).
        property_accessor: ``"set"`` / ``"del"`` / ``"get"`` when ``func`` is one
            accessor of a wrapped getset property (``Tensor.real`` and friends),
            else ``None``. Mutating property SETTERS return ``None`` from a call
            that rewrites the receiver, so the wrapper reconstructs the receiver
            as the logged output for names in
            ``_MUTATING_TENSOR_PROPERTY_SETTERS``.

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
    mutates_receiver = _func_mutates_receiver(func_name)
    inplace_param_index = _positional_inplace_index(func)
    reconstructs_receiver_output = func_name in {"__setitem__", "zero_", "__delitem__"}
    has_inplace_signature = (
        func_name.endswith("_")
        or func_name.startswith("__i")
        or func_name in {"__setitem__", "__delitem__"}
        or is_mutating_property_setter
    )
    force_distinct_return = func_name == "identity"
    # ``TensorBase.__new__`` is the one wrapped callable whose ORIGINAL refuses
    # to run under any python TorchDispatchMode when handed a strict Tensor
    # subclass cls (see pause_own_dispatch_modes); every other op pays nothing.
    constructs_tensor_subclass = func_name == "__new__"
    # FAST-PATH-only gate (direct top-level ``__new__``/``_make_subclass``/
    # ``as_subclass`` calls made with logging disabled -- e.g. a user model or
    # intervention hook constructing a subclass during a replay/validation
    # pass, or TorchLens's own paused internal calls): ``None`` whenever the
    # active torch build tolerates subclass construction under a dispatch
    # mode (``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE``, the common case -- zero
    # cost there) or ``func_name`` is not one of the three. Deliberately
    # DISTINCT from ``constructs_tensor_subclass`` above: the LOGGED path's
    # existing ``__new__``-only pause and its ``mode_paused_interior`` gap
    # accounting are untouched by this.
    _fast_path_subclass_cls_index = (
        None
        if HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
        else _FAST_PATH_SUBCLASS_CLS_ARG_INDEX.get(func_name)
    )
    # Decoration-time constant: ``propagate_detached_saved_activation`` is a
    # guaranteed no-op for any name outside the propagation allowlist, but its
    # ARGUMENTS (two tensor collections, each with a BFS fall-back for nested
    # args such as ``register_hook``'s callable) were evaluated eagerly on
    # every paused internal call. Gating on the closure constant skips exactly
    # the calls the allowlist check inside the helper would discard.
    is_detached_propagation_func = func_name in _DETACHED_ACTIVATION_PROPAGATION_FUNCS
    canonical_capture_callable = None
    if func_name != "data" or property_accessor == "del":
        canonical_capture_callable = (func, func_name)
    # See the barcode-transparency note inside ``wrapped_func`` (R16-5).
    is_barcode_transparent = func_name == "as_subclass"

    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        """Dispatch a decorated torch callable through the logging gate."""
        # ---- Fast path ----
        # When logging is off, pass through with minimal overhead.
        # DeviceContext injection is still needed even when not logging,
        # because the user's model may rely on torch.device('meta') context.
        # r43 hon2_4: op-logging is OWNER-thread-scoped. A NON-owner thread running torch ops
        # during a capture (a worker formatting ``str(tensor)``, a DataLoader thread) must NOT
        # be logged into the owner's Trace -- doing so tags its temporaries with capture labels
        # (a false cross-thread ceiling) and corrupts owner-op attribution (an observed crash).
        # Cross-thread tensor->host escapes are still observed by the mode-independent belt
        # (tensor-method patches), which is independent of this wrapper.
        if not _state._logging_enabled:
            # Deferred payload clones: pending aliases stay zero-copy for the
            # process lifetime, so the fast path carries the SAME pre-execution
            # interception as the logging path — any wrapped call that can
            # write through a tensor argument first copies out pending aliases
            # sharing those storages. Disarmed cost is one dict-truthiness
            # check; with pending aliases, only mutation-signature calls pay
            # the storage lookups.
            if _COW_PENDING and (
                mutates_receiver
                or is_mutating_property_setter
                or reconstructs_receiver_output
                or "out" in kwargs
                or _call_requests_inplace(inplace_param_index, args, kwargs)
            ):
                materialize_deferred_for_call(_collect_tensor_args(args, kwargs))
            if needs_device_injection:
                kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            if (
                _fast_path_subclass_cls_index is not None
                and _fast_path_constructs_strict_subclass(args, _fast_path_subclass_cls_index)
                and _any_owned_dispatch_mode_active()
            ):
                raise SubclassConstructionUnderDispatchModeError(
                    f"torchlens cannot construct the Tensor subclass "
                    f"{args[_fast_path_subclass_cls_index]!r} via {func_name!r} while a "
                    "TorchLens dispatch mode is active on this torch build. "
                    "Remedy: upgrade torch (HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE reports "
                    "False here) or avoid constructing/converting into a custom Tensor "
                    "subclass inside a model forward, an intervention hook, or a "
                    "validate_forward_pass replay on this build.",
                    code="subclass_ctor_under_dispatch_mode_unsupported",
                    func_name=func_name,
                )
            out = func(*args, **kwargs)
            fast_collector = _state._active_fast_run_collector
            if fast_collector is not None and fast_collector.wants_function(func_name):
                fast_collector.capture_function(func_name, out)
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
            # r45 hon2_1: while a runnable capture is armed, a NON-owner thread's op that consumes
            # a captured tensor as an operand ceilings replay proof to ``unverifiable`` (the
            # worker-DERIVED cross-thread escape sibling: fresh worker-side storage the
            # owner-only census never registered). ``_nonowner_belt_armed`` is False for every
            # plain trace and the whole steady state, so the disarmed hot path pays one bool read.
            if (
                _state._nonowner_belt_armed
                and active_trace is not None
                and owner_thread_id != current_thread_id
            ):
                observe_nonowner_operands(args, kwargs)
            if needs_device_injection:
                kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            out = func(*args, **kwargs)
            if is_detached_propagation_func and has_detached_saved_activations():
                propagate_detached_saved_activation(
                    func_name,
                    _collect_tensor_args(args, kwargs),
                    _collect_output_tensors(out),
                )
            return out

        trace = cast(Any, active_trace)
        if needs_device_injection:
            kwargs = _maybe_inject_device_kwarg(func_name, kwargs)

        # Skip logging inside vmap/functorch transforms — internal TorchLens
        # operations (safe_copy, torch.equal, .item()) don't have vmap batching
        # rules and will crash. The original function is already vmap-compatible.
        # Warn once per forward pass so the user knows their Trace is
        # missing whatever runs inside the transform.
        if _is_inside_functorch_transform():
            if not _state._functorch_warning_emitted:
                _state._functorch_warning_emitted = True
                trace._raw_transform_escape_detected = True
                _warn_functorch_region_not_logged()
            # A raw transform interior is outside the witness claim, but the witness-off
            # route retains its original logging state and avoids the context-manager cost.
            if _state._completeness_witness_mode == "shadow":
                with _state.pause_logging():
                    if _state._escape_detector_mode == "shadow":
                        with expected_original_call(func, f"torch_func:{func_name}:functorch"):
                            return func(*args, **kwargs)
                    return func(*args, **kwargs)
            else:
                if _state._escape_detector_mode == "shadow":
                    with expected_original_call(func, f"torch_func:{func_name}:functorch"):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)

        # Skip logging inside a Dynamo-traced region, for the same reason as the
        # functorch guard above: TorchLens' internal operations (safe_copy,
        # torch.equal, .item(), memory accounting) read tensor VALUES, and the
        # tensors flowing through a compiled region are data-free FakeTensors.
        # Tracing through the wrapper used to die with a raw, unexplained
        # ``InternalTorchDynamoError: 'FakeTensor' object has no attribute
        # 'fake_mode'``. TorchLens already unwraps compiled *submodules* before
        # capture (see _capture_state_helpers.unwrap_compiled_submodules), but a
        # compiled *callable* held as a plain attribute or called as a free
        # function cannot be swapped out, so this is the boundary for those.
        # Degrading to a pass-through matches the documented contract: log the
        # eager source module; torch.compile internals are not traced.
        if _is_inside_dynamo_compilation():
            # ``_raw_dynamo_region_detected`` is the specific cause and outranks every
            # other verification reason: a compiled region also spawns compile threads
            # and leaves unaccounted aten dispatches, so without it the Trace reports a
            # true-but-misleading reason such as ``owner_thread_tripwire_changed``.
            # ``_raw_transform_escape_detected`` additionally licenses the
            # unattributable-model-output tolerance the functorch boundary uses, which is
            # what lets the forward finish at all when the output leaves the region.
            trace._raw_dynamo_region_detected = True
            trace._raw_transform_escape_detected = True
            if not _state._dynamo_warning_emitted:
                _state._dynamo_warning_emitted = True
                _warn_dynamo_region_not_logged()
            if _state._completeness_witness_mode == "shadow":
                with _state.pause_logging():
                    return func(*args, **kwargs)
            return func(*args, **kwargs)

        # F20 lazy-buffer completion: torch's lazy pre-hook materializes
        # pending UninitializedBuffer/Parameter slots through wrapped calls
        # whose RECEIVER is the pending tensor (.data device read/setter,
        # empty_like on the data alias). A pending tensor has no
        # shape/storage to log and the plumbing is pre-forward
        # materialization, not model dataflow (the materialized buffer
        # registers at the module-entry gate); pass the call through
        # unlogged, mirroring the functorch/dynamo region guards above.
        if args and isinstance(args[0], torch.Tensor) and _is_uninitialized_param(args[0]):
            return func(*args, **kwargs)

        # Usage stats: count every decorated function call during logging.
        if _state._collect_usage_stats:
            _state._function_call_counts[func_name] = (
                _state._function_call_counts.get(func_name, 0) + 1
            )
            _state._function_call_models.setdefault(func_name, set()).add(
                _state._current_model_name
            )

        # Reset barcode; skip metadata-only functions that would cause recursion.
        # R16-5: ``as_subclass`` is barcode-TRANSPARENT. Torch's default
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
        if is_unlogged_func:
            if _diagnostic_edge_armed():
                wrapper_name = f"torch_func:{func_name}:not_logged"
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

        # Register buffer tensors on first encounter. Buffers are tagged with
        # _tl.address during model prep but don't get _tl.label_raw
        # until the first function actually uses them.
        for t in arg_tensorlike:
            if isinstance(t, torch.nn.Parameter):
                continue
            # The r81 identity ladder + F20 lazy late-index live in
            # buffer_writes.resolve_or_late_index_buffer_address.
            address = resolve_or_late_index_buffer_address(trace, t)
            if address is not None and get_tensor_label(t) is None:
                log_source_tensor(trace, t, "buffer", address)

        # Intercept print functions to show TorchLens label info in repr.
        if is_print_func and arg_tensorlike:
            # r39 hon2_1: stringifying a captured tensor extracts its VALUES into the returned
            # string (a genuine tensor->host value escape the user can fold back into control
            # flow -- a string NaN guard). ``print_override`` runs that extraction under
            # ``pause_logging()``, blinding the ordinary ``.numpy()``/``.item()`` escape
            # observers, so record the source here BEFORE the paused format, through the same
            # attribution ladder. A runnable capture then ceilings a changed-input run whose
            # stringified source differs, exactly like a ``.numpy()`` escape.
            for stringified in arg_tensorlike:
                record_host_string_escape_source(trace, stringified)
            # r43 hon2_4: ``print_override`` formats under a GLOBAL ``pause_logging()``; a
            # NON-OWNER thread must NEVER flip that toggle mid-forward (it blinds owner op
            # capture -> the observed crash). The owner keeps ``print_override``; a worker
            # calls the original torch string function unchanged (the captured-tensor ceiling
            # was already applied by ``record_host_string_escape_source`` above).
            if string_escape_is_owner_thread(trace):
                out = print_override(args[0], func_name)
                return out
            return func(*args, **kwargs)

        # Snapshot args before the call in case in-place ops mutate them.
        if trace.save_arg_values:
            arg_copies = tuple([copy_arg_tree(arg) for arg in args])
            kwarg_copies = {k: copy_arg_tree(v) for k, v in kwargs.items()}
        else:
            arg_copies = args
            kwarg_copies = kwargs

        buffer_snapshots = snapshot_buffer_args(trace, func_name, arg_tensorlike, kwargs)

        # ---- Storage-rebinding setter pre-call snapshot (r28 reconcile) ----
        # ``t.data = rhs`` rebinds the receiver onto RHS's storage. Whether the
        # rebind SWAPS the storage object (rhs on foreign storage) or PRESERVES
        # it (rhs a view of the receiver's own storage) decides the ancestry
        # barrier below, and is only observable BEFORE the setter runs.
        rebind_setter_swaps_storage = False
        if (
            is_storage_rebinding_setter
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
        ):
            receiver_storage_key = _untyped_storage_key(args[0])
            rhs_storage_key = _untyped_storage_key(args[1])
            # Fail closed: an unreadable storage on either side counts as a swap.
            rebind_setter_swaps_storage = (
                receiver_storage_key is None
                or rhs_storage_key is None
                or receiver_storage_key != rhs_storage_key
            )

        # ---- Deferred-clone interception (clone-on-write) ----
        # Any wrapped call that can WRITE through a tensor argument must first
        # copy out pending deferred payload aliases sharing those storages —
        # this runs BEFORE the mutation, so saved bytes stay capture-time
        # exact. Triggers cover every wrapped mutation surface: in-place
        # methods and augmented-assignment dunders (``mutates_receiver``),
        # ``__setitem__``/``zero_``/``__delitem__``, mutating property setters,
        # ``out=`` destinations, and ``inplace=True`` conveniences (``F.relu``
        # / ``F.dropout``) whose actual underscore mutation may run below this
        # wrapper — including POSITIONALLY-passed ``inplace`` (torch's
        # ``Hardswish.forward`` runs ``F.hardswish(input, self.inplace)``, and
        # ``torch._C._nn.hardswish_`` below it is not a wrapped surface, so
        # this wrapper is the ONLY interception point; missing it left stale
        # pending aliases that tripped the belt at the next mutating call —
        # the mobilenet_v3_small failure). NOT ``has_inplace_signature`` —
        # that flag is true for EVERY dunder (``"__add__".endswith("_")``)
        # and is only ever meaningful gated behind a same-object return.
        # Storage-rebinding ``.data=`` writes no bytes but rides along via its
        # property-setter signature — a spurious materialization is merely a
        # wasted clone, never a correctness risk.
        if _COW_PENDING and (
            mutates_receiver
            or is_mutating_property_setter
            or reconstructs_receiver_output
            or "out" in kwargs
            or _call_requests_inplace(inplace_param_index, args, kwargs)
        ):
            materialize_deferred_for_call(arg_tensorlike)

        # ---- Execute the original function ----
        # Write a unique barcode BEFORE the call. If any inner wrapped functions
        # execute during this call, they will overwrite it. After the call,
        # matching barcode => this is the bottom-level (leaf) function.
        func_call_barcode = make_random_barcode()
        trace._wrapper_runtime_ws.current_func_barcode = func_call_barcode
        _save_rng = getattr(trace, "save_rng_states", False)
        rng_states = log_current_rng_states(torch_only=True) if _save_rng else {}
        autocast_state = log_current_autocast_state()
        func_call_id = _state.next_func_call_id()
        register_call_input_container_snapshots(
            trace,
            args,
            kwargs,
            func_call_id=func_call_id,
            event_index=func_call_id,
        )
        from ...intervention.runtime import snapshot_call_inputs_for_inplace_intervention_site

        call_input_snapshots = snapshot_call_inputs_for_inplace_intervention_site(
            func_name=func_name,
            args=args,
            kwargs=kwargs,
            trace=trace,
            func_call_id=func_call_id,
        )
        expected_token = None
        pauses_owned_modes = (
            constructs_tensor_subclass
            and args
            and isinstance(args[0], type)
            and args[0] is not torch.Tensor
            and issubclass(args[0], torch.Tensor)
        )
        # W3 F8: per-op duration must measure the USER op, not TorchLens
        # bookkeeping. The clock starts here -- after RNG/autocast snapshots
        # and container/intervention-site registration -- and stops right
        # after the call returns, so ``func_duration`` no longer
        # systematically overstates cheap ops in instrumented captures.
        # Attempted-op marker (snoop D5): pushed BEFORE execution (one dict
        # read when echo is off); semantics on EchoSession.attempted_push.
        _echo = trace.__dict__.get("_echo_session")
        _echo_depth = _echo.attempted_push(func_name, args, kwargs) if _echo is not None else 0
        # W0.3 (TN-D12, hygiene -- not a misattribution fix): the markers
        # open exactly where the op clock opens, so a range never encloses
        # TorchLens's own pre-call bookkeeping (RNG/autocast snapshots,
        # container registration).
        op_marker_tokens = _push_op_markers(trace, func_name, func_call_id)
        func_exec_start = time.time()
        device_memory_before = None
        if getattr(trace, "track_device_memory", False):
            # Observe item 15 (deferred import per the spine layer lint).
            from ...observe._device_memory import read_before as _dm_read_before

            device_memory_before = _dm_read_before(trace)
        mode_pause = pause_own_dispatch_modes() if pauses_owned_modes else nullcontext(())
        paused_modes: tuple[Any, ...] = ()
        try:
            with mode_pause as paused_modes:
                if _diagnostic_edge_armed():
                    with expected_original_call(
                        func,
                        f"torch_func:{func_name}:logged",
                        func_name=func_name,
                        func_call_id=func_call_id,
                        call_barcode=func_call_barcode,
                    ) as expected_token:
                        out_orig = func(*args, **kwargs)
                else:
                    out_orig = func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            # Observe item 15: the OOM raised BEFORE this op committed -- name
            # the ATTEMPTED call, never the last committed op.
            if getattr(trace, "track_device_memory", False):
                from ...observe._device_memory import settle_op_bracket

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
            if (
                pauses_owned_modes
                and not HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE
                and "already associated to a python object" in str(exc)
            ):
                raise SubclassConstructionUnderDispatchModeError(
                    f"torchlens could not construct the Tensor subclass via "
                    f"{func_name!r} even after pausing its own dispatch mode on this "
                    f"torch build (original torch error: {exc}). "
                    "Remedy: upgrade torch (HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE reports "
                    "False here), or avoid this construction while other TorchLens "
                    "instrumentation observes the same forward on this build.",
                    code="subclass_ctor_under_dispatch_mode_unsupported",
                    func_name=func_name,
                ) from exc
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
            from ...observe._device_memory import settle_op_bracket

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

        # __setitem__, zero_, __delitem__ modify in-place and return None;
        # treat the first arg (the modified tensor) as the output. Mutating
        # property setters (``t.real = rhs`` and friends, round-31 M6) have the
        # exact same shape: real dataflow, ``None`` return, mutated receiver.
        if reconstructs_receiver_output or (
            is_mutating_property_setter
            and out_orig is None
            and len(args) > 0
            and isinstance(args[0], torch.Tensor)
        ):
            out_orig = args[0]

        # ---- In-place detection and safe copy ----
        same_object_returned = len(args) > 0 and id(out_orig) == id(args[0])
        record_is_inplace = get_tensor_label(out_orig) is not None
        # True in-place ops (add_, mul_, etc.) modify the tensor and return self.
        # No-op functions (to(same_dtype), contiguous() on contiguous tensor)
        # also return self but don't modify anything.
        # Both cases need safe_copy so logging doesn't overwrite the original's
        # label, but only true in-place ops should propagate the new label back.
        was_inplace = same_object_returned and has_inplace_signature
        # The internal identity-forcing decorator (_state._decorated_identity)
        # exists precisely to MINT a distinct logged tensor at module boundaries
        # (nn.Identity / pass-through outputs). Unlike user-visible no-ops such as
        # x.contiguous(), it must NOT preserve the input's Python object identity,
        # otherwise the module exit re-reads the input's label and the boundary
        # node (e.g. identity_1_2) never attaches to the module's output_ops.
        if same_object_returned:
            # Create a distinct tensor object for logging — otherwise attaching
            # _tl.label_raw on the output would clobber the input's label.
            # Snapshot the USER op's live autograd node first (round-31 M5):
            # the safe copy's grad_fn is TorchLens's own ``CloneBackward``
            # bookkeeping, and it must never replace the operation's recorded
            # autograd metadata (``grad_fn_class_*`` / handle).
            # TorchLens bookkeeping read: for a same-object in-place return the
            # output IS the user's receiver (a registered buffer for BN's
            # ``num_batches_tracked.add_(1)``), so an unmarked ``grad_fn`` read
            # would record a phantom declared-state fact (r65 unread-bit).
            with internal_scalar_read():
                live_user_grad_fn = out_orig.grad_fn if isinstance(out_orig, torch.Tensor) else None
            out_orig = safe_copy(out_orig)
            if live_user_grad_fn is not None and isinstance(out_orig, torch.Tensor):
                try:
                    setattr(out_orig, "tl_user_grad_fn", live_user_grad_fn)
                except AttributeError:
                    pass
            out_orig = _parameter_mutation_output_for_logging(
                trace,
                out_orig,
                source=args[0],
                was_inplace=was_inplace,
            )

        if canonical_capture_callable is None:
            capture_func, capture_func_name = _canonical_capture_callable(
                func, func_name, property_accessor
            )
        else:
            capture_func, capture_func_name = canonical_capture_callable
        # r28 reconcile: a storage-rebinding setter is RECORDED as the canonical
        # single-argument ``detach(rhs)`` call -- the receiver's OLD value has
        # zero dataflow into the result (the rebind replaces every value), so
        # only the RHS argument is logged. The live receiver still gets the
        # op's label through the same-object in-place propagation below.
        log_args, log_kwargs = args, kwargs
        log_arg_copies, log_kwarg_copies = arg_copies, kwarg_copies
        if is_storage_rebinding_setter and len(args) >= 2:
            log_args = (args[1],)
            log_kwargs = {}
            log_arg_copies = (arg_copies[1],) if len(arg_copies) >= 2 else log_args
            log_kwarg_copies = {}
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
        if getattr(trace, "intervention_ready", False):
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
                        # r28 reconcile: a storage-SWAPPING ``.data=`` rebind
                        # threads its consumers to the RHS producer (correct
                        # dataflow), but verdict-steering attribution must
                        # never root THROUGH the swap -- the r79/r81 belt
                        # posture. Register the op label as an ancestry
                        # barrier so layout/witness rooting fails closed
                        # exactly like the pre-M6 unattributed break.
                        if is_storage_rebinding_setter and rebind_setter_swaps_storage:
                            from .completeness_witness import record_storage_rebind_barrier

                            record_storage_rebind_barrier(trace, out_label)
                        # W3 F1: the write may have gone through a VIEW; every
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

            # W3 F6: the module-boundary identity mint (force_distinct_return)
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

            # W3 F1 (out= family): an ``out=`` destination may itself be a view
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
            # W1-BUF-2 (weightsfree memo D7): on an ADMITTED meta capture the
            # fused mutator's Python decomposition strips the USER call of its
            # bottom-level status (the inner allocator calls claim it), so the
            # declared buffer writes would silently vanish. The snapshot-holding
            # call is the ONE that resolved the buffer args, so it records its
            # own writes; the decomposition's inner calls never carry
            # snapshots (their names are not mutator-classified).
            from ...capture._weightsfree_admission import weightsfree_meta_active

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

        if out_orig is not out_before_hooks:
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


# ---------------------------------------------------------------------------
# get_arg_names — now writes to _state._arg_names instead of self
# ---------------------------------------------------------------------------


def get_arg_names(orig_func: Callable[..., Any], func_name: str) -> None:
    """Extract argument names for a function and store in ``_state._arg_names``.

    Tries ``inspect.signature`` first (works for Python functions). Falls back
    to docstring parsing for C builtins whose signature isn't introspectable.

    Stores under the EXACT registered name: ``add``, ``add_``, and ``__add__``
    have meaningfully different signatures (``__add__(self, other)`` is a
    2-arg dunder; ``torch.add(input, other, *, alpha, out)`` is not), and the
    historical underscore-stripped shared key (#82) let whichever registered
    last overwrite the rest, recording wrong ``arg_names`` metadata (W3 audit
    F9). Lookup falls back to the stripped key for names whose own
    introspection stored nothing, preserving the old best-effort behavior.

    Skipped for property-like attributes (``real``, ``imag``, ``T``, etc.) that
    aren't callable in the normal sense.
    """
    if func_name in ["real", "imag", "T", "mT", "data", "H"]:
        return

    storage_key = func_name

    try:
        params = inspect.signature(orig_func).parameters
        argnames = []
        for name, param in params.items():
            if name in ("cls", "self"):
                continue
            # #123: Use Parameter.kind instead of naive asterisk stripping
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                argnames.append(f"*{name}")
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                argnames.append(f"**{name}")
            else:
                argnames.append(name)
        # A purely-variadic signature ((*args, **kwargs) on opaque C dunders)
        # names nothing; prefer the docstring parse, and store nothing when
        # that also fails -- honest-unknown beats a wrong borrowed signature.
        # A genuinely empty signature (zero-arg methods) still stores ().
        if not argnames or not all(name.startswith("*") for name in argnames):
            _state._arg_names[storage_key] = tuple(argnames)
            return
    except (ValueError, TypeError):
        # TypeError: Python 3.14+ deferred annotation evaluation (PEP 649)
        # can fail when class-level names (e.g. Tensor.bool) shadow builtins
        # during inspect.signature() annotation resolution. Falls back to
        # docstring parsing below.
        pass

    # Fallback: parse argument names from the docstring's first line.
    # C builtins typically have docstrings like "add(input, other, *, alpha=1)".
    docstring = orig_func.__doc__
    if (type(docstring) is not str) or (len(docstring) == 0):
        return

    paren_start, paren_end = docstring.find("("), docstring.find(")")
    argstring = docstring[paren_start + 1 : paren_end]
    arg_list = argstring.split(",")
    arg_list = [arg.strip(" ") for arg in arg_list]
    argnames = []
    for arg in arg_list:
        argname = arg.split("=")[0]
        if argname in ["*", "/", "//", ""]:
            continue
        argname = argname.replace("*", "")
        argnames.append(argname)
    argnames = tuple([arg for arg in argnames if arg not in ["self", "cls"]])  # type: ignore[assignment]
    _state._arg_names[storage_key] = argnames  # type: ignore[assignment]


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
    # NOTE: Do NOT guard on `_orig_to_decorated` being non-empty here.
    # A prior partial failure may have populated the dict without completing
    # decoration. Using _is_decorated (set at end of this function) ensures
    # retry after partial failure (#138).

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
    # by ORIGINALS (B8-4). ``DeviceContext.__torch_function__`` always receives
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

    b8-sol R56-8: a fork DURING a traced forward (e.g. a fork-start
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


def _warm_derived_identity_caches() -> None:
    """Warm torch's identity-keyed DERIVED caches while the namespace holds originals.

    B8-4 invariant: no torchlens path may FIRST-call a cached torch
    introspection table while wrappers are installed. Called by BOTH install
    paths -- full decoration (``decorate_all_once``) and re-install after a
    prior ``unwrap_torch()`` -- BEFORE any wrapper setattr, so every covered
    table stays keyed by originals for the whole wrapped epoch. Covered:

    - the two ``torch.overrides`` tables (the original B8-4 pair);
    - torch's ``_device_constructors()`` set -- ``DeviceContext.__torch_function__``
      always receives the ORIGINAL C function, so a wrapper-keyed set silently
      skips device injection for stale pre-wrap factory references (R56); the
      warm also collects ``_DEVICE_CONSTRUCTOR_NAMES`` for the wrapper-side
      injection path;
    - dynamo's identity-keyed rule tables, when dynamo is already imported
      (never force-imported -- tables a user materializes mid-epoch are
      dropped at ``unwrap_torch()`` instead).
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


def _dynamo_identity_rule_caches() -> list[Any]:
    """Return dynamo's identity-keyed rule caches, without importing dynamo.

    ``torch._dynamo.trace_rules`` memoizes rule tables keyed by the live
    callable objects resolved from the torch namespace at materialization
    time. Like the ``torch.overrides`` tables (B8-4) these are DERIVED caches:
    the attribute-identity restore census cannot see them, so a table built
    while torchlens wrappers are installed silently poisons ``torch.compile``
    for the rest of the process (``lookup(torch.cos)`` degrades to
    ``SkipFunctionVariable`` after ``unwrap_torch()``). Feature-detected via
    ``sys.modules`` + ``getattr`` -- never force-imports dynamo, and degrades
    to an empty list on any future torch that renames the getters.
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
    died on ``_VariableFunctionsClass.cos`` — B8-1a). The install-site stamp
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
    r36 smuggling defense (tests/test_r36_tensor_method_smuggling.py, LOCKED)
    pins as denied on REAL identity with the wrappers module visible. The
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
    # Pre-compute type objects for efficient isinstance-like checks below.
    # These MUST come from ``types`` constants, not live torch attributes
    # (e.g. ``type(torch.mean)``): when this helper runs after decoration
    # (late torchvision import), the torch attributes are already wrappers and
    # probing them would misclassify every pristine C callable.
    function_class = types.FunctionType  # <class 'function'>
    builtin_class = types.BuiltinFunctionType  # <class 'builtin_function_or_method'>
    method_class = types.MethodDescriptorType  # <class 'method_descriptor'>
    wrapper_class = types.WrapperDescriptorType  # <class 'wrapper_descriptor'>
    getset_class = types.GetSetDescriptorType  # <class 'getset_descriptor'> (properties)

    # --- Pass 1: Collect argument names before any decoration ---
    # inspect.signature() must run against the pristine torch namespace.
    # Python 3.14+ (PEP 649) evaluates annotations lazily; if we decorate
    # Tensor.bool first, then inspect Tensor.dim_order, the annotation
    # bool | list[...] resolves bool to our wrapper -> TypeError (#138).
    for namespace_name, func_name in func_pairs:
        # Exact-name dedup (W3 F9): ``add``, ``add_``, and ``__add__`` have
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

        if type(orig_func) in [function_class, builtin_class, method_class, wrapper_class]:
            # --- Shared-original deduplication ---
            # If this exact C builtin was already wrapped under a different namespace
            # (e.g. torch.cos and torch._VF.cos share the same id()), reuse the
            # existing wrapper. Creating a second wrapper would break the 1:1 mapping
            # in _orig_to_decorated and the JIT builtin table.
            if id(orig_func) in _state._orig_to_decorated:
                existing = _state._orig_to_decorated[id(orig_func)]
                try:
                    _setattr_ignoring_advisories(local_func_namespace, func_name, existing)
                except (AttributeError, TypeError):
                    pass
                continue

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
            new_property = property(getter_dec, setter_dec, deleter_dec, doc=func_name)
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
    CaptureContextError
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
    CaptureContextError
        If ``_active_trace`` is set or logging is enabled.
    """

    if _state._active_trace is None and not _state._logging_enabled:
        return
    trace = _state._active_trace
    model_label = getattr(trace, "model_label", None) or getattr(trace, "model_class_name", None)
    raise CaptureContextError(
        "unwrap_torch() was called while a TorchLens capture is still active"
        + (f" for model {model_label!r}" if model_label else ""),
        code="unwrap_during_active_capture",
        remedy=(
            "let the capture finish before removing the wrappers (pass "
            "unwrap_when_done=True to tl.trace, or call unwrap_torch() after "
            "trace() returns) — unwrapping mid-forward silently truncates the Trace"
        ),
        owner_thread_id=_state._active_owner_thread_id,
        calling_thread_id=threading.get_ident(),
    )


def _buries_live_wrapper(current: Any) -> bool:
    """Return whether a foreign callable buries a live torchlens wrapper.

    ``_unwrap_torch_locked`` restores a namespace slot only when the CURRENT
    attribute is a known torchlens wrapper; a third-party wrapper installed on
    top of ours is (correctly) left untouched, but that leaves the torchlens
    wrapper LIVE inside its ``__wrapped__``/closure chain with zero diagnostic
    (B8-6). This walk detects the burial so teardown can name it. Bounded and
    cheap: it runs only for drifted slots (rare), never on the hot path.
    """

    seen: set[int] = set()
    stack: list[Any] = [current]
    visited = 0
    while stack and visited < 32:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        visited += 1
        if obj is not current and id(obj) in _state._decorated_to_orig:
            return True
        wrapped = getattr(obj, "__wrapped__", None)
        if callable(wrapped):
            stack.append(wrapped)
        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    content = cell.cell_contents
                except ValueError:
                    continue
                if callable(content):
                    stack.append(content)
    return False


def _unwrap_torch_locked() -> None:
    """Remove torchlens wrappers; caller holds ``_wrapper_install_lock``."""

    _state._logging_enabled = False
    _state._active_trace = None
    reset_detector_tables()
    _state._escape_detector_mode = "off"
    _state._completeness_witness_mode = "off"
    from .belt import restore_belt_references

    restore_belt_references()
    from .backward import uninstall_autograd_wrappers

    uninstall_autograd_wrappers()

    if not _state._decorated_to_orig:
        _state._is_decorated = False
        return

    buried_sites: list[str] = []
    for namespace_name, func_name in get_orig_torch_funcs():
        # r-b4 R26-5b: install tolerates namespace drift; teardown/re-install must
        # too, or unwrap_torch() dies mid-loop on the exact drift install absorbs,
        # leaving torch partially wrapped (a process-global leak).
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            # B8-6: a drifted slot whose foreign wrapper chains to a live
            # torchlens wrapper stays buried past this teardown -- collect it
            # so the user learns unwrap did NOT fully restore that callable.
            if (
                id(current) not in _state._orig_to_decorated
                and callable(current)
                and _buries_live_wrapper(current)
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

    if buried_sites:
        from ..._errors import TorchLensWarning

        shown = ", ".join(buried_sites[:5])
        more = f" (+{len(buried_sites) - 5} more)" if len(buried_sites) > 5 else ""
        warnings.warn(
            f"unwrap_torch() left {len(buried_sites)} torch callable(s) with a "
            f"torchlens wrapper buried under a third-party wrapper: {shown}{more}. "
            "TorchLens never clobbers foreign patches, so those slots still run "
            "the torchlens wrapper underneath. Remove or reinstall the outer "
            "wrapper around the restored original to fully unwrap.",
            TorchLensWarning,
            stacklevel=3,
        )

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

    from .belt import sweep_stale_belt_references

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
    # B8-4 holds per EPOCH: unwrap_torch() cleared the derived identity caches
    # (device constructors, any dynamo rule tables), so they must be re-warmed
    # against the restored originals BEFORE the setattr loop repoints the
    # namespace -- the historical asymmetry left epoch 2+ correct only when
    # something happened to materialize them between unwrap and re-wrap.
    _warm_derived_identity_caches()

    for namespace_name, func_name in get_orig_torch_funcs():
        # r-b4 R26-5b: install tolerates namespace drift; teardown/re-install must
        # too, or unwrap_torch() dies mid-loop on the exact drift install absorbs,
        # leaving torch partially wrapped (a process-global leak).
        local_func_namespace = get_optional_torch_namespace(namespace_name)
        if local_func_namespace is None or not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        decorated = None
        if id(current) in _state._orig_to_decorated:
            decorated = _state._orig_to_decorated[id(current)]
        elif id(current) in _state._decorated_to_orig:
            decorated = current
        if decorated is None:
            continue
        try:
            _setattr_ignoring_advisories(local_func_namespace, func_name, decorated)
        except (AttributeError, TypeError):
            pass

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
