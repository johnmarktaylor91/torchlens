"""Lazy torch function wrapping for capture-time operation interception.

Wrappers persist after first installation and branch on ``_state._logging_enabled``.
This module also patches detached torch references and torch transform boundaries.
"""

import inspect
import sys
import sysconfig
import time
import types
import weakref
import warnings
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Literal, TYPE_CHECKING, cast

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
from ...constants import get_orig_torch_funcs
from ...data_classes.func_call_location import FuncCallLocation
from ._tl import (
    get_buffer_address,
    get_tensor_label,
    is_decorated_function,
    mark_decorated_function,
    set_tensor_label,
)
from ...data_classes.internal_types import FuncExecutionContext
from ...utils.introspection import get_vars_of_type_from_obj, nested_getattr
from ...utils._torch_compat import (
    get_current_function_mode_stack,
    get_device_constructors,
    get_device_context_type,
    fix_tensor_sequence_slot,
    get_functorch_maybe_current_level,
    get_jit_builtin_table,
    get_optional_torch_namespace,
    get_torch_function_mode_stack_length,
    mark_torch_capability_missing,
)
from ...utils.display import identity
from ...utils.rng import log_current_autocast_state, log_current_rng_states
from ...utils.hashing import make_random_barcode
from ...utils.arg_handling import copy_arg_tree
from ...utils.tensor_utils import print_override, safe_copy
from .ops import (
    _walk_output_tensors_with_paths,
    apply_live_hooks_to_outputs,
    log_function_output_tensors,
    register_call_input_container_snapshots,
)
from .buffer_writes import (
    record_op_buffer_writes,
    resolve_registered_buffer_address,
    snapshot_buffer_args,
)
from .escape_detection import (
    EscapeDetectorMode,
    expected_original_call,
    mark_expected_original_accounted,
    reset_detector_tables,
)
from .completeness_witness import (
    CompletenessWitnessMode,
    completeness_scope_for_wrapper,
    record_uncaptured_owner_callsite,
)
from .sources import log_source_tensor

if TYPE_CHECKING:
    pass


DetachedPatchPolicy = Literal["scoped", "legacy", "full"]
"""Supported detached-reference discovery policies."""

_RELEASE_DEFAULT_PATCH_POLICY: DetachedPatchPolicy = "legacy"
"""Release default; scoped remains opt-in until its certification soak completes."""


def _diagnostic_edge_armed() -> bool:
    """Return whether either exact wrapper-edge diagnostic is enabled.

    Returns
    -------
    bool
        ``True`` when a shared one-shot token is required.
    """

    return (
        _state._escape_detector_mode == "shadow"
        or _state._completeness_witness_mode == "shadow"
        or _state._runnable_ledger_armed
    )


_KNOWN_TORCH_FREE_PREFIXES = (
    "PIL",
    "Pillow",
    "dill",
    "graphviz",
    "mpmath",
    "pydot",
    "sympy",
)
_LEGACY_DETACHED_SKIP_PREFIXES: tuple[str, ...] = (
    "torch.",
    "numpy.",
    "pytest",
    "pluggy",
    "setuptools",
)
_STDLIB_PATHS = tuple(
    path
    for path in (
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
    )
    if path
)


@dataclass(frozen=True)
class PatchReport:
    """Summary of one detached-reference discovery pass.

    Parameters
    ----------
    policy:
        Effective discovery policy.
    epoch:
        Wrapper lifecycle epoch.
    module_identities_scanned:
        Number of module identities shallow-scanned.
    deep_modules_scanned:
        Number of modules receiving class/default inspection.
    direct_attributes_inspected:
        Number of direct module attributes inspected.
    slots_patched:
        Number of identity-matching slots replaced and ledgered.
    source_files_opened:
        Number of source files successfully opened by this pass. Scoped and full
        always report zero because only legacy uses source-gated deep scanning.
    """

    policy: DetachedPatchPolicy
    epoch: int
    module_identities_scanned: int = 0
    deep_modules_scanned: int = 0
    direct_attributes_inspected: int = 0
    slots_patched: int = 0
    source_files_opened: int = 0


@dataclass(frozen=True)
class _MutationLedgerEntry:
    """One reversible identity-conditional foreign-slot mutation."""

    owner_ref: Callable[[], Any | None]
    slot_kind: Literal["module", "class", "defaults", "kwdefault", "model"]
    slot_key: str | None
    original: Any
    replacement: Any
    epoch: int


# ---------------------------------------------------------------------------
# CPython slot fixup for Tensor sequence protocol
# ---------------------------------------------------------------------------


def _nvtx_range_push(name: str) -> bool:
    """Push an NVTX range if CUDA NVTX support is available.

    Parameters
    ----------
    name:
        Range label.

    Returns
    -------
    bool
        Whether a corresponding pop should be attempted.
    """

    try:
        torch.cuda.nvtx.range_push(name)  # type: ignore[no-untyped-call]
    except Exception:
        return False
    return True


def _nvtx_range_pop(enabled: bool) -> None:
    """Pop a previously pushed NVTX range.

    Parameters
    ----------
    enabled:
        Whether a push succeeded.
    """

    if not enabled:
        return
    try:
        torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]
    except Exception:
        return


def _fix_tensor_sequence_slot() -> None:
    """Clear the stale sq_item C slot on torch.Tensor after dunder changes."""

    fix_tensor_sequence_slot()


def _is_inside_functorch_transform() -> bool:
    """Return True if inside a vmap/grad/etc. functorch transform."""
    maybe_current_level = get_functorch_maybe_current_level()
    if maybe_current_level is None:
        return False
    return maybe_current_level() is not None


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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(namespace, attr_name, decorated)
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
        namespace_key = namespace_name.removeprefix("torch.")
        namespace = nested_getattr(torch, namespace_key)
        if not hasattr(namespace, attr_name):
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(namespace, attr_name, decorated)
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
    tensors = []
    needs_bfs = False
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensors.append(arg)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                elif not isinstance(item, (int, float, bool, str, type(None))):
                    needs_bfs = True
        elif isinstance(arg, dict):
            for val in arg.values():
                if isinstance(val, torch.Tensor):
                    tensors.append(val)
        elif not isinstance(arg, (int, float, bool, str, type(None), torch.dtype, torch.device)):
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
    # Rare: dict, custom object, etc. — fall back to BFS.
    return get_vars_of_type_from_obj(
        out, which_type=torch.Tensor, subclass_exceptions=[torch.nn.Parameter]
    )


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

    _add_tensor_backward_hook(trace, tensor, raw_label)


def torch_func_decorator(func: Callable[..., Any], func_name: str) -> Callable[..., Any]:
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
    barcode is written to ``trace._current_func_barcode``.  If the
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

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapped_func(*args: Any, **kwargs: Any) -> Any:
        """Dispatch a decorated torch callable through the logging gate."""
        # ---- Fast path ----
        # When logging is off, pass through with minimal overhead.
        # DeviceContext injection is still needed even when not logging,
        # because the user's model may rely on torch.device('meta') context.
        if not _state._logging_enabled or _state._active_trace is None:
            kwargs = _maybe_inject_device_kwarg(func_name, kwargs)
            return func(*args, **kwargs)

        trace = cast(Any, _state._active_trace)
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
                import warnings

                warnings.warn(
                    "TorchLens detected a functorch/vmap/grad/jacfwd transform "
                    "during this forward pass. Operations that run inside the "
                    "transform are not logged. The returned Trace will only "
                    "contain operations that ran OUTSIDE the transform.",
                    UserWarning,
                    stacklevel=2,
                )
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

        # Usage stats: count every decorated function call during logging.
        if _state._collect_usage_stats:
            _state._function_call_counts[func_name] = (
                _state._function_call_counts.get(func_name, 0) + 1
            )
            _state._function_call_models.setdefault(func_name, set()).add(
                _state._current_model_name
            )

        # Reset barcode; skip metadata-only functions that would cause recursion.
        trace._current_func_barcode = 0
        if func_name in funcs_not_to_log:
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

        # Register buffer tensors on first encounter. Buffers are tagged with
        # _tl.address during model prep but don't get _tl.label_raw
        # until the first function actually uses them.
        for t in arg_tensorlike:
            if isinstance(t, torch.nn.Parameter):
                continue
            address = get_buffer_address(t)
            if address is None:
                address = resolve_registered_buffer_address(trace, t)
            if address is not None and get_tensor_label(t) is None:
                log_source_tensor(trace, t, "buffer", address)

        # Intercept print functions to show TorchLens label info in repr.
        if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
            out = print_override(args[0], func_name)
            return out

        # Snapshot args before the call in case in-place ops mutate them.
        if trace.save_arg_values:
            arg_copies = tuple([copy_arg_tree(arg) for arg in args])
            kwarg_copies = {k: copy_arg_tree(v) for k, v in kwargs.items()}
        else:
            arg_copies = args
            kwarg_copies = kwargs

        buffer_snapshots = snapshot_buffer_args(trace, func_name, arg_tensorlike, kwargs)

        # ---- Execute the original function ----
        # Write a unique barcode BEFORE the call. If any inner wrapped functions
        # execute during this call, they will overwrite it. After the call,
        # matching barcode => this is the bottom-level (leaf) function.
        func_call_barcode = make_random_barcode()
        trace._current_func_barcode = func_call_barcode
        capture_start_time = time.time()
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
        nvtx_pushed = (
            _nvtx_range_push(f"torchlens::{func_name}")
            if getattr(trace, "emit_nvtx", False)
            else False
        )
        expected_token = None
        try:
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
        finally:
            _nvtx_range_pop(nvtx_pushed)
        return_value = out_orig
        exec_ctx = FuncExecutionContext(
            time_elapsed=time.time() - capture_start_time,
            rng_states=rng_states,
            autocast_state=autocast_state,
        )
        is_bottom_level_func = trace._current_func_barcode == func_call_barcode

        # __setitem__, zero_, __delitem__ modify in-place and return None;
        # treat the first arg (the modified tensor) as the output.
        if func_name in ["__setitem__", "zero_", "__delitem__"]:
            out_orig = args[0]

        # ---- In-place detection and safe copy ----
        same_object_returned = len(args) > 0 and id(out_orig) == id(args[0])
        record_is_inplace = get_tensor_label(out_orig) is not None
        # True in-place ops (add_, mul_, etc.) modify the tensor and return self.
        # No-op functions (to(same_dtype), contiguous() on contiguous tensor)
        # also return self but don't modify anything.
        # Both cases need safe_copy so logging doesn't overwrite the original's
        # label, but only true in-place ops should propagate the new label back.
        was_inplace = same_object_returned and (
            func_name.endswith("_")
            or func_name.startswith("__i")
            or func_name in {"__setitem__", "__delitem__"}
        )
        # The internal identity-forcing decorator (_state._decorated_identity)
        # exists precisely to MINT a distinct logged tensor at module boundaries
        # (nn.Identity / pass-through outputs). Unlike user-visible no-ops such as
        # x.contiguous(), it must NOT preserve the input's Python object identity,
        # otherwise the module exit re-reads the input's label and the boundary
        # node (e.g. identity_1_2) never attaches to the module's output_ops.
        force_distinct_return = func_name == "identity"
        if same_object_returned:
            # Create a distinct tensor object for logging — otherwise attaching
            # _tl.label_raw on the output would clobber the input's label.
            out_orig = safe_copy(out_orig)

        out_before_hooks = out_orig
        out_orig = apply_live_hooks_to_outputs(
            trace,
            func,
            func_name,
            args,
            kwargs,
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
            # Hide TorchLens bookkeeping dispatches only from the opt-in user-op census.
            if _state._completeness_witness_mode == "shadow":
                with _state.pause_logging():
                    call_emitted_op = log_function_output_tensors(
                        trace,
                        func,
                        func_name,
                        args,
                        kwargs,
                        arg_copies,
                        kwarg_copies,
                        out_orig,
                        exec_ctx,
                        is_bottom_level_func,
                        func_call_id,
                    )
            else:
                call_emitted_op = log_function_output_tensors(
                    trace,
                    func,
                    func_name,
                    args,
                    kwargs,
                    arg_copies,
                    kwarg_copies,
                    out_orig,
                    exec_ctx,
                    is_bottom_level_func,
                    func_call_id,
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
                    if isinstance(return_value, torch.Tensor):
                        set_tensor_label(return_value, out_label)
                        _register_inplace_live_grad_hook(trace, return_value, out_label)

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
        if is_bottom_level_func:
            record_op_buffer_writes(trace, func_name, buffer_snapshots, producer_label)

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
    setattr(wrapped_func, "__tl_detector_excluded__", func_name in funcs_not_to_log)

    return wrapped_func


# ---------------------------------------------------------------------------
# get_arg_names — now writes to _state._arg_names instead of self
# ---------------------------------------------------------------------------


def get_arg_names(orig_func: Callable[..., Any], func_name: str) -> None:
    """Extract argument names for a function and store in ``_state._arg_names``.

    Tries ``inspect.signature`` first (works for Python functions). Falls back
    to docstring parsing for C builtins whose signature isn't introspectable.

    Stores under the underscore-stripped name (e.g. ``"add"`` for both ``add``
    and ``add_``) so callers can look up via ``func_name.strip("_")`` consistently (#82).

    Skipped for property-like attributes (``real``, ``imag``, ``T``, etc.) that
    aren't callable in the normal sense.
    """
    if func_name in ["real", "imag", "T", "mT", "data", "H"]:
        return

    storage_key = func_name.strip("_")

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

    # Pre-compute type objects for efficient isinstance-like checks below.
    function_class = type(lambda: 0)  # <class 'function'>
    builtin_class = type(torch.mean)  # <class 'builtin_function_or_method'>
    method_class = type(torch.Tensor.__add__)  # <class 'method_descriptor'>
    wrapper_class = type(torch.Tensor.__getitem__)  # <class 'method-wrapper'>
    getset_class = type(torch.Tensor.real)  # <class 'getset_descriptor'> (properties)

    # --- Pass 1: Collect argument names before any decoration ---
    # inspect.signature() must run against the pristine torch namespace.
    # Python 3.14+ (PEP 649) evaluates annotations lazily; if we decorate
    # Tensor.bool first, then inspect Tensor.dim_order, the annotation
    # bool | list[...] resolves bool to our wrapper -> TypeError (#138).
    for namespace_name, func_name in get_orig_torch_funcs():
        if func_name.strip("_") in _state._arg_names:
            continue
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        get_arg_names(orig_func, func_name)

    # --- Pass 2: Decorate all functions ---
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
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
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        setattr(local_func_namespace, func_name, existing)
                except (AttributeError, TypeError):
                    pass
                continue

            recorded_name = _recorded_func_name(namespace_name, func_name)
            new_func = torch_func_decorator(orig_func, recorded_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, new_func)
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
            getter_dec = torch_func_decorator(getter_orig, func_name)
            setter_dec = torch_func_decorator(setter_orig, func_name)
            deleter_dec = torch_func_decorator(deleter_orig, func_name)
            mark_decorated_function(getter_dec)
            mark_decorated_function(setter_dec)
            mark_decorated_function(deleter_dec)
            new_property = property(getter_dec, setter_dec, deleter_dec, doc=func_name)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, new_property)
                # #31: Only add mapper entries if setattr succeeded — otherwise
                # we'd have dangling entries pointing to an uninstalled property.
                cast(dict[int, Any], _state._orig_to_decorated)[id(orig_func)] = new_property
                cast(dict[int, Any], _state._decorated_to_orig)[id(new_property)] = orig_func
                cast(dict[Any, Any], _state._decorated_func_mapper)[new_property] = orig_func
                cast(dict[Any, Any], _state._decorated_func_mapper)[orig_func] = new_property
            except (AttributeError, TypeError):
                pass

    # ---- JIT builtin table registration ----
    # torch.jit._builtins._builtin_table maps id(func) -> ATen op name.
    # We must register our wrappers so JIT recognizes them as the same ops.
    # Without this, torch.jit.script fails on any code using wrapped functions.
    _register_jit_builtin_wrappers()

    # ---- DeviceContext bypass setup ----
    # Collect names of factory functions (zeros, ones, empty, etc.) that accept
    # a device kwarg. The lru_cache must be cleared first so _device_constructors()
    # re-evaluates with our wrapped functions (otherwise it returns stale refs).
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

    # Create the decorated identity — a no-op that forces a new log entry at
    # module boundaries (nn.Identity, pass-through outputs).  Stored on _state
    # instead of monkey-patching torch.identity (which doesn't exist in PyTorch
    # type stubs and causes mypy errors).
    _state._decorated_identity = torch_func_decorator(identity, "identity")
    _decorate_transform_builders()
    _decorate_direct_transforms()
    _state._is_decorated = True

    # Wrapping __getitem__ on torch.Tensor pollutes the C-level sq_item slot,
    # making PySequence_Check(tensor) return True.  Clear it so torch.tensor()
    # doesn't try to iterate 0-d tensor elements as sequences.
    _fix_tensor_sequence_slot()


def _weak_owner_ref(owner: Any) -> Callable[[], Any | None]:
    """Return a weak owner reference, with a conservative strong fallback.

    Parameters
    ----------
    owner:
        Object whose slot TorchLens may mutate.

    Returns
    -------
    Callable[[], Any | None]
        Zero-argument owner resolver used during conditional reversal.
    """

    try:
        return weakref.ref(owner)
    except TypeError:
        return lambda: owner


def _record_mutation(
    owner: Any,
    slot_kind: Literal["module", "class", "defaults", "kwdefault", "model"],
    slot_key: str | None,
    original: Any,
    replacement: Any,
) -> None:
    """Append one mutation to the current epoch ledger.

    Parameters
    ----------
    owner:
        Mutated module, class, function, or model object.
    slot_kind:
        Mutation category used for reversal.
    slot_key:
        Attribute/default key, or ``None`` for positional defaults.
    original:
        Identity/value present immediately before TorchLens wrote.
    replacement:
        Exact identity/value TorchLens installed.
    """

    _state._detached_patch_ledger.append(
        _MutationLedgerEntry(
            _weak_owner_ref(owner),
            slot_kind,
            slot_key,
            original,
            replacement,
            _state._detached_patch_epoch,
        )
    )


def _reverse_detached_reference_ledger() -> None:
    """Conditionally reverse mutations from the current wrapper epoch.

    A slot is restored only when it still contains the exact replacement
    TorchLens installed. User mutations made after patching are preserved.
    """

    for entry in reversed(_state._detached_patch_ledger):
        owner = entry.owner_ref()
        if owner is None:
            continue
        try:
            if entry.slot_kind in {"module", "model"}:
                owner_dict = vars(owner)
                if owner_dict.get(entry.slot_key) is entry.replacement:
                    owner_dict[cast(str, entry.slot_key)] = entry.original
            elif entry.slot_kind == "class":
                if vars(owner).get(entry.slot_key) is entry.replacement:
                    setattr(owner, cast(str, entry.slot_key), entry.original)
            elif entry.slot_kind == "defaults":
                if getattr(owner, "__defaults__", None) is entry.replacement:
                    owner.__defaults__ = entry.original
            elif entry.slot_kind == "kwdefault":
                kwdefaults = getattr(owner, "__kwdefaults__", None)
                if (
                    isinstance(kwdefaults, dict)
                    and kwdefaults.get(entry.slot_key) is entry.replacement
                ):
                    kwdefaults[cast(str, entry.slot_key)] = entry.original
        except (AttributeError, KeyError, TypeError):
            continue
    _state._detached_patch_ledger.clear()


def _reset_detached_patch_epoch_state() -> None:
    """Clear identity caches that cannot cross wrapper epochs."""

    _state._crawled_module_keys.clear()
    _state._crawled_module_identities.clear()
    _state._detached_positive_module_ids.clear()
    _state._detached_positive_modules.clear()


def unwrap_torch() -> None:
    """Remove torchlens wrappers and restore original torch callables.

    After calling this, ``torch.cos``, ``torch.Tensor.__add__``, etc. are the
    originals shipped by PyTorch.  TorchLens logging will not work until
    ``wrap_torch()`` is called (or ``trace`` auto-wraps).

    Safe to call multiple times — no-op if already unwrapped.
    """
    _state._logging_enabled = False
    _state._active_trace = None
    reset_detector_tables()
    _state._escape_detector_mode = "off"
    _state._completeness_witness_mode = "off"
    _state._detached_patch_policy = _RELEASE_DEFAULT_PATCH_POLICY
    _state._detached_patch_modules = ()
    from .backward import uninstall_autograd_wrappers

    uninstall_autograd_wrappers()

    if not _state._decorated_to_orig:
        _state._is_decorated = False
        _reverse_detached_reference_ledger()
        _reset_detached_patch_epoch_state()
        return

    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    for namespace_name, func_name, _transform_kind in TRANSFORM_BUILDER_SITES:
        namespace_key = namespace_name.removeprefix("torch.")
        local_func_namespace = (
            torch if namespace_name == "torch" else nested_getattr(torch, namespace_key)
        )
        if not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    for namespace_name, func_name, _transform_kind, _label_name in DIRECT_TRANSFORM_SITES:
        namespace_key = namespace_name.removeprefix("torch.")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        current = getattr(local_func_namespace, func_name)
        orig = _state._decorated_to_orig.get(id(current))
        if orig is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, orig)
        except (AttributeError, TypeError):
            pass

    _reverse_detached_reference_ledger()
    _state._is_decorated = False
    _reset_detached_patch_epoch_state()

    # Restoring Tensor.__getitem__ doesn't clear the stale sq_item slot.
    _fix_tensor_sequence_slot()

    # Torch's ``_device_constructors()`` is an lru_cache keyed on nothing; it
    # memoizes the SET of factory callables that ``DeviceContext.__torch_function__``
    # injects a device into. ``wrap_torch`` cleared and re-populated it so the set
    # held our WRAPPED callables. Now that the originals are restored, that cache is
    # stale (it still points at the replaced wrappers), so torch's device-context
    # dispatch would no longer recognise the restored ``torch.empty``/``zeros``/...
    # as device constructors -- silently breaking ``with torch.device('meta'): ...``
    # after an unwrap. Clear it so torch re-evaluates against the restored originals.
    device_constructors = get_device_constructors()
    if device_constructors is not None:
        try:
            device_constructors.cache_clear()
        except (AttributeError, TypeError):
            pass


def _resolve_patch_policy(
    policy: DetachedPatchPolicy | Literal["default"] | None,
) -> DetachedPatchPolicy:
    """Resolve a public/compatibility detached-reference policy.

    Parameters
    ----------
    policy:
        Requested policy. ``None`` preserves the current epoch choice, while
        deprecated ``"default"`` resolves to the release default.

    Returns
    -------
    DetachedPatchPolicy
        Effective typed policy.

    Raises
    ------
    ValueError
        If the policy name is unsupported.
    """

    if policy is None:
        current = _state._detached_patch_policy
        if current in {"scoped", "legacy", "full"}:
            return cast(DetachedPatchPolicy, current)
        return _RELEASE_DEFAULT_PATCH_POLICY
    if policy == "default":
        warnings.warn(
            "Detached patch policy 'default' is deprecated; omit patch_policy to use the "
            "release default.",
            DeprecationWarning,
            stacklevel=3,
        )
        return _RELEASE_DEFAULT_PATCH_POLICY
    if policy not in {"scoped", "legacy", "full"}:
        raise ValueError("patch_policy must be 'scoped', 'legacy', or 'full'.")
    return policy


def _configure_patch_policy(
    policy: DetachedPatchPolicy | Literal["default"] | None,
    modules: tuple[str, ...],
) -> DetachedPatchPolicy:
    """Apply monotone process-level patch configuration for this epoch.

    Parameters
    ----------
    policy:
        Optional explicitly requested policy.
    modules:
        Additive module/package prefixes for scoped deep scanning.

    Returns
    -------
    DetachedPatchPolicy
        Effective policy after configuration.
    """

    effective = _resolve_patch_policy(policy)
    if policy is not None:
        _state._detached_patch_policy = effective
    if modules:
        normalized = tuple(dict.fromkeys((*_state._detached_patch_modules, *modules)))
        _state._detached_patch_modules = normalized
    return cast(DetachedPatchPolicy, _state._detached_patch_policy)


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
    patch_policy: DetachedPatchPolicy | Literal["default"] | None = None,
    patch_modules: tuple[str, ...] = (),
    escape_detector: EscapeDetectorMode | None = None,
    completeness_witness: bool | CompletenessWitnessMode | None = None,
) -> None:
    """Install (or re-install) torchlens wrappers on all torch functions.

    If this is the first call, performs full decoration (equivalent to
    ``decorate_all_once`` + ``patch_detached_references``).  If wrappers were
    previously removed via ``unwrap_torch()``, re-installs them from the
    cached maps without re-creating wrapper objects.

    Safe to call multiple times. Patching is process-global, so policy and
    allowlist configuration also apply process-wide for the current epoch.

    Parameters
    ----------
    patch_policy:
        ``"legacy"`` preserves the release-default broad crawl, ``"full"``
        deep-scans every eligible module, and ``"scoped"`` performs exact
        shallow discovery plus bounded provenance/allowlist deep scanning.
    patch_modules:
        Additive exact module names or package prefixes for scoped deep scanning.
    escape_detector:
        Opt-in callable diagnostic mode. ``"shadow"`` reports exact raw-call
        escapes and marks traces unverified; the release default is ``"off"``.
    completeness_witness:
        Opt-in aten dispatcher census. ``True`` or ``"shadow"`` reports
        unaccounted dispatches and marks traces unverified; default is off.
    """
    from .backward import install_autograd_wrappers

    effective_policy = _configure_patch_policy(patch_policy, patch_modules)
    _configure_escape_detector(escape_detector)
    _configure_completeness_witness(completeness_witness)

    if _state._is_decorated:
        install_autograd_wrappers()
        if patch_policy is not None or patch_modules:
            patch_detached_references(policy=effective_policy, modules=patch_modules)
        return

    _state._detached_patch_epoch += 1
    _reset_detached_patch_epoch_state()

    if not _state._orig_to_decorated:
        # First time: full decoration
        decorate_all_once()
        install_autograd_wrappers()
        patch_detached_references(policy=effective_policy, modules=patch_modules)
        return

    # Re-install from existing maps (after a prior unwrap_torch)
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                setattr(local_func_namespace, func_name, decorated)
        except (AttributeError, TypeError):
            pass

    _decorate_transform_builders()
    _decorate_direct_transforms()

    # Recreate decorated identity in case wrapper references shifted
    _state._decorated_identity = torch_func_decorator(identity, "identity")
    _state._is_decorated = True
    install_autograd_wrappers()
    patch_detached_references(policy=effective_policy, modules=patch_modules)

    # Re-wrapping __getitem__ pollutes sq_item again; clear it.
    _fix_tensor_sequence_slot()


@contextmanager
def wrapped(
    *,
    patch_policy: DetachedPatchPolicy | Literal["default"] | None = None,
    patch_modules: tuple[str, ...] = (),
    escape_detector: EscapeDetectorMode | None = None,
    completeness_witness: bool | CompletenessWitnessMode | None = None,
) -> Iterator[None]:
    """Context manager: wrap torch on entry, unwrap on exit.

    Usage::

        with torchlens.wrapped():
            log = torchlens.trace(model, x)
        # torch is clean again here

    Parameters
    ----------
    patch_policy:
        Process-level detached-reference policy for this wrapper epoch.
    patch_modules:
        Additive scoped deep-scan module/package prefixes.
    escape_detector:
        Optional ``"off"`` or diagnostic ``"shadow"`` mode.
    completeness_witness:
        Optional bool or ``"off"``/``"shadow"`` dispatcher witness mode.
    """
    wrap_torch(
        patch_policy=patch_policy,
        patch_modules=patch_modules,
        escape_detector=escape_detector,
        completeness_witness=completeness_witness,
    )
    try:
        yield
    finally:
        unwrap_torch()


# ---------------------------------------------------------------------------
# sys.modules deep crawl
# ---------------------------------------------------------------------------


def patch_detached_references(
    full: bool | None = None,
    *,
    policy: DetachedPatchPolicy | Literal["default"] | None = None,
    modules: Collection[str] = (),
    model: Any | None = None,
) -> PatchReport:
    """Crawl ``sys.modules`` and replace stale references to original torch
    functions with their decorated counterparts.

    **Why this is needed**: Code like ``from torch import cos`` captures a
    reference to the *original* ``torch.cos`` before decoration. After
    ``decorate_all_once()`` replaces ``torch.cos``, the importing module
    still holds the old reference. This crawl fixes those stale references.

    **Four crawl levels**:

    1. **Module-level attributes** — ``import torch; my_cos = torch.cos`` style.
       Checks each attribute in the module's ``__dict__`` against
       ``_orig_to_decorated`` by ``id()``.

    2. **Class-level attributes** — Classes defined in other modules that store
       torch function references as class attributes or custom_methods. Crawls
       ``vars(cls)`` for each class found in the module.

    3. **Function defaults** — Functions that use torch functions as default
       argument values (e.g. ``def f(act=torch.relu)``). Patches both
       ``__defaults__`` and ``__kwdefaults__``.

    4. **Model instance attributes** — Handled separately by
       ``patch_model_instance()`` at ``trace`` time, since model
       instances may not exist yet when this function runs.

    ``legacy`` preserves the release-default Level-1 broad scan and source-gated
    Level-2/3 behavior. ``full`` deep-scans every eligible module. ``scoped``
    shallow-scans exact module identities and deep-scans only exact-positive,
    model-provenance, prior-positive, and allowlisted candidates; it never reads
    source files.

    Parameters
    ----------
    full:
        Deprecated compatibility spelling. ``True`` selects ``full`` and
        ``False`` selects the release default. Cannot be combined with ``policy``.
    policy:
        Explicit detached-reference patching policy.
    modules:
        Additive exact module names or package prefixes for scoped deep scanning.
    model:
        Root model whose class/forward provenance contributes scoped candidates.

    Returns
    -------
    PatchReport
        Structured discovery and mutation counts.
    """
    if full is not None and policy is not None:
        raise ValueError("full and policy cannot be supplied together.")
    requested_policy: DetachedPatchPolicy | Literal["default"] | None = policy
    if full is not None:
        requested_policy = "full" if full else _RELEASE_DEFAULT_PATCH_POLICY
        warnings.warn(
            "full= is deprecated; use policy='full' or omit policy for the release default.",
            DeprecationWarning,
            stacklevel=2,
        )
    module_names = tuple(modules)
    effective_policy = _resolve_patch_policy(requested_policy)
    mapping = _state._orig_to_decorated
    if not mapping:
        return PatchReport(effective_policy, _state._detached_patch_epoch)

    live_modules = _distinct_live_modules()
    new_modules = [
        (key, module) for key, module in live_modules if not _module_identity_was_crawled(module)
    ]
    counters = {
        "module_identities_scanned": 0,
        "deep_modules_scanned": 0,
        "direct_attributes_inspected": 0,
        "slots_patched": 0,
    }
    source_open_counter = [0]
    deep_candidates: dict[int, tuple[str, types.ModuleType]] = {}
    scoped_hot_ids = _scoped_hot_module_ids(model, module_names)
    force_full_scan = requested_policy == "full"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod_key, mod in live_modules:
            should_scan = force_full_scan or (mod_key, mod) in new_modules
            if effective_policy == "scoped" and id(mod) in scoped_hot_ids:
                should_scan = True
            if not should_scan:
                continue
            _state._crawled_module_keys.add(mod_key)
            _remember_crawled_module_identity(mod)
            if _should_skip_detached_module_key(mod_key, effective_policy):
                continue
            if _safe_module_name(mod, mod_key).startswith("torchlens"):
                continue
            try:
                mod_dict = vars(mod)
            except TypeError:
                continue
            counters["module_identities_scanned"] += 1
            exact_hit = False
            for attr_name, attr_val in list(mod_dict.items()):
                counters["direct_attributes_inspected"] += 1
                replacement = mapping.get(id(attr_val))
                if replacement is None:
                    continue
                try:
                    if mod_dict.get(attr_name) is not attr_val:
                        continue
                    mod_dict[attr_name] = replacement
                except (KeyError, TypeError):
                    continue
                _record_mutation(mod, "module", attr_name, attr_val, replacement)
                counters["slots_patched"] += 1
                exact_hit = True
            if exact_hit:
                _remember_positive_module(mod)
            if effective_policy == "scoped":
                if exact_hit or id(mod) in scoped_hot_ids:
                    deep_candidates[id(mod)] = (mod_key, mod)
            elif _should_deep_scan_detached_module(
                mod,
                effective_policy,
                source_open_counter=source_open_counter,
            ):
                deep_candidates[id(mod)] = (mod_key, mod)

        crawled_class_ids: set[int] = set()
        for _mod_key, mod in deep_candidates.values():
            counters["deep_modules_scanned"] += 1
            try:
                values = list(vars(mod).values())
            except TypeError:
                continue
            for attr_val in values:
                is_type = _safe_is_type(attr_val)
                if is_type and id(attr_val) not in crawled_class_ids:
                    crawled_class_ids.add(id(attr_val))
                    counters["slots_patched"] += _patch_class_attributes(attr_val, mapping)
                if not is_type and _safe_is_callable(attr_val):
                    counters["slots_patched"] += _patch_function_defaults(attr_val, mapping)

    return PatchReport(
        policy=effective_policy,
        epoch=_state._detached_patch_epoch,
        module_identities_scanned=counters["module_identities_scanned"],
        deep_modules_scanned=counters["deep_modules_scanned"],
        direct_attributes_inspected=counters["direct_attributes_inspected"],
        slots_patched=counters["slots_patched"],
        source_files_opened=source_open_counter[0],
    )


def _distinct_live_modules() -> list[tuple[str, types.ModuleType]]:
    """Return one stable sys.modules entry per live module identity."""

    result: list[tuple[str, types.ModuleType]] = []
    seen: set[int] = set()
    for key, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType) or id(module) in seen:
            continue
        seen.add(id(module))
        result.append((key, module))
    return result


def _module_identity_was_crawled(module: types.ModuleType) -> bool:
    """Return whether this exact live module identity was already scanned."""

    reference = _state._crawled_module_identities.get(id(module))
    return reference is not None and reference() is module


def _remember_crawled_module_identity(module: types.ModuleType) -> None:
    """Record one scanned identity, weakly when the owner supports it."""

    _state._crawled_module_identities[id(module)] = _weak_owner_ref(module)


def _remember_positive_module(module: types.ModuleType) -> None:
    """Retain one exact-hit scoped module as a weak hot candidate."""

    if id(module) in _state._detached_positive_module_ids:
        return
    _state._detached_positive_module_ids.add(id(module))
    _state._detached_positive_modules.append(_weak_owner_ref(module))


def _safe_module_name(module: types.ModuleType, fallback: str) -> str:
    """Return a defensive module name without triggering lazy-module failures."""

    try:
        name = module.__name__
    except Exception:
        return fallback
    return name if isinstance(name, str) else fallback


def _module_matches_allowlist(name: str, modules: Collection[str]) -> bool:
    """Return whether ``name`` matches an exact module or package prefix."""

    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in modules)


def _scoped_hot_module_ids(model: Any | None, modules: Collection[str]) -> set[int]:
    """Return current scoped deep/shallow hot module identities."""

    names = set(modules) | set(_state._detached_patch_modules)
    if model is not None:
        try:
            model_modules = tuple(model.modules())
        except (AttributeError, TypeError):
            model_modules = (model,)
        for model_module in model_modules:
            cls = type(model_module)
            cls_module = getattr(cls, "__module__", None)
            if isinstance(cls_module, str):
                names.add(cls_module)
            forward = getattr(cls, "forward", None)
            forward_module = getattr(forward, "__module__", None)
            if isinstance(forward_module, str):
                names.add(forward_module)
    live_positive_refs: list[Callable[[], Any | None]] = []
    hot_ids: set[int] = set()
    for reference in _state._detached_positive_modules:
        positive_module = reference()
        if positive_module is None:
            continue
        live_positive_refs.append(reference)
        hot_ids.add(id(positive_module))
    _state._detached_positive_modules[:] = live_positive_refs
    _state._detached_positive_module_ids.clear()
    _state._detached_positive_module_ids.update(hot_ids)
    for key, module in _distinct_live_modules():
        name = _safe_module_name(module, key)
        if _module_matches_allowlist(name, names):
            hot_ids.add(id(module))
    return hot_ids


def _safe_is_type(value: Any) -> bool:
    """Return ``isinstance(value, type)`` without propagating foreign errors."""

    try:
        return isinstance(value, type)
    except Exception:
        return False


def _safe_is_callable(value: Any) -> bool:
    """Return ``callable(value)`` without propagating foreign errors."""

    try:
        return callable(value)
    except Exception:
        return False


def _patch_class_attributes(cls: type[Any], mapping: dict[int, Any]) -> int:
    """Patch direct raw callable identities in one class dictionary."""

    try:
        cls_dict = vars(cls)
    except TypeError:
        return 0
    patched = 0
    for name, value in list(cls_dict.items()):
        replacement = mapping.get(id(value))
        if replacement is None:
            continue
        try:
            if vars(cls).get(name) is not value:
                continue
            setattr(cls, name, replacement)
        except (AttributeError, TypeError):
            continue
        _record_mutation(cls, "class", name, value, replacement)
        patched += 1
    return patched


def _should_skip_detached_module_key(mod_key: str, policy: DetachedPatchPolicy) -> bool:
    """Return whether a sys.modules key should be skipped before module lookup.

    Parameters
    ----------
    mod_key:
        Key from ``sys.modules``.
    policy:
        Detached-reference patch policy.

    Returns
    -------
    bool
        True if the module key is known not to need detached-reference patching.
    """

    prefixes = _LEGACY_DETACHED_SKIP_PREFIXES
    if policy == "legacy":
        prefixes = prefixes + _KNOWN_TORCH_FREE_PREFIXES
    return mod_key.startswith(prefixes) or ".dist-info" in mod_key


def _should_deep_scan_detached_module(
    mod: types.ModuleType,
    policy: DetachedPatchPolicy,
    *,
    source_open_counter: list[int] | None = None,
) -> bool:
    """Return whether Level 2/3 detached-reference scans should run for a module.

    Parameters
    ----------
    mod:
        Module object being scanned.
    policy:
        Detached-reference patch policy.
    source_open_counter:
        Optional single-item counter incremented for successful legacy source opens.

    Returns
    -------
    bool
        True when class-attribute and function-default introspection should run.
    """

    if policy == "full":
        return True
    if _module_file_is_stdlib(mod):
        return False
    has_torch = _module_source_mentions_torch(mod, source_open_counter=source_open_counter)
    return has_torch is not False


def _safe_module_file(mod: types.ModuleType) -> str | None:
    """Return ``mod.__file__`` as a string without triggering import side effects.

    ``getattr(mod, "__file__", None)`` only suppresses ``AttributeError``, but some
    lazy-import shims (e.g. SpeechBrain's ``LazyModule``) raise ``ImportError`` (or
    other exceptions) from ``__getattr__`` when an optional dependency is missing.
    The ``sys.modules`` crawl in :func:`patch_detached_references` only needs a
    readable file path, so guard broadly and treat any failure as "no file".

    Parameters
    ----------
    mod:
        Module object to inspect.

    Returns
    -------
    str | None
        The module file path when available as a string, else ``None``.
    """

    try:
        mod_file = getattr(mod, "__file__", None)
    except Exception:
        return None
    return mod_file if isinstance(mod_file, str) else None


def _module_file_is_stdlib(mod: types.ModuleType) -> bool:
    """Return whether a module file lives under the Python stdlib directory.

    Parameters
    ----------
    mod:
        Module object to inspect.

    Returns
    -------
    bool
        True when ``mod.__file__`` is inside the configured stdlib paths.
    """

    mod_file = _safe_module_file(mod)
    if not isinstance(mod_file, str):
        return False
    for stdlib_path in _STDLIB_PATHS:
        try:
            if mod_file.startswith(stdlib_path):
                return "site-packages" not in mod_file and "dist-packages" not in mod_file
        except TypeError:
            continue
    return False


def _module_source_mentions_torch(
    mod: types.ModuleType,
    *,
    source_open_counter: list[int] | None = None,
) -> bool | None:
    """Return whether a module's Python source contains ``b"torch"``.

    Parameters
    ----------
    mod:
        Module object to inspect.
    source_open_counter:
        Optional single-item counter incremented after a source file is opened.

    Returns
    -------
    bool | None
        True if readable source contains ``b"torch"``, False if readable
        source does not, and None when no conservative classification is
        possible.
    """

    mod_file = _safe_module_file(mod)
    if not isinstance(mod_file, str) or not mod_file.endswith(".py"):
        return None
    cached = _state._detached_source_has_torch.get(mod_file)
    if cached is not None or mod_file in _state._detached_source_has_torch:
        return cached
    try:
        with open(mod_file, "rb") as source_file:
            if source_open_counter is not None:
                source_open_counter[0] += 1
            has_torch = b"torch" in source_file.read()
    except OSError:
        _state._detached_source_has_torch[mod_file] = None
        return None
    _state._detached_source_has_torch[mod_file] = has_torch
    return has_torch


def clear_patch_detached_references_cache() -> None:
    """Clear caches used by ``patch_detached_references``.

    Returns
    -------
    None
        Cache state is cleared in place.
    """

    _state._crawled_module_keys.clear()
    _state._dir_cache.clear()
    _state._detached_source_has_torch.clear()


def _patch_function_defaults(func: Any, mapping: dict[int, Any]) -> int:
    """Patch ``__defaults__`` and ``__kwdefaults__`` of a function if they contain
    original torch function references.

    This handles the case where a function uses a torch function as a default
    argument value, e.g. ``def f(out=torch.relu)``. The default still
    points to the pre-decoration original; we replace it with the wrapper.

    Returns
    -------
    int
        Number of positional-default tuples and keyword-default slots patched.
    """
    patched = 0
    try:
        defaults = getattr(func, "__defaults__", None)
    except Exception:
        return 0
    if defaults is not None and not isinstance(defaults, tuple):
        return 0
    if defaults is not None:
        new_defaults = []
        changed = False
        for d in defaults:
            if id(d) in mapping:
                new_defaults.append(mapping[id(d)])
                changed = True
            else:
                new_defaults.append(d)
        if changed:
            replacement_defaults = tuple(new_defaults)
            try:
                if getattr(func, "__defaults__", None) is not defaults:
                    return patched
                func.__defaults__ = replacement_defaults
            except (AttributeError, TypeError):
                pass
            else:
                _record_mutation(func, "defaults", None, defaults, replacement_defaults)
                patched += 1

    try:
        kwdefaults = getattr(func, "__kwdefaults__", None)
    except Exception:
        return patched
    if kwdefaults is not None and isinstance(kwdefaults, dict):
        for k, v in list(kwdefaults.items()):
            if id(v) in mapping:
                replacement = mapping[id(v)]
                try:
                    if kwdefaults.get(k) is not v:
                        continue
                    kwdefaults[k] = replacement
                except (TypeError, KeyError):
                    pass
                else:
                    _record_mutation(func, "kwdefault", k, v, replacement)
                    patched += 1
    return patched


def patch_model_instance(model: Any) -> None:
    """Level 4 crawl: patch detached torch function references on a model instance.

    Scans ``vars(model)`` and all submodules for instance attributes that are
    original torch functions and replaces them with decorated versions. This
    catches patterns like ``self.act = torch.relu`` in ``__init__``, where the
    reference was captured before decoration.

    Skips dunder attributes to avoid accidentally replacing internal PyTorch
    machinery (e.g. ``__class__``).
    """
    mapping = _state._orig_to_decorated
    if not mapping:
        return
    for module in model.modules():
        try:
            mod_dict = vars(module)
        except TypeError:
            continue
        for attr_name, attr_val in list(mod_dict.items()):
            if attr_name.startswith("__") or not callable(attr_val):
                continue
            decorated_func = mapping.get(id(attr_val))
            if decorated_func is not None:
                try:
                    if mod_dict.get(attr_name) is not attr_val:
                        continue
                    mod_dict[attr_name] = decorated_func
                except (TypeError, KeyError):
                    pass
                else:
                    _record_mutation(module, "model", attr_name, attr_val, decorated_func)
