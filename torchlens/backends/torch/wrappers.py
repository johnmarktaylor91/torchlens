"""Lazy torch function wrapping with explicit wrap/unwrap lifecycle.

This module implements the core interception mechanism for TorchLens.  Torch
functions are wrapped **lazily** — ``import torchlens`` has no side effects.
Wrapping happens on the first call to ``trace()`` (or any other
entry point that needs logging) and stays in place afterward.  Users can
explicitly control the lifecycle via ``wrap_torch()`` / ``unwrap_torch()`` /
``wrapped()``.

Every function listed in ``ORIG_TORCH_FUNCS`` is replaced with a thin wrapper
that checks a single boolean (``_state._logging_enabled``) on each call:

  - **Logging off** (default): one branch check, near-zero overhead, original function called.
  - **Logging on**: all tensor outputs are captured into the active ``Trace``.

Key design decisions:

1. **Lazy wrapping** — torch is clean after ``import torchlens``.  Wrapping is
   triggered automatically on first use and persists until ``unwrap_torch()`` is
   called.  The toggle makes persistent wrapping safe for production use.

2. **Shared originals reuse wrappers**: If ``torch.cos`` and ``torch._VF.cos`` point to the
   same C builtin, only one wrapper is created and both namespaces point to it. This keeps
   ``_orig_to_decorated`` / JIT builtin table mappings 1:1.

3. **sys.modules crawl** (``patch_detached_references``): Code like ``from torch import cos``
   captures a reference to the *original* function before decoration. The crawl replaces
   these stale references in module dicts, class dicts, and function defaults.

4. **JIT compatibility**: Decorated wrappers are registered in ``torch.jit._builtins._builtin_table``
   so ``torch.jit.script`` recognizes them as known ATen ops.

5. **DeviceContext bypass**: Python wrappers bypass PyTorch's C-level ``TorchFunctionMode``
   dispatch, so ``torch.device('meta')`` context managers won't inject ``device`` kwargs
   automatically. We detect active ``DeviceContext`` and inject the kwarg ourselves.
"""

import ctypes
import inspect
import sys
import time
import types
import weakref
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, TYPE_CHECKING, cast

import torch

from ... import _state
from ...constants import ORIG_TORCH_FUNCS
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
from .buffer_writes import record_op_buffer_writes, snapshot_buffer_args
from .sources import log_source_tensor

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


_PATCHED_MODEL_INSTANCES: weakref.WeakSet[Any] = weakref.WeakSet()


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


#
# When __getitem__ is replaced on a C extension type (like torch.Tensor) with
# a Python function, CPython sets the sq_item slot in tp_as_sequence.  This
# makes PySequence_Check(tensor) return True, which causes torch.tensor() to
# try iterating 0-d tensor elements as sequences -- calling len() which raises
# TypeError.  The sq_item slot is NEVER cleared by restoring the original
# wrapper_descriptor or by delattr, because CPython's update_one_slot only
# restores the exact slot the wrapper_descriptor wraps (mp_subscript), not
# the collateral sq_item slot.
#
# We fix this by nulling sq_item directly via ctypes after any decoration or
# undecoration cycle.  This is safe because tensor indexing uses mp_subscript
# (mapping protocol), not sq_item (sequence protocol).


class _PySequenceMethods(ctypes.Structure):
    """Minimal ctypes mirror of CPython's PySequenceMethods struct."""

    _fields_ = [
        ("sq_length", ctypes.c_void_p),
        ("sq_concat", ctypes.c_void_p),
        ("sq_repeat", ctypes.c_void_p),
        ("sq_item", ctypes.c_void_p),
        ("was_sq_slice", ctypes.c_void_p),
        ("sq_ass_item", ctypes.c_void_p),
        ("was_sq_ass_slice", ctypes.c_void_p),
        ("sq_contains", ctypes.c_void_p),
        ("sq_inplace_concat", ctypes.c_void_p),
        ("sq_inplace_repeat", ctypes.c_void_p),
    ]


class _PyTypeObject(ctypes.Structure):
    """Partial ctypes mirror of CPython's PyTypeObject up to tp_as_sequence.

    Layout is stable across CPython 3.8+ (tp_vectorcall_offset replaced
    tp_print in 3.8; all earlier fields are pointer-sized regardless).
    """

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ssize_t),
        ("tp_name", ctypes.c_char_p),
        ("tp_basicsize", ctypes.c_ssize_t),
        ("tp_itemsize", ctypes.c_ssize_t),
        ("tp_dealloc", ctypes.c_void_p),
        ("tp_vectorcall_offset", ctypes.c_ssize_t),
        ("tp_getattr", ctypes.c_void_p),
        ("tp_setattr", ctypes.c_void_p),
        ("tp_as_async", ctypes.c_void_p),
        ("tp_repr", ctypes.c_void_p),
        ("tp_as_number", ctypes.c_void_p),
        ("tp_as_sequence", ctypes.POINTER(_PySequenceMethods)),
        ("tp_as_mapping", ctypes.c_void_p),
    ]


def _fix_tensor_sequence_slot() -> None:
    """Clear the stale sq_item C slot on torch.Tensor after dunder changes.

    Wrapping ``__getitem__`` on a C extension type pollutes the ``sq_item``
    slot in ``tp_as_sequence``, making ``PySequence_Check(tensor)`` return
    ``True``.  This breaks ``torch.tensor([0-d_tensor, ...])`` because the
    C code then calls ``len()`` on each element.  Clearing ``sq_item`` to
    NULL restores the clean-state behavior where tensors are NOT treated as
    sequences.  Tensor indexing is unaffected because it goes through
    ``mp_subscript`` (mapping protocol).

    Safe to call multiple times.  Fails silently on non-CPython or if the
    struct layout doesn't match (verified via ``tp_name``).
    """
    if sys.implementation.name != "cpython":
        return
    try:
        type_obj = _PyTypeObject.from_address(id(torch.Tensor))
        # Verify struct layout by checking tp_name
        if type_obj.tp_name != b"Tensor":
            return
        if type_obj.tp_as_sequence:
            type_obj.tp_as_sequence.contents.sq_item = None
    except Exception:
        pass  # Best-effort; non-CPython or unexpected layout


def _is_inside_functorch_transform() -> bool:
    """Return True if inside a vmap/grad/etc. functorch transform."""
    try:
        from torch._C._functorch import maybe_current_level

        return maybe_current_level() is not None
    except (ImportError, AttributeError):
        return False


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
            if not _state._logging_enabled or _state._active_trace is None:
                return raw_built(*call_args, **call_kwargs)

            trace = cast(Any, _state._active_trace)
            capture_start_time = time.time()
            save_rng = getattr(trace, "save_rng_states", False)
            rng_states = log_current_rng_states(torch_only=True) if save_rng else {}
            autocast_state = log_current_autocast_state()
            func_call_id = _state.next_func_call_id()
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
        namespace_key = namespace_name.removeprefix("torch.")
        namespace = torch if namespace_name == "torch" else nested_getattr(torch, namespace_key)
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


def _get_active_device() -> str | None:
    """Return the device from the innermost active ``DeviceContext``, or ``None``.

    Walks the ``TorchFunctionMode`` stack in reverse (innermost first) to find
    the most recently pushed ``DeviceContext``. This is the same device that
    PyTorch's C dispatch would inject — but since our Python wrappers bypass
    C dispatch, we must query it manually.
    """
    global _DeviceContext
    if _DeviceContext is None:
        from torch.utils._device import DeviceContext

        _DeviceContext = DeviceContext
    try:
        from torch.overrides import _get_current_function_mode_stack

        for mode in reversed(_get_current_function_mode_stack()):  # type: ignore[no-untyped-call]
            if isinstance(mode, _DeviceContext):
                return str(mode.device)
    except (ImportError, AttributeError):
        pass
    return None


def _maybe_inject_device_kwarg(func_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inject ``device`` kwarg for factory functions when a ``DeviceContext`` is active.

    Python wrappers bypass PyTorch's C-level ``TorchFunctionMode`` dispatch, so
    ``torch.device('meta')`` context (used by e.g. HuggingFace ``from_pretrained``)
    won't inject the device kwarg automatically. We replicate that injection here.

    Only applies to known factory functions (``torch.zeros``, ``torch.ones``, etc.)
    whose names were collected into ``_DEVICE_CONSTRUCTOR_NAMES`` at decoration time.
    """
    # Early exit: not a factory function, or caller already specified device
    if not (_DEVICE_CONSTRUCTOR_NAMES and func_name in _DEVICE_CONSTRUCTOR_NAMES):
        return kwargs
    if "device" in kwargs:
        return kwargs
    try:
        from torch.overrides import _len_torch_function_stack  # type: ignore[attr-defined]

        if _len_torch_function_stack() > 0:
            device = _get_active_device()
            if device is not None:
                return {**kwargs, "device": device}
    except (ImportError, AttributeError):
        pass
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
                import warnings

                warnings.warn(
                    "TorchLens detected a functorch/vmap/grad/jacfwd transform "
                    "during this forward pass. Operations that run inside the "
                    "transform are not logged. The returned Trace will only "
                    "contain operations that ran OUTSIDE the transform.",
                    UserWarning,
                    stacklevel=2,
                )
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

        buffer_snapshots = snapshot_buffer_args(trace, func_name, arg_tensorlike)

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
        nvtx_pushed = (
            _nvtx_range_push(f"torchlens::{func_name}")
            if getattr(trace, "emit_nvtx", False)
            else False
        )
        try:
            out_orig = func(*args, **kwargs)
        finally:
            _nvtx_range_pop(nvtx_pushed)
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
        # True in-place ops (add_, mul_, etc.) modify the tensor and return self.
        # No-op functions (to(same_dtype), contiguous() on contiguous tensor)
        # also return self but don't modify anything.
        # Both cases need safe_copy so logging doesn't overwrite the original's
        # label, but only true in-place ops should propagate the new label back.
        was_inplace = same_object_returned and (
            func_name.endswith("_") or func_name.startswith("__i")
        )
        if same_object_returned:
            # Create a distinct tensor object for logging — otherwise attaching
            # _tl.label_raw on the output would clobber the input's label.
            out_orig = safe_copy(out_orig)

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
        )

        # Log all output tensors (excluding Parameters, which are source tensors).
        # Fast inline check for the common single-tensor output case.
        if getattr(trace, "intervention_ready", False):
            output_tensors = [entry[0] for entry in _walk_output_tensors_with_paths(out_orig)]
        else:
            output_tensors = _collect_output_tensors(out_orig)

        if len(output_tensors) > 0:
            log_function_output_tensors(
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

            # For true in-place ops, propagate the newly assigned label back
            # to the original tensor so subsequent operations see it.
            if was_inplace and not isinstance(args[0], torch.nn.Parameter):
                out_label = get_tensor_label(out_orig)
                if out_label is not None:
                    set_tensor_label(args[0], out_label)

        producer_label = None
        if isinstance(out_orig, torch.Tensor):
            producer_label = get_tensor_label(out_orig)
        elif output_tensors:
            producer_label = get_tensor_label(output_tensors[0])
        if is_bottom_level_func:
            record_op_buffer_writes(trace, func_name, buffer_snapshots, producer_label)

        return out_orig

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

    try:
        import torch.jit._builtins as _jit_builtins

        builtin_table = cast(Any, _jit_builtins._builtin_table)
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
    except (ImportError, AttributeError):
        pass  # JIT internals may change across PyTorch versions


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
    for namespace_name, func_name in ORIG_TORCH_FUNCS:
        if func_name.strip("_") in _state._arg_names:
            continue
        namespace_key = namespace_name.replace("torch.", "")
        local_func_namespace = nested_getattr(torch, namespace_key)
        if not hasattr(local_func_namespace, func_name):
            continue
        orig_func = getattr(local_func_namespace, func_name)
        get_arg_names(orig_func, func_name)

    # --- Pass 2: Decorate all functions ---
    for namespace_name, func_name in ORIG_TORCH_FUNCS:
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

            new_func = torch_func_decorator(orig_func, func_name)
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
    try:
        from torch.utils._device import _device_constructors

        _device_constructors.cache_clear()
        for ctor in _device_constructors():
            name = getattr(ctor, "__name__", None)
            if name:
                _DEVICE_CONSTRUCTOR_NAMES.add(name)
    except (ImportError, AttributeError):
        pass

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


def _replace_detached_references(mapping: dict[int, Any]) -> None:
    """Crawl ``sys.modules`` and replace callable references using ``mapping``.

    ``mapping`` may be either original->decorated or decorated->original. This
    keeps the sys.modules crawl logic symmetric so ``undecorate`` can reverse
    the same module/class/default-arg patching done during normal decoration.
    """
    if not mapping:
        return

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        if hasattr(mod, "__name__") and getattr(mod, "__name__", "").startswith("torchlens"):
            continue

        try:
            mod_dict = vars(mod)
        except TypeError:
            continue

        for attr_name, attr_val in list(mod_dict.items()):
            if id(attr_val) in mapping:
                try:
                    mod_dict[attr_name] = mapping[id(attr_val)]
                except (TypeError, KeyError):
                    pass
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    is_type = isinstance(attr_val, type)
            except Exception:
                is_type = False
            if is_type:
                try:
                    cls_dict = vars(attr_val)
                except TypeError:
                    continue
                for cls_attr_name, cls_attr_val in list(cls_dict.items()):
                    if id(cls_attr_val) in mapping:
                        try:
                            setattr(attr_val, cls_attr_name, mapping[id(cls_attr_val)])
                        except (AttributeError, TypeError):
                            pass

            try:
                is_callable = callable(attr_val) and not is_type
            except Exception:
                is_callable = False
            if is_callable:
                _patch_function_defaults(attr_val, mapping)


def unwrap_torch() -> None:
    """Remove torchlens wrappers and restore original torch callables.

    After calling this, ``torch.cos``, ``torch.Tensor.__add__``, etc. are the
    originals shipped by PyTorch.  TorchLens logging will not work until
    ``wrap_torch()`` is called (or ``trace`` auto-wraps).

    Safe to call multiple times — no-op if already unwrapped.
    """
    _state._logging_enabled = False
    _state._active_trace = None
    from .backward import uninstall_autograd_wrappers

    uninstall_autograd_wrappers()

    if not _state._decorated_to_orig:
        _state._is_decorated = False
        _PATCHED_MODEL_INSTANCES.clear()
        return

    for namespace_name, func_name in ORIG_TORCH_FUNCS:
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

    _replace_detached_references(_state._decorated_to_orig)
    _state._is_decorated = False
    _PATCHED_MODEL_INSTANCES.clear()

    # Restoring Tensor.__getitem__ doesn't clear the stale sq_item slot.
    _fix_tensor_sequence_slot()


def wrap_torch() -> None:
    """Install (or re-install) torchlens wrappers on all torch functions.

    If this is the first call, performs full decoration (equivalent to
    ``decorate_all_once`` + ``patch_detached_references``).  If wrappers were
    previously removed via ``unwrap_torch()``, re-installs them from the
    cached maps without re-creating wrapper objects.

    Safe to call multiple times — no-op if already wrapped.
    """
    from .backward import install_autograd_wrappers

    if _state._is_decorated:
        install_autograd_wrappers()
        return

    if not _state._orig_to_decorated:
        # First time: full decoration
        decorate_all_once()
        install_autograd_wrappers()
        patch_detached_references()
        return

    # Re-install from existing maps (after a prior unwrap_torch)
    for namespace_name, func_name in ORIG_TORCH_FUNCS:
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

    _replace_detached_references(_state._orig_to_decorated)
    # Recreate decorated identity in case wrapper references shifted
    _state._decorated_identity = torch_func_decorator(identity, "identity")
    _state._is_decorated = True
    install_autograd_wrappers()
    patch_detached_references()

    # Re-wrapping __getitem__ pollutes sq_item again; clear it.
    _fix_tensor_sequence_slot()


@contextmanager
def wrapped() -> Iterator[None]:
    """Context manager: wrap torch on entry, unwrap on exit.

    Usage::

        with torchlens.wrapped():
            log = torchlens.trace(model, x)
        # torch is clean again here
    """
    wrap_torch()
    try:
        yield
    finally:
        unwrap_torch()


# Keep old names as private aliases for internal use
undecorate_all_globally = unwrap_torch
redecorate_all_globally = wrap_torch


# ---------------------------------------------------------------------------
# sys.modules deep crawl
# ---------------------------------------------------------------------------


def patch_detached_references() -> None:
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

    Incremental: uses ``_state._crawled_module_keys`` to skip already-scanned
    modules. Called on every ``trace`` to catch lazily-imported modules.
    """
    new_keys = set(sys.modules.keys()) - _state._crawled_module_keys
    if not new_keys:
        return

    mapping = _state._orig_to_decorated
    if not mapping:
        return

    crawled_class_ids: set[int] = set()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod_key in new_keys:
            _state._crawled_module_keys.add(mod_key)
            if (
                mod_key.startswith(("torch.", "numpy.", "pytest", "pluggy", "setuptools"))
                or ".dist-info" in mod_key
            ):
                continue
            mod = sys.modules.get(mod_key)
            if mod is None:
                continue
            # Skip torchlens internals — we don't want to patch our own references.
            if hasattr(mod, "__name__") and getattr(mod, "__name__", "").startswith("torchlens"):
                continue

            try:
                mod_dict = vars(mod)
            except TypeError:
                continue

            for attr_name, attr_val in list(mod_dict.items()):
                # Level 1: module-level references (e.g. ``from torch import cos``)
                if id(attr_val) in mapping:
                    try:
                        mod_dict[attr_name] = mapping[id(attr_val)]
                    except (TypeError, KeyError):
                        pass
                    continue

                # Level 2: class-level attributes that hold torch function references
                attr_type = type(attr_val)
                if issubclass(attr_type, type):
                    try:
                        is_type = isinstance(attr_val, type)
                    except Exception:
                        is_type = False
                else:
                    is_type = False
                if is_type:
                    cls_id = id(attr_val)
                    if cls_id in crawled_class_ids:
                        continue
                    crawled_class_ids.add(cls_id)
                    try:
                        cls_dict = vars(attr_val)
                    except TypeError:
                        continue
                    for cls_attr_name, cls_attr_val in list(cls_dict.items()):
                        if id(cls_attr_val) in mapping:
                            try:
                                setattr(attr_val, cls_attr_name, mapping[id(cls_attr_val)])
                            except (AttributeError, TypeError):
                                pass

                # Level 3: function default arguments (e.g. ``def f(act=torch.relu)``)
                try:
                    is_callable = callable(attr_val) and not is_type
                except Exception:
                    is_callable = False
                if is_callable:
                    _patch_function_defaults(attr_val, mapping)


def clear_patch_detached_references_cache() -> None:
    """Clear caches used by ``patch_detached_references``.

    Returns
    -------
    None
        Cache state is cleared in place.
    """

    _state._crawled_module_keys.clear()
    _state._dir_cache.clear()


def _patch_function_defaults(func: Any, mapping: dict[int, Any]) -> None:
    """Patch ``__defaults__`` and ``__kwdefaults__`` of a function if they contain
    original torch function references.

    This handles the case where a function uses a torch function as a default
    argument value, e.g. ``def f(out=torch.relu)``. The default still
    points to the pre-decoration original; we replace it with the wrapper.
    """
    try:
        defaults = getattr(func, "__defaults__", None)
    except Exception:
        return
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
            try:
                func.__defaults__ = tuple(new_defaults)
            except (AttributeError, TypeError):
                pass

    try:
        kwdefaults = getattr(func, "__kwdefaults__", None)
    except Exception:
        return
    if kwdefaults is not None and isinstance(kwdefaults, dict):
        for k, v in list(kwdefaults.items()):
            if id(v) in mapping:
                try:
                    kwdefaults[k] = mapping[id(v)]
                except (TypeError, KeyError):
                    pass


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
    try:
        if model in _PATCHED_MODEL_INSTANCES:
            return
    except TypeError:
        pass

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
                    mod_dict[attr_name] = decorated_func
                except (TypeError, KeyError):
                    pass
    try:
        _PATCHED_MODEL_INSTANCES.add(model)
    except TypeError:
        pass
