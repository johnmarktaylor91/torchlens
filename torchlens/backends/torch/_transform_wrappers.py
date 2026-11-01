"""Transform-boundary wrappers: torch.func builders and autograd.functional direct calls.

A functorch or ``torch.func`` transform (``vmap``, ``grad``, ``jacobian``, ...)
runs its inner operations without batching rules for TorchLens's own
bookkeeping, so TorchLens records the transform as one boundary op and does
not log its interior. This module decorates the transform builders and the
direct-call transforms; ``wrappers.decorate_all_once`` installs them and
``wrappers`` re-exports the names it did before the split.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from functools import partial, wraps
from typing import Any, cast

from ... import _state
from ..._data_classes.internal_types import FuncExecutionContext
from ..._utils._torch_compat import get_optional_torch_namespace
from ..._utils.rng import log_current_autocast_state, log_current_rng_states
from ._callable_location import _callable_source_location
from ._ops_emission import apply_live_hooks_to_outputs, log_function_output_tensors
from ._output_tensors import _collect_output_tensors
from ._tl import mark_decorated_function


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
            save_rng = trace._options_save_rng
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
        save_rng = trace._options_save_rng
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
