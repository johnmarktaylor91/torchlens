"""Transform-boundary metadata shared by every TorchLens boundary op.

Split out of ``wrappers.py`` along its boundary seam for the file-size ratchet, logic
unchanged: the transform tags a callable carries, its code location, the serializable
builder config of a ``torch.func`` transform, and the ``__tl_transform_*__`` metadata that
marks a callable as a transform boundary (``torch.func`` builders, ``autograd.functional``,
``autograd.grad``, the foreign-kernel ``autograd.Function`` and the TorchScript call).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from ...data_classes.func_call_location import FuncCallLocation

__all__ = (
    "_callable_code_location",
    "_callable_source_location",
    "_set_transform_metadata",
    "_transform_builder_config",
    "_transform_tags",
)


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
