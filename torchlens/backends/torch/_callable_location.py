"""Lazy source location of the user callable a transform boundary op wraps.

Split out of ``wrappers.py`` for its file-size ratchet, logic unchanged: the
``FuncCallLocation`` recorded as ``__tl_transform_fn_source__`` by the transform-boundary
metadata (``torch.func`` builders, ``autograd.functional``, ``autograd.grad``, the
foreign-kernel ``autograd.Function`` and the TorchScript call). ``wrappers`` re-imports it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from ..._data_classes.func_call_location import FuncCallLocation

__all__ = ("_callable_source_location",)


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
