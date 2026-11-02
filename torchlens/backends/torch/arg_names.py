"""Argument-name introspection for wrapped torch callables (``_state._arg_names``).

Records, under the EXACT registered name, the parameter names of every function
the torch backend decorates, from ``inspect.signature`` when the callable is
introspectable and from the docstring's first line for C builtins that are not.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ... import _state

__all__ = ["get_arg_names"]


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

    argnames = _arg_names_from_signature(orig_func)
    if argnames is None:
        argnames = _arg_names_from_docstring(orig_func)
    if argnames is not None:
        _state._arg_names[func_name] = argnames


def _arg_names_from_signature(orig_func: Callable[..., Any]) -> tuple[str, ...] | None:
    """Return the names ``inspect.signature`` gives ``orig_func``, or ``None`` to fall back.

    A purely-variadic signature (``(*args, **kwargs)`` on opaque C dunders) names
    nothing; the caller then prefers the docstring parse, and stores nothing when
    that also fails (honest-unknown beats a wrong borrowed signature). A genuinely
    empty signature (zero-arg methods) still stores ``()``.
    """

    try:
        params = inspect.signature(orig_func).parameters
    except (ValueError, TypeError):
        # TypeError: Python 3.14+ deferred annotation evaluation (PEP 649)
        # can fail when class-level names (e.g. Tensor.bool) shadow builtins
        # during inspect.signature() annotation resolution.
        return None
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
    if not argnames or not all(name.startswith("*") for name in argnames):
        return tuple(argnames)
    return None


def _arg_names_from_docstring(orig_func: Callable[..., Any]) -> tuple[str, ...] | None:
    """Parse argument names from the docstring's first line, or ``None`` when absent.

    C builtins typically have docstrings like ``"add(input, other, *, alpha=1)"``.
    """

    docstring = orig_func.__doc__
    if (type(docstring) is not str) or (len(docstring) == 0):
        return None
    paren_start, paren_end = docstring.find("("), docstring.find(")")
    argstring = docstring[paren_start + 1 : paren_end]
    argnames = []
    for arg in (part.strip(" ") for part in argstring.split(",")):
        argname = arg.split("=")[0]
        if argname in ["*", "/", "//", ""]:
            continue
        argnames.append(argname.replace("*", ""))
    return tuple(arg for arg in argnames if arg not in ["self", "cls"])
