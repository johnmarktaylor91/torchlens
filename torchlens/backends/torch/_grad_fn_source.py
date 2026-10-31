"""Best-effort source metadata for autograd grad-fn classes.

Introspection helpers that ``backward.py`` uses when it builds a ``GradFn``:
whether a grad-fn looks user-defined, and the source file, line, signature and
docstring of its class and its ``__init__``/``forward``/``backward`` methods
(split out of ``backward.py`` under its size ratchet).
"""

from __future__ import annotations

import inspect
from typing import Any


def _grad_fn_is_custom(grad_fn_handle: Any) -> bool:
    """Return whether a grad_fn_handle appears to come from user/custom autograd code.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    bool
        True when the type's module path is outside PyTorch's built-in autograd
        and nn namespaces.
    """
    class_module = type(grad_fn_handle).__module__
    if class_module == "torch.autograd.function":
        return True
    builtin_prefixes = ("torch.autograd", "torch.nn", "torch", "builtins")
    return not class_module.startswith(builtin_prefixes)


def _safe_source_file_and_line(obj: Any) -> tuple[str | None, int | None]:
    """Return source file and first line for an object when introspection works.

    Parameters
    ----------
    obj:
        Object to inspect.

    Returns
    -------
    tuple[str | None, int | None]
        Source file and line, or ``(None, None)`` when unavailable.
    """

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None, None
    return source_file, source_line


def _safe_signature(obj: Any) -> str | None:
    """Return an inspect signature string when available.

    Parameters
    ----------
    obj:
        Callable object to inspect.

    Returns
    -------
    str | None
        Signature string, or ``None`` when unavailable.
    """

    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _grad_fn_source_metadata(grad_fn_cls: type[Any]) -> dict[str, Any]:
    """Return best-effort source metadata for a grad-fn class.

    Parameters
    ----------
    grad_fn_cls:
        Runtime class of the autograd grad-fn object.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``GradFn`` for source metadata fields.
    """

    class_source_file, class_source_line = _safe_source_file_and_line(grad_fn_cls)
    init_method = getattr(grad_fn_cls, "__init__", None)
    forward_method = getattr(grad_fn_cls, "forward", None)
    backward_method = getattr(grad_fn_cls, "backward", None)
    init_source_file, init_source_line = _safe_source_file_and_line(init_method)
    forward_source_file, forward_source_line = _safe_source_file_and_line(forward_method)
    backward_source_file, backward_source_line = _safe_source_file_and_line(backward_method)
    return {
        "class_source_file": class_source_file,
        "class_source_line": class_source_line,
        "class_docstring": grad_fn_cls.__doc__,
        "init_source_file": init_source_file,
        "init_source_line": init_source_line,
        "init_signature": _safe_signature(init_method),
        "init_docstring": getattr(init_method, "__doc__", None),
        "forward_source_file": forward_source_file,
        "forward_source_line": forward_source_line,
        "forward_signature": _safe_signature(forward_method),
        "forward_docstring": getattr(forward_method, "__doc__", None),
        "backward_source_file": backward_source_file,
        "backward_source_line": backward_source_line,
        "backward_signature": _safe_signature(backward_method),
        "backward_docstring": getattr(backward_method, "__doc__", None),
    }
