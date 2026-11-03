"""Public-spelling ``repr`` for ``tl.Edit`` (the ``Edit`` dataclass).

A builtin edit prints as the public constructor call that builds it, so the
printed text is the call a user would type (``tl.clip_norm(1.0)``). Edits that
no public constructor reproduces from their stored arguments print as
``tl.Edit(...)`` with their identity fields.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from .._utils._display_constructor import clipped, value_text
from .._utils.fail_open import fail_open

if TYPE_CHECKING:
    from .types import Edit

#: Public namespaces searched for a builtin constructor, in order, with the
#: prefix each one prints under (``import torchlens as tl``).
_PUBLIC_NAMESPACES: tuple[tuple[str, str], ...] = (
    ("torchlens", "tl"),
    ("torchlens.intervention", "tl.intervention"),
)


def render_edit_repr(spec: Edit) -> str:
    """Return the public spelling of one edit.

    Parameters
    ----------
    spec:
        Edit to render.

    Returns
    -------
    str
        ``tl.<constructor>(args)`` for a builtin edit, else ``tl.Edit(...)``.
    """

    return fail_open(lambda: clipped(_edit_repr(spec)), lambda _error: "tl.Edit(...)")


def _edit_repr(spec: Edit) -> str:
    """Render known public constructors or the bounded identity fallback."""

    public = _public_constructor(spec)
    if public is not None:
        path, target = public
        parts = [_render_value(value) for value in spec.args]
        order = list(_signature(target).parameters)
        keyed = sorted(
            spec.kwargs, key=lambda item: order.index(item[0]) if item[0] in order else len(order)
        )
        parts.extend(
            f"{key}={_render_value(value)}"
            for key, value in keyed
            if not _is_signature_default(target, key, value)
        )
        return f"{path}({', '.join(parts)})"
    fields = [repr(spec.name)]
    if spec.portability != "builtin":
        fields.append(f"portability={spec.portability!r}")
    if spec.signal is not None:
        fields.append(f"signal={spec.signal!r}")
    if spec.args:
        fields.append(f"args=({', '.join(_render_value(value) for value in spec.args)},)")
    if spec.kwargs:
        rendered = ", ".join(f"({key!r}, {_render_value(value)})" for key, value in spec.kwargs)
        fields.append(f"kwargs=({rendered},)")
    return f"tl.Edit({', '.join(fields)})"


def _public_constructor(spec: Edit) -> tuple[str, Callable[..., Any]] | None:
    """Return the public constructor path that rebuilds ``spec``, if any.

    Only builtin edits qualify: their stored ``(args, kwargs)`` are exactly the
    constructor arguments the saved-spec loader re-invokes. A selection recipe
    is not a constructor argument, so an edit carrying one never qualifies.

    Parameters
    ----------
    spec:
        Edit to inspect.

    Returns
    -------
    tuple[str, Callable[..., Any]] | None
        Dotted public path such as ``"tl.clip_norm"`` and the constructor, or
        ``None``.
    """

    if spec.portability != "builtin" or spec.selection_recipe is not None:
        return None
    from .helpers import builtin_helper_constructor

    target = builtin_helper_constructor(spec.name)
    if target is None:
        return None
    import importlib

    for module_name, prefix in _PUBLIC_NAMESPACES:
        namespace = importlib.import_module(module_name)
        try:
            candidate = getattr(namespace, spec.name)
        except (AttributeError, ImportError):
            # Lazy facades raise (typed) AttributeError for absent or
            # redirected names and ImportError for gated extras.
            continue
        if candidate is target:
            return f"{prefix}.{spec.name}", target
    return None


def _is_signature_default(target: Callable[..., Any], key: str, value: Any) -> bool:
    """Return whether ``value`` equals the constructor's default for ``key``.

    Parameters
    ----------
    target:
        Public constructor.
    key:
        Keyword argument name.
    value:
        Stored keyword value.

    Returns
    -------
    bool
        ``True`` when printing the keyword would only restate its default.
    """

    parameter = _signature(target).parameters.get(key)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return False
    default = parameter.default
    if isinstance(value, torch.Tensor) or isinstance(default, torch.Tensor):
        return False
    return type(default) is type(value) and bool(default == value)


@lru_cache(maxsize=128)
def _signature(target: Callable[..., Any]) -> inspect.Signature:
    """Inspect each helper signature once, including repeated default lookups."""

    return inspect.signature(target)


_render_value = value_text
