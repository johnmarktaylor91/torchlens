"""The per-capture scan for Dynamo-compiled callables held as plain module attributes.

Every capture reads every module's instance dictionary for a compiled callable
(a ``torch.compile`` result assigned to an attribute), because a hot compiled
callable can bypass the Python torch wrappers entirely. The scan is a
trace-time observation and stays one: nothing is remembered between captures,
so a callable compiled, assigned or removed between two traces is seen by the
next one. What shrinks is the work per attribute: a value whose exact type
cannot carry the compile marker is skipped on a type test instead of two
function calls.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from torch import nn

from ..._plain_attr_fidelity import _PLAIN_ATTR_IGNORED_NAMES
from ..._utils._model_walk import model_named_modules
from ..._utils._torch_compat import is_dynamo_compiled_callable

#: Exact builtin types whose instances have no ``__dict__``: they can hold no
#: ``_torchdynamo_orig_callable`` marker and are never an ``nn.Module``.
_MARKERLESS_TYPES = frozenset(
    {type(None), bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset}
)


def cannot_carry_compile_marker(value: Any) -> bool:
    """Return whether ``value``'s exact type rules out Dynamo's compile marker.

    True for the exact builtin types with no instance ``__dict__`` and for an
    exact ``OrderedDict`` (every ``nn.Module`` owns about ten, its hook
    registries) whose instance ``__dict__`` is empty. ``type()`` never runs
    the value's own attribute hooks. A ``True`` answer is exactly the case in
    which ``isinstance(value, nn.Module)`` and
    :func:`~torchlens._utils._torch_compat.is_dynamo_compiled_callable` are
    both ``False``; anything else gets ``False`` and takes the full test.

    Parameters
    ----------
    value
        One value of a module's instance dictionary.

    Returns
    -------
    bool
        Whether the value can be skipped without the marker lookup.
    """

    value_type = type(value)
    if value_type in _MARKERLESS_TYPES:
        return True
    return value_type is OrderedDict and not value.__dict__


def compiled_plain_callable_sites(model: nn.Module) -> tuple[str, ...]:
    """Return plain module attributes carrying Dynamo-compiled callables.

    Parameters
    ----------
    model:
        Model whose module instance dictionaries are inspected.

    Returns
    -------
    tuple[str, ...]
        Stable module-attribute paths for compiled non-module callables.

    Notes
    -----
    The direct ``vars`` walk never executes descriptors. This inventory is what
    makes cache-hit handling independent of the transient ``is_compiling()`` flag:
    a hot compiled callable can bypass Python torch wrappers entirely.
    """
    sites: list[str] = []
    try:
        modules = model_named_modules(model)
    except Exception:
        modules = (("", model),)
    for module_name, module in modules:
        try:
            attributes = vars(module)
        except (TypeError, AttributeError):
            continue
        for attr_name, value in attributes.items():
            value_type = type(value)
            if value_type in _MARKERLESS_TYPES or (
                value_type is OrderedDict and not value.__dict__
            ):
                continue  # ``cannot_carry_compile_marker``, inline: this loop runs per attribute
            if attr_name in _PLAIN_ATTR_IGNORED_NAMES or isinstance(value, nn.Module):
                continue
            if is_dynamo_compiled_callable(value):
                prefix = module_name or "<root>"
                sites.append(f"{prefix}.{attr_name}")
    return tuple(dict.fromkeys(sites))
