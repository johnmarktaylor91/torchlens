"""Function family resolvers: the op fields derived from the function name or the callable.

The ``function`` family of ``torchlens/_trace_core/field_table.py`` holds columns that
depend on nothing but the wrapped callable and its name. Their per-op computation is
replaced by one resolution per name (:func:`name_family`) or per builtin callable
(:func:`callable_family`), stored here and answered by one dict read per op. Each value
is the expression the per-op code evaluated, on the same object:

* ``name_family(func_name)`` is ``(_normalize_func_name(func_name), inplace_named)``,
  where ``inplace_named`` is the test ``_log_output_tensor_info`` applies to the name.
  Both are pure functions of a string.
* ``callable_family(func, func_name)`` is the callable's qualified name and transform
  descriptors. It is kept only for builtin callables not bound to an instance: they
  carry no instance attributes, so every ``__tl_*`` read answers its default for the
  life of the object, and keeping them alive costs nothing (they are module or type
  members). Any other callable is resolved again on every call, as before.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

from ..._capture.arg_positions import _normalize_func_name
from ..._utils._mutation_names import _is_inplace_augmented_assignment_dunder

__all__ = (
    "NAME_FAMILY",
    "CALLABLE_FAMILY",
    "TRANSFORM_CONFIG",
    "NO_CONFIG",
    "name_family",
    "callable_family",
)

TRANSFORM_CONFIG = "__tl_transform_config__"
"""Transform-boundary config attribute; its builder kwargs are snapshotted per op."""

NO_CONFIG = object()
"""Marks "no transform config on this function" (the op records a fresh empty dict)."""

TRANSFORM_FUNC_NAMES = frozenset(
    {
        "vmap",
        "grad",
        "grad_and_value",
        "jacrev",
        "jacfwd",
        "hessian",
        "vjp",
        "jvp",
        "linearize",
        "vjpfn",
        "jvpfn",
        "linearizefn",
    }
)
"""Function names reserved for torch.func transform boundary ops."""

NAME_FAMILY: dict[str, tuple[str, bool]] = {}
"""``func_name -> (layer_type, inplace_named)``; read with ``NAME_FAMILY.get(name)``."""

CALLABLE_FAMILY: dict[str, tuple[Any, ...]] = {}
"""``func_name -> (func, *callable values)`` for unbound builtin callables only."""

_BUILTIN_CALLABLE_TYPES = (
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)


def name_family(func_name: str) -> tuple[str, bool]:
    """Resolve and keep ``(layer_type, inplace_named)`` for ``func_name``."""

    inplace_named = _is_inplace_augmented_assignment_dunder(func_name) or (
        func_name.endswith("_") and not func_name.startswith("__")
    )
    family = NAME_FAMILY[func_name] = (_normalize_func_name(func_name), inplace_named)
    return family


def callable_family(func: Callable[..., Any], func_name: str) -> tuple[Any, ...]:
    """Resolve the callable's values: ``(func, qualname, is_transform, kind, chain,
    fn_name, fn_qualname, fn_source, config)``, with the per-op expressions.

    Kept in :data:`CALLABLE_FAMILY` only for an unbound builtin callable (its ``config``
    is then always :data:`NO_CONFIG`); the caller checks ``family[0] is func`` on a hit.
    """

    is_transform = bool(getattr(func, "__tl_is_transform_boundary__", False)) or (
        func_name in TRANSFORM_FUNC_NAMES
    )
    family = (
        func,
        getattr(func, "__qualname__", None),
        is_transform,
        getattr(func, "__tl_transform_kind__", None) or (func_name if is_transform else None),
        tuple(getattr(func, "__tl_transform_tags__", ())),
        getattr(func, "__tl_transform_fn_name__", None),
        getattr(func, "__tl_transform_fn_qualname__", None),
        getattr(func, "__tl_transform_fn_source__", None),
        getattr(func, TRANSFORM_CONFIG, NO_CONFIG),
    )
    if type(func) in _BUILTIN_CALLABLE_TYPES and _unbound(func) and family[8] is NO_CONFIG:
        CALLABLE_FAMILY[func_name] = family
    return family


def _unbound(func: Callable[..., Any]) -> bool:
    """Return whether a builtin callable holds no instance (a module or no ``__self__``)."""

    owner = getattr(func, "__self__", None)
    return owner is None or isinstance(owner, (types.ModuleType, type))
