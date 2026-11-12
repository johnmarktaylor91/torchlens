"""Descriptive family of the shared op fields: names and transform descriptors of the callable.

One of the per-family modules split out of ``_ops_shared_fields`` along the
families of ``torchlens/_trace_core/field_table.py``: the ``function`` columns
derived from the callable alone (qualified name, argument names, transform
descriptors), so a lane moving output facts and strings to first read edits this
module alone.
"""

from collections.abc import Callable
from typing import Any

from ... import _state as _st
from ..._capture._arg_snapshot import snapshot_mutable_arg
from ._ops_function_family import (
    CALLABLE_FAMILY,
    NO_CONFIG,
    TRANSFORM_FUNC_NAMES,
    callable_family,
)

__all__ = (
    "DESCRIPTIVE_FIELDS",
    "TRANSFORM_FUNC_NAMES",
    "_set_descriptive_fields",
    "observe_descriptive",
    "resolve_descriptive",
)

#: The fields :func:`resolve_descriptive` returns, in the order the record writes them.
DESCRIPTIVE_FIELDS = (
    "func_qualname",
    "arg_names",
    "is_transform",
    "transform_kind",
    "transform_chain",
    "transform_config",
    "transform_fn_name",
    "transform_fn_qualname",
    "transform_fn_source",
)


def observe_descriptive(func: Callable[..., Any], func_name: str) -> tuple[Any, ...]:
    """Return what the call must record: the callable's function family, config copied.

    The family is the per-function constant part (``_ops_function_family``: kept per
    unbound builtin callable, otherwise resolved on this call from the callable's
    attributes). A transform config is a caller-owned container, so it is copied here,
    at the call, and the copy is the one :func:`resolve_descriptive` stores.
    """

    family = CALLABLE_FAMILY.get(func_name)
    if family is None or family[0] is not func:
        family = callable_family(func, func_name)
        if family[8] is not NO_CONFIG:
            family = (*family[:8], snapshot_mutable_arg(family[8]))
    return family


def resolve_descriptive(family: tuple[Any, ...], func_name: str) -> tuple[Any, ...]:
    """Return the :data:`DESCRIPTIVE_FIELDS` values of one op from its recorded family.

    ``arg_names`` is read from the decoration registry, the exact-name lookup first
    (add / add_ / __add__ have distinct signatures). The underscore-stripped key remains
    a best-effort fallback for NON-dunder names whose own introspection stored nothing;
    a dunder whose signature is opaque stays honestly unknown (empty) rather than
    borrowing the namesake torch function's different signature.
    """

    _, qualname, is_transform, kind, chain, fn_name, fn_qualname, fn_source, config = family
    arg_names = _st._arg_names.get(func_name)
    if arg_names is None:
        arg_names = (
            () if func_name.startswith("__") else _st._arg_names.get(func_name.strip("_"), ())
        )
    return (
        qualname,
        arg_names,
        is_transform,
        kind,
        chain,
        {} if config is NO_CONFIG else config,
        fn_name,
        fn_qualname,
        fn_source,
    )


def _set_descriptive_fields(
    fields_dict: dict[str, Any], func: Callable[..., Any], func_name: str
) -> None:
    """Record the callable's qualified name, argument names and transform descriptors.

    The values are :func:`resolve_descriptive` over :func:`observe_descriptive`; keys
    keep the order the per-op code wrote them in.
    """

    (
        fields_dict["func_qualname"],
        fields_dict["arg_names"],
        fields_dict["is_transform"],
        fields_dict["transform_kind"],
        fields_dict["transform_chain"],
        fields_dict["transform_config"],
        fields_dict["transform_fn_name"],
        fields_dict["transform_fn_qualname"],
        fields_dict["transform_fn_source"],
    ) = resolve_descriptive(observe_descriptive(func, func_name), func_name)
