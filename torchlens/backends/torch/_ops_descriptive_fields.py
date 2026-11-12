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
    "TRANSFORM_FUNC_NAMES",
    "_set_descriptive_fields",
)


def _set_descriptive_fields(
    fields_dict: dict[str, Any], func: Callable[..., Any], func_name: str
) -> None:
    """Record the callable's qualified name, argument names and transform descriptors.

    The callable's own values come from its function-family resolution
    (``_ops_function_family.callable_family``: kept per unbound builtin callable,
    otherwise resolved on this call). ``arg_names`` is read live, the exact-name lookup
    first (add / add_ / __add__ have distinct signatures). The underscore-stripped key
    remains a best-effort fallback for NON-dunder names whose own introspection stored
    nothing; a dunder whose signature is opaque stays honestly unknown (empty) rather
    than borrowing the namesake torch function's different signature.
    """

    family = CALLABLE_FAMILY.get(func_name)
    if family is None or family[0] is not func:
        family = callable_family(func, func_name)
    _, qualname, is_transform, kind, chain, fn_name, fn_qualname, fn_source, config = family
    fields_dict["func_qualname"] = qualname
    arg_names = _st._arg_names.get(func_name)
    if arg_names is None:
        arg_names = (
            () if func_name.startswith("__") else _st._arg_names.get(func_name.strip("_"), ())
        )
    fields_dict["arg_names"] = arg_names
    # Keys keep the order the per-op code wrote them in.
    fields_dict["is_transform"] = is_transform
    fields_dict["transform_kind"] = kind
    fields_dict["transform_chain"] = chain
    fields_dict["transform_config"] = {} if config is NO_CONFIG else snapshot_mutable_arg(config)
    fields_dict["transform_fn_name"] = fn_name
    fields_dict["transform_fn_qualname"] = fn_qualname
    fields_dict["transform_fn_source"] = fn_source
