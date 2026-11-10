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

# Transform-boundary config attribute; its builder kwargs are snapshotted per op.
_TRANSFORM_CONFIG = "__tl_transform_config__"
# Marks "no transform config on this function"; most ops have none, and the fresh
# empty dict they record needs no snapshot.
_NO_CONFIG = object()
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


__all__ = (
    "TRANSFORM_FUNC_NAMES",
    "_set_descriptive_fields",
)


def _set_descriptive_fields(
    fields_dict: dict[str, Any], func: Callable[..., Any], func_name: str
) -> None:
    """Record the callable's qualified name, argument names and transform descriptors.

    ``arg_names`` is the exact-name lookup first (add / add_ / __add__ have distinct
    signatures). The underscore-stripped key remains a best-effort fallback for
    NON-dunder names whose own introspection stored nothing; a dunder whose
    signature is opaque stays honestly unknown (empty) rather than borrowing the
    namesake torch function's different signature. It is inline, not a helper:
    once per op (tests/test_capture_row_commit_cost.py pins calls).
    """

    fields_dict["func_qualname"] = getattr(func, "__qualname__", None)
    arg_names = _st._arg_names.get(func_name)
    if arg_names is None:
        arg_names = (
            () if func_name.startswith("__") else _st._arg_names.get(func_name.strip("_"), ())
        )
    fields_dict["arg_names"] = arg_names
    is_transform = bool(getattr(func, "__tl_is_transform_boundary__", False)) or (
        func_name in TRANSFORM_FUNC_NAMES
    )
    fields_dict["is_transform"] = is_transform
    fields_dict["transform_kind"] = getattr(func, "__tl_transform_kind__", None) or (
        func_name if is_transform else None
    )
    fields_dict["transform_chain"] = tuple(getattr(func, "__tl_transform_tags__", ()))
    config = getattr(func, _TRANSFORM_CONFIG, _NO_CONFIG)
    fields_dict["transform_config"] = {} if config is _NO_CONFIG else snapshot_mutable_arg(config)
    fields_dict["transform_fn_name"] = getattr(func, "__tl_transform_fn_name__", None)
    fields_dict["transform_fn_qualname"] = getattr(func, "__tl_transform_fn_qualname__", None)
    fields_dict["transform_fn_source"] = getattr(func, "__tl_transform_fn_source__", None)
