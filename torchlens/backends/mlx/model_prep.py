"""MLX model preparation helpers for technical-preview capture."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass
class MLXModuleTree:
    """Discovered MLX object-module tree.

    Parameters
    ----------
    root
        Root MLX module.
    metadata
        TorchLens module metadata keyed by primary address.
    address_by_id
        Primary module address keyed by object identity.
    modules_by_class
        Module instance addresses grouped by class for scoped ``__call__`` wrapping.
    param_owner_by_address
        Owning module address keyed by parameter address.
    param_address_by_id
        Primary parameter address keyed by exact MLX array identity.
    call_counts
        Per-primary-address call counts recorded during capture.
    forward_args_by_call
        Forward args keyed by ``(primary_address, call_index)``.
    """

    root: object
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    modules_by_class: dict[type[Any], dict[int, str]]
    param_owner_by_address: dict[str, str]
    param_address_by_id: dict[int, str]
    call_counts: dict[str, int] = field(default_factory=dict)
    forward_args_by_call: dict[tuple[str, int], tuple[tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=dict
    )


def iter_named_modules(model: object) -> Iterator[tuple[str, object]]:
    """Yield an MLX module tree using public MLX traversal.

    Parameters
    ----------
    model:
        Root MLX module.

    Yields
    ------
    tuple[str, object]
        Dotted module address and module object.
    """

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for name, module in _iter_mapping_or_sequence(named_modules()):
            yield str(name), module
        return
    yield "", model


def discover_mlx_module_tree(model: object) -> MLXModuleTree | None:
    """Discover an MLX object-module hierarchy with deterministic aliases.

    Parameters
    ----------
    model
        Candidate MLX module.

    Returns
    -------
    MLXModuleTree | None
        Discovered module tree, or ``None`` for raw functions/plain callables.
    """

    if inspect.isfunction(model) or inspect.ismethod(model) or not callable(model):
        return None
    if not _is_mlx_module(model):
        return None

    modules_by_id: dict[int, object] = {}
    addresses_by_id: dict[int, list[str]] = {}
    for raw_address, module in iter_named_modules(model):
        if not _is_mlx_module(module):
            continue
        address = "self" if raw_address == "" else str(raw_address)
        modules_by_id[id(module)] = module
        addresses_by_id.setdefault(id(module), []).append(address)
    if id(model) not in addresses_by_id:
        modules_by_id[id(model)] = model
        addresses_by_id[id(model)] = ["self"]

    address_by_id: dict[int, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    modules_by_class: dict[type[Any], dict[int, str]] = {}
    alias_to_primary: dict[str, str] = {}
    for module_id, addresses in addresses_by_id.items():
        primary = _primary_address(addresses)
        module = modules_by_id[module_id]
        address_by_id[module_id] = primary
        modules_by_class.setdefault(type(module), {})[module_id] = primary
        all_addresses = sorted(set(addresses), key=_address_sort_key)
        for alias in all_addresses:
            alias_to_primary[alias] = primary
        metadata[primary] = {
            **_module_source_metadata(module),
            "cls": type(module),
            "class_name": type(module).__name__,
            "class_qualname": f"{type(module).__module__}.{type(module).__qualname__}",
            "address_children": [],
            "all_addresses": all_addresses,
            "training": bool(getattr(module, "training", False)),
            "forward_pre_hooks": [],
            "forward_hooks": [],
            "backward_pre_hooks": [],
            "backward_hooks": [],
            "full_backward_pre_hooks": [],
            "full_backward_hooks": [],
            "custom_attributes": {},
            "custom_methods": [],
            "_module_object": module,
        }

    for primary, item in metadata.items():
        item["address_children"] = _direct_children(primary, metadata)

    param_owner_by_address: dict[str, str] = {}
    param_address_by_id: dict[int, str] = {}
    for param_address, value in _iter_mlx_parameter_tree(model):
        owner_alias = param_address.rsplit(".", 1)[0] if "." in param_address else "self"
        owner = alias_to_primary.get(owner_alias, owner_alias)
        primary_param_address = _join_module_address(owner, param_address.rsplit(".", 1)[-1])
        param_owner_by_address[primary_param_address] = owner
        param_address_by_id.setdefault(id(value), primary_param_address)

    return MLXModuleTree(
        root=model,
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=modules_by_class,
        param_owner_by_address=param_owner_by_address,
        param_address_by_id=param_address_by_id,
    )


def prepare_model_once(model: object) -> object:
    """Apply one-time MLX preparation.

    Parameters
    ----------
    model:
        MLX model.

    Returns
    -------
    object
        The unchanged model.
    """

    return model


def prepare_model_session(session: object, model: object) -> object:
    """Apply per-session MLX preparation.

    Parameters
    ----------
    session:
        MLX backend session.
    model:
        MLX model.

    Returns
    -------
    object
        The unchanged model.
    """

    return model


def cleanup_model_session(session: object, prepared_model: object) -> None:
    """Clean up per-session MLX preparation.

    Parameters
    ----------
    session:
        MLX backend session.
    prepared_model:
        Prepared model object.
    """

    return None


def _iter_mapping_or_sequence(value: object) -> Iterator[tuple[object, object]]:
    """Yield key/value pairs from MLX traversal outputs.

    Parameters
    ----------
    value
        Mapping-like or iterable traversal result.

    Yields
    ------
    tuple[object, object]
        Traversal key and value.
    """

    if isinstance(value, dict):
        yield from value.items()
        return
    yield from value  # type: ignore[misc]


def _is_mlx_module(value: object) -> bool:
    """Return whether ``value`` is an ``mlx.nn.Module``.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when MLX is installed and ``value`` is an MLX module.
    """

    try:
        import mlx.nn as nn
    except ImportError:
        return False
    return isinstance(value, nn.Module)


def _primary_address(addresses: list[str]) -> str:
    """Return the deterministic primary address for module aliases.

    Parameters
    ----------
    addresses
        All observed addresses for one module object.

    Returns
    -------
    str
        Primary TorchLens address.
    """

    if "self" in addresses:
        return "self"
    return sorted(set(addresses), key=_address_sort_key)[0]


def _address_sort_key(address: str) -> tuple[tuple[int, int | str], ...]:
    """Return a natural-ish sort key for dotted MLX addresses.

    Parameters
    ----------
    address
        Dotted module address.

    Returns
    -------
    tuple[tuple[int, int | str], ...]
        Sortable key.
    """

    key: list[tuple[int, int | str]] = []
    for part in address.split("."):
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def _direct_children(parent: str, metadata: dict[str, dict[str, Any]]) -> list[str]:
    """Return direct child module addresses for ``parent``.

    Parameters
    ----------
    parent
        Parent address.
    metadata
        Module metadata keyed by primary address.

    Returns
    -------
    list[str]
        Direct child addresses.
    """

    children: list[str] = []
    for address in sorted(metadata, key=_address_sort_key):
        if address == parent:
            continue
        nearest = _nearest_existing_parent(address, metadata)
        if nearest == parent:
            children.append(address)
    return children


def _nearest_existing_parent(address: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    """Return the closest existing parent address for ``address``.

    Parameters
    ----------
    address
        Child address.
    metadata
        Module metadata keyed by primary address.

    Returns
    -------
    str | None
        Parent address, or ``None`` for root.
    """

    if address == "self":
        return None
    parts = address.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in metadata:
            return candidate
    return "self" if "self" in metadata else None


def _iter_mlx_parameter_tree(model: object) -> Iterator[tuple[str, object]]:
    """Yield flattened MLX parameter addresses and arrays.

    Parameters
    ----------
    model
        MLX module whose parameter tree should be flattened.

    Yields
    ------
    tuple[str, object]
        Dotted parameter address and MLX array.
    """

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        yield from _flatten_parameter_tree(parameters(), "")


def _flatten_parameter_tree(value: object, prefix: str) -> Iterator[tuple[str, object]]:
    """Yield array leaves from an MLX parameter tree.

    Parameters
    ----------
    value
        Parameter tree value.
    prefix
        Dotted address prefix.

    Yields
    ------
    tuple[str, object]
        Parameter address and array.
    """

    if _is_mlx_array(value):
        yield prefix, value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = _join_module_address(prefix, str(key)) if prefix else str(key)
            yield from _flatten_parameter_tree(item, child_prefix)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_prefix = _join_module_address(prefix, str(index)) if prefix else str(index)
            yield from _flatten_parameter_tree(item, child_prefix)


def _is_mlx_array(value: object) -> bool:
    """Return whether ``value`` is an MLX array.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when ``value`` is an MLX array.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _module_source_metadata(module: object) -> dict[str, Any]:
    """Return best-effort source metadata for an MLX module.

    Parameters
    ----------
    module
        MLX module object.

    Returns
    -------
    dict[str, Any]
        Source metadata compatible with TorchLens module logs.
    """

    cls = type(module)
    init = getattr(cls, "__init__", None)
    call = getattr(cls, "__call__", None)
    return {
        "class_source_file": inspect.getsourcefile(cls),
        "class_source_line": _source_line(cls),
        "init_source_file": inspect.getsourcefile(init) if init is not None else None,
        "init_source_line": _source_line(init),
        "forward_source_file": inspect.getsourcefile(call) if call is not None else None,
        "forward_source_line": _source_line(call),
        "class_docstring": inspect.getdoc(cls),
        "init_signature": _signature_string(init),
        "init_docstring": inspect.getdoc(init) if init is not None else None,
        "forward_signature": _signature_string(call),
        "forward_docstring": inspect.getdoc(call) if call is not None else None,
    }


def _source_line(obj: object) -> int | None:
    """Return the first source line for ``obj`` when inspectable.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    int | None
        First source line, or ``None``.
    """

    if obj is None:
        return None
    try:
        return inspect.getsourcelines(cast(Any, obj))[1]
    except (OSError, TypeError):
        return None


def _signature_string(obj: object) -> str | None:
    """Return ``obj``'s signature string when inspectable.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    str | None
        Signature string, or ``None``.
    """

    if obj is None:
        return None
    try:
        return str(inspect.signature(cast(Any, obj)))
    except (TypeError, ValueError):
        return None


def _join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child attribute name.

    Returns
    -------
    str
        Joined module address.
    """

    return child_name if parent in {"", "self"} else f"{parent}.{child_name}"


__all__ = [
    "MLXModuleTree",
    "cleanup_model_session",
    "discover_mlx_module_tree",
    "iter_named_modules",
    "prepare_model_once",
    "prepare_model_session",
]
