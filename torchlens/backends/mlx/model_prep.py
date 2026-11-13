"""MLX model preparation helpers for MLX capture."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .._finalize import (
    join_module_address as _join_module_address,
    module_source_metadata as _module_source_metadata,
    nearest_metadata_parent,
)
from .._module_returns import ModuleReturnLedger


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
    module_returns
        Leaves each module call returned, recorded at module exit.
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
    module_returns: ModuleReturnLedger = field(default_factory=ModuleReturnLedger)


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
        metadata[primary] = _module_record(module, all_addresses)

    for address, container in _list_containers(model, alias_to_primary).items():
        metadata[address] = _module_record(container, [address])

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


def _module_record(module: object, all_addresses: list[str]) -> dict[str, Any]:
    """Return the TorchLens module metadata for one module or container object.

    Parameters
    ----------
    module
        MLX module, or the list/tuple/dict container holding sub-modules.
    all_addresses
        Every address of ``module``, primary first.

    Returns
    -------
    dict[str, Any]
        Module metadata keyed the way the shared module finalization reads it.
    """

    return {
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


def _list_containers(model: object, module_addresses: dict[str, str]) -> dict[str, object]:
    """Return the non-module containers that hold sub-modules, keyed by address.

    MLX keeps sub-modules in plain Python lists (``nn.Sequential.layers``) and
    dicts, so a child's address (``block.layers.0``) runs through a container
    that is not itself a module. The container gets its own module record, as
    a torch ``ModuleList`` does, so every address's parent is a recorded module
    and addresses stay unchanged.

    Parameters
    ----------
    model
        Root MLX module.
    module_addresses
        Every discovered module address (aliases included).

    Returns
    -------
    dict[str, object]
        Container object keyed by its dotted address.
    """

    containers: dict[str, object] = {}
    for address in sorted(module_addresses, key=_address_sort_key):
        if address == "self":
            continue
        node: object | None = model
        parts = address.split(".")
        for depth, part in enumerate(parts[:-1], start=1):
            node = _child_value(node, part)
            if node is None:
                break
            prefix = ".".join(parts[:depth])
            if prefix in module_addresses or _is_mlx_module(node):
                continue
            if isinstance(node, (list, tuple, dict)):
                containers.setdefault(prefix, node)
    return containers


def _child_value(node: object, part: str) -> object | None:
    """Return ``node``'s child named by one address segment, or ``None``.

    Parameters
    ----------
    node
        MLX module (a ``dict`` subclass) or a list/tuple/dict container.
    part
        One dotted-address segment.

    Returns
    -------
    object | None
        The child value, or ``None`` when the segment does not resolve.
    """

    if isinstance(node, dict):
        return node.get(part)
    if isinstance(node, (list, tuple)):
        if not part.isdigit() or int(part) >= len(node):
            return None
        return node[int(part)]
    return getattr(node, part, None)


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
        nearest = nearest_metadata_parent(address, metadata)
        if nearest == parent:
            children.append(address)
    return children


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
