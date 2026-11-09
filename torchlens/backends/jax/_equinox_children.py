"""Equinox child discovery: module fields, list and tuple fields, and their containers.

Torch holds a list of child modules in a ``ModuleList`` and Flax NNX in an ``nnx.List``,
both modules whose address is the field name, so a child's address ``blocks.0`` has a
parent ``blocks``. An Equinox module keeps such children in a plain list or tuple field.
The shared hierarchy reads a module's parent from its dotted address, so the field gets a
container entry here: a never-called module named after the field, typed ``list`` or
``tuple``, whose children are the indexed modules.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Any

_EMPTY_HOOK_FIELDS = (
    "forward_pre_hooks",
    "forward_hooks",
    "backward_pre_hooks",
    "backward_hooks",
    "full_backward_pre_hooks",
    "full_backward_hooks",
)


def is_equinox_module(value: object) -> bool:
    """Return whether ``value`` is an Equinox module instance.

    Parameters
    ----------
    value
        Candidate root callable.

    Returns
    -------
    bool
        True when Equinox is importable and ``value`` is an ``eqx.Module``.
    """

    try:
        import equinox as eqx
    except ImportError:
        return False
    return isinstance(value, eqx.Module)


def iter_equinox_module_fields(module: Any) -> Iterator[tuple[str, Any]]:
    """Yield direct Equinox module children from module fields and list or tuple fields.

    Parameters
    ----------
    module
        Equinox module instance.

    Yields
    ------
    tuple[str, Any]
        Child name relative to ``module`` (``field`` or ``field.index``) and child module.
    """

    if not is_dataclass(module):
        return
    for field in fields(module):
        child = getattr(module, field.name, None)
        items = enumerate(child) if isinstance(child, (list, tuple)) else ((None, child),)
        for index, item in items:
            if is_equinox_module(item):
                yield (field.name if index is None else f"{field.name}.{index}"), item


def with_sequence_containers(
    model: Any, metadata: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return ``metadata`` with a container entry for every list or tuple field of modules.

    Parameters
    ----------
    model
        Equinox module root (its fields locate each container's list or tuple).
    metadata
        Module metadata keyed by primary address, as the Equinox walk builds it.

    Returns
    -------
    dict[str, dict[str, Any]]
        New mapping: each parent lists the container in place of its indexed children,
        each container lists those children and sits just before the first of them.
    """

    entries = {address: dict(entry) for address, entry in metadata.items()}
    containers: dict[str, dict[str, Any]] = {}
    for address, entry in metadata.items():
        prefix = "" if address == "self" else f"{address}."
        children: list[str] = []
        for child in entry["address_children"]:
            field_name, _, index = child[len(prefix) :].partition(".")
            if not index:
                children.append(child)
                continue
            container = prefix + field_name
            if container not in containers:
                containers[container] = _container_entry(container, _resolve(model, container))
            containers[container]["address_children"].append(child)
            if container not in children:
                children.append(container)
        entries[address]["address_children"] = children
    before = {entry["address_children"][0]: name for name, entry in containers.items()}
    result: dict[str, dict[str, Any]] = {}
    for address, entry in entries.items():
        if address in before:
            result[before[address]] = containers[before[address]]
        result[address] = entry
    return result


def _resolve(model: Any, address: str) -> Any:
    """Return the object at a primary module address (attribute names and list indices)."""

    value = model
    for part in address.split("."):
        is_index = part.isdigit() and isinstance(value, (list, tuple))
        value = value[int(part)] if is_index else getattr(value, part)
    return value


def _container_entry(address: str, container: Any) -> dict[str, Any]:
    """Return the metadata entry of a list or tuple field that holds modules."""

    kind = type(container)
    return {
        "cls": kind,
        "class_name": kind.__name__,
        "class_qualname": f"{kind.__module__}.{kind.__qualname__}",
        "address_children": [],
        "all_addresses": [address],
        "training": False,
        **{name: [] for name in _EMPTY_HOOK_FIELDS},
        "custom_attributes": {},
        "custom_methods": [],
    }
