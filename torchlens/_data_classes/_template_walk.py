"""Generic walks over replay templates (tuples, lists, dict values, dataclass fields).

Imports nothing from torchlens, so callers anywhere in the package can share
it; ``cleanup`` supplies the leaf test and the leaf replacement.

Both walks visit nodes in the order ``is_hit``/``replace_leaf`` first, then
tuple or list, then dict, then dataclass instance. Per node they dispatch on
the exact class first and cache each dataclass's field names per call, so a
template pays ``dataclasses.fields`` once per class instead of once per node.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from typing import Any

#: Exact leaf classes that are never tuples, lists, dicts or dataclasses.
_ATOMIC_CLASSES = frozenset({str, int, float, bool, bytes, complex, type(None)})
_ABSENT = object()


def dataclass_field_names(value: Any, cache: dict[type, tuple[str, ...] | None]) -> Any:
    """Return the field names of dataclass instance ``value``, or ``None`` for anything else.

    ``cache`` maps a class to its answer; callers own it and drop it after one walk.
    """

    cls = value.__class__
    names = cache.get(cls, _ABSENT)
    if names is _ABSENT:
        if is_dataclass(value) and not isinstance(value, type):
            names = tuple(field.name for field in fields(value))
        else:
            names = None
        cache[cls] = names
    return names


def template_has_leaf(
    value: Any,
    is_hit: Callable[[Any], bool],
    cache: dict[type, tuple[str, ...] | None] | None = None,
) -> bool:
    """Return whether ``map_template_leaves`` would visit a node ``is_hit`` accepts.

    Walks the same nodes in the same order, building nothing.
    """

    if is_hit(value):
        return True
    cls = value.__class__
    if cls in _ATOMIC_CLASSES:
        return False
    if cache is None:
        cache = {}
    if isinstance(value, (tuple, list)):
        items: Any = value
    elif isinstance(value, dict):
        items = value.values()
    else:
        names = dataclass_field_names(value, cache)
        if names is None:
            return False
        items = (
            item
            for item in (getattr(value, name, _ABSENT) for name in names)
            if item is not _ABSENT
        )
    return any(template_has_leaf(item, is_hit, cache) for item in items)


def map_template_leaves(
    value: Any,
    replace_leaf: Callable[[Any], Any],
    cache: dict[type, tuple[str, ...] | None] | None = None,
) -> Any:
    """Rebuild ``value`` with every node ``replace_leaf`` swaps replaced.

    ``replace_leaf`` returns its argument to keep a node; anything else
    replaces the node without descending into it. Tuples, lists, dicts and
    dataclass instances are rebuilt; other values are returned as they are.
    """

    replaced = replace_leaf(value)
    if replaced is not value or value.__class__ in _ATOMIC_CLASSES:
        return replaced
    if cache is None:
        cache = {}
    if isinstance(value, tuple):
        return tuple([map_template_leaves(item, replace_leaf, cache) for item in value])
    if isinstance(value, list):
        return [map_template_leaves(item, replace_leaf, cache) for item in value]
    if isinstance(value, dict):
        return {key: map_template_leaves(item, replace_leaf, cache) for key, item in value.items()}
    names = dataclass_field_names(value, cache)
    if names is None:
        return value
    updates = {}
    for name in names:
        item = getattr(value, name, _ABSENT)
        if item is not _ABSENT:
            updates[name] = map_template_leaves(item, replace_leaf, cache)
    return replace(value, **updates)
