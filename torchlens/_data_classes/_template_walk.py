"""Generic walks over replay templates (tuples, lists, dict values, dataclass fields).

Imports nothing from torchlens, so callers anywhere in the package can share
it; ``cleanup`` supplies the leaf test and the leaf replacement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from typing import Any


def template_has_leaf(value: Any, is_hit: Callable[[Any], bool]) -> bool:
    """Return whether ``map_template_leaves`` would visit a node ``is_hit`` accepts.

    Walks the same nodes in the same order, building nothing.
    """

    if is_hit(value):
        return True
    if isinstance(value, (tuple, list)):
        return any(template_has_leaf(item, is_hit) for item in value)
    if isinstance(value, dict):
        return any(template_has_leaf(item, is_hit) for item in value.values())
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            template_has_leaf(getattr(value, field.name), is_hit)
            for field in fields(value)
            if hasattr(value, field.name)
        )
    return False


def map_template_leaves(value: Any, replace_leaf: Callable[[Any], Any]) -> Any:
    """Rebuild ``value`` with every node ``replace_leaf`` swaps replaced.

    ``replace_leaf`` returns its argument to keep a node; anything else
    replaces the node without descending into it. Tuples, lists, dicts and
    dataclass instances are rebuilt; other values are returned as they are.
    """

    replaced = replace_leaf(value)
    if replaced is not value:
        return replaced
    if isinstance(value, tuple):
        return tuple(map_template_leaves(item, replace_leaf) for item in value)
    if isinstance(value, list):
        return [map_template_leaves(item, replace_leaf) for item in value]
    if isinstance(value, dict):
        return {key: map_template_leaves(item, replace_leaf) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: map_template_leaves(getattr(value, field.name), replace_leaf)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value
