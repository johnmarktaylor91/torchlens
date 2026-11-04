"""Replace intervention-template references to removed parent ops.

Split out of ``cleanup``: pure walks over args/kwargs templates (tuples,
lists, dict values, dataclass fields) used when ops are removed.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

from ..intervention.types import ParentRef, Unsupported


def _scrub_template(template: Any, labels_to_remove: set[str]) -> Any:
    """Return ``template`` with removed-parent refs replaced, rebuilt only if it names one.

    An untouched template keeps its object: every removal runs before the
    step-9 rename, which rebuilds each template into a fresh copy.
    """

    if _template_mentions_removed_parent(template, labels_to_remove):
        return _replace_removed_parent_refs(template, labels_to_remove)
    return template


def _template_mentions_removed_parent(value: Any, labels_to_remove: set[str]) -> bool:
    """Return whether ``_replace_removed_parent_refs`` would replace any leaf.

    Walks the same nodes in the same way (tuples, lists, dict values, set
    dataclass fields) without building anything.
    """

    if isinstance(value, ParentRef) and value.parent_label in labels_to_remove:
        return True
    if isinstance(value, (tuple, list)):
        return any(_template_mentions_removed_parent(item, labels_to_remove) for item in value)
    if isinstance(value, dict):
        return any(
            _template_mentions_removed_parent(item, labels_to_remove) for item in value.values()
        )
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _template_mentions_removed_parent(getattr(value, field.name), labels_to_remove)
            for field in fields(value)
            if hasattr(value, field.name)
        )
    return False


def _replace_removed_parent_refs(value: Any, labels_to_remove: set[str]) -> Any:
    """Replace template parent refs to removed labels with unsupported leaves.

    Args:
        value: Template or nested component.
        labels_to_remove: Removed layer labels.

    Returns:
        Template value with stale parent refs replaced.
    """

    if isinstance(value, ParentRef) and value.parent_label in labels_to_remove:
        return Unsupported(reason="removed_parent_ref", value_type="ParentRef")
    if isinstance(value, tuple):
        return tuple(_replace_removed_parent_refs(item, labels_to_remove) for item in value)
    if isinstance(value, list):
        return [_replace_removed_parent_refs(item, labels_to_remove) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_removed_parent_refs(item, labels_to_remove) for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _replace_removed_parent_refs(getattr(value, field.name), labels_to_remove)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value
