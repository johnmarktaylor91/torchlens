"""Raw-to-final label renaming inside replay templates and label-bearing records.

Split from ``labeling.py`` (step 9 and step 10 helpers). Each walk dispatches on
the exact class before the ``isinstance`` ladder and reads dataclass field
names from a per-call cache, so a template pays ``dataclasses.fields`` once per
class rather than once per node. Node order, rebuilt types and results are those
of the plain ladder: tuple, list, dict, then dataclass instance. The caller passes
the ``ParentRef`` class in, so this module imports nothing from ``intervention``
and stays outside the package's import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .._data_classes._template_walk import dataclass_field_names

#: Exact leaf classes that are never containers, dataclasses or callables.
_ATOMIC_CLASSES = frozenset({int, float, bool, bytes, complex, type(None)})
#: String fields of intervention records that hold a layer label.
_LABEL_FIELDS = frozenset(
    {"parent_label", "child_label", "target_label", "call_label", "site_label", "layer_label"}
)
_ABSENT = object()
_FieldCache = dict[type, Any]


def _rename_template_parent_refs(
    value: Any, mapping: dict[str, str], parent_ref_cls: type, cache: _FieldCache | None = None
) -> Any:
    """Rename ``ParentRef.parent_label`` leaves in replay templates.

    Args:
        value: Template or nested template component.
        mapping: Raw-to-final layer-label mapping.
        parent_ref_cls: The ``ParentRef`` class.
        cache: Per-walk dataclass field-name cache (built when ``None``).

    Returns:
        Template value with parent references rewritten where possible.
    """

    cls = value.__class__
    if cls is parent_ref_cls:
        # Exactly what ``replace`` builds for this two-field init-only class.
        label = value.parent_label
        return parent_ref_cls(mapping.get(label, label), value._capture_layout)
    if cls is str or cls in _ATOMIC_CLASSES:
        return value
    if isinstance(value, parent_ref_cls):
        return replace(value, parent_label=mapping.get(value.parent_label, value.parent_label))
    if cache is None:
        cache = {}
    if isinstance(value, tuple):
        return tuple(
            [_rename_template_parent_refs(item, mapping, parent_ref_cls, cache) for item in value]
        )
    if isinstance(value, list):
        return [
            _rename_template_parent_refs(item, mapping, parent_ref_cls, cache) for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
            for key, item in value.items()
        }
    names = dataclass_field_names(value, cache)
    if names is None:
        return value
    updates = {}
    for name in names:
        item = getattr(value, name, _ABSENT)
        if item is not _ABSENT:
            updates[name] = _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
    return replace(value, **updates)


def _rename_label_dataclass(
    value: Any, mapping: dict[str, str], cache: _FieldCache | None = None
) -> Any:
    """Rename known label fields on frozen intervention dataclasses.

    Args:
        value: Dataclass-like record or nested container.
        mapping: Raw-to-final layer-label mapping.
        cache: Per-walk dataclass field-name cache (built when ``None``).

    Returns:
        Record with label-bearing string fields rewritten.
    """

    cls = value.__class__
    if cls is str:
        return mapping.get(value, value)
    if cls in _ATOMIC_CLASSES or callable(value):
        return value
    if cache is None:
        cache = {}
    if isinstance(value, tuple):
        return tuple([_rename_label_dataclass(item, mapping, cache) for item in value])
    if isinstance(value, list):
        return [_rename_label_dataclass(item, mapping, cache) for item in value]
    if isinstance(value, dict):
        return {
            _rename_label_dataclass(key, mapping, cache): _rename_label_dataclass(
                item, mapping, cache
            )
            for key, item in value.items()
        }
    names = dataclass_field_names(value, cache)
    if names is not None:
        updates = {}
        for name in names:
            field_value = getattr(value, name, _ABSENT)
            if field_value is _ABSENT:
                continue
            if name in _LABEL_FIELDS and isinstance(field_value, str):
                updates[name] = mapping.get(field_value, field_value)
            else:
                updates[name] = _rename_label_dataclass(field_value, mapping, cache)
        return replace(value, **updates)
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _rename_replay_fields(
    layer_entry: Any,
    mapping: dict[str, str],
    set_entry_field: Callable[[str, Any], None],
    cache: _FieldCache,
    parent_ref_cls: type,
) -> None:
    """Rename labels in one Op's edge uses, replay templates and edit records.

    Args:
        layer_entry: Op (or test double) whose fields are renamed.
        mapping: Raw-to-final layer-label mapping.
        set_entry_field: Writer for one renamed field of ``layer_entry``.
        cache: Dataclass field-name cache shared across the rename sweep.
        parent_ref_cls: The ``ParentRef`` class.
    """

    edge_uses = getattr(layer_entry, "_edge_uses", None)
    if edge_uses:
        set_entry_field(
            "_edge_uses", [_rename_label_dataclass(record, mapping, cache) for record in edge_uses]
        )
    args_template = getattr(layer_entry, "args_template", None)
    if args_template is not None:
        set_entry_field(
            "args_template",
            _rename_template_parent_refs(args_template, mapping, parent_ref_cls, cache),
        )
    kwargs_template = getattr(layer_entry, "kwargs_template", None)
    if kwargs_template is not None:
        set_entry_field(
            "kwargs_template",
            _rename_template_parent_refs(kwargs_template, mapping, parent_ref_cls, cache),
        )
    interventions = getattr(layer_entry, "edits", None)
    if interventions:
        set_entry_field(
            "edits", [_rename_label_dataclass(record, mapping, cache) for record in interventions]
        )
