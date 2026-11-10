"""Raw-to-final label renaming inside replay templates and label-bearing records.

Split from ``labeling.py`` (step 9 and step 10 helpers). Each walk dispatches on
the exact class before the ``isinstance`` ladder and reads dataclass field
names from a per-call cache, so a template pays ``dataclasses.fields`` once per
class rather than once per node. Node order, rebuilt types and results are those
of the plain ladder: tuple, list, dict, then dataclass instance. The caller passes
the ``ParentRef`` class in, so this module imports nothing from ``intervention``
and stays outside the package's import cycle.

A class that reaches the dataclass branch is remembered in the same cache
(``_record_plan``), so its later instances skip the ladder, and a field holding
an exact string, an atom, an empty tuple or a ``ParentRef`` is answered in the
field loop instead of by a recursive call. The values are those of the ladder;
only the number of Python calls differs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .._data_classes._template_walk import dataclass_field_names
from .._utils._record_group_table import record_member

#: Exact leaf classes that are never containers, dataclasses or callables.
_ATOMIC_CLASSES = frozenset({int, float, bool, bytes, complex, type(None)})
#: String fields of intervention records that hold a layer label.
_LABEL_FIELDS = frozenset(
    {"parent_label", "child_label", "target_label", "call_label", "site_label", "layer_label"}
)
_ABSENT = object()
_NO_STATE: dict[str, Any] = {}
_CONTAINERS = (tuple, list, dict)
_FieldCache = dict[Any, Any]
#: Field names of a record class, and whether its instance dict answers them.
_RecordPlan = tuple[tuple[str, ...], bool]


def _record_plan(
    value: Any, cache: _FieldCache, key: tuple[Any, type], ladder: tuple[type, ...], calls: bool
) -> _RecordPlan | None:
    """Decide once per class whether ``value`` reaches a walk's dataclass branch.

    Args:
        value: First instance of its class the walk meets.
        cache: The walk's cache; the answer is stored under ``key``.
        key: ``(walk marker, class)``.
        ladder: Classes the walk tests before its dataclass branch.
        calls: Whether the walk returns a callable unchanged before that branch.

    Returns:
        ``(field names, direct)`` for a class whose instances the walk rebuilds
        field by field, else ``None``. ``direct`` says a field the instance
        ``__dict__`` holds is what ``getattr`` returns: the class has the plain
        attribute lookup and no descriptor under a field name.
    """

    plan = None
    if not isinstance(value, ladder) and not (calls and callable(value)):
        names = dataclass_field_names(value, cache)
        if names is not None:
            cls = value.__class__
            direct = (
                hasattr(value, "__dict__")
                and cls.__getattribute__ is object.__getattribute__
                and not any(hasattr(getattr(cls, name, None), "__get__") for name in names)
            )
            plan = (names, direct)
    cache[key] = plan
    return plan


def _rebuild_template_record(
    value: Any,
    plan: _RecordPlan,
    mapping: dict[str, str],
    parent_ref_cls: type[Any],
    cache: _FieldCache,
) -> Any:
    """Rebuild one template dataclass node with its ``ParentRef`` leaves renamed.

    Args:
        value: Dataclass instance the template walk reached.
        plan: The class's ``_record_plan``.
        mapping: Raw-to-final layer-label mapping.
        parent_ref_cls: The ``ParentRef`` class.
        cache: The walk's cache.

    Returns:
        ``dataclasses.replace`` of ``value`` with every present field walked.
    """

    names, direct = plan
    state = value.__dict__ if direct else _NO_STATE
    updates = {}
    for name in names:
        if name in state:
            item = state[name]
        else:
            item = getattr(value, name, _ABSENT)
            if item is _ABSENT:
                continue
        item_cls = item.__class__
        if item_cls is str or item_cls in _ATOMIC_CLASSES or (item_cls is tuple and not item):
            updates[name] = item
        elif item_cls is parent_ref_cls:
            label = item.parent_label
            updates[name] = parent_ref_cls(
                mapping[label] if label in mapping else label, item._capture_layout
            )
        else:
            updates[name] = _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
    return replace(value, **updates)


def _rename_template_sequence(
    value: Any, mapping: dict[str, str], parent_ref_cls: type[Any], cache: _FieldCache
) -> list[Any]:
    """Return the walked items of an exact tuple or list template node.

    Args:
        value: Exact ``tuple`` or ``list``.
        mapping: Raw-to-final layer-label mapping.
        parent_ref_cls: The ``ParentRef`` class.
        cache: The walk's cache.

    Returns:
        A new list holding each item as the template walk returns it.
    """

    items = list(value)
    for index, item in enumerate(value):
        item_cls = item.__class__
        if item_cls is parent_ref_cls:
            label = item.parent_label
            items[index] = parent_ref_cls(
                mapping[label] if label in mapping else label, item._capture_layout
            )
        elif item_cls is not str and item_cls not in _ATOMIC_CLASSES:
            items[index] = _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
    return items


def _rename_template_parent_refs(
    value: Any, mapping: dict[str, str], parent_ref_cls: type[Any], cache: _FieldCache | None = None
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
    if cache is None:
        cache = {}
    if cls is tuple or cls is list:
        items = _rename_template_sequence(value, mapping, parent_ref_cls, cache)
        return tuple(items) if cls is tuple else items
    plan_key = (parent_ref_cls, cls)
    plan = (
        cache[plan_key]
        if plan_key in cache
        else _record_plan(value, cache, plan_key, (parent_ref_cls, *_CONTAINERS), False)
    )
    if plan is not None:
        return _rebuild_template_record(value, plan, mapping, parent_ref_cls, cache)
    if isinstance(value, parent_ref_cls):
        return replace(value, parent_label=mapping.get(value.parent_label, value.parent_label))
    if isinstance(value, (tuple, list)):
        items = [
            _rename_template_parent_refs(item, mapping, parent_ref_cls, cache) for item in value
        ]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, dict):
        return {
            key: _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
            for key, item in value.items()
        }
    names = dataclass_field_names(value, cache)
    if names is not None:
        updates = {}
        for name in names:
            item = getattr(value, name, _ABSENT)
            if item is not _ABSENT:
                updates[name] = _rename_template_parent_refs(item, mapping, parent_ref_cls, cache)
        value = replace(value, **updates)
    return value


def _rebuild_label_record(
    value: Any, plan: _RecordPlan, mapping: dict[str, str], cache: _FieldCache
) -> Any:
    """Rebuild one label-bearing dataclass record with its labels renamed.

    Args:
        value: Dataclass instance the label walk reached.
        plan: The class's ``_record_plan``.
        mapping: Raw-to-final layer-label mapping.
        cache: The walk's cache.

    Returns:
        ``dataclasses.replace`` of ``value`` with every present field walked.
    """

    names, direct = plan
    state = value.__dict__ if direct else _NO_STATE
    updates = {}
    for name in names:
        if name in state:
            item = state[name]
        else:
            item = getattr(value, name, _ABSENT)
            if item is _ABSENT:
                continue
        item_cls = item.__class__
        if item_cls is str:
            updates[name] = mapping[item] if item in mapping else item
        elif item_cls in _ATOMIC_CLASSES or (item_cls is tuple and not item):
            updates[name] = item
        elif name in _LABEL_FIELDS and isinstance(item, str):
            updates[name] = mapping.get(item, item)
        else:
            updates[name] = _rename_label_dataclass(item, mapping, cache)
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
    if cls in _ATOMIC_CLASSES:
        return value
    if cache is None:
        cache = {}
    plan_key = (None, cls)
    plan = (
        cache[plan_key]
        if plan_key in cache
        else _record_plan(value, cache, plan_key, _CONTAINERS, True)
    )
    if plan is not None:
        return _rebuild_label_record(value, plan, mapping, cache)
    if callable(value):
        return value
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
    parent_ref_cls: type[Any],
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
    args_template = record_member(layer_entry, "args_template", None)
    if args_template is not None:
        set_entry_field(
            "args_template",
            _rename_template_parent_refs(args_template, mapping, parent_ref_cls, cache),
        )
    kwargs_template = record_member(layer_entry, "kwargs_template", None)
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
