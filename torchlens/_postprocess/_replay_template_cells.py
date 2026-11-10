"""Replay argument templates as a derived on-read family.

Capture observes each op's ``args_template`` and ``kwargs_template`` with raw
labels in its ``ParentRef`` leaves. Step 9 does not rename them: it keeps the
raw templates in the family's side columns, indexed by store row, with one
``TemplateLabels`` per trace, and writes the field's shared marker into each
set template cell (``REPLAY_TEMPLATES``, a ``SideColumnFamily``). The first
read of a template runs :func:`resolve_template`. The resolver rebuilds the
template with final labels, as step 9 used to, and caches on each
``ParentRef`` the display key of the op it refers to, as step 17 used to. A
template nobody reads is never rebuilt and costs no object beyond the raw
template capture already made.

``TemplateLabels`` holds the raw-to-final label map and a weak reference to
the trace, whose all-keys lookup gives the display keys. Pickled (with a
pending template of a replayable op), it carries the map plus the display
keys read from the live trace, so the load resolves to the same template.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from .._trace_core.side_columns import SideColumnFamily
from .._utils._display_keys import display_key
from .._utils._record_group_table import record_member
from .._utils.fail_open import fail_open
from ._label_rename import _rename_label_dataclass, _rename_template_parent_refs

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = (
    "REPLAY_TEMPLATES",
    "TemplateLabels",
    "carry_entry_templates",
    "carry_templates",
    "resolve_template",
    "rename_replay_entry",
    "template_labels",
)

#: The two stored fields of the family.
TEMPLATE_FIELDS = ("args_template", "kwargs_template")


class TemplateLabels:
    """The label context every carried template of one trace shares.

    Parameters
    ----------
    mapping:
        The trace's raw-to-final layer-label map (``_raw_to_final_layer_labels``).
    parent_ref_cls:
        The ``ParentRef`` class (passed in: this module imports nothing from
        ``intervention``).
    trace_ref:
        Weak reference to the trace whose all-keys lookup gives display keys.
    keys:
        Display key by final label, carried by a pickled context instead of a
        trace.
    """

    __slots__ = ("field_cache", "keys", "mapping", "parent_ref_cls", "trace_ref")

    def __init__(
        self,
        mapping: dict[str, str],
        parent_ref_cls: type[Any],
        trace_ref: Any = None,
        keys: dict[str, str] | None = None,
    ) -> None:
        """Bind the context to its label map and its trace."""

        self.mapping = mapping
        self.parent_ref_cls = parent_ref_cls
        self.trace_ref = trace_ref
        self.keys = keys
        self.field_cache: dict[Any, Any] = {}

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Carry the label map and the display keys, never the trace."""

        return TemplateLabels, (self.mapping, self.parent_ref_cls, None, self.display_keys())

    def records(self) -> Any:
        """Return the live trace's all-keys lookup, or ``None`` once the trace is gone."""

        trace = self.trace_ref() if self.trace_ref is not None else None
        return None if trace is None else getattr(trace, "_layer_dict_all_keys", None)

    def display_keys(self) -> dict[str, str] | None:
        """Return the display key of every final label the live trace resolves.

        Without a live trace, the keys this context was unpickled with.
        """

        records = self.records()
        if records is None:
            return self.keys
        keys = {}
        for label in self.mapping.values():
            record = records.get(label)
            if record is not None:
                keys[label] = display_key(record)
        return keys


def resolve_template(template: Any, labels: TemplateLabels, name: str) -> Any:
    """Return the template step 9 and step 17 would have stored for one raw template.

    Each read builds a new template: the args and kwargs cells of one op
    resolve to equal but distinct objects, as the per-field rename did.
    """

    renamed = _rename_template_parent_refs(
        template, labels.mapping, labels.parent_ref_cls, labels.field_cache
    )
    records = labels.records()
    if records is not None:
        fail_open(lambda: _bind_keys(renamed, records.get, _record_key), _ignore)
    elif labels.keys is not None:
        fail_open(lambda: _bind_keys(renamed, labels.keys.get, str), _ignore)
    return renamed


#: The family. Its cells cache their answer, so ``op.x is op.x`` holds.
REPLAY_TEMPLATES = SideColumnFamily("replay_templates", TEMPLATE_FIELDS, resolve_template)


def _record_key(record: Any) -> str:
    """Return the display key of a looked-up record."""

    return display_key(record)


def _ignore(_error: Exception) -> None:
    """Degraded form of a failed binding walk: the remaining keys stay unbound."""


def _bind_keys(value: Any, lookup: Any, key_of: Any) -> None:
    """Walk a template and cache each parent reference's display key in place.

    ``lookup(label)`` returns what the key is read from, or ``None`` when the
    label has none; ``key_of`` turns that into the key. Node order and class
    tests are those of the former step 17 walk.
    """

    name = type(value).__name__
    if name == "ParentRef":
        found = lookup(value.parent_label)
        if found is not None:
            object.__setattr__(value, "_display_key", key_of(found))
    elif name == "CapturedArgTemplate":
        _bind_keys(value.args, lookup, key_of)
        _bind_keys(value.kwargs, lookup, key_of)
    elif isinstance(value, (tuple, list)):
        for member in value:
            _bind_keys(member, lookup, key_of)
    elif isinstance(value, dict):
        for member in value.values():
            _bind_keys(member, lookup, key_of)


def template_labels(trace: Trace, parent_ref_cls: type[Any]) -> TemplateLabels:
    """Return the label context for the carried templates of ``trace``."""

    try:
        trace_ref = weakref.ref(trace)
    except TypeError:  # a trace double without weak-reference support
        trace_ref = _StrongRef(trace)
    return TemplateLabels(trace._raw_to_final_layer_labels, parent_ref_cls, trace_ref)


class _StrongRef:
    """Callable holding an object that refuses weak references."""

    __slots__ = ("target",)

    def __init__(self, target: Any) -> None:
        """Hold ``target``."""

        self.target = target

    def __call__(self) -> Any:
        """Return the held object."""

        return self.target


def carry_templates(table: Any, columns: dict[str, list[Any]], labels: TemplateLabels) -> None:
    """Hold each set template of ``table`` in the side columns and mark its cell.

    ``table`` is step 9's raw-row table (``store``, ``row_ids``, ``rows``,
    ``fids``); ``columns`` holds each template column as read before step 9
    wrote anything, indexed like ``table.rows``. A ``None`` template stays
    ``None``.
    """

    store, row_ids, rows = table.store, table.row_ids, table.rows
    side = REPLAY_TEMPLATES.bind(store, labels)
    aligned = row_ids == list(range(len(store)))
    for name in TEMPLATE_FIELDS:
        fid = table.fids[name]
        templates = columns[name]
        if aligned:
            by_row = templates
        else:
            by_row = [None] * len(store)
            for row_id, template in zip(row_ids, templates, strict=True):
                by_row[row_id] = template
        marker = REPLAY_TEMPLATES.hold_column(side, name, by_row)
        for row, template in zip(rows, templates, strict=True):
            if template is not None:
                row[fid] = marker


#: Key of the sweep's ``TemplateLabels`` in the per-op path's field-name cache.
_LABELS_KEY = "replay_template_labels"


def carry_entry_templates(
    trace: Trace,
    layer_entry: Any,
    set_entry_field: Callable[[str, Any], None],
    field_cache: dict[Any, Any],
    parent_ref_cls: type[Any],
) -> None:
    """Per-op form of :func:`carry_templates` for traces off the column path.

    An op on a shared row store has its set templates held in the side
    columns; a record with no shared row store (a detached row, a test double)
    cannot hold them, so its templates are renamed now. The sweep shares one
    ``TemplateLabels``, kept in its field-name cache.
    """

    labels = field_cache.get(_LABELS_KEY)
    if labels is None or labels.mapping is not trace._raw_to_final_layer_labels:
        labels = field_cache[_LABELS_KEY] = template_labels(trace, parent_ref_cls)
    place = _row_place(layer_entry)
    side = None
    if place is not None:
        try:
            side = REPLAY_TEMPLATES.bind(place[0], labels)
        except TypeError:  # a detached single-row store holds no side columns
            side = None
    for name in TEMPLATE_FIELDS:
        template = record_member(layer_entry, name, None)
        if template is None:
            continue
        if side is not None:
            set_entry_field(name, REPLAY_TEMPLATES.hold(side, name, place[1], template))
        else:
            set_entry_field(name, resolve_template(template, labels, name))


def _row_place(record: Any) -> tuple[Any, int] | None:
    """Return ``(store, row)`` of a record that reads its fields from a row store.

    An ``Op`` holds its store in the ``_core`` slot; other facade records in
    their instance dict.
    """

    store = getattr(record, "_core", None)
    if store is not None:
        return store, record._row
    instance_dict = getattr(record, "__dict__", {})
    if CORE_KEY in instance_dict:
        return instance_dict[CORE_KEY], instance_dict[ROW_KEY]
    return None


def rename_replay_entry(
    trace: Trace,
    layer_entry: Any,
    set_entry_field: Callable[[str, Any], None],
    field_cache: dict[Any, Any],
    parent_ref_cls: type[Any],
) -> None:
    """Step 9 for one op off the column path: the replay fields.

    Renames the labels in the op's edge uses and edit records now and carries
    its templates (:func:`carry_entry_templates`).

    Args:
        trace: Trace being postprocessed.
        layer_entry: Op (or test double) whose fields are renamed.
        set_entry_field: Writer for one field of ``layer_entry``.
        field_cache: Dataclass field-name cache shared across the rename sweep.
        parent_ref_cls: The ``ParentRef`` class.
    """

    mapping = trace._raw_to_final_layer_labels
    for name in ("_edge_uses", "edits"):
        records = getattr(layer_entry, name, None)
        if records:
            set_entry_field(
                name, [_rename_label_dataclass(record, mapping, field_cache) for record in records]
            )
    carry_entry_templates(trace, layer_entry, set_entry_field, field_cache, parent_ref_cls)
