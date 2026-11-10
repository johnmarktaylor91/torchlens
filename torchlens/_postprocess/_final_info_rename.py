"""Step 9 by column: raw-to-final label renaming over the building rows.

The column form of ``labeling._replace_layer_names_for_layer_entry``. Each
function renames one field (or one small family) for every op, reading the
column the caller already took from the raw rows and writing the renamed
value back into each row. The renamed values are those of the per-op
function: a new container per cell, except that the members of one
equivalence class keep sharing one renamed set and the members of one
recurrence group one renamed list.

The caller passes in the ``ParentRef`` class and the conditional view
projection, so this module imports nothing from the package's import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import repeat
from typing import TYPE_CHECKING, Any

from ._conditional_child_rename import _rename_children_by_cond, _rename_elif_children
from ._label_rename import _rename_label_dataclass
from ._replay_template_cells import TemplateLabels, carry_templates, template_labels

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

_Rows = list[list[Any]]
_Lookup = Callable[[str], str]
_ProjectViews = Callable[[Any], tuple[Any, Any, Any]]

#: Label containers renamed with the lookup key of each label's op.
_OP_KEYED = (
    "parents",
    "root_ancestors",
    "children",
    "input_ancestors",
    "output_descendants",
    "internal_source_parents",
    "internal_source_ancestors",
)
#: Renamed fields the per-op function reads by attribute name.
RENAMED_ATTRS = ("_edge_uses", "edits", "out_versions_by_child")
#: Renamed fields the per-op function reads by member name.
RENAMED_MEMBERS = (
    *_OP_KEYED,
    "conditional_entry_children",
    "equivalent_ops",
    "recurrent_ops",
    "buffer_source",
    "parent_arg_positions",
    "conditional_arm_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
    "args_template",
    "kwargs_template",
)


def _rename_containers(rows: _Rows, fid: int, column: list[Any], lookup: _Lookup) -> None:
    """Rename one column of label containers: a new container of the same type per cell."""

    for row, labels in zip(rows, column, strict=True):
        if not labels:
            continue
        cls = labels.__class__
        if cls is list or (cls is not set and isinstance(labels, list)):
            row[fid] = list(map(lookup, labels))
        else:
            row[fid] = cls(map(lookup, labels))


def _rename_equivalent_ops(rows: _Rows, fid: int, column: list[Any], lookup: _Lookup) -> None:
    """Rename each equivalence class once; its members keep sharing one container.

    Capture hands every member of a class the same set, so the renamed set is
    remembered by the identity of the raw one. ``column`` keeps every raw set
    alive for the pass, so an identity cannot be reused.
    """

    renamed_by_id: dict[int, Any] = {}
    for row, group, identity in zip(rows, column, map(id, column), strict=True):
        if not group:
            continue
        if identity in renamed_by_id:
            row[fid] = renamed_by_id[identity]
        else:
            row[fid] = renamed_by_id[identity] = group.__class__(map(lookup, group))


def _rename_recurrent_ops(
    rows: _Rows, fid: int, column: list[Any], raw_labels: list[Any], lookup: _Lookup
) -> None:
    """Rename each recurrence group once; its members share one renamed list.

    Loop detection hands every member its own equal list, so the renamed list
    is remembered under every raw member label at the group's first op.
    """

    renamed_by_member: dict[Any, list[str]] = {}
    for row, group, raw_label in zip(rows, column, raw_labels, strict=True):
        if not group:
            continue
        cls = group.__class__
        if cls is not list and not isinstance(group, list):
            row[fid] = cls(map(lookup, group))  # legacy non-list value: no sharing
            continue
        if raw_label in renamed_by_member:
            renamed = renamed_by_member[raw_label]
        else:
            renamed = list(map(lookup, group))
            for member in group:
                renamed_by_member[member] = renamed
        row[fid] = renamed


def _rename_buffer_sources(
    rows: _Rows, fid: int, column: list[Any], mapping: dict[str, str]
) -> None:
    """Rename the one label each buffer's ``buffer_source`` holds."""

    for row, label in zip(rows, column, strict=True):
        if label is None:
            continue
        if (label.__class__ is str or isinstance(label, str)) and label in mapping:
            row[fid] = mapping[label]


def _rename_arg_positions(column: list[Any], mapping: dict[str, str]) -> None:
    """Rename the parent labels of every ``parent_arg_positions`` dict in place."""

    for positions in column:
        if not positions:
            continue
        for kind in ("args", "kwargs"):
            labels = positions[kind]
            if labels.__class__ is dict:
                for key in labels:
                    labels[key] = mapping[labels[key]]
            else:
                for key, label in labels.items():
                    labels[key] = mapping[label]


def _rename_out_versions(rows: _Rows, fid: int, column: list[Any], mapping: dict[str, str]) -> None:
    """Rename the child-label keys of every ``out_versions_by_child`` dict."""

    for row, versions in zip(rows, column, strict=True):
        if versions:
            row[fid] = {mapping[child]: version for child, version in versions.items()}


def _rename_conditional_children(
    rows: _Rows,
    fids: dict[str, int],
    columns: dict[str, list[Any]],
    mapping: dict[str, str],
    project_views: _ProjectViews,
) -> None:
    """Rename the conditional child facts and re-project the three views.

    An op that carries ``conditional_arm_children`` gets it renamed and its
    then, elif and else views projected from the renamed value (an empty one
    projects three empty views); an op without it gets each view that is set
    renamed on its own.
    """

    lookup = mapping.__getitem__
    f_arm = fids["conditional_arm_children"]
    f_then = fids["conditional_then_children"]
    f_elif = fids["conditional_elif_children"]
    f_else = fids["conditional_else_children"]
    for row, arms, then, elifs, other in zip(
        rows,
        columns["conditional_arm_children"],
        columns["conditional_then_children"],
        columns["conditional_elif_children"],
        columns["conditional_else_children"],
        strict=True,
    ):
        if arms is None:
            if elifs:
                row[f_elif] = _rename_elif_children(elifs, mapping)
            if then:
                row[f_then] = list(map(lookup, then))
            if other:
                row[f_else] = list(map(lookup, other))
        elif arms.__class__ is dict and not arms:
            row[f_arm], row[f_then], row[f_elif], row[f_else] = {}, [], {}, []
        else:
            renamed = row[f_arm] = _rename_children_by_cond(arms, mapping)
            row[f_then], row[f_elif], row[f_else] = project_views(renamed)


def _rename_replay_columns(
    rows: _Rows,
    fids: dict[str, int],
    columns: dict[str, list[Any]],
    mapping: dict[str, str],
    labels: TemplateLabels,
) -> None:
    """Rename labels in the edge uses and edit records; carry the replay templates.

    The templates are not renamed here: each set template cell becomes a carried
    cell that renames it, and binds its display keys, on first read.
    """

    cache: dict[Any, Any] = {}
    for name in ("_edge_uses", "edits"):
        fid = fids[name]
        for row, records in zip(rows, columns[name], strict=True):
            if records:
                row[fid] = list(
                    map(_rename_label_dataclass, records, repeat(mapping), repeat(cache))
                )
    carry_templates(rows, fids, columns, labels)


def rename_label_columns(
    trace: Trace,
    table: Any,
    columns: dict[str, list[Any]],
    parent_ref_cls: type[Any],
    project_views: _ProjectViews,
) -> None:
    """Replace the raw labels in every op's fields with final labels.

    Parameters
    ----------
    trace:
        Trace being postprocessed; supplies the three raw-to-final mappings.
    table:
        The ops as raw rows: ``rows`` is each op's list of cells in op order,
        ``fids`` the field id by column name.
    columns:
        The columns named in ``RENAMED_ATTRS`` and ``RENAMED_MEMBERS`` plus
        ``_label_raw``, each read in op order before any write.
    parent_ref_cls:
        The ``ParentRef`` class.
    project_views:
        Projects the then, elif and else views of a ``conditional_arm_children``.
    """

    rows, fids = table.rows, table.fids
    mapping = trace._raw_to_final_layer_labels
    layer_mapping = trace._raw_to_final_parent_layer_labels
    op_lookup = trace._raw_to_final_op_labels.__getitem__
    for name in _OP_KEYED:
        _rename_containers(rows, fids[name], columns[name], mapping.__getitem__)
    name = "conditional_entry_children"
    _rename_containers(rows, fids[name], columns[name], layer_mapping.__getitem__)
    _rename_equivalent_ops(rows, fids["equivalent_ops"], columns["equivalent_ops"], op_lookup)
    _rename_recurrent_ops(
        rows, fids["recurrent_ops"], columns["recurrent_ops"], columns["_label_raw"], op_lookup
    )
    _rename_buffer_sources(rows, fids["buffer_source"], columns["buffer_source"], mapping)
    _rename_arg_positions(columns["parent_arg_positions"], mapping)
    name = "out_versions_by_child"
    _rename_out_versions(rows, fids[name], columns[name], mapping)
    _rename_conditional_children(rows, fids, columns, layer_mapping, project_views)
    labels = template_labels(trace, parent_ref_cls)
    _rename_replay_columns(rows, fids, columns, mapping, labels)
